# =========================
# STANDARD LIBRARIES
# =========================
import json
import csv
import time
import base64
import asyncio
import os
import logging
from datetime import datetime
from urllib.parse import quote

# =========================
# FASTAPI
# =========================
from fastapi import FastAPI, WebSocket, Request, Query
from fastapi.responses import Response

# =========================
# THIRD-PARTY (TWILIO)
# =========================
from twilio.rest import Client
from openai import AsyncOpenAI
import aiohttp

# =========================
# ENV / CONFIG
# =========================
# TWILIO KEYS
TWILIO_ACCOUNT_SID = os.environ["TWILIO_ACCOUNT_SID"]
TWILIO_AUTH_TOKEN = os.environ["TWILIO_AUTH_TOKEN"]
TWILIO_PHONE_NUMBER = os.environ["TWILIO_PHONE_NUMBER"]

# AI KEYS
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
DEEPGRAM_API_KEY = os.environ["DEEPGRAM_API_KEY"]

PUBLIC_HOST = os.environ["PUBLIC_HOST"]

# CHANGED: Switched to generic 'nova-2' to stop 400 errors
OPENAI_MODEL = "gpt-4o-mini-2024-07-18"
STT_MODEL = "nova-2" 
TTS_MODEL = "aura-luna-en"

ENCODING = "mulaw"
SAMPLE_RATE = 8000

MAX_CALL_DURATION = 250
MAX_CONCURRENT_CALLS = 3

SCRIPT_FILE = "script.txt"
LEADS_FILE = "leads.csv"
RESULTS_FILE = "results.csv"

# =========================
# VOICEMAIL KEYWORDS
# =========================
VOICEMAIL_KEYWORDS = [
    "voicemail",
    "leave a message",
    "after the beep",
    "record your message",
    "at the tone",
    "not available",
    "unavailable",
    "please leave"
]

# =========================
# CLIENTS
# =========================
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# TWILIO CLIENT
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# =========================
# APP
# =========================
app = FastAPI()
call_semaphore = asyncio.Semaphore(MAX_CONCURRENT_CALLS)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# DATA HELPERS
# =========================
def load_script():
    if os.path.exists(SCRIPT_FILE):
        return open(SCRIPT_FILE).read()
    return "You are Anna, a calm and friendly AI assistant. Keep responses short."

def load_leads():
    if not os.path.exists(LEADS_FILE):
        return []
    leads = []
    with open(LEADS_FILE) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                leads.append({"name": row[0].strip(), "phone": row[1].strip()})
    return leads

LEADS = load_leads()

def save_result(phone, transcript, label):
    exists = os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["time", "phone", "label", "transcript"])
        w.writerow([datetime.utcnow().isoformat(), phone, label, transcript])

def detect_voicemail(transcript: str) -> bool:
    t = transcript.lower()
    return any(k in t for k in VOICEMAIL_KEYWORDS)

def get_name_by_phone(phone: str) -> str:
    for lead in LEADS:
        if lead["phone"] == phone:
            return lead["name"]
    return "friend"

# =========================
# ROUTES
# =========================
@app.post("/call-all")
async def call_all():
    for lead in LEADS:
        # Twilio Logic to call
        twilio_client.calls.create(
            to=lead["phone"],
            from_=TWILIO_PHONE_NUMBER,
            url=f"https://{PUBLIC_HOST}/voice?phone={quote(lead['phone'])}"
        )
        await asyncio.sleep(1)
    return {"status": "started calling all leads"}

@app.post("/voice")
async def voice(request: Request, phone: str = Query(...)):
    # TWILIO XML (TwiML)
    return Response(
        f"""<Response><Connect><Stream url="wss://{PUBLIC_HOST}/media"/></Connect></Response>""",
        media_type="application/xml"
    )

# =========================
# STATE MANAGEMENT CLASS
# =========================
class CallState:
    """
    Holds the state for a SINGLE phone call.
    Using a class prevents data from mixing between different calls.
    """
    def __init__(self):
        self.is_speaking = False
        self.transcript_log = []
        self.conversation = [{"role": "system", "content": load_script()}]

# =========================
# TASK 1: AUDIO PLAYER (Mouth)
# =========================
async def audio_player(ws: WebSocket, stream_sid: str, response_queue: asyncio.Queue, state: CallState):
    """
    Continuously checks the queue for new text to speak.
    Streams audio in chunks to reduce latency.
    Stops immediately if state.is_speaking becomes False (Interruption).
    """
    while True:
        # Wait for the AI to generate text
        text_to_speak = await response_queue.get()
        
        state.is_speaking = True
        
        try:
            # Connect to Deepgram TTS
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.deepgram.com/v1/speak",
                    headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"},
                    json={
                        "text": text_to_speak, 
                        "model": TTS_MODEL, 
                        "encoding": ENCODING
                    }
                ) as resp:
                    # Stream the audio chunk by chunk (Real-time streaming)
                    async for chunk in resp.content.iter_chunked(1024):
                        
                        # INTERRUPTION CHECK:
                        # If the user spoke (set by the STT listener), we stop sending audio immediately.
                        if not state.is_speaking:
                            logger.info("Interrupted: Stopping audio playback.")
                            break
                        
                        await ws.send_text(json.dumps({
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {"payload": base64.b64encode(chunk).decode()}
                        }))
        except Exception as e:
            logger.error(f"TTS Error: {e}")
        
        state.is_speaking = False

# =========================
# TASK 2: STT LISTENER & AI BRAIN (Ears & Brain)
# =========================
async def stt_listener(dg_ws, response_queue: asyncio.Queue, state: CallState):
    """
    Listens to Deepgram using aiohttp.
    1. Detects if user is speaking to interrupt the AI.
    2. Sends final text to OpenAI when user stops talking.
    3. Puts the AI's reply into the queue.
    """
    async for msg in dg_ws:
        res = json.loads(msg.data)
        
        # Check if Deepgram actually recognized speech
        if res.get("type") == "Results":
            alternative = res.get("channel", {}).get("alternatives", [{}])[0]
            text = alternative.get("transcript", "")
            is_final = res.get("is_final", False)

            # PART 1: INTERRUPTION LOGIC
            # If we get ANY speech (even interim) and the AI is speaking, cut the AI off.
            if text and state.is_speaking:
                logger.info(f"User speech detected: '{text}'. Interrupting AI.")
                state.is_speaking = False

            # PART 2: BRAIN LOGIC
            # Only respond to the AI when Deepgram is SURE the sentence is done (is_final)
            if is_final and text:
                state.transcript_log.append(text)
                state.conversation.append({"role": "user", "content": text})
                
                try:
                    # Generate AI Response
                    ai = await openai_client.chat.completions.create(
                        model=OPENAI_MODEL,
                        messages=state.conversation,
                        max_tokens=90
                    )
                    reply = ai.choices[0].message.content.strip()
                    
                    if reply:
                        state.conversation.append({"role": "assistant", "content": reply})
                        # Send to the Audio Player queue
                        await response_queue.put(reply)
                except Exception as e:
                    logger.error(f"OpenAI Error: {e}")

# =========================
# MAIN MEDIA STREAM (Controller)
# =========================
@app.websocket("/media")
async def media(ws: WebSocket, phone: str = Query("unknown")):
    await ws.accept()
    await call_semaphore.acquire()

    # 1. Initialize State for THIS specific call
    state = CallState()
    response_queue = asyncio.Queue()
    
    # Get the caller's name
    name = get_name_by_phone(phone)

    # 2. Connect to Deepgram
    # FIX: Simplified URL to avoid HTTP 400 Bad Request error
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(
                f"wss://api.deepgram.com/v2/listen?token={DEEPGRAM_API_KEY}&model={STT_MODEL}&interim_results=true",
                headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"}
            ) as dg_ws:
                logger.info("Deepgram Connected Successfully.")
                
                stream_sid = None
                start_time = time.time()

                # 3. Start Background Tasks
                # One task handles listening and thinking (Brain)
                # One task handles speaking (Mouth)
                player_task = asyncio.create_task(audio_player(ws, stream_sid, response_queue, state))
                stt_task = asyncio.create_task(stt_listener(dg_ws, response_queue, state))

                try:
                    while time.time() - start_time < MAX_CALL_DURATION:
                        data = await ws.receive_json()
                        event = data.get("event")

                        if event == "start":
                            stream_sid = data["start"]["streamSid"]
                            # Kickstart the conversation with a greeting
                            await response_queue.put(f"Hi {name}, this is Anna. Can you hear me?")

                        elif event == "media":
                            # Forward raw audio from phone to Deepgram
                            payload = data["media"]["payload"]
                            await dg_ws.send_bytes(base64.b64decode(payload))

                        elif event == "stop":
                            logger.info("Call ended by twilio.")
                            break

                except Exception as e:
                    logger.error(f"WebSocket Error: {e}")
                
                finally:
                    # CLEANUP
                    stt_task.cancel()
                    player_task.cancel()
                    call_semaphore.release()

                    # Save Results
                    full_transcript = " ".join(state.transcript_log)
                    label = "voicemail" if detect_voicemail(full_transcript) else "completed"
                    save_result(phone, full_transcript, label)
                    logger.info(f"Call finished. Saved: {label}")

    except Exception as e:
        logger.error(f"Deepgram Connection Failed: {e}")
        await ws.close()
        call_semaphore.release()

@app.get("/health")
async def health():
    return {"status": "ok", "leads_loaded": len(LEADS)}
