import os, asyncio, tempfile, traceback
from typing import Optional

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import whisper
from openai import AsyncOpenAI, APIError, APITimeoutError, RateLimitError
from pythonosc.udp_client import SimpleUDPClient

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# CORS (tighten origin if you want)
app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=False,
  allow_methods=["*"],
  allow_headers=["*"],
)

# Load Whisper once (use "tiny"/"base" depending on RAM/latency)
model = whisper.load_model("base")

# Async OpenAI client
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# OSC client (best-effort)
try:
  osc_client: Optional[SimpleUDPClient] = SimpleUDPClient("127.0.0.1", 5005)
except Exception:
  osc_client = None

@app.get("/")
async def root():
  return {"ok": True}

@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
  if not file:
    return JSONResponse(status_code=400, content={"error": "no_file"})

  # Unique temp file per request (keep extension so ffmpeg decodes)
  suffix = os.path.splitext(file.filename or "")[1] or ".webm"
  tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
  tmp_path = tmp.name
  try:
    # Stream write to disk
    while True:
      chunk = await file.read(1024 * 1024)
      if not chunk:
        break
      tmp.write(chunk)
    tmp.flush(); tmp.close()

    # Whisper in thread (don’t block event loop)
    def do_transcribe(path: str) -> str:
      result = model.transcribe(path)
      return (result or {}).get("text", "").strip()

    try:
      text = await asyncio.wait_for(asyncio.to_thread(do_transcribe, tmp_path), timeout=60.0)
    except asyncio.TimeoutError:
      return JSONResponse(status_code=504, content={"error": "transcription_timeout"})
    except Exception as e:
      print("Whisper error:", e); traceback.print_exc()
      return JSONResponse(status_code=500, content={"error": "transcription_failed"})

    # OpenAI (async) with timeouts + friendly fallbacks
    gpt_output = ""
    if text:
      try:
        resp = await asyncio.wait_for(
          client.chat.completions.create(
            model="gpt-4o-mini",           # swap if you prefer another model
            messages=[{"role": "user", "content": text}],
            temperature=0.7,
          ),
          timeout=40.0
        )
        gpt_output = (resp.choices[0].message.content or "").strip()
      except RateLimitError:
        gpt_output = "I'm being rate-limited right now. Please try again in a moment."
      except APITimeoutError:
        gpt_output = "The AI took too long to respond. Let's try again."
      except APIError as e:
        print("OpenAI API error:", e)
        gpt_output = "I hit an API error. Please try again."
      except asyncio.TimeoutError:
        gpt_output = "The AI took too long to respond. Let's try again."
      except Exception as e:
        print("OpenAI unknown error:", e); traceback.print_exc()
        gpt_output = "Something went wrong generating a reply."

    # OSC send (don’t crash request)
    try:
      if osc_client and gpt_output:
        osc_client.send_message("/chat_output", gpt_output)
    except Exception as e:
      print("OSC send failed:", e)

    return {"transcription": text or "", "gpt_output": gpt_output or ""}

  finally:
    try: os.remove(tmp_path)
    except Exception: pass
