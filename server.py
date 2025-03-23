from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import whisper
from openai import OpenAI
import os

# Initialize FastAPI app
app = FastAPI()

# Load Whisper model for transcription and language detection
model = whisper.load_model("tiny")  # Keep it lightweight for Railway

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Allow requests from any origin (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/transcribe/")
async def transcribe_only(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        with open("temp_trigger.wav", "wb") as f:
            f.write(audio_bytes)

        result = model.transcribe("temp_trigger.wav")
        return {
            "language": result["language"],
            "text": result["text"]
        }

    except Exception as e:
        print("🔥 WAKE WORD ERROR:", e)
        return {"error": str(e)}
