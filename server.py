from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import whisper
import openai
from io import BytesIO
import os

app = FastAPI()

# Load whisper model (tiny to stay under Railway free tier limits)
model = whisper.load_model("tiny")

# Set OpenAI API key from env var
openai.api_key = os.getenv("OPENAI_API_KEY")

# Allow requests from anywhere (you can limit this later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
    audio_bytes = await file.read()

    # Save uploaded file to disk
    with open("temp.wav", "wb") as f:
        f.write(audio_bytes)

    # Transcribe using Whisper (file path expected!)
    result = model.transcribe("temp.wav")
    transcribed_text = result["text"]

    # Generate GPT response
    response = openai.ChatCompletion.create(
        model="gpt-4",  # or your custom model if fine-tuned
        messages=[
            {"role": "system", "content": "You are Re.born, a reflective, poetic agent."},
            {"role": "user", "content": transcribed_text}
        ]
    )

    gpt_output = response["choices"][0]["message"]["content"]

    return {
        "transcription": transcribed_text,
        "gpt_output": gpt_output
    }
