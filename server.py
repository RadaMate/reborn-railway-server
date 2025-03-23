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
    # Read uploaded file into memory
    audio_bytes = await file.read()
    audio_file = BytesIO(audio_bytes)

    # Transcribe audio using whisper
    result = model.transcribe(audio_file)
    transcribed_text = result["text"]

    # Send transcription to OpenAI (GPT-4 or GPT-3.5)
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are Re.born, a reflective conversational agent."},
            {"role": "user", "content": transcribed_text}
        ]
    )

    gpt_output = response["choices"][0]["message"]["content"]

    return {
        "transcription": transcribed_text,
        "gpt_output": gpt_output
    }
