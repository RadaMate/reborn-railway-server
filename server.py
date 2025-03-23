from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import whisper
from openai import OpenAI
import os

app = FastAPI()
model = whisper.load_model("tiny")  # Keep memory usage low for Railway

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()

        # Save uploaded file temporarily
        with open("temp.wav", "wb") as f:
            f.write(audio_bytes)

        # Transcribe using Whisper
        result = model.transcribe("temp.wav")
        transcribed_text = result["text"]

        # Send transcription to OpenAI (GPT)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are Re.born, a poetic, reflective GPT trained to explore motherhood, memory, and metamorphosis."},
                {"role": "user", "content": transcribed_text}
            ]
        )

        gpt_output = response.choices[0].message.content

        return {
            "transcription": transcribed_text,
            "gpt_output": gpt_output
        }

    except Exception as e:
        print("🔥 ERROR:", e)
        return {"error": str(e)}
