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

@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
    try:
        # Save uploaded audio file
        audio_bytes = await file.read()
        with open("temp.wav", "wb") as f:
            f.write(audio_bytes)

        # Transcribe and detect language
        result = model.transcribe("temp.wav")
        transcribed_text = result["text"]
        detected_language = result["language"]

        # Set system prompt based on language
        if detected_language == "bg":
            system_prompt = "Ти си Re.born – поетична, съзерцателна GPT, обучена да размишлява за майчинството, паметта и работата. Отговаряй на български език с нежност и дълбочина."
        else:
            system_prompt = "You are Re.born — a poetic, reflective GPT trained to explore motherhood, memory, and metamorphosis. Respond in English with care and imagination."

        # Generate GPT response using fine-tuned model
        response = client.chat.completions.create(
            model="ft:gpt-3.5-turbo-1106:re-born::BEK8G87T",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcribed_text}
            ]
        )

        gpt_output = response.choices[0].message.content

        return {
            "language": detected_language,
            "transcription": transcribed_text,
            "gpt_output": gpt_output
        }

    except Exception as e:
        print("🔥 ERROR:", e)
        return {"error": str(e)}
