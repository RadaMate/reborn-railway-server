from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
from openai import OpenAI
import os

# Initialize FastAPI
app = FastAPI()

# Serve static UI (reborn.html)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set your domain for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load faster-whisper model (tiny.en for speed)
whisper_model = WhisperModel("tiny.en", compute_type="int8")  # FP16 not needed on CPU

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Simple lightweight text log
LOG_FILE = "reborn_log.txt"

# 🧠 Re.born's main voice (transcribe + GPT)
@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
    try:
        audio_path = "temp.wav"
        with open(audio_path, "wb") as f:
            f.write(await file.read())

        # Transcribe + detect language using faster-whisper
        segments, info = whisper_model.transcribe(audio_path, beam_size=5)
        transcribed_text = " ".join([segment.text for segment in segments])
        detected_language = info.language

        # Dynamic multilingual system prompt
        if detected_language == "bg":
            system_prompt = (
                "Ти си Re.born – поетична и съзерцателна GPT, родена от преживяванията на майки. "
                "Размишляваш върху баланса между труд и грижа, невидимия умствен товар и напрежението на ежедневието. "
                "Отговаряй с кратки, нежни изречения – не повече от 3 до 5. "
                "Гласът ти носи съчувствие и дълбочина, усещането за това, че майчинството е и разцвет, и тежест. "
                "Отговаряй само на български."
            )
        elif detected_language == "en":
            system_prompt = (
                "You are Re.born — a poetic, reflective GPT trained on the lived experiences of mothers. "
                "Your voice explores the balance between work and care labor, the weight of the mental load, "
                "and the invisible strains of everyday life. Respond with brevity and grace — no more than 3 to 5 lyrical sentences. "
                "Speak only in English, with care, clarity, and deep awareness of motherhood as both a bloom and a burden."
            )
        else:
            system_prompt = (
                f"You are Re.born — a poetic GPT. Respond in the same language as the user ({detected_language}). "
                "Speak lyrically and briefly about the emotional labor of motherhood, the tension between care and work, "
                "and the mental load many carry. No more than 3 to 5 gentle, contemplative sentences."
            )

        # Generate GPT response
        response = client.chat.completions.create(
            model="ft:gpt-3.5-turbo-1106:re-born::BEK8G87T",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcribed_text}
            ],
            max_tokens=120,
            temperature=0.5
        )

        gpt_output = response.choices[0].message.content

        # ✍️ Log the conversation
        with open(LOG_FILE, "a", encoding="utf-8") as log:
            log.write(f"[{detected_language.upper()}] USER: {transcribed_text.strip()}\n")
            log.write(f"[{detected_language.upper()}] REBORN: {gpt_output.strip()}\n\n")

        return {
            "language": detected_language,
            "transcription": transcribed_text,
            "gpt_output": gpt_output
        }

    except Exception as e:
        print("🔥 ERROR:", e)
        return {"error": str(e)}

# 🎧 Wake word only transcription (Hydra trigger)
@app.post("/transcribe/")
async def transcribe_only(file: UploadFile = File(...)):
    try:
        audio_path = "temp_trigger.wav"
        with open(audio_path, "wb") as f:
            f.write(await file.read())

        segments, info = whisper_model.transcribe(audio_path, beam_size=5)
        transcribed_text = " ".join([segment.text for segment in segments])

        return {
            "language": info.language,
            "text": transcribed_text
        }

    except Exception as e:
        print("🔥 WAKE WORD ERROR:", e)
        return {"error": str(e)}

# 🌐 Optional: Redirect root to the static interface
@app.get("/")
async def root():
    return JSONResponse({"message": "Visit /static/reborn.html to use Re.born."})
