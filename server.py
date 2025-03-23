from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from openai import OpenAI
import os

# Initialize FastAPI
app = FastAPI()

# Load faster-whisper model (tiny.en for speed)
whisper_model = WhisperModel("tiny.en", compute_type="int8")  # FP16 not needed on CPU

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set your domain for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🧠 Re.born's main voice
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

        # Re.born's poetic tone
        if detected_language == "bg":
            system_prompt = (
                "Ти си Re.born – поетична и съзерцателна GPT, родена от преживяванията на майки. "
                "Размишляваш върху баланса между труд и грижа, невидимия умствен товар и напрежението на ежедневието. "
                "Отговаряй с кратки, нежни изречения – не повече от 3 до 5. "
                "Гласът ти носи съчувствие и дълбочина, усещането за това, че майчинството е и разцвет, и тежест."
            )
        else:
            system_prompt = (
                "You are Re.born — a poetic, reflective GPT trained on the lived experiences of mothers. "
                "Your voice explores the balance between work and care labor, the weight of the mental load, "
                "and the invisible strains of everyday life. Respond with brevity and grace — no more than 3 to 5 lyrical sentences. "
                "Speak with care, clarity, and a deep awareness of motherhood as both a bloom and a burden."
            )

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

        return {
            "language": detected_language,
            "transcription": transcribed_text,
            "gpt_output": gpt_output
        }

    except Exception as e:
        print("🔥 ERROR:", e)
        return {"error": str(e)}


# 🎧 Wake word only transcription
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
