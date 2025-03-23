from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import whisper
from openai import OpenAI
import os

# Initialize FastAPI app
app = FastAPI()

# Load lightweight Whisper model
model = whisper.load_model("tiny")

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Allow web access from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with domain for security in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🎙️ Main interaction route – GPT response
@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        with open("temp.wav", "wb") as f:
            f.write(audio_bytes)

        # Transcribe audio + detect language
        result = model.transcribe("temp.wav")
        transcribed_text = result["text"]
        detected_language = result["language"]

        # Language-aware poetic system prompt
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

        # Generate GPT response with tighter max_tokens
        response = client.chat.completions.create(
            model="ft:gpt-3.5-turbo-1106:re-born::BEK8G87T",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcribed_text}
            ],
            max_tokens=120  # keep it quick + poetic
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

# 🛎 Wake word detection-only route
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
