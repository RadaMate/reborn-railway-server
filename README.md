# Re.born Voice Server (FastAPI + Whisper + GPT)

This is a deployable FastAPI server that:
- Accepts audio (wav) via POST
- Transcribes using Whisper (tiny model)
- Sends transcription to OpenAI (GPT-4 or GPT-3.5)
- Returns both transcription and GPT response

## 📡 API Endpoint

POST `/upload-audio/`
- Body: `multipart/form-data` with file=`.wav` or `.mp3`
- Returns: `{ "transcription": "...", "gpt_output": "..." }`

## 🌐 CORS Enabled

This server is ready to receive requests from web frontends like a mobile browser or web app.

## 🧠 Requirements

- Python 3.10+
- FFmpeg installed (needed by Whisper)
