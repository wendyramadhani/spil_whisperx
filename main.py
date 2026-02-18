import os
import shutil
import torch
import whisperx
import uvicorn
import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
from whisperx.diarize import DiarizationPipeline
from dotenv import load_dotenv

load_dotenv()

# --- KONFIGURASI SERVER ---
# Masukkan Token Hugging Face
HF_TOKEN = os.getenv("HF_TOKEN")

# Setup Device
has_gpu = torch.cuda.is_available()
device = "cuda" if has_gpu else "cpu"

if has_gpu:
    compute_type = "float16" # A100 wajib float16 biar ngebut
    batch_size = 16 
    model_name = "large-v2"  # Model terbaik
    print(f"🚀 MODE SERVER: GPU {torch.cuda.get_device_name(0)} Aktif!")
else:
    # Fallback kalau GPU tidak terdeteksi (Jaga-jaga)
    compute_type = "int8"
    batch_size = 1
    model_name = "tiny" 
    print("⚠️ WARNING: Server jalan di CPU. Cek driver NVIDIA anda!")

# Global Variables
ml_models = {}

# KUNCI PENGAMAN GPU (Agar tidak crash kalau banyak request barengan)
gpu_lock = asyncio.Lock()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- 1. Load Whisper ---
    print(f"1. Loading Whisper ({model_name})...")
    model = whisperx.load_model(model_name, device, compute_type=compute_type)
    
    # --- 2. Load Alignment (PENTING: Ganti ke 'id' untuk Indonesia) ---
    print("2. Loading Alignment (Bahasa Indonesia)...")
    # Menggunakan 'id' agar timestamp lebih akurat untuk percakapan indo
    model_a, metadata = whisperx.load_align_model(language_code="en", device=device)
    
    # --- 3. Load Diarization ---
    print("3. Loading Diarization (Pyannote)...")
    diarize_model = None
    if HF_TOKEN:
        try:
            diarize_model = DiarizationPipeline(use_auth_token=HF_TOKEN, device=device)
            print("✅ Diarization Loaded!")
        except Exception as e:
            print(f"⚠️ Gagal load Diarization: {e}")
    else:
        print("⚠️ Token HF tidak ditemukan, Diarization OFF.")

    # Simpan ke memori global
    ml_models["transcribe"] = model
    ml_models["align"] = model_a
    ml_models["metadata"] = metadata
    ml_models["diarize"] = diarize_model
    
    print("✅ SERVER SIAP MENERIMA REQUEST!")
    yield
    
    # Cleanup saat server mati
    ml_models.clear()
    if device == "cuda": 
        torch.cuda.empty_cache()

app = FastAPI(title="WhisperX API Server", lifespan=lifespan)

@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    # Kunci Pintu: Hanya boleh 1 proses dalam 1 waktu
    if gpu_lock.locked():
        print("⏳ GPU sedang sibuk, request masuk antrean...")
        
    async with gpu_lock: # <--- Request lain harus nunggu di sini sampai ini selesai
        temp_filename = f"temp_{file.filename}"
        try:
            # Ambil model dari memory
            model = ml_models["transcribe"]
            model_a = ml_models["align"]
            metadata = ml_models["metadata"]
            diarize_model = ml_models.get("diarize")

            # Simpan file sementara
            with open(temp_filename, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            print(f"🎤 Processing Start: {file.filename}")
            
            # A. Transcribe
            audio = whisperx.load_audio(temp_filename)
            result = model.transcribe(audio, batch_size=batch_size)
            
            # B. Align (Pakai model 'id' yang sudah diload)
            # Note: Jika audio terdeteksi bukan 'id', idealnya load ulang model align.
            # Tapi demi kecepatan server, kita paksa pakai model 'id' default.
            result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
            
            # C. Diarization
            if diarize_model:
                print("   ...Diarization Running...")
                diarize_segments = diarize_model(audio)
                result = whisperx.assign_word_speakers(diarize_segments, result)
            
            # D. Formatting
            formatted_output = []
            for segment in result["segments"]:
                formatted_output.append({
                    "speaker": segment.get("speaker", "Unknown"),
                    "content": segment["text"].strip()
                })
                
            return formatted_output

        except Exception as e:
            print(f"❌ Error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
            
        finally:
            # Bersih-bersih file
            if os.path.exists(temp_filename): 
                os.remove(temp_filename)
            
            # Bersih-bersih VRAM GPU (WAJIB DI SERVER)
            if device == "cuda":
                torch.cuda.empty_cache()
            
            print(f"✅ Processing Done: {file.filename}")

if __name__ == "__main__":
    # Reload=False untuk production
    # Workers=1 karena WhisperX tidak thread-safe (kita pakai Lock di atas)
    uvicorn.run(app, host="0.0.0.0", port=8000)