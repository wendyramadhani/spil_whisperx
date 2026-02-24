import os
import shutil
import torch
import whisperx
import uvicorn
import asyncio
import random
import uuid
import gc
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
from whisperx.diarize import DiarizationPipeline
from dotenv import load_dotenv

# --- 1. KUNCI KONSISTENSI AI (Mencegah Halusinasi Acak) ---
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --- 2. LOAD ENVIRONMENT VARIABLES ---
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
WHISPER_LANG = os.getenv("WHISPER_LANG", "id")
BATCH_SIZE = int(os.getenv("WHISPER_BATCH_SIZE", 16))
MIN_SPEAKERS = int(os.getenv("WHISPER_MIN_SPEAKERS", 2))
MAX_SPEAKERS = int(os.getenv("WHISPER_MAX_SPEAKERS", 2))
TEMPERATURE = float(os.getenv("WHISPER_TEMPERATURE", 0.0))
INITIAL_PROMPT = os.getenv("WHISPER_PROMPT", "")

# --- 3. SETUP DEVICE & PARAMETER GPU ---
has_gpu = torch.cuda.is_available()
device = "cuda" if has_gpu else "cpu"

if has_gpu:
    compute_type = "float16" # A100 wajib float16 biar ngebut
    model_name = "large-v2"  # Model terbaik
    print(f"🚀 MODE SERVER: GPU {torch.cuda.get_device_name(0)} Aktif!")
else:
    compute_type = "int8"
    model_name = "tiny" 
    print("⚠️ WARNING: Server jalan di CPU. Cek driver NVIDIA anda!")

# Tampilkan Konfigurasi Aktif
print("\n=== KONFIGURASI SERVER AKTIF ===")
print(f"Bahasa      : {WHISPER_LANG}")
print(f"Batch Size  : {BATCH_SIZE}")
print(f"Speakers    : {MIN_SPEAKERS} sampai {MAX_SPEAKERS}")
print(f"Temperature : {TEMPERATURE}")
if INITIAL_PROMPT:
    print(f"Prompt Aktif: {INITIAL_PROMPT[:50]}...")
print("================================\n")

# Global Variables
ml_models = {}
gpu_lock = asyncio.Lock()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- A. Menyiapkan Opsi Anti-Halusinasi ---
    asr_options = {"temperatures": [TEMPERATURE]}
    if INITIAL_PROMPT:
        asr_options["initial_prompt"] = INITIAL_PROMPT

    # --- B. Load Model Whisper ---
    print(f"1. Loading Whisper ({model_name})...")
    model = whisperx.load_model(
        model_name, 
        device, 
        compute_type=compute_type, 
        asr_options=asr_options
    )
    
    # --- C. Load Model Alignment ---
    print(f"2. Loading Alignment ({WHISPER_LANG})...")
    model_a, metadata = whisperx.load_align_model(language_code=WHISPER_LANG, device=device)
    
    # --- D. Load Diarization ---
    print("3. Loading Diarization (Pyannote)...")
    diarize_model = None
    if HF_TOKEN:
        try:
            diarize_model = DiarizationPipeline(device=device)
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
    
    print("\n✅ SERVER SIAP MENERIMA REQUEST!")
    yield
    
    # Cleanup saat server mati
    ml_models.clear()
    gc.collect()
    if device == "cuda": 
        torch.cuda.empty_cache()

app = FastAPI(title="WhisperX API Server", lifespan=lifespan)

@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    if gpu_lock.locked():
        print("⏳ GPU sedang sibuk, request masuk antrean...")
        
    async with gpu_lock: 
        # Menggunakan UUID agar nama file unik (Mencegah file tertimpa jika request barengan)
        unik_id = uuid.uuid4().hex
        temp_filename = f"temp_{unik_id}_{file.filename}"
        audio = None # Inisialisasi awal untuk cleanup
        
        try:
            # Simpan file sementara
            with open(temp_filename, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            print(f"🎤 Processing Start: {file.filename}")
            
            # Load Audio
            audio = whisperx.load_audio(temp_filename)
            
            # 1. Transcribe
            result = ml_models["transcribe"].transcribe(
                audio, 
                batch_size=BATCH_SIZE, 
                language=WHISPER_LANG
            )
            
            # 2. Align
            result = whisperx.align(
                result["segments"], 
                ml_models["align"], 
                ml_models["metadata"], 
                audio, 
                device, 
                return_char_alignments=False
            )
            
            # 3. Diarization (Pakai Min/Max dari env)
            if ml_models.get("diarize"):
                print("   ...Diarization Running...")
                diarize_segments = ml_models["diarize"](
                    audio, 
                    min_speakers=MIN_SPEAKERS, 
                    max_speakers=MAX_SPEAKERS
                )
                result = whisperx.assign_word_speakers(diarize_segments, result)
            
            # 4. Formatting Output
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
            # --- PEMBERSIHAN EKSTREM ---
            # 1. Hapus file audio fisik
            if os.path.exists(temp_filename): 
                os.remove(temp_filename)
                
            # 2. Hapus variabel besar dari memori Python
            try:
                del audio
                del result
            except:
                pass
                
            # 3. Kuras sisa cache GPU dan RAM
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            
            print(f"✅ Processing Done: {file.filename} | Memory Cleared.")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 