import os
import sys
import moviepy as mp
from pathlib import Path
# import whisper # Original whisper library
import torch
import json

from tqdm import tqdm
from src.config import Config # Import Config to access WHISPER_INITIAL_PROMPT

# In uv, sys.executable is usually .venv\Scripts\python.exe
# We want to get to .venv\Lib\site-packages
executable_path = Path(sys.executable)
if "Scripts" in executable_path.parts:
    venv_base = executable_path.parent.parent
else:
    venv_base = executable_path.parent # Fallback for different structures

site_packages = venv_base / "Lib" / "site-packages"
nvidia_dir = site_packages / "nvidia"

# Folders to check
cuda_paths = [
    nvidia_dir / "cublas" / "bin",
    nvidia_dir / "cudnn" / "bin",
    # Sometimes needed for newer CTranslate2 versions
    nvidia_dir / "cuda_runtime" / "bin"
]

print("--- Checking for Faster Whisper DLLs ---")
for p in cuda_paths:
    if p.exists():
        print(f"Found and adding to DLL path: {p}")
        os.add_dll_directory(str(p))
        # Fallback: Also add to environment PATH for older DLL loading logic
        os.environ["PATH"] = str(p) + os.pathsep + os.environ["PATH"]
    else:
        print(f"NOT FOUND: {p}")

print("--- End of DLL Check ---")
# --- END OF WINDOWS DLL FIX ---

# Now it is safe to import Faster Whisper
from faster_whisper import WhisperModel

class AudioProcessor:
    def __init__(self, model_size: str, temp_dir: Path):
        self.model_size = model_size
        self.temp_dir = temp_dir
        # Detect if CUDA (NVIDIA GPU) is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"--- AudioProcessor using device: {self.device} ---")

        # Initialize Faster Whisper model once
        # compute_type="float16" for GPU, "int8" for CPU for better performance
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        print(f"--- Loading Faster Whisper model '{self.model_size}' on {self.device} with compute_type='{self.compute_type}' ---")
        self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)


    def extract_audio(self, video_path: Path) -> Path:
        """Extracts audio from video file. Skips if the file already exists."""
        audio_path = self.temp_dir / f"{video_path.stem}.mp3"
        
        if audio_path.exists():
            print(f"--- Audio file already exists: {audio_path.name}. Skipping extraction. ---")
            return audio_path

        print(f"--- Extracting audio to: {audio_path.name} ---")
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        video = mp.VideoFileClip(str(video_path))
        video.audio.write_audiofile(str(audio_path))
        return audio_path

    def transcribe(self, audio_path: Path) -> list:
        """Transcribes audio or loads existing transcript from JSON using Faster Whisper."""
        transcript_path = self.temp_dir / f"{audio_path.stem}_transcript.json"

        if transcript_path.exists():
            print(f"--- Transcript already exists: {transcript_path.name}. Loading from file. ---")
            with open(transcript_path, 'r', encoding='utf-8') as f:
                return json.load(f)

        print(f"--- Transcribing audio: {audio_path.name} using Faster Whisper ---")

        segments, info = self.model.transcribe(
            str(audio_path),
            language='th',
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            initial_prompt=Config.WHISPER_INITIAL_PROMPT
        )

        transcript = []

        # --- NEW PROGRESS BAR LOGIC ---
        # info.duration gives the total length of the audio file in seconds
        total_duration = round(info.duration, 2)

        # Wrap the iteration in a tqdm block
        with tqdm(total=total_duration, unit=" audio sec", desc="Transcribing") as pbar:
            previous_end = 0.0

            for segment in segments:
                transcript.append({
                    "text": segment.text,
                    "start": segment.start,
                    "end": segment.end
                })

                # Calculate how many seconds of audio this specific segment covers
                # and push the progress bar forward by that amount
                increment = segment.end - previous_end
                pbar.update(increment)
                previous_end = segment.end
        # --- END PROGRESS BAR LOGIC ---

        print(f"\n--- Saving transcript to: {transcript_path.name} ---")
        with open(transcript_path, 'w', encoding='utf-8') as f:
            json.dump(transcript, f, ensure_ascii=False, indent=2)

        return transcript
