import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

class Config:
    # Provider Selection
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "lm_studio").lower()

    # LM Studio Settings
    LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
    LM_STUDIO_API_KEY = os.getenv("LM_STUDIO_API_KEY", "not-needed")
    LM_STUDIO_MODEL_NAME = os.getenv("LM_STUDIO_MODEL_NAME")

    # NVIDIA Settings
    NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
    NVIDIA_MODEL_NAME = os.getenv("NVIDIA_MODEL_NAME")

    # Whisper Settings
    WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "large-v3") # Default to large-v3 for better accuracy
    WHISPER_INITIAL_PROMPT = os.getenv("WHISPER_INITIAL_PROMPT") # New: Initial prompt for Whisper

    # Project Paths
    ROOT_DIR = Path(__file__).parent.parent
    INPUT_VIDEO_PATH = ROOT_DIR / os.getenv("INPUT_VIDEO_PATH", "data/test.mp4")
    OUTPUT_DIR = ROOT_DIR / os.getenv("OUTPUT_DIR", "output")
    TEMP_DIR = ROOT_DIR / os.getenv("TEMP_DIR", "temp")
    
    # Ensure directories exist
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    TEMP_DIR.mkdir(exist_ok=True, parents=True)

    # Transcribe Settings
    TRANSCRIBE_CHUNK_SIZE = int(os.getenv("TRANSCRIBE_CHUNK_SIZE", "2000"))
    TRANSCRIBE_CHUNK_OVERLAP = int(os.getenv("TRANSCRIBE_CHUNK_OVERLAP", "200"))

    # Clip Duration Settings
    MIN_CLIP_DURATION = int(os.getenv("MIN_CLIP_DURATION", "10"))
    MAX_CLIP_DURATION = int(os.getenv("MAX_CLIP_DURATION", "120"))
