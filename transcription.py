import os
import re
import sys
from typing import Dict
import whisperx
from dotenv import load_dotenv

# Load config from .env file
load_dotenv()

# Config & Constants
HF_TOKEN: str = os.getenv("HF_TOKEN", "")
AUDIO_SAMPLE_RATE: int = 16000
DEVICE: str = "cuda"
BATCH_SIZE: int = 4
COMPUTE_TYPE: str = "float16"
WHISPER_MODEL_SIZE: str = "large-v3"

# Fix TTRPG named entities mistranscribed by Whisper
DND_DICTIONARY: Dict[str, str] = {
    "kewerath": "Quewerath",
    "malekaj": "Malekai",
    "wrota baldyra": "Wrota Baldura",
    "fajrun": "Faerûn"
}

# Prompt injection to prime the model with game context
TRANCRIPTION_PROMPT: str = (
    "Gramy w Dungeons and Dragons. Miejsca: Kraina Ni-on. "
    "Pojęcia: rzut na percepcję, rzut obronny, klasa pancerza, "
    "obrażenia kłute, krytyk, mistrz gry."
)


def apply_custom_dictionary(text: str) -> str:
    """Replace common mispronouncments using the local dict."""
    for wrong_term, correct_term in DND_DICTIONARY.items():
        text = re.sub(wrong_term, correct_term, text, flags=re.IGNORECASE)
    return text


def get_last_timestamp(output_path: str) -> float:
    """Parse output file backward to find where the last session ended."""
    if not os.path.exists(output_path):
        return 0.0
    
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in reversed(lines):
                # Match [start_time - end_time] structure
                match = re.search(r'\[.*? - (\d+(?:\.\d+)?)s\]', line)
                if match:
                    return float(match.group(1))
    except (IOError, ValueError) as e:
        print(f"[Warning] Failed to read last timestamp ({e}). Restarting from 0.0.")
    
    return 0.0


def transcribe_session() -> None:
    """Main pipeline execution: WhisperX transcription + speaker diarization."""
    if len(sys.argv) < 2:
        print('Usage: python transcribe_rpg.py "your_audio_file.m4a"')
        sys.exit(1)

    audio_path: str = sys.argv[1]

    if not os.path.exists(audio_path):
        print(f"Error: File '{audio_path}' not found.")
        sys.exit(1)
        
    if not HF_TOKEN:
        print("Error: Missing HF_TOKEN in .env file.")
        sys.exit(1)

    base_name: str = os.path.splitext(audio_path)[0]
    output_path: str = f"{base_name}_transcript.txt"

    last_timestamp: float = get_last_timestamp(output_path)
    asr_options = {"initial_prompt": TRANCRIPTION_PROMPT}

    print(f"Loading WhisperX ({WHISPER_MODEL_SIZE})...")
    model = whisperx.load_model(WHISPER_MODEL_SIZE, DEVICE, compute_type=COMPUTE_TYPE, asr_options=asr_options)

    print("Loading audio track...")
    audio = whisperx.load_audio(audio_path)

    # Check if we need to append or write fresh
    if last_timestamp > 0:
        print(f"\n>>> Save-state found! Resuming from: {last_timestamp}s <<<")
        start_sample = int(last_timestamp * AUDIO_SAMPLE_RATE)
        audio = audio[start_sample:]
        write_mode = "a"
    else:
        print("\nStarting fresh transcription...")
        write_mode = "w"

    print("Running transcription...")
    result = model.transcribe(audio, batch_size=BATCH_SIZE, language="pl")

    print("Aligning words with timeline...")
    model_alignment, metadata = whisperx.load_align_model(language_code=result["language"], device=DEVICE)
    result = whisperx.align(result["segments"], model_alignment, metadata, audio, DEVICE, return_char_alignments=False)

    print("Running Speaker Diarization...")
    try:
        from whisperx.diarize import DiarizationPipeline, assign_word_speakers
        diarize_model = DiarizationPipeline(token=HF_TOKEN, device=DEVICE)
    except Exception as e:
        print(f"\nFailed to load Diarization Pipeline: {e}")
        sys.exit(1)

    diarize_segments = diarize_model(audio, min_speakers=1, max_speakers=4)
    result = assign_word_speakers(diarize_segments, result)

    print(f"Saving outputs to {output_path}...")
    with open(output_path, write_mode, encoding="utf-8") as file:
        if last_timestamp > 0:
            file.write("\n\n--- [TRANSCRIPTION RESUMED] ---\n\n")

        for segment in result["segments"]:
            speaker: str = segment.get("speaker", "Unknown Speaker")
            start: float = round(segment['start'], 2)
            end: float = round(segment['end'], 2)
            text: str = segment['text'].strip()
            
            # Clean text syntax
            text = apply_custom_dictionary(text)
            file.write(f"[{start}s - {end}s] {speaker}: {text}\n")

    print(f"\n✅ Done. Output saved to: {output_path}")


if __name__ == "__main__":
    transcribe_session()