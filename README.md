# dnd-transcription
TTRPG Audio Transcription; Diarization Pipeline

A highly optimized Python script designed to transcribe long TTRPG (Dungeons & Dragons, Call of Cthulhu) audio sessions. It features precise time-alignments, automated speaker diarization (speaker naming), a session resume mechanism, and custom dictionary entity matching to fix fantasy/niche vocabulary.

## Core Features

- **WhisperX Integration:** Uses `large-v3` with optimized batching for fast, state-of-the-art transcription.
- **Speaker Diarization:** Leverages Hugging Face pipelines (`pyannote.audio`) to identify and split tracks by speaker.
- **Smart Resume Mechanism:** Parses existing output transcripts to automatically find the last recorded timestamp and resume processing from that exact second (prevents redundant processing and saves GPU time).
- **Domain-Specific Vocabulary Correction:** Built-in regex dictionary to automatically catch and correct mistranscribed fantasy names, locations, and mechanical gaming terms.
- **Initial Context Injection:** Utilizes Whisper prompts to prime the model for specific game jargon before processing starts.

## Tech Stack & Dependencies

- **Language:** Python 3.10+
- **ML Frameworks:** WhisperX, PyTorch (CUDA required)
- **APIs/Models:** Hugging Face Hub (`pyannote.audio` diarization)
- **Environment Management:** `python-dotenv` for secure token handling

**##Quick Start
**
1. Clone the repository:
   ```bash
   git clone [https://github.com/michosg15/dnd-transcription.git](https://github.com/michosg15/dnd-transcription.git)
   cd dnd-transcription
```

2. Install dependencies:
Make sure you have PyTorch installed with CUDA support, then run:
```bash
pip install whisperx python-dotenv
```

3. Setup environment variables:
Copy the example file:
```bash
cp .env.example .env
```
Open `.env` and paste your Hugging Face User Access Token (required for the diarization pipeline).

4. Run the script:
```bash
python transcribe_rpg.py "Your_Session_File.m4a"
```
