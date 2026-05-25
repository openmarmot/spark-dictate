# spark-dictate

A lightweight, cross-platform push-to-talk dictation tool that sends audio from your local microphone to **whisper.cpp** running on a DGX Spark (or other NVIDIA hardware) and copies the transcription to your clipboard.

## Features

- Hold-to-talk dictation (Right Option on macOS / Right Alt on Windows/Linux)
- Local audio capture and processing (16kHz, gain amplification, silence padding)
- GPU-accelerated transcription via whisper.cpp on your DGX Spark
- Cross-platform clipboard output (macOS, Windows, Linux)
- Minimal dependencies, no cloud required
- Optional local LLM answers: say *"question ..."* or *"query ..."* to send the rest of your speech to a local LLM (OpenAI-compatible chat endpoint)
- Optional spoken LLM answers: LLM responses are automatically sent to a local Kokoro (or compatible) TTS server and played aloud

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Data Flow                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   User holds Right Option/Alt                                   │
│            ↓                                                    │
│   Local Microphone Recording (16kHz mono)                       │
│            ↓                                                    │
│   Audio Processing: 4x gain + 0.5s silence padding             │
│            ↓                                                    │
│   HTTP POST to DGX Spark /v1/audio/transcriptions              │
│            ↓                                                    │
│   whisper.cpp (GPU inference on DGX Spark)                      │
│            ↓                                                    │
│   Returns JSON with transcribed text                            │
│            ↓                                                    │
│   Text copied to system clipboard                               │
│            ↓                                                    │
│   User pastes into any application (Ctrl+V / ⌘V)               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Note**: When you prefix speech with "question" or "query", the transcribed text is instead sent to your local LLM. The LLM reply is copied to the clipboard and (if configured) spoken via the TTS server before normal dictation resumes.

## Requirements

**Client machine (Mac/Windows/Linux):**
- Python 3.10+
- Microphone

**DGX Spark (server):**
- [whisper.cpp](https://github.com/ggml-org/whisper.cpp) built with CUDA support
- Model file (e.g., `ggml-large-v3-turbo-q8_0.bin`)
- *(Optional for spoken LLM answers)* Kokoro FastAPI GPU container (`ghcr.io/remsky/kokoro-fastapi-gpu:latest`) on port 8880

## Setup

### 1. Server (DGX Spark)

Install whisper.cpp and start the server:

```bash
./server/whisper.cpp/start_whisper_cuda.sh
```

#### Optional: Text-to-Speech (Kokoro)

To enable spoken answers for LLM queries, run the Kokoro FastAPI TTS server (GPU) on your DGX Spark:

```bash
docker run -d --gpus all \
  -p 8880:8880 \
  --name kokoro-tts \
  ghcr.io/remsky/kokoro-fastapi-gpu:latest
```

During client first-run setup you will be prompted for the TTS base URL, e.g. `http://192.168.1.45:8880/v1`. Supported voices include `af_heart` (default), `am_adam`, etc. (see Kokoro docs).

### 2. Client

```bash
./client/start_client.sh
```

On first launch, you will be prompted for:
- DGX Spark whisper.cpp server (e.g. `192.168.1.45:8025`)
- Local LLM base URL + model (for "question"/"query" prefix)
- Optional TTS base URL + model/voice (for speaking LLM answers; press Enter to skip)

## Usage

### Basic Dictation

1. Hold **Right Option (⌥)** on Mac or **Right Alt** on Windows/Linux
2. Speak
3. Release to transcribe — text is copied to clipboard
4. Paste wherever you need it

### LLM Queries ("question" / "query" prefix)

Prefix your dictation with **"question"** or **"query"** to send the rest of the utterance to your local LLM instead of copying the raw transcription.

**Examples** (just speak naturally):

- "question what is the capital of France?"
- "query how do I list running docker containers?"
- "question explain the difference between a list and a tuple in python in one sentence"

What happens:
- The prefix word is stripped.
- The remainder is sent as a prompt to the configured LLM (`LLM_BASE_URL` + `LLM_MODEL`).
- The LLM's response is:
  1. Copied to the system clipboard (so you can paste the answer).
  2. Sent to the TTS server (if `TTS_BASE_URL` is configured and `TTS_ENABLED` is true) and played through your speakers/headphones.

This gives you a completely local voice-to-voice Q&A loop with no cloud services.

**Tips**
- The trigger is case-insensitive and only looks at the first word.
- Set `"LLM_ENABLED": false` in `client_config.json` if you want to disable this feature entirely (plain dictation only).
- Long answers will be spoken in full before the next dictation can start.

## Configuration

All settings are stored in `client/code/client_config.json` (created on first run with interactive prompts; gitignored). Delete the file or edit it directly to reconfigure.

### LLM / "question" / "query" settings
- `LLM_BASE_URL` — OpenAI-compatible chat completions endpoint (e.g. `http://10.12.0.50:8000/v1`)
- `LLM_MODEL` — model name to use for queries
- `LLM_ENABLED` — set to `false` to completely disable the "question"/"query" prefix feature (plain dictation only)

### TTS settings (spoken LLM answers)
- `TTS_BASE_URL` — OpenAI-compatible audio endpoint (e.g. `http://192.168.1.45:8880/v1` for Kokoro)
- `TTS_MODEL` — usually `kokoro`
- `TTS_VOICE` — voice ID (`af_heart` is a good default)
- `TTS_ENABLED` — set to `false` to disable playback while keeping the LLM text copy

TTS is only used for LLM query responses (see Usage section above). Press Enter when prompted for the TTS URL during first-run setup to leave it disabled.

## Troubleshooting

**No audio captured:** Ensure your microphone is set as the default input device.

**Transcription empty or ".":** Check that the server is running and reachable. Try adjusting `GAIN` if your microphone is quiet.

**"No clipboard tool found" on Linux:** Install `wl-clipboard` (Wayland) or `xclip` (X11).

## License

Public domain (Unlicense)
