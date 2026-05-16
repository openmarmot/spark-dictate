# tts-spark

A lightweight, cross-platform push-to-talk dictation tool that sends audio from your local microphone to **whisper.cpp** running on a DGX Spark (or other NVIDIA hardware) and copies the transcription to your clipboard.

## Features

- Hold-to-talk dictation (Right Option on macOS / Right Alt on Windows/Linux)
- Local audio capture and processing (16kHz, gain amplification, silence padding)
- GPU-accelerated transcription via whisper.cpp on your DGX Spark
- Cross-platform clipboard output (macOS, Windows, Linux)
- Minimal dependencies, no cloud required

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

## Requirements

**Client machine (Mac/Windows/Linux):**
- Python 3.10+
- Microphone

**DGX Spark (server):**
- [whisper.cpp](https://github.com/ggml-org/whisper.cpp) built with CUDA support
- Model file (e.g., `ggml-large-v3-turbo-q8_0.bin`)

## Setup

### 1. Server (DGX Spark)

Install whisper.cpp and start the server:

```bash
./server/whisper.cpp/start_whisper_cuda.sh
```

### 2. Client

```bash
./client/start_client.sh
```

On first launch, enter your DGX Spark's IP and port (e.g., `192.168.1.45:8025`).

## Usage

1. Hold **Right Option (⌥)** on Mac or **Right Alt** on Windows/Linux
2. Speak
3. Release to transcribe — text is copied to clipboard
4. Paste wherever you need it

## Configuration

Edit the config section in `client/code/client.py`:

| Setting    | Default              | Description                        |
|------------|----------------------|------------------------------------|
| `MODEL_NAME` | `"whisper-large-v3"` | Model name sent to server         |
| `HOTKEY`   | `Key.alt_r`          | Push-to-talk key (Right Option/Alt)|
| `GAIN`     | `4.0`                | Audio amplification (3–8 typical)  |

## Troubleshooting

**No audio captured:** Ensure your microphone is set as the default input device.

**Transcription empty or ".":** Check that the server is running and reachable. Try adjusting `GAIN` if your microphone is quiet.

**"No clipboard tool found" on Linux:** Install `wl-clipboard` (Wayland) or `xclip` (X11).

## License

Public domain (Unlicense)
