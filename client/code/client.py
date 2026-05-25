#!/usr/bin/env python3
import os
import tempfile
import time
import threading
import signal
import sounddevice as sd
import requests
import subprocess
import numpy as np
from pynput import keyboard
import wave
import platform
import json

# ========================= CONFIG =========================
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "client_config.json")
HOTKEY = keyboard.Key.alt_r  # Right Option (⌥) on Mac / Right Alt on Windows/Linux
def _fix_llm_url(u):
    u=(u or "").strip()
    if u and not u.startswith(("http://","https://")):u="http://"+u
    return u.rstrip("/")
def load_client_config():
    cfg = {
        "MODEL_NAME": "whisper-large-v3",
        "GAIN": 4.0,
        "DGX_SERVER": None,
        "LLM_BASE_URL": None,
        "LLM_MODEL": "cyankiwi/MiniMax-M2.7-AWQ-4bit",
        "LLM_ENABLED": True,
        "TTS_BASE_URL": None,
        "TTS_MODEL": "kokoro",
        "TTS_VOICE": "af_heart",
        "TTS_ENABLED": True
    }
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                loaded = json.load(f)
            cfg.update({k: v for k, v in loaded.items() if k in cfg})
        except Exception:
            pass
    needs_save = False
    if cfg.get("LLM_BASE_URL"):
        fixed=_fix_llm_url(cfg["LLM_BASE_URL"])
        if fixed!=cfg.get("LLM_BASE_URL"):
            cfg["LLM_BASE_URL"]=fixed
            needs_save=True
    if cfg.get("TTS_BASE_URL"):
        fixed=_fix_llm_url(cfg["TTS_BASE_URL"])
        if fixed!=cfg.get("TTS_BASE_URL"):
            cfg["TTS_BASE_URL"]=fixed
            needs_save=True
    if not cfg.get("DGX_SERVER"):
        server_input = input("\nEnter DGX Spark whisper.cpp server (e.g. 192.168.1.45:8025) [default: localhost:8025]: ").strip()
        if not server_input:
            server_input = "localhost:8025"
        cfg["DGX_SERVER"] = server_input
        needs_save = True
    if not cfg.get("LLM_BASE_URL"):
        llm_url = input("\nEnter local OpenAI-compatible LLM base URL (e.g. http://10.12.0.50:8000/v1) [default: http://localhost:8000/v1]: ").strip()
        if not llm_url:
            llm_url = "http://localhost:8000/v1"
        cfg["LLM_BASE_URL"] = _fix_llm_url(llm_url)
        needs_save = True
    if not cfg.get("LLM_MODEL"):
        llm_model = input("Enter LLM model name [default: cyankiwi/MiniMax-M2.7-AWQ-4bit]: ").strip()
        if not llm_model:
            llm_model = "cyankiwi/MiniMax-M2.7-AWQ-4bit"
        cfg["LLM_MODEL"] = llm_model
        needs_save = True
    if "LLM_ENABLED" not in cfg:
        cfg["LLM_ENABLED"] = True
        needs_save = True
    if "TTS_ENABLED" not in cfg:
        cfg["TTS_ENABLED"] = True
        needs_save = True
    if not cfg.get("TTS_BASE_URL"):
        tts_url = input("\nEnter TTS server base URL (OpenAI-compatible, e.g. http://192.168.1.45:8880/v1) [press Enter to skip]: ").strip()
        if tts_url:
            cfg["TTS_BASE_URL"] = _fix_llm_url(tts_url)
            needs_save = True
    if cfg.get("TTS_BASE_URL"):
        if not cfg.get("TTS_MODEL"):
            tts_model = input("Enter TTS model name [default: kokoro]: ").strip()
            if not tts_model:
                tts_model = "kokoro"
            cfg["TTS_MODEL"] = tts_model
            needs_save = True
        if not cfg.get("TTS_VOICE"):
            tts_voice = input("Enter TTS voice [default: af_heart]: ").strip()
            if not tts_voice:
                tts_voice = "af_heart"
            cfg["TTS_VOICE"] = tts_voice
            needs_save = True
    if needs_save:
        try:
            with open(CONFIG_PATH, "w") as f:
                json.dump({k: cfg[k] for k in ["MODEL_NAME", "GAIN", "DGX_SERVER", "LLM_BASE_URL", "LLM_MODEL", "LLM_ENABLED", "TTS_BASE_URL", "TTS_MODEL", "TTS_VOICE", "TTS_ENABLED"]}, f, indent=2)
            print(f"✅ Saved config to {CONFIG_PATH}")
        except Exception as e:
            print("⚠️  Could not save config:", e)
    return cfg

config = load_client_config()
MODEL_NAME = config["MODEL_NAME"]
GAIN = config["GAIN"]
DGX_BASE_URL = f"http://{config['DGX_SERVER']}"
LLM_BASE_URL = _fix_llm_url(config.get("LLM_BASE_URL"))
LLM_MODEL = config["LLM_MODEL"]
LLM_ENABLED = bool(config.get("LLM_ENABLED", True))
TTS_BASE_URL = _fix_llm_url(config.get("TTS_BASE_URL")) if config.get("TTS_BASE_URL") else None
TTS_MODEL = config.get("TTS_MODEL", "kokoro")
TTS_VOICE = config.get("TTS_VOICE", "af_heart")
TTS_ENABLED = bool(config.get("TTS_ENABLED", True))
print(f"✅ Connected to: {DGX_BASE_URL}")
print(f"🤖 LLM ready: {LLM_MODEL} @ {LLM_BASE_URL}")
print(f"   LLM queries: {'enabled' if LLM_ENABLED else 'disabled'}")
if TTS_BASE_URL:
    print(f"🔊 TTS ready: {TTS_MODEL}/{TTS_VOICE} @ {TTS_BASE_URL}")
    print(f"   Spoken LLM answers: {'enabled' if TTS_ENABLED else 'disabled'}")
else:
    print("🔊 TTS: disabled (no base URL configured)")
print()
# ========================================================

def ask_llm(prompt):
    try:
        payload = {
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Provide concise, direct answers."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1024,
            "temperature": 0.6
        }
        resp = requests.post(
            _fix_llm_url(LLM_BASE_URL)+"/chat/completions",
            json=payload,
            timeout=120
        )
        if resp.status_code == 200:
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            if content:
                return content
            return "(no response from LLM)"
        else:
            print(f"LLM error {resp.status_code}: {resp.text[:200]}")
            return f"(LLM error: {resp.status_code})"
    except Exception as e:
        print("LLM query failed:", e)
        return f"(LLM unavailable: {e})"

# ====================== END LLM ======================

# ====================== TTS (Kokoro / OpenAI-compatible) ======================
def play_wav(path):
    """Play a WAV file using sounddevice (reuses existing deps, cross-platform)."""
    try:
        with wave.open(path, 'rb') as wf:
            sr = wf.getframerate()
            nch = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())
            # Convert to float32 [-1,1]
            if sampwidth == 2:
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            elif sampwidth == 1:
                audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128) / 128.0
            elif sampwidth == 4:
                audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                audio = np.frombuffer(frames, dtype=np.float32)
            if nch > 1:
                audio = audio.reshape(-1, nch)
            sd.play(audio, samplerate=sr)
            sd.wait()
        print("🔊 TTS playback complete")
    except Exception as e:
        print("Playback failed:", e)


def tts_speak(text):
    """Send text to configured Kokoro TTS server and play the returned WAV."""
    if not text or not text.strip():
        return
    if not TTS_ENABLED or not TTS_BASE_URL:
        return
    try:
        payload = {
            "model": TTS_MODEL,
            "input": text.strip(),
            "voice": TTS_VOICE,
            "response_format": "wav"
        }
        url = _fix_llm_url(TTS_BASE_URL) + "/audio/speech"
        print(f"🔊 Sending to TTS ({TTS_MODEL}/{TTS_VOICE}) ...")
        resp = requests.post(url, json=payload, timeout=120)
        if resp.status_code == 200 and resp.content:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(resp.content)
                tmp_path = tmp.name
            try:
                play_wav(tmp_path)
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
        else:
            print(f"TTS error {resp.status_code}: {resp.text[:200] if resp.text else 'no content'}")
    except Exception as e:
        print("TTS request failed:", e)
# ====================== END TTS ======================

# Detect OS only once at startup
SYSTEM = platform.system()
print(f"🚀 Push-to-talk dictation ready on {SYSTEM}")

recording = False
audio_data = []
stream = None
lock = threading.Lock()

def callback(indata, frames, time_info, status):
    if status:
        print("Audio status:", status)
    with lock:
        audio_data.append(indata.copy())

def start_recording():
    global stream, audio_data, recording
    with lock:
        audio_data = []
        recording = True
    print("🎤 Recording... (hold Right ⌥ / Alt)")
    try:
        stream = sd.InputStream(samplerate=16000, channels=1, dtype='float32', callback=callback)
        stream.start()
    except Exception as e:
        print("Failed to start microphone:", e)
        recording = False

def stop_recording():
    global stream, recording
    print("⏹️  Stopping recording...")
    with lock:
        recording = False
    if stream:
        stream.stop()
        stream.close()
        stream = None
    transcribe_and_paste()

def transcribe_and_paste():
    if not audio_data:
        print("No audio recorded")
        return

    # Combine + gain + peak
    audio_array = np.concatenate(audio_data, axis=0).flatten()
    peak = np.max(np.abs(audio_array))
    print(f"🔊 Peak audio level: {peak:.4f} (max 1.0)")

    boosted = (audio_array * GAIN).clip(-1.0, 1.0)

    # 0.5s silence padding
    silence = np.zeros(int(16000 * 0.5), dtype=np.int16)
    pcm = (boosted * 32767).astype(np.int16)
    padded = np.concatenate([silence, pcm, silence])

    padded_audio = padded.tobytes()

    # Temp WAV for server
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        with wave.open(tmp_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(padded_audio)

    try:
        print("📤 Sending to whisper.cpp...")
        files = {'file': open(tmp_path, 'rb')}
        data = {
            'model': MODEL_NAME,
            'language': 'en',
            'temperature': '0.0',
            'response_format': 'json',
            'prompt': 'hello'
        }

        response = requests.post(
            f"{DGX_BASE_URL}/v1/audio/transcriptions",
            files=files,
            data=data,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            text = result.get("text", "").strip()
            #print(f"🔍 Raw server response: {result}")

            if text and text != "." and text != "":
                lower_text = text.lower().lstrip()
                if LLM_ENABLED and (lower_text.startswith("question") or lower_text.startswith("query")):
                    # extract query (after first word)
                    parts = text.split(None, 1)
                    query = parts[1] if len(parts) > 1 else text
                    print(f"❓ Query detected: {query}")
                    text = ask_llm(query)
                    print(f"🤖 LLM response: {text}")
                else:
                    print(f"✅ Transcribed: {text}")

                # Cross-platform clipboard (OS detected only once at startup)
                if SYSTEM == "Darwin":        # macOS
                    subprocess.run(['pbcopy'], input=text.encode('utf-8'), check=True)
                elif SYSTEM == "Windows":
                    subprocess.run(['clip'], input=text.encode('utf-8'), check=True)
                elif SYSTEM == "Linux":
                    try:
                        subprocess.run(['wl-copy'], input=text.encode('utf-8'), check=True)
                    except FileNotFoundError:
                        try:
                            subprocess.run(['xclip', '-selection', 'clipboard'], input=text.encode('utf-8'), check=True)
                        except FileNotFoundError:
                            print("⚠️  No clipboard tool found (install xclip or wl-clipboard)")

                print("📋 Copied to clipboard — just hit Ctrl+V (or ⌘V on Mac)")

                # Speak LLM answers (after clipboard feedback so user sees paste instruction immediately)
                if LLM_ENABLED and (lower_text.startswith("question") or lower_text.startswith("query")):
                    tts_speak(text)
            else:
                print("⚠️  Only '.' returned")
        else:
            print(f"Error from server: {response.status_code} {response.text}")
    except Exception as e:
        print("Transcription failed:", e)
    finally:
        os.unlink(tmp_path)

def on_press(key):
    global recording
    if key == HOTKEY and not recording:
        threading.Thread(target=start_recording, daemon=True).start()

def on_release(key):
    if key == HOTKEY and recording:
        threading.Thread(target=stop_recording, daemon=True).start()

def signal_handler(sig, frame):
    print("\n👋 Shutting down cleanly...")
    if stream:
        stream.stop()
        stream.close()
    os._exit(0)

# ====================== MAIN ======================
print("   Hold Right Option (⌥) / Right Alt to speak → release\n")

signal.signal(signal.SIGINT, signal_handler)

with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    try:
        listener.join()
    except KeyboardInterrupt:
        signal_handler(None, None)
