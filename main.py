import sys
import os
import argparse
import json
import subprocess
import tempfile
import time
import urllib.request
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

PROMPT_PREFIX = """\
You are a professional text extractor. Extract the readable text from the following html page, after the cut. Do not extract headers, menu, footers, only the main text of the page. Remember that the text will be passed to a text-to-speech engine, and the final output will be read to the user, who wants to know the content of the html page as if he was reading a magazine article. Only write the extracted text, no preable, no 'ok, here is the content of the page'

------------------

"""

LLAMA_SERVER_CMD = [
    "llama-server",
    "-hf", "ggml-org/gpt-oss-20b-GGUF",
    "--n-cpu-moe", "12",
    "-c", "32768",
    "--jinja",
    "--no-mmap",
    "--offline",
    "--port", "8192",
]

LLAMA_SERVER_URL = "http://127.0.0.1:8192"

SPEAKERS = ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ryan", "Aiden", "Ono_Anna", "Sohee"]


def download_html(url: str) -> str:
    print(f"Downloading {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r:
        return r.read().decode("utf-8", errors="replace")


def extract_text(html: str) -> str:
    """Start llama-server, query it to extract text from HTML, then shut it down."""
    print("Starting llama-server ...")
    proc = subprocess.Popen(
        LLAMA_SERVER_CMD,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        # Poll /health until the server is ready.
        deadline = time.monotonic() + 120
        ready = False
        while time.monotonic() < deadline:
            try:
                req = urllib.request.urlopen(f"{LLAMA_SERVER_URL}/health", timeout=2)
                if req.status == 200:
                    ready = True
                    break
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(1)
        if not ready:
            raise RuntimeError("llama-server failed to become ready within 120s")

        print("Extracting readable text (LLM) ...")
        payload = json.dumps({
            "model": "gpt-oss-20b",
            "messages": [{"role": "user", "content": PROMPT_PREFIX + html}],
        }).encode()
        req = urllib.request.Request(
            f"{LLAMA_SERVER_URL}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            raise RuntimeError(f"llama-server returned {e.code}: {body}") from e
        return data["choices"][0]["message"]["content"].strip()
    finally:
        proc.terminate()
        proc.wait()


def generate_tts(text: str, speaker: str, language: str, wav_path: str) -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"TTS: device={device}, speaker={speaker}, language={language}")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device_map=device,
        dtype=torch.bfloat16,
    )
    wavs, sr = model.generate_custom_voice(text=text, language=language, speaker=speaker)
    sf.write(wav_path, wavs[0], sr)
    print(f"WAV written (sample rate: {sr})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a webpage to an MP3 audiobook using local LLM and TTS."
    )
    parser.add_argument("url", help="URL of the webpage to convert")
    parser.add_argument(
        "output", nargs="?", default="output.mp3",
        help="Output MP3 file (default: output.mp3)"
    )
    parser.add_argument(
        "--speaker", default="Aiden", choices=SPEAKERS,
        help="TTS voice speaker (default: Aiden)"
    )
    parser.add_argument(
        "--language", default="Auto",
        help="Language for TTS, e.g. English, Italian, Auto (default: Auto)"
    )
    args = parser.parse_args()

    html = download_html(args.url)
    text = extract_text(html)

    if not text:
        sys.exit("Error: LLM returned empty text.")

    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    success = False
    try:
        generate_tts(text, args.speaker, args.language, wav_path)
        print(f"Encoding {args.output} ...")
        subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path, "-b:a", "128k", args.output],
            check=True,
        )
        success = True
    finally:
        if success and os.path.exists(wav_path):
            os.unlink(wav_path)

    print(f"Done -> {args.output}")


if __name__ == "__main__":
    main()
