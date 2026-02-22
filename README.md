# selfcast

Convert webpages to audiobooks using fully local LLM and TTS inference. No cloud APIs, no data leaves your machine.

The pipeline:

1. Downloads a webpage
2. Pre-cleans HTML with [trafilatura](https://github.com/adbar/trafilatura) to strip boilerplate (nav, scripts, styles, footers), reducing tokens by 70-90%
3. Extracts readable text using a local LLM ([llama.cpp](https://github.com/ggml-org/llama.cpp) + [gpt-oss-20b](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF))
4. Chunks the text for TTS (~3000 chars per chunk) with "Part X of Y" announcements and silence gaps between chunks
5. Generates speech using [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) on Apple MPS
6. Encodes the result to MP3 via ffmpeg

## Requirements

- macOS with Apple Silicon (tested on MacBook Air M4, 16 GB)
- The LLM and TTS inference parameters are tuned for this machine. If you have more or less RAM you may need to adjust the model, context size, or batch settings in `main.py`.

## Installation

Install system dependencies with Homebrew:

```bash
brew install ffmpeg llama.cpp uv
```

Install Python dependencies:

```bash
uv sync
```

On first run the LLM and TTS models will be downloaded from Hugging Face automatically.

## Usage

```bash
uv run main.py <url> [output.mp3] [--speaker SPEAKER] [--language LANGUAGE]
```

Examples:

```bash
# Basic usage (outputs to output.mp3)
uv run main.py https://example.com/article

# Custom output file
uv run main.py https://example.com/article article.mp3

# Choose a different voice and language
uv run main.py https://example.com/article article.mp3 --speaker Vivian --language English
```

### Options

| Option | Default | Description |
|---|---|---|
| `url` | (required) | URL of the webpage to convert |
| `output` | `output.mp3` | Output MP3 file path |
| `--speaker` | `Aiden` | TTS voice: Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee |
| `--language` | `Auto` | Language: Auto, English, Italian, Chinese, Japanese, Korean, German, French, Russian, Portuguese, Spanish |

## How it works

The script first uses trafilatura to strip boilerplate from the HTML, drastically reducing the token count so that even large pages (e.g. Wikipedia) fit within the LLM's 32K context window. It then starts a temporary `llama-server` instance to extract article text, and shuts it down to free memory before loading the TTS model. Long texts are automatically split into chunks (~3000 chars each) with "Part X of Y" prefixes and 2-second silence gaps between them. Both models run on Apple MPS (Metal Performance Shaders) for GPU-accelerated inference. The two models are too large to fit in memory simultaneously on 16 GB, hence the sequential approach.

## Benchmarks

Tested on a MacBook Air M4, 16 GB RAM.

| Article | HTML size | After trafilatura | Reduction | TTS chunks | Audio length | TTS time |
|---|---|---|---|---|---|---|
| [Guido van Rossum](https://en.wikipedia.org/wiki/Guido_van_Rossum) | 178K chars | 13K chars | 93% | 2 | 6:09 | ~19 min |
| [Python (programming language)](https://en.wikipedia.org/wiki/Python_(programming_language)) | 604K chars | 77K chars | 87% | 2 | 5:23 | ~19 min |

## License

[GPL-3.0](LICENSE)
