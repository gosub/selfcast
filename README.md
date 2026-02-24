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

There are three subcommands: **url** (convert a single webpage), **follow** (subscribe to a feed), and **feed** (batch-process feeds).

### Single URL mode

```bash
uv run main.py url <url> [output.mp3] [--speaker SPEAKER] [--language LANGUAGE] [--save-text]
```

Examples:

```bash
# Basic usage (outputs to output.mp3)
uv run main.py url https://example.com/article

# Custom output file and voice
uv run main.py url https://example.com/article article.mp3 --speaker Vivian --language English

# Save intermediate text files for inspection
uv run main.py url https://example.com/article article.mp3 --save-text
```

| Option | Default | Description |
|---|---|---|
| `url` | (required) | URL of the webpage to convert |
| `output` | `output.mp3` | Output MP3 file path |
| `--speaker` | `Aiden` | TTS voice: Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee |
| `--language` | `Auto` | Language: Auto, English, Italian, Chinese, Japanese, Korean, German, French, Russian, Portuguese, Spanish |
| `--save-text` | off | Save raw HTML, trafilatura output, and LLM output as separate files |

### Follow mode

Subscribe to a new RSS/Atom feed by adding it to an OPML file. The feed title is auto-detected.

```bash
uv run main.py follow <feed-url> [--opml-file feeds.opml] [--title TITLE]
```

Examples:

```bash
# Add a feed (creates feeds.opml if it doesn't exist)
uv run main.py follow https://example.com/feed.xml

# Custom OPML file and title override
uv run main.py follow https://example.com/feed.xml --opml-file my-feeds.opml --title "My Blog"
```

| Option | Default | Description |
|---|---|---|
| `feed_url` | (required) | RSS/Atom feed URL to subscribe to |
| `--opml-file` | `feeds.opml` | OPML file to add the feed to (created if missing) |
| `--title` | auto-detected | Override the feed title |

### Feed mode

Process new articles from RSS/Atom feeds listed in an OPML file. Each run downloads only unprocessed articles and generates podcast RSS feeds you can subscribe to.

```bash
uv run main.py feed <opml-file> [--output-dir feeds/] [--speaker SPEAKER] [--language LANGUAGE] [--base-url URL] [--save-text]
```

Example:

```bash
uv run main.py feed my-feeds.opml --output-dir feeds/ --speaker Aiden --language English
```

| Option | Default | Description |
|---|---|---|
| `opml_file` | (required) | Path to OPML file listing RSS/Atom feeds |
| `--output-dir` | `feeds/` | Output directory for generated audio and RSS |
| `--speaker` | `Aiden` | TTS voice |
| `--language` | `Auto` | Language for TTS |
| `--base-url` | (none) | Public URL prefix for podcast enclosure URLs (e.g. `https://myserver.com/feeds/`) |
| `--save-text` | off | Save intermediate text files for each article |

The feed pipeline runs in three batched phases to stay within 16 GB RAM:
1. **Discovery** — parse OPML, fetch feeds, download new articles, clean with trafilatura
2. **LLM extraction** — start llama-server once, process all articles, then shut down
3. **TTS generation** — load TTS model once, generate audio for all articles

Output structure (entry folders are date-prefixed for chronological sorting):

```
feeds/
  state.json                              # tracks processed articles
  feed.xml                                # root RSS combining all feeds
  blog-name/
    feed.xml                              # per-feed podcast RSS
    2026-02-20-article-title/
      2026-02-20-article-title.mp3
    2026-02-22-another-article/
      2026-02-22-another-article.mp3
```

Point your podcast app at `feeds/feed.xml` (or a per-feed `feed.xml`) to subscribe. Use `--base-url` to set absolute URLs if serving over HTTP.

## How it works

The script first uses trafilatura to strip boilerplate from the HTML, drastically reducing the token count so that even large pages (e.g. Wikipedia) fit within the LLM's 32K context window. It then starts a temporary `llama-server` instance to generate a spoken preamble ("Selfcast rendering of: Title. By: Author.") and extract article text, and shuts it down to free memory before loading the TTS model. Long texts are automatically split into chunks (~3000 chars each) with "Part X of Y" prefixes and 2-second silence gaps between them. Both models run on Apple MPS (Metal Performance Shaders) for GPU-accelerated inference. The two models are too large to fit in memory simultaneously on 16 GB, hence the sequential approach.

In feed mode, the same pipeline runs in batch: all LLM work happens with the server loaded once, then all TTS work with the model loaded once, making it efficient to process many articles in a single run.

## Benchmarks

Tested on a MacBook Air M4, 16 GB RAM.

| Article | HTML size | After trafilatura | Reduction | TTS chunks | Audio length | TTS time |
|---|---|---|---|---|---|---|
| [Guido van Rossum](https://en.wikipedia.org/wiki/Guido_van_Rossum) | 178K chars | 13K chars | 93% | 2 | 6:09 | ~19 min |
| [Python (programming language)](https://en.wikipedia.org/wiki/Python_(programming_language)) | 604K chars | 77K chars | 87% | 2 | 5:23 | ~19 min |
| [How to Do Great Work](https://paulgraham.com/greatwork.html) | 80K chars | 67K chars | 16% | 22 | 1:07:17 | ~3h 14min |
| [Things That Aren't Doing the Thing](https://strangestloop.io/essays/things-that-arent-doing-the-thing) | 3K chars | 929 chars | 71% | 1 | 1:08 | ~3 min |

## License

[GPL-3.0](LICENSE)
