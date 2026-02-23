import sys
import os
import re
import argparse
import contextlib
import json
import subprocess
import tempfile
import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import formatdate

import numpy as np
import torch
import soundfile as sf
import trafilatura
import feedparser
from qwen_tts import Qwen3TTSModel

SYSTEM_PROMPT = """\
You are preparing article text for a text-to-speech engine. \
The user will provide article text extracted from a webpage. \
Keep the original text as faithful as possible — only make minimal edits \
so it renders well in audio.

Rules:
- Preserve the original wording, structure, and meaning.
- Remove reference markers like [1], [2], [citation needed], etc.
- Spell out abbreviations on first use (e.g. "GPL" -> "G P L").
- Convert bullet points and numbered lists into flowing prose.
- Remove any leftover navigation text, headers, or footers.
- Do NOT summarize, paraphrase, or rewrite sentences unless necessary for audio clarity.
- Do not add any preamble like "Here is the text" — output ONLY the cleaned article."""

PREAMBLE_SYSTEM_PROMPT = """\
You generate a short spoken introduction for an audiobook rendering of a webpage. \
The user will provide article metadata (title, author, site, URL) and a text snippet. \
Use the metadata to produce a single line in this exact format:
Selfcast rendering of: <title>. By: <author>.
If the author is missing, uncertain, or looks like a generic/placeholder name \
(e.g. "admin", "editor", "staff", the site name itself), \
use: From: <website or site name or domain from the URL>.
Output ONLY this single line, nothing else."""

LLAMA_SERVER_CMD = [
    "llama-server",
    "-hf", "ggml-org/gpt-oss-20b-GGUF",
    "--n-cpu-moe", "12",
    "-c", "32768",
    "--jinja",
    "--no-mmap",
    "--port", "8192",
]

LLAMA_SERVER_URL = "http://127.0.0.1:8192"

SPEAKERS = ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ryan", "Aiden", "Ono_Anna", "Sohee"]

# Max characters per TTS chunk (~3000 chars ≈ 20-25K tokens for Qwen3-TTS)
TTS_CHUNK_MAX_CHARS = 3000
# Seconds of silence between TTS chunks
SILENCE_BETWEEN_CHUNKS = 2.0


# ---------------------------------------------------------------------------
# Shared pipeline functions
# ---------------------------------------------------------------------------

def download_html(url: str) -> str:
    print(f"Downloading {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r:
        html = r.read().decode("utf-8", errors="replace")
    print(f"HTML size: {len(html):,} chars")
    return html


def clean_html(html: str) -> tuple[str, dict]:
    """Strip boilerplate, scripts, styles, nav — keep article text and metadata."""
    text = trafilatura.extract(html, include_comments=False)
    if not text:
        text = html  # fallback to raw HTML

    meta = trafilatura.extract_metadata(html)
    metadata = {
        "title": meta.title if meta else None,
        "author": meta.author if meta else None,
        "sitename": meta.sitename if meta else None,
        "url": meta.url if meta else None,
    }
    print(f"After trafilatura: {len(text):,} chars ({100 - len(text) / len(html) * 100:.0f}% reduction)")
    return text, metadata


@contextlib.contextmanager
def llama_server():
    """Start llama-server, wait for readiness, yield, then shut down."""
    print("Starting llama-server ...")
    proc = subprocess.Popen(
        LLAMA_SERVER_CMD,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
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
        print("llama-server ready.")
        yield
    finally:
        proc.terminate()
        proc.wait()
        print("llama-server stopped.")


def llama_query(messages: list[dict]) -> str:
    """Send a chat completion request to the already-running llama-server."""
    payload = json.dumps({
        "model": "gpt-oss-20b",
        "messages": messages,
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


def generate_preamble(text: str, metadata: dict, url: str | None = None) -> str:
    """Ask the LLM to produce a short spoken intro line using metadata."""
    print("Generating preamble (LLM) ...")
    meta_lines = []
    for key in ("title", "author", "sitename"):
        if metadata.get(key):
            meta_lines.append(f"{key}: {metadata[key]}")
    # Always include the original URL (not just trafilatura's)
    effective_url = url or metadata.get("url")
    if effective_url:
        meta_lines.append(f"url: {effective_url}")
    user_content = "\n".join(meta_lines) + "\n\n" + text[:500]
    return llama_query([
        {"role": "system", "content": PREAMBLE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ])


def extract_text(cleaned: str, metadata: dict, url: str | None = None) -> tuple[str, str]:
    """Start llama-server, query it for preamble + text extraction, then shut it down."""
    with llama_server():
        preamble = generate_preamble(cleaned, metadata, url=url)
        print(f"Preamble: {preamble}")

        print("Extracting readable text (LLM) ...")
        text = llama_query([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": cleaned},
        ])
        return preamble, text


def chunk_text(text: str, max_chars: int = TTS_CHUNK_MAX_CHARS) -> list[str]:
    """Split text into chunks by paragraphs, each up to max_chars."""
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        # If adding this paragraph would exceed the limit, flush current chunk
        if current and current_len + len(para) + 2 > max_chars:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0
        current.append(para)
        current_len += len(para) + 2  # +2 for "\n\n" separator

    if current:
        chunks.append("\n\n".join(current))

    # Add "Part X of Y" prefix to each chunk
    total = len(chunks)
    if total > 1:
        chunks = [f"Part {i + 1} of {total}. {chunk}" for i, chunk in enumerate(chunks)]

    return chunks


def load_tts_model():
    """Load the Qwen3-TTS model and return it."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"TTS: loading model on {device} ...")
    return Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device_map=device,
        dtype=torch.bfloat16,
    )


def generate_tts(text: str, speaker: str, language: str, wav_path: str,
                  preamble: str | None = None, model=None) -> None:
    if model is None:
        model = load_tts_model()

    chunks = chunk_text(text)
    print(f"TTS: {len(chunks)} chunk(s) to generate")

    all_audio: list[np.ndarray] = []
    sr = None

    tts_start = time.monotonic()

    # Generate preamble as chunk 0
    if preamble:
        print(f"  Generating preamble ({len(preamble)} chars) ...")
        wavs, chunk_sr = model.generate_custom_voice(
            text=preamble, language=language, speaker=speaker,
        )
        if sr is None:
            sr = chunk_sr
        all_audio.append(wavs[0])
        silence_samples = int(sr * SILENCE_BETWEEN_CHUNKS)
        all_audio.append(np.zeros(silence_samples, dtype=wavs[0].dtype))

    for i, chunk in enumerate(chunks):
        print(f"  Generating chunk {i + 1}/{len(chunks)} ({len(chunk)} chars) ...")
        wavs, chunk_sr = model.generate_custom_voice(
            text=chunk, language=language, speaker=speaker,
        )
        if sr is None:
            sr = chunk_sr
        all_audio.append(wavs[0])

        # Insert silence between chunks (not after the last one)
        if i < len(chunks) - 1:
            silence_samples = int(sr * SILENCE_BETWEEN_CHUNKS)
            all_audio.append(np.zeros(silence_samples, dtype=wavs[0].dtype))

    tts_elapsed = time.monotonic() - tts_start
    combined = np.concatenate(all_audio)
    sf.write(wav_path, combined, sr)
    print(f"WAV written (sample rate: {sr}, {len(combined) / sr:.1f}s, TTS took {tts_elapsed:.0f}s)")


def encode_mp3(wav_path: str, mp3_path: str) -> None:
    """Encode WAV to MP3 via ffmpeg."""
    print(f"Encoding {mp3_path} ...")
    subprocess.run(
        ["ffmpeg", "-y", "-i", wav_path, "-b:a", "128k", mp3_path],
        check=True,
    )


# ---------------------------------------------------------------------------
# Feed mode helpers
# ---------------------------------------------------------------------------

def slugify(text: str) -> str:
    """Convert text to a filesystem-safe slug."""
    slug = text.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")
    return slug[:80]


def parse_opml(path: str) -> list[dict]:
    """Parse an OPML file and return a list of feed dicts."""
    tree = ET.parse(path)
    feeds = []
    for outline in tree.iter("outline"):
        xml_url = outline.get("xmlUrl")
        if xml_url:
            feeds.append({
                "title": outline.get("title") or outline.get("text") or xml_url,
                "xml_url": xml_url,
                "html_url": outline.get("htmlUrl", ""),
            })
    return feeds


def load_state(path: str) -> dict:
    """Load state from JSON file, or return empty state."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"processed": {}}


def save_state(path: str, state: dict) -> None:
    """Atomically write state to JSON file."""
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp_path, path)


def entry_date(entry) -> str:
    """Extract publication date from a feed entry as YYYY-MM-DD, or today."""
    for attr in ("published_parsed", "updated_parsed"):
        tp = getattr(entry, attr, None) or entry.get(attr)
        if tp:
            try:
                return time.strftime("%Y-%m-%d", tp)
            except (TypeError, ValueError):
                pass
    return datetime.now().strftime("%Y-%m-%d")


def generate_podcast_rss(feed_dir: str, entries: list[dict],
                         feed_title: str, base_url: str | None = None) -> str:
    """Generate an RSS 2.0 podcast feed XML file. Returns the path written."""
    ITUNES_NS = "http://www.itunes.com/dtds/podcast-1.0.dtd"

    rss = ET.Element("rss", version="2.0")
    rss.set("xmlns:itunes", ITUNES_NS)
    channel = ET.SubElement(rss, "channel")
    ET.SubElement(channel, "title").text = f"Selfcast: {feed_title}"
    ET.SubElement(channel, "description").text = (
        f"Selfcast audio rendering of {feed_title}"
    )

    # Sort entries by timestamp (newest first)
    sorted_entries = sorted(entries, key=lambda e: e.get("timestamp", ""), reverse=True)

    for entry in sorted_entries:
        item = ET.SubElement(channel, "item")
        ET.SubElement(item, "title").text = entry["title"]
        if entry.get("link"):
            ET.SubElement(item, "link").text = entry["link"]
            ET.SubElement(item, "guid").text = entry["link"]

        mp3_path = entry["mp3"]
        if base_url:
            enclosure_url = base_url.rstrip("/") + "/" + mp3_path
        else:
            enclosure_url = mp3_path
        mp3_size = os.path.getsize(mp3_path) if os.path.exists(mp3_path) else 0
        ET.SubElement(item, "enclosure", url=enclosure_url,
                      type="audio/mpeg", length=str(mp3_size))

        if entry.get("timestamp"):
            try:
                dt = datetime.fromisoformat(entry["timestamp"])
                ET.SubElement(item, "pubDate").text = formatdate(
                    dt.timestamp(), usegmt=True
                )
            except (ValueError, OSError):
                pass

    feed_path = os.path.join(feed_dir, "feed.xml")
    tree = ET.ElementTree(rss)
    ET.indent(tree, space="  ")
    tree.write(feed_path, encoding="unicode", xml_declaration=True)
    print(f"RSS feed written: {feed_path}")
    return feed_path


def generate_root_rss(output_dir: str, state: dict,
                      base_url: str | None = None) -> str:
    """Generate a root RSS feed combining all entries across all feeds."""
    ITUNES_NS = "http://www.itunes.com/dtds/podcast-1.0.dtd"

    rss = ET.Element("rss", version="2.0")
    rss.set("xmlns:itunes", ITUNES_NS)
    channel = ET.SubElement(rss, "channel")
    ET.SubElement(channel, "title").text = "Selfcast"
    ET.SubElement(channel, "description").text = "All Selfcast audio renderings"

    all_entries = list(state.get("processed", {}).values())
    all_entries.sort(key=lambda e: e.get("timestamp", ""), reverse=True)

    for entry in all_entries:
        item = ET.SubElement(channel, "item")
        title = entry["title"]
        if entry.get("feed"):
            title = f"{entry['title']} ({entry['feed']})"
        ET.SubElement(item, "title").text = title
        if entry.get("link"):
            ET.SubElement(item, "link").text = entry["link"]
            ET.SubElement(item, "guid").text = entry["link"]

        mp3_path = entry["mp3"]
        if base_url:
            enclosure_url = base_url.rstrip("/") + "/" + mp3_path
        else:
            enclosure_url = mp3_path
        mp3_size = os.path.getsize(mp3_path) if os.path.exists(mp3_path) else 0
        ET.SubElement(item, "enclosure", url=enclosure_url,
                      type="audio/mpeg", length=str(mp3_size))

        if entry.get("timestamp"):
            try:
                dt = datetime.fromisoformat(entry["timestamp"])
                ET.SubElement(item, "pubDate").text = formatdate(
                    dt.timestamp(), usegmt=True
                )
            except (ValueError, OSError):
                pass

    feed_path = os.path.join(output_dir, "feed.xml")
    tree = ET.ElementTree(rss)
    ET.indent(tree, space="  ")
    tree.write(feed_path, encoding="unicode", xml_declaration=True)
    print(f"Root RSS feed written: {feed_path}")
    return feed_path


# ---------------------------------------------------------------------------
# URL mode (single article)
# ---------------------------------------------------------------------------

def url_main(args) -> None:
    raw_html = download_html(args.url)
    cleaned, metadata = clean_html(raw_html)
    preamble, text = extract_text(cleaned, metadata, url=args.url)

    if not text:
        sys.exit("Error: LLM returned empty text.")

    if args.save_text:
        base = os.path.splitext(args.output)[0]
        for suffix, content in [
            (".1-raw.html", raw_html),
            (".2-trafilatura.txt", cleaned),
            (".3-llm.txt", (preamble + "\n\n" + text) if preamble else text),
        ]:
            path = base + suffix
            with open(path, "w") as f:
                f.write(content)
            print(f"Saved {path}")

    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    success = False
    try:
        generate_tts(text, args.speaker, args.language, wav_path, preamble=preamble)
        encode_mp3(wav_path, args.output)
        success = True
    finally:
        if success and os.path.exists(wav_path):
            os.unlink(wav_path)

    print(f"Done -> {args.output}")


# ---------------------------------------------------------------------------
# Feed mode (batch from OPML)
# ---------------------------------------------------------------------------

def feed_main(args) -> None:
    feeds = parse_opml(args.opml_file)
    if not feeds:
        sys.exit("Error: no feeds found in OPML file.")
    print(f"Found {len(feeds)} feed(s) in OPML.")

    state_path = os.path.join(args.output_dir, "state.json")
    os.makedirs(args.output_dir, exist_ok=True)
    state = load_state(state_path)

    # Phase 1: Discovery
    print("\n=== Phase 1: Discovery ===")
    work_list = []
    for feed_info in feeds:
        print(f"\nFetching feed: {feed_info['title']} ...")
        try:
            feed = feedparser.parse(feed_info["xml_url"])
        except Exception as e:
            print(f"  Error fetching feed: {e}")
            continue

        feed_slug = slugify(feed_info["title"])
        for entry in feed.entries:
            entry_key = entry.get("id", entry.get("link", ""))
            if not entry_key:
                continue
            if entry_key in state.get("processed", {}):
                continue

            entry_title = entry.get("title", "untitled")
            entry_link = entry.get("link", "")
            date_prefix = entry_date(entry)
            entry_slug = f"{date_prefix}-{slugify(entry_title)}"
            output_dir = os.path.join(args.output_dir, feed_slug, entry_slug)

            print(f"  New: {entry_title}")
            try:
                raw_html = download_html(entry_link)
                cleaned, metadata = clean_html(raw_html)
            except Exception as e:
                print(f"  Error downloading {entry_link}: {e}")
                continue

            work_list.append({
                "entry_key": entry_key,
                "feed_title": feed_info["title"],
                "feed_slug": feed_slug,
                "entry_title": entry_title,
                "entry_slug": entry_slug,
                "entry_link": entry_link,
                "raw_html": raw_html,
                "cleaned": cleaned,
                "metadata": metadata,
                "output_dir": output_dir,
            })

    if not work_list:
        print("\nNo new articles to process.")
        return

    print(f"\n{len(work_list)} new article(s) to process.")

    # Phase 2: LLM extraction (server loaded once)
    print("\n=== Phase 2: LLM extraction ===")
    with llama_server():
        for i, item in enumerate(work_list):
            print(f"\n[{i + 1}/{len(work_list)}] {item['entry_title']}")
            try:
                item["preamble"] = generate_preamble(item["cleaned"], item["metadata"],
                                                       url=item["entry_link"])
                print(f"Preamble: {item['preamble']}")

                print("Extracting readable text (LLM) ...")
                item["text"] = llama_query([
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": item["cleaned"]},
                ])
            except Exception as e:
                print(f"  LLM error: {e}")
                item["text"] = None

    # Remove items that failed LLM extraction
    work_list = [item for item in work_list if item.get("text")]

    if not work_list:
        print("\nAll articles failed LLM extraction.")
        return

    # Phase 3: TTS generation (model loaded once)
    print("\n=== Phase 3: TTS generation ===")
    model = load_tts_model()

    for i, item in enumerate(work_list):
        print(f"\n[{i + 1}/{len(work_list)}] {item['entry_title']}")
        os.makedirs(item["output_dir"], exist_ok=True)
        mp3_path = os.path.join(item["output_dir"], item["entry_slug"] + ".mp3")

        if args.save_text:
            base = os.path.join(item["output_dir"], item["entry_slug"])
            for suffix, content in [
                (".1-raw.html", item["raw_html"]),
                (".2-trafilatura.txt", item["cleaned"]),
                (".3-llm.txt", (item["preamble"] + "\n\n" + item["text"])
                 if item.get("preamble") else item["text"]),
            ]:
                path = base + suffix
                with open(path, "w") as f:
                    f.write(content)
                print(f"Saved {path}")

        fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            generate_tts(
                item["text"], args.speaker, args.language, wav_path,
                preamble=item.get("preamble"), model=model,
            )
            encode_mp3(wav_path, mp3_path)
        except Exception as e:
            print(f"  TTS/encode error: {e}")
            continue
        finally:
            if os.path.exists(wav_path):
                os.unlink(wav_path)

        # Update state after each successful MP3
        state.setdefault("processed", {})[item["entry_key"]] = {
            "feed": item["feed_title"],
            "feed_slug": item["feed_slug"],
            "title": item["entry_title"],
            "slug": item["entry_slug"],
            "link": item["entry_link"],
            "mp3": mp3_path,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        save_state(state_path, state)
        print(f"Done -> {mp3_path}")

    # Generate per-feed RSS and root RSS
    print("\n=== Generating RSS feeds ===")
    affected_feeds = {item["feed_slug"]: item["feed_title"] for item in work_list}
    for feed_slug, feed_title in affected_feeds.items():
        feed_entries = [
            v for v in state["processed"].values()
            if v.get("feed_slug") == feed_slug
        ]
        feed_dir = os.path.join(args.output_dir, feed_slug)
        generate_podcast_rss(feed_dir, feed_entries, feed_title,
                             base_url=args.base_url)

    generate_root_rss(args.output_dir, state, base_url=args.base_url)

    print(f"\nAll done. {len(work_list)} article(s) processed.")


# ---------------------------------------------------------------------------
# Follow mode (add a feed to OPML)
# ---------------------------------------------------------------------------

def follow_main(args) -> None:
    feed_url = args.feed_url
    opml_path = args.opml_file

    # Fetch the feed to get its title
    print(f"Fetching {feed_url} ...")
    feed = feedparser.parse(feed_url)
    if feed.bozo and not feed.entries:
        sys.exit(f"Error: could not parse feed at {feed_url}")
    title = args.title or feed.feed.get("title", feed_url)
    html_url = feed.feed.get("link", "")

    if os.path.exists(opml_path):
        tree = ET.parse(opml_path)
        root = tree.getroot()
        body = root.find("body")
        # Check for duplicates
        for outline in body.iter("outline"):
            if outline.get("xmlUrl") == feed_url:
                print(f"Feed already exists in {opml_path}: {title}")
                return
    else:
        root = ET.Element("opml", version="2.0")
        head = ET.SubElement(root, "head")
        ET.SubElement(head, "title").text = "Selfcast feeds"
        body = ET.SubElement(root, "body")
        tree = ET.ElementTree(root)

    outline = ET.SubElement(body, "outline",
                            text=title, title=title,
                            xmlUrl=feed_url)
    if html_url:
        outline.set("htmlUrl", html_url)

    ET.indent(tree, space="  ")
    tree.write(opml_path, encoding="unicode", xml_declaration=True)
    print(f"Added \"{title}\" to {opml_path} ({len(feed.entries)} entries in feed)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert webpages to audiobooks using local LLM and TTS."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- url subcommand ---
    url_parser = subparsers.add_parser("url", help="Convert a single URL to MP3")
    url_parser.add_argument("url", help="URL of the webpage to convert")
    url_parser.add_argument(
        "output", nargs="?", default="output.mp3",
        help="Output MP3 file (default: output.mp3)"
    )
    url_parser.add_argument(
        "--speaker", default="Aiden", choices=SPEAKERS,
        help="TTS voice speaker (default: Aiden)"
    )
    url_parser.add_argument(
        "--language", default="Auto",
        help="Language for TTS, e.g. English, Italian, Auto (default: Auto)"
    )
    url_parser.add_argument(
        "--save-text", action="store_true",
        help="Save pipeline intermediate text files alongside the output"
    )

    # --- feed subcommand ---
    feed_parser = subparsers.add_parser("feed", help="Process new articles from RSS/Atom feeds")
    feed_parser.add_argument("opml_file", help="Path to OPML file listing feeds")
    feed_parser.add_argument(
        "--output-dir", default="feeds",
        help="Output directory for feeds (default: feeds/)"
    )
    feed_parser.add_argument(
        "--speaker", default="Aiden", choices=SPEAKERS,
        help="TTS voice speaker (default: Aiden)"
    )
    feed_parser.add_argument(
        "--language", default="Auto",
        help="Language for TTS, e.g. English, Italian, Auto (default: Auto)"
    )
    feed_parser.add_argument(
        "--base-url", default=None,
        help="Public URL prefix for podcast enclosure URLs"
    )
    feed_parser.add_argument(
        "--save-text", action="store_true",
        help="Save pipeline intermediate text files for each article"
    )

    # --- follow subcommand ---
    follow_parser = subparsers.add_parser("follow", help="Add a feed to an OPML file")
    follow_parser.add_argument("feed_url", help="RSS/Atom feed URL to follow")
    follow_parser.add_argument(
        "--opml-file", default="feeds.opml",
        help="OPML file to add the feed to (default: feeds.opml)"
    )
    follow_parser.add_argument(
        "--title", default=None,
        help="Override the feed title (default: auto-detected from feed)"
    )

    args = parser.parse_args()

    if args.command == "url":
        url_main(args)
    elif args.command == "feed":
        feed_main(args)
    elif args.command == "follow":
        follow_main(args)


if __name__ == "__main__":
    main()
