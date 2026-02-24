import sys
import os
import re
import argparse
import contextlib
import hashlib
import json
import shutil
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
# Max characters per LLM input chunk (~24K chars ≈ safe margin within 32K token context)
LLM_CHUNK_MAX_CHARS = 24000
# Seconds of silence between TTS chunks
SILENCE_BETWEEN_CHUNKS = 2.0


# ---------------------------------------------------------------------------
# Checkpointing helpers
# ---------------------------------------------------------------------------

def checkpoint_dir(key: str) -> str:
    """Return checkpoint directory for a given key (URL, entry link, etc.)."""
    h = hashlib.sha256(key.encode()).hexdigest()[:12]
    d = os.path.join(".selfcast-cache", h)
    os.makedirs(d, exist_ok=True)
    return d


def _write_checkpoint(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


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


def _split_long_text(text: str, max_chars: int) -> list[str]:
    """Split text into segments each ≤ max_chars using cascading delimiters."""
    if len(text) <= max_chars:
        return [text]

    # Try splitting by cascading delimiters: paragraphs, lines, sentences, words
    for sep in ("\n\n", "\n", ". "):
        parts = text.split(sep)
        if len(parts) == 1:
            continue
        segments: list[str] = []
        current = parts[0]
        for part in parts[1:]:
            candidate = current + sep + part
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    # Recursively split if a single accumulated segment is still too long
                    segments.extend(_split_long_text(current, max_chars))
                current = part
        if current:
            segments.extend(_split_long_text(current, max_chars))
        return segments

    # Last resort: split on word boundaries
    words = text.split()
    segments = []
    current = words[0] if words else ""
    for word in words[1:]:
        if len(current) + 1 + len(word) <= max_chars:
            current += " " + word
        else:
            if current:
                segments.append(current)
            current = word
    if current:
        segments.append(current)
    return segments


def llm_clean_text(cleaned: str) -> str:
    """Send text through the LLM for cleanup, chunking if it exceeds LLM context."""
    chunks = _split_long_text(cleaned, LLM_CHUNK_MAX_CHARS)
    if len(chunks) > 1:
        print(f"Text too long ({len(cleaned):,} chars), splitting into {len(chunks)} LLM chunks")
    results = []
    for i, chunk in enumerate(chunks):
        if len(chunks) > 1:
            print(f"  LLM chunk {i + 1}/{len(chunks)} ({len(chunk):,} chars) ...")
        result = llama_query([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": chunk},
        ])
        results.append(result)
    return "\n\n".join(results)


def extract_text(cleaned: str, metadata: dict, url: str | None = None) -> tuple[str, str]:
    """Start llama-server, query it for preamble + text extraction, then shut it down."""
    with llama_server():
        preamble = generate_preamble(cleaned, metadata, url=url)
        print(f"Preamble: {preamble}")

        print("Extracting readable text (LLM) ...")
        text = llm_clean_text(cleaned)
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
        # Break up oversized paragraphs that exceed max_chars on their own
        if len(para) > max_chars:
            sub_parts = _split_long_text(para, max_chars)
        else:
            sub_parts = [para]
        for part in sub_parts:
            # If adding this part would exceed the limit, flush current chunk
            if current and current_len + len(part) + 2 > max_chars:
                chunks.append("\n\n".join(current))
                current = []
                current_len = 0
            current.append(part)
            current_len += len(part) + 2  # +2 for "\n\n" separator

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
                  preamble: str | None = None, model=None,
                  checkpoint_dir: str | None = None) -> None:
    if model is None:
        model = load_tts_model()

    chunks = chunk_text(text)
    total_items = len(chunks) + (1 if preamble else 0)
    print(f"TTS: {total_items} chunk(s) to generate" +
          (" (including preamble)" if preamble else ""))

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    def _chunk_path(idx: int) -> str | None:
        if not checkpoint_dir:
            return None
        return os.path.join(checkpoint_dir, f"chunk-{idx:03d}.wav")

    all_audio: list[np.ndarray] = []
    sr = None
    tts_start = time.monotonic()
    chunk_idx = 0

    def _process_chunk(label: str, tts_text: str) -> None:
        nonlocal sr, chunk_idx
        cp = _chunk_path(chunk_idx)
        if cp and os.path.exists(cp):
            print(f"  {label}: loaded from checkpoint")
            audio, chunk_sr = sf.read(cp, dtype="float32")
        else:
            print(f"  {label} ...")
            wavs, chunk_sr = model.generate_custom_voice(
                text=tts_text, language=language, speaker=speaker,
            )
            audio = wavs[0]
            if cp:
                sf.write(cp, audio, chunk_sr)
        if sr is None:
            sr = chunk_sr
        all_audio.append(audio)
        chunk_idx += 1

    # Generate preamble as chunk 0
    if preamble:
        _process_chunk(f"Generating preamble ({len(preamble)} chars)", preamble)
        silence_samples = int(sr * SILENCE_BETWEEN_CHUNKS)
        all_audio.append(np.zeros(silence_samples, dtype=np.float32))

    for i, chunk in enumerate(chunks):
        _process_chunk(
            f"Generating chunk {i + 1}/{len(chunks)} ({len(chunk)} chars)", chunk)
        # Insert silence between chunks (not after the last one)
        if i < len(chunks) - 1:
            silence_samples = int(sr * SILENCE_BETWEEN_CHUNKS)
            all_audio.append(np.zeros(silence_samples, dtype=np.float32))

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
# Shared LLM → TTS → MP3 pipeline
# ---------------------------------------------------------------------------

def _llm_tts_pipeline(cleaned: str, metadata: dict, cp_dir: str,
                       output: str, speaker: str, language: str,
                       keep_checkpoints: bool, url: str | None = None) -> None:
    """Shared LLM → TTS → MP3 pipeline with checkpointing."""
    # LLM stage
    llm_path = os.path.join(cp_dir, "3-llm.txt")
    preamble_path = os.path.join(cp_dir, "4-preamble.txt")
    if os.path.exists(llm_path):
        print("Loaded LLM text from checkpoint")
        with open(llm_path) as f:
            text = f.read()
        preamble = None
        if os.path.exists(preamble_path):
            with open(preamble_path) as f:
                preamble = f.read().strip()
    else:
        preamble, text = extract_text(cleaned, metadata, url=url)
        _write_checkpoint(llm_path, text)
        if preamble:
            _write_checkpoint(preamble_path, preamble)

    if not text:
        sys.exit("Error: LLM returned empty text.")

    # TTS + encode + cleanup
    chunks_dir = os.path.join(cp_dir, "chunks")
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    success = False
    try:
        generate_tts(text, speaker, language, wav_path,
                     preamble=preamble, checkpoint_dir=chunks_dir)
        encode_mp3(wav_path, output)
        success = True
    finally:
        if os.path.exists(wav_path):
            os.unlink(wav_path)
        if success and not keep_checkpoints:
            shutil.rmtree(cp_dir)
            print(f"Cleaned up checkpoint dir: {cp_dir}")

    print(f"Done -> {output}")


# ---------------------------------------------------------------------------
# One-shot feed helper
# ---------------------------------------------------------------------------

def _add_to_feed(mp3_path: str, title: str, key: str,
                  feed_dir: str, link: str | None = None) -> None:
    """Add an MP3 to the one-shot podcast feed."""
    state_path = os.path.join(feed_dir, "state.json")
    os.makedirs(feed_dir, exist_ok=True)
    state = load_state(state_path)
    state.setdefault("processed", {})[key] = {
        "feed": "One-shot",
        "feed_slug": "one-shot",
        "title": title,
        "slug": slugify(title),
        "link": link or "",
        "mp3": mp3_path,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    save_state(state_path, state)
    generate_root_rss(feed_dir, state)
    print(f"Added to feed: {feed_dir}/feed.xml")


# ---------------------------------------------------------------------------
# URL mode (single article)
# ---------------------------------------------------------------------------

def url_main(args) -> None:
    cp_dir = checkpoint_dir(args.url)

    # Stage 1: download
    raw_path = os.path.join(cp_dir, "1-raw.html")
    if os.path.exists(raw_path):
        print("Loaded raw HTML from checkpoint")
        with open(raw_path) as f:
            raw_html = f.read()
    else:
        raw_html = download_html(args.url)
        _write_checkpoint(raw_path, raw_html)

    # Stage 2: trafilatura
    traf_path = os.path.join(cp_dir, "2-trafilatura.txt")
    if os.path.exists(traf_path):
        print("Loaded trafilatura text from checkpoint")
        with open(traf_path) as f:
            cleaned = f.read()
        metadata = {}  # not available from checkpoint
    else:
        cleaned, metadata = clean_html(raw_html)
        _write_checkpoint(traf_path, cleaned)

    # Determine output path
    title = metadata.get("title") or os.path.basename(args.url)
    if args.add_to_feed:
        slug = slugify(title)
        one_shot_dir = os.path.join(args.add_to_feed, "one-shot")
        os.makedirs(one_shot_dir, exist_ok=True)
        output = os.path.join(one_shot_dir, f"{slug}.mp3")
    else:
        output = args.output

    # Stage 3+: LLM → TTS → MP3
    _llm_tts_pipeline(cleaned, metadata, cp_dir,
                       output, args.speaker, args.language,
                       args.keep_checkpoints, url=args.url)

    if args.add_to_feed:
        _add_to_feed(output, title, args.url, args.add_to_feed,
                      link=args.url)


# ---------------------------------------------------------------------------
# Text mode (local text file)
# ---------------------------------------------------------------------------

def text_main(args) -> None:
    input_path = os.path.abspath(args.input)
    cp_dir = checkpoint_dir(input_path)

    # Stage 1: read source text (no checkpoint needed — it's a local file)
    with open(input_path) as f:
        cleaned = f.read()

    # Determine output path
    title = os.path.splitext(os.path.basename(args.input))[0]
    metadata = {"title": title}
    if args.add_to_feed:
        slug = slugify(title)
        one_shot_dir = os.path.join(args.add_to_feed, "one-shot")
        os.makedirs(one_shot_dir, exist_ok=True)
        output = os.path.join(one_shot_dir, f"{slug}.mp3")
    else:
        output = args.output

    # Stage 2+: LLM → TTS → MP3
    _llm_tts_pipeline(cleaned, metadata, cp_dir,
                       output, args.speaker, args.language,
                       args.keep_checkpoints)

    if args.add_to_feed:
        _add_to_feed(output, title, input_path, args.add_to_feed)


# ---------------------------------------------------------------------------
# PDF mode (local PDF file)
# ---------------------------------------------------------------------------

def pdf_main(args) -> None:
    input_path = os.path.abspath(args.input)
    cp_dir = checkpoint_dir(input_path)

    # Stage 1: extract text from PDF
    extracted_path = os.path.join(cp_dir, "1-extracted.txt")
    if os.path.exists(extracted_path):
        print("Loaded extracted text from checkpoint")
        with open(extracted_path) as f:
            cleaned = f.read()
    else:
        from pypdf import PdfReader
        print(f"Extracting text from {args.input} ...")
        reader = PdfReader(input_path)
        pages = [page.extract_text() or "" for page in reader.pages]
        cleaned = "\n\n".join(pages)
        print(f"Extracted {len(cleaned):,} chars from {len(reader.pages)} pages")
        _write_checkpoint(extracted_path, cleaned)

    # Determine output path
    title = os.path.splitext(os.path.basename(args.input))[0]
    metadata = {"title": title}
    if args.add_to_feed:
        slug = slugify(title)
        one_shot_dir = os.path.join(args.add_to_feed, "one-shot")
        os.makedirs(one_shot_dir, exist_ok=True)
        output = os.path.join(one_shot_dir, f"{slug}.mp3")
    else:
        output = args.output

    # Stage 2+: LLM → TTS → MP3
    _llm_tts_pipeline(cleaned, metadata, cp_dir,
                       output, args.speaker, args.language,
                       args.keep_checkpoints)

    if args.add_to_feed:
        _add_to_feed(output, title, input_path, args.add_to_feed)


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

    # Phase 1: Discovery (with checkpointing for download + trafilatura)
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

            cp_dir = checkpoint_dir(entry_link)

            print(f"  New: {entry_title}")

            # Stage 1: download (checkpointed)
            raw_path = os.path.join(cp_dir, "1-raw.html")
            if os.path.exists(raw_path):
                print(f"    Loaded raw HTML from checkpoint")
                with open(raw_path) as f:
                    raw_html = f.read()
            else:
                try:
                    raw_html = download_html(entry_link)
                    _write_checkpoint(raw_path, raw_html)
                except Exception as e:
                    print(f"  Error downloading {entry_link}: {e}")
                    continue

            # Stage 2: trafilatura (checkpointed)
            traf_path = os.path.join(cp_dir, "2-trafilatura.txt")
            if os.path.exists(traf_path):
                print(f"    Loaded trafilatura text from checkpoint")
                with open(traf_path) as f:
                    cleaned = f.read()
                metadata = {}
            else:
                cleaned, metadata = clean_html(raw_html)
                _write_checkpoint(traf_path, cleaned)

            work_list.append({
                "entry_key": entry_key,
                "feed_title": feed_info["title"],
                "feed_slug": feed_slug,
                "entry_title": entry_title,
                "entry_slug": entry_slug,
                "entry_link": entry_link,
                "cleaned": cleaned,
                "metadata": metadata,
                "output_dir": output_dir,
                "cp_dir": cp_dir,
            })

    if not work_list:
        print("\nNo new articles to process.")
        return

    print(f"\n{len(work_list)} new article(s) to process.")

    # Phase 2: LLM extraction (server loaded once, with checkpointing)
    print("\n=== Phase 2: LLM extraction ===")
    need_llm = any(
        not os.path.exists(os.path.join(item["cp_dir"], "3-llm.txt"))
        for item in work_list
    )
    if need_llm:
        ctx = llama_server()
    else:
        ctx = contextlib.nullcontext()

    with ctx:
        for i, item in enumerate(work_list):
            print(f"\n[{i + 1}/{len(work_list)}] {item['entry_title']}")
            cp_dir = item["cp_dir"]
            llm_path = os.path.join(cp_dir, "3-llm.txt")
            preamble_path = os.path.join(cp_dir, "4-preamble.txt")

            if os.path.exists(llm_path):
                print("  Loaded LLM text from checkpoint")
                with open(llm_path) as f:
                    item["text"] = f.read()
                item["preamble"] = None
                if os.path.exists(preamble_path):
                    with open(preamble_path) as f:
                        item["preamble"] = f.read().strip()
            else:
                try:
                    item["preamble"] = generate_preamble(
                        item["cleaned"], item["metadata"],
                        url=item["entry_link"])
                    print(f"  Preamble: {item['preamble']}")

                    print("  Extracting readable text (LLM) ...")
                    item["text"] = llm_clean_text(item["cleaned"])
                    _write_checkpoint(llm_path, item["text"])
                    if item["preamble"]:
                        _write_checkpoint(preamble_path, item["preamble"])
                except Exception as e:
                    print(f"  LLM error: {e}")
                    item["text"] = None

    # Remove items that failed LLM extraction
    work_list = [item for item in work_list if item.get("text")]

    if not work_list:
        print("\nAll articles failed LLM extraction.")
        return

    # Phase 3: TTS generation (model loaded once, with checkpointing)
    print("\n=== Phase 3: TTS generation ===")
    model = load_tts_model()

    for i, item in enumerate(work_list):
        print(f"\n[{i + 1}/{len(work_list)}] {item['entry_title']}")
        os.makedirs(item["output_dir"], exist_ok=True)
        mp3_path = os.path.join(item["output_dir"], item["entry_slug"] + ".mp3")
        chunks_dir = os.path.join(item["cp_dir"], "chunks")

        fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            generate_tts(
                item["text"], args.speaker, args.language, wav_path,
                preamble=item.get("preamble"), model=model,
                checkpoint_dir=chunks_dir,
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

        # Clean up checkpoint dir for this article
        if not args.keep_checkpoints:
            shutil.rmtree(item["cp_dir"])
            print(f"  Cleaned up checkpoint dir: {item['cp_dir']}")

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
        "--keep-checkpoints", action="store_true",
        help="Don't delete the checkpoint dir after success (for debugging)"
    )
    url_parser.add_argument(
        "--add-to-feed", nargs="?", const="feeds", default=None,
        metavar="DIR",
        help="Add the rendered MP3 to a podcast feed in DIR (default: feeds/)"
    )

    # --- text subcommand ---
    text_parser = subparsers.add_parser("text", help="Convert a local text file to MP3")
    text_parser.add_argument("input", help="Path to the text file")
    text_parser.add_argument(
        "output", nargs="?", default="output.mp3",
        help="Output MP3 file (default: output.mp3)"
    )
    text_parser.add_argument(
        "--speaker", default="Aiden", choices=SPEAKERS,
        help="TTS voice speaker (default: Aiden)"
    )
    text_parser.add_argument(
        "--language", default="Auto",
        help="Language for TTS, e.g. English, Italian, Auto (default: Auto)"
    )
    text_parser.add_argument(
        "--keep-checkpoints", action="store_true",
        help="Don't delete the checkpoint dir after success (for debugging)"
    )
    text_parser.add_argument(
        "--add-to-feed", nargs="?", const="feeds", default=None,
        metavar="DIR",
        help="Add the rendered MP3 to a podcast feed in DIR (default: feeds/)"
    )

    # --- pdf subcommand ---
    pdf_parser = subparsers.add_parser("pdf", help="Convert a local PDF file to MP3")
    pdf_parser.add_argument("input", help="Path to the PDF file")
    pdf_parser.add_argument(
        "output", nargs="?", default="output.mp3",
        help="Output MP3 file (default: output.mp3)"
    )
    pdf_parser.add_argument(
        "--speaker", default="Aiden", choices=SPEAKERS,
        help="TTS voice speaker (default: Aiden)"
    )
    pdf_parser.add_argument(
        "--language", default="Auto",
        help="Language for TTS, e.g. English, Italian, Auto (default: Auto)"
    )
    pdf_parser.add_argument(
        "--keep-checkpoints", action="store_true",
        help="Don't delete the checkpoint dir after success (for debugging)"
    )
    pdf_parser.add_argument(
        "--add-to-feed", nargs="?", const="feeds", default=None,
        metavar="DIR",
        help="Add the rendered MP3 to a podcast feed in DIR (default: feeds/)"
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
        "--keep-checkpoints", action="store_true",
        help="Don't delete checkpoint dirs after success (for debugging)"
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
    elif args.command == "text":
        text_main(args)
    elif args.command == "pdf":
        pdf_main(args)
    elif args.command == "feed":
        feed_main(args)
    elif args.command == "follow":
        follow_main(args)


if __name__ == "__main__":
    main()
