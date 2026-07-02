import os
import re
import time
import json
import ast
import tempfile
import shutil

import requests
import streamlit as st
from faster_whisper import WhisperModel
from bs4 import BeautifulSoup
from moviepy.editor import VideoFileClip

# ---- Compatibility shim ----
# moviepy 1.0.3's resize() calls PIL.Image.ANTIALIAS, which was removed in
# Pillow 10.0+ (renamed to Image.LANCZOS). Restore the old attribute name
# as an alias so moviepy keeps working regardless of installed Pillow version.
from PIL import Image
if not hasattr(Image, 'ANTIALIAS'):
    Image.ANTIALIAS = Image.LANCZOS

# ==================== CONFIGURATION ====================
# Apify is used only to download the source YouTube video. Kept hardcoded
# for now per current testing setup — rotate/move to a secret before sharing
# this app publicly.
APIFY_API_TOKEN = "apify_api_ddMBOcNe4LMijVOsErHSJDfvgUMlfE1Pi3LQ"
APIFY_ACTOR_ID = "streamers~youtube-video-downloader"

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"

SHORTS_WIDTH = 1080
SHORTS_HEIGHT = 1920
SHORTS_MAX_DURATION = 59  # seconds, YouTube Shorts cap is 60s


# ==================== YOUTUBE URL HELPERS ====================

def clean_youtube_url(url):
    url = url.strip()
    patterns = [
        r'(?:youtube\.com\/watch\?v=)([\w-]+)',
        r'(?:youtu\.be\/)([\w-]+)',
        r'(?:youtube\.com\/embed\/)([\w-]+)',
        r'(?:youtube\.com\/v\/)([\w-]+)',
        r'(?:youtube\.com\/shorts\/)([\w-]+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return f"https://www.youtube.com/watch?v={match.group(1)}"
    return url


def is_valid_youtube_url(url):
    patterns = [
        r'^https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+',
        r'^https?://(?:www\.)?youtu\.be/[\w-]+',
        r'^https?://(?:www\.)?youtube\.com/embed/[\w-]+',
        r'^https?://(?:www\.)?youtube\.com/shorts/[\w-]+'
    ]
    return any(re.match(p, url) for p in patterns)


# ==================== DOWNLOAD (APIFY) ====================

def _find_best_video_url(video_data):
    """
    Prefer the stable Apify-hosted copy ('downloadedFileUrl') over signed
    googlevideo.com URLs, which are IP-locked to the server that requested
    them and will 403 when fetched from elsewhere.
    Returns (url, needs_apify_auth).
    """
    stable_fields = ['downloadedFileUrl', 'fileUrl']
    for field in stable_fields:
        value = video_data.get(field)
        if isinstance(value, str) and value.startswith('http'):
            return value, True

    priority_fields = [
        'muxedUrl', 'combinedUrl', 'videoUrl', 'hdUrl', 'sdUrl',
        'downloadUrl', 'url', 'download', 'link', 'videoOnlyUrl'
    ]
    for field in priority_fields:
        value = video_data.get(field)
        if isinstance(value, str) and value.startswith('http'):
            return value, False

    for key in ('video', 'file', 'result', 'formats'):
        nested = video_data.get(key)
        if isinstance(nested, dict):
            for sub_field in ('url', 'downloadUrl', 'muxedUrl'):
                if nested.get(sub_field):
                    return nested[sub_field], False
        elif isinstance(nested, list) and nested:
            first = nested[0]
            if isinstance(first, dict):
                for sub_field in ('url', 'downloadUrl'):
                    if first.get(sub_field):
                        return first[sub_field], False

    for key, value in video_data.items():
        if isinstance(value, str) and value.startswith('http') and 'googlevideo.com' in value:
            if 'audio' not in key.lower():
                return value, False

    audio_only = video_data.get('audioOnlyUrl')
    if audio_only:
        return audio_only, False

    return None, False


def download_file_from_url(download_url, output_filename, extra_headers=None, progress_callback=None):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'video/webm,video/ogg,video/*;q=0.9,*/*;q=0.8',
    }
    if extra_headers:
        headers.update(extra_headers)

    resp = requests.get(download_url, stream=True, headers=headers, allow_redirects=True)
    resp.raise_for_status()

    total_size = int(resp.headers.get('content-length', 0))
    downloaded = 0

    with open(output_filename, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size and progress_callback:
                    progress_callback(min(downloaded / total_size, 1.0))

    return output_filename


def download_video_apify(youtube_url, output_filename, status_placeholder=None, progress_bar=None):
    cleaned_url = clean_youtube_url(youtube_url)
    if not is_valid_youtube_url(cleaned_url):
        raise RuntimeError(f"Invalid YouTube URL: {cleaned_url}")

    input_data = {"videos": [{"url": cleaned_url}]}
    api_base = "https://api.apify.com/v2"
    headers = {
        'Authorization': f'Bearer {APIFY_API_TOKEN}',
        'Content-Type': 'application/json'
    }

    start_url = f"{api_base}/actors/{APIFY_ACTOR_ID}/runs"
    response = requests.post(start_url, json=input_data, headers=headers)
    if response.status_code != 201:
        raise RuntimeError(f"Failed to start actor (status {response.status_code}): {response.text[:400]}")

    run_id = response.json()['data']['id']

    max_attempts = 120
    attempts = 0
    status_data = None

    while attempts < max_attempts:
        status_response = requests.get(f"{api_base}/actor-runs/{run_id}", headers=headers)
        if status_response.status_code != 200:
            time.sleep(5)
            attempts += 1
            continue

        status_data = status_response.json()
        current_status = status_data['data']['status']
        if status_placeholder:
            status_placeholder.text(f"Downloading video... actor status: {current_status}")

        if current_status == 'SUCCEEDED':
            break
        elif current_status in ['FAILED', 'ABORTED', 'TIMED-OUT']:
            raise RuntimeError(f"Actor run {current_status}")

        attempts += 1
        time.sleep(5)

    if attempts >= max_attempts:
        raise RuntimeError("Actor run timed out after 10 minutes")

    dataset_id = status_data['data']['defaultDatasetId']
    results_response = requests.get(f"{api_base}/datasets/{dataset_id}/items", headers=headers)
    results = results_response.json()

    if not results:
        raise RuntimeError("No results returned from actor")

    video_data = results[0]
    download_url, needs_apify_auth = _find_best_video_url(video_data)

    if not download_url:
        raise RuntimeError(f"No download URL found. Fields received: {list(video_data.keys())}")

    if status_placeholder:
        status_placeholder.text("Downloading video file...")

    auth_header = {'Authorization': f'Bearer {APIFY_API_TOKEN}'} if needs_apify_auth else None

    def _progress(frac):
        if progress_bar:
            progress_bar.progress(frac)

    return download_file_from_url(download_url, output_filename, extra_headers=auth_header, progress_callback=_progress)


# ==================== TRANSCRIBE ====================

@st.cache_resource(show_spinner=False)
def load_whisper_model(model_name="base"):
    # CPU + int8 quantization keeps this fast and light on Streamlit Cloud's
    # free-tier resources. No torch/triton/CUDA needed at all.
    return WhisperModel(model_name, device="cpu", compute_type="int8")


def transcribe_video(video_path, model_name="base"):
    model = load_whisper_model(model_name)
    audio_path = os.path.join(os.path.dirname(video_path), "temp_audio.wav")
    os.system(f'ffmpeg -y -i "{video_path}" -ar 16000 -ac 1 -b:a 64k -f mp3 "{audio_path}"')

    segments_iter, info = model.transcribe(audio_path, beam_size=5)
    transcription = []
    for seg in segments_iter:
        transcription.append({
            'start': seg.start,
            'end': seg.end,
            'text': seg.text.strip()
        })
    return transcription


# ==================== FIND RELEVANT SEGMENTS (GROQ) ====================

def _extract_balanced_block(text, start_index, open_char, close_char):
    depth = 0
    in_string = False
    escaped = False
    for i in range(start_index, len(text)):
        ch = text[i]
        if ch == "\\" and not escaped:
            escaped = True
            continue
        if ch == '"' and not escaped:
            in_string = not in_string
        if not in_string:
            if ch == open_char:
                depth += 1
            elif ch == close_char:
                depth -= 1
                if depth == 0:
                    return text[start_index:i + 1]
        escaped = False
    raise ValueError("Unbalanced JSON block")


def _extract_json_payload(text):
    text = text.strip()
    last_json_object = None
    for match in re.finditer(r"\{", text):
        try:
            candidate = _extract_balanced_block(text, match.start(), '{', '}')
        except ValueError:
            continue
        if '"conversations"' in candidate or "'conversations'" in candidate:
            last_json_object = candidate
    if last_json_object:
        return last_json_object

    for opener, closer in [('[', ']'), ('{', '}')]:
        for match in reversed(list(re.finditer(re.escape(opener), text))):
            try:
                return _extract_balanced_block(text, match.start(), opener, closer)
            except ValueError:
                continue
    raise ValueError('No JSON payload found in model response')


def _estimate_tokens(text):
    return max(1, len(text) // 4)


def _chunk_transcript(transcript, max_tokens_per_chunk=4000):
    chunks, current_chunk, current_tokens = [], [], 0
    for seg in transcript:
        seg_tokens = _estimate_tokens(json.dumps(seg))
        if current_chunk and current_tokens + seg_tokens > max_tokens_per_chunk:
            chunks.append(current_chunk)
            current_chunk, current_tokens = [], 0
        current_chunk.append(seg)
        current_tokens += seg_tokens
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def _call_groq(prompt_transcript, user_query, groq_api_key, max_retries=4):
    prompt = f"""You are an expert video editor who can read video transcripts and perform video editing. Given a transcript with segments, your task is to identify all the conversations related to a user query. A group of continuous segments in the transcript is a conversation.

Guidelines:
1. The conversation should be relevant to the user query and include more than one segment for context.
2. Include all before/after segments needed to make it complete.
3. Do not cut off mid-sentence or mid-idea.
4. Choose multiple conversations if relevant.
5. Match start/end using the segment timestamps given.
6. This transcript may be a partial chunk of a longer video. Only use segments given below — do not invent timestamps outside this range. If nothing is relevant, return an empty list.
7. Prefer segments that are self-contained enough to work as a standalone short video (roughly 15-59 seconds long).

Output format: {{ "conversations": [{{"start": "s1", "end": "e1"}}, {{"start": "s2", "end": "e2"}}] }}

Important: respond with valid JSON only. No extra text, explanation, or markdown. If nothing found, return {{ "conversations": [] }}.

Transcript:
{prompt_transcript}

User query:
{user_query}"""

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {groq_api_key}"}
    data = {
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Return only the JSON object described above. No explanation, no markdown, no extra text."}
        ],
        "model": GROQ_MODEL,
        "temperature": 0.2,
        "max_tokens": 1024,
        "top_p": 1,
        "stream": False,
        "response_format": {"type": "json_object"},
        "stop": None
    }

    for attempt in range(1, max_retries + 1):
        response = requests.post(GROQ_API_URL, headers=headers, json=data)

        if response.status_code in (429, 413):
            wait = min(2 ** attempt, 30)
            time.sleep(wait)
            continue

        try:
            response_data = response.json()
        except ValueError:
            raise RuntimeError(f"Non-JSON response from Groq: {response.text}")

        if response.status_code != 200:
            raise RuntimeError(f"Groq API request failed ({response.status_code}): {response_data}")

        if not response_data.get("choices"):
            raise RuntimeError(f"Unexpected Groq API response structure: {response_data}")

        choice = response_data["choices"][0]
        message = choice.get("message") or choice
        raw_content = message.get("content") if isinstance(message, dict) else None
        if raw_content is None:
            raise RuntimeError(f"Missing message content in Groq response: {response_data}")

        raw_content = raw_content.strip()
        json_text = _extract_json_payload(raw_content)

        try:
            conversations = json.loads(json_text)
        except ValueError:
            conversations = ast.literal_eval(json_text)

        if isinstance(conversations, list):
            return conversations
        return conversations.get("conversations", [])

    raise RuntimeError("Exceeded max retries due to repeated rate limiting.")


def get_relevant_segments(transcript, user_query, groq_api_key, status_placeholder=None,
                           max_tokens_per_chunk=4000, delay_between_chunks=2.0):
    chunks = _chunk_transcript(transcript, max_tokens_per_chunk=max_tokens_per_chunk)
    all_conversations = []
    for i, chunk in enumerate(chunks, start=1):
        if status_placeholder:
            status_placeholder.text(f"Analyzing transcript with Groq... chunk {i}/{len(chunks)}")
        try:
            all_conversations.extend(_call_groq(chunk, user_query, groq_api_key))
        except RuntimeError as exc:
            st.warning(f"Chunk {i} failed and was skipped: {exc}")
        if i < len(chunks):
            time.sleep(delay_between_chunks)
    return all_conversations


# ==================== CREATE SHORTS ====================

def _to_vertical(clip):
    target_ratio = SHORTS_WIDTH / SHORTS_HEIGHT
    clip_ratio = clip.w / clip.h
    if clip_ratio > target_ratio:
        new_width = int(clip.h * target_ratio)
        x_center = clip.w / 2
        clip = clip.crop(x1=x_center - new_width / 2, x2=x_center + new_width / 2)
    else:
        new_height = int(clip.w / target_ratio)
        y_center = clip.h / 2
        clip = clip.crop(y1=y_center - new_height / 2, y2=y_center + new_height / 2)
    return clip.resize((SHORTS_WIDTH, SHORTS_HEIGHT))


def create_shorts(original_video_path, segments, output_dir, fade_duration=0.4, status_placeholder=None):
    os.makedirs(output_dir, exist_ok=True)
    video = VideoFileClip(original_video_path)
    created_files = []

    for i, seg in enumerate(segments, start=1):
        try:
            start = float(seg['start'])
            end = float(seg['end'])
        except (KeyError, ValueError, TypeError):
            continue

        if end <= start:
            continue

        if end - start > SHORTS_MAX_DURATION:
            end = start + SHORTS_MAX_DURATION

        if status_placeholder:
            status_placeholder.text(f"Rendering short {i}/{len(segments)}...")

        clip = video.subclip(start, end)
        clip = _to_vertical(clip)
        clip = clip.fadein(fade_duration).fadeout(fade_duration)

        output_path = os.path.join(output_dir, f"short_{i}.mp4")
        clip.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=30,
                              verbose=False, logger=None)
        created_files.append({
            "path": output_path,
            "start": start,
            "end": end,
            "duration": end - start
        })

    video.close()
    return created_files


# ==================== STREAMLIT APP ====================

st.set_page_config(page_title="Clip Anything — YouTube Shorts Generator", page_icon="🎬", layout="wide")

st.title("🎬 Clip Anything")
st.caption("Paste a YouTube link, describe what you're looking for, and get vertical short clips ready to download.")

with st.sidebar:
    st.header("Settings")
    groq_api_key = st.text_input("Groq API Key", type="password", help="Get one at console.groq.com")
    whisper_model_name = st.selectbox(
        "Transcription model (Whisper)",
        options=["tiny", "base", "small", "medium"],
        index=1,
        help="Larger models are more accurate but slower."
    )
    st.divider()
    st.caption("The video is downloaded via a hosted service. No extra configuration needed for that step.")

st.subheader("1. YouTube video")
youtube_url = st.text_input("YouTube video URL", placeholder="https://www.youtube.com/watch?v=...")

st.subheader("2. What should the shorts be about?")
user_query = st.text_area(
    "Describe the moments you want clipped out",
    value="Find the most engaging, self-contained moments in this video",
    height=80
)

generate_clicked = st.button("Generate Shorts", type="primary", use_container_width=True)

if generate_clicked:
    if not groq_api_key:
        st.error("Please enter your Groq API key in the sidebar.")
    elif not youtube_url or not is_valid_youtube_url(clean_youtube_url(youtube_url)):
        st.error("Please enter a valid YouTube video URL.")
    else:
        work_dir = tempfile.mkdtemp(prefix="clip_anything_")
        input_video_path = os.path.join(work_dir, "input_video.mp4")
        shorts_dir = os.path.join(work_dir, "shorts")

        status = st.empty()
        progress_bar = st.progress(0.0)

        try:
            # Step 1: Download
            status.text("Starting download...")
            download_video_apify(youtube_url, input_video_path, status_placeholder=status, progress_bar=progress_bar)
            progress_bar.progress(1.0)
            status.text("Download complete.")

            # Step 2: Transcribe
            status.text(f"Transcribing video with Whisper ({whisper_model_name})... this can take a while.")
            transcription = transcribe_video(input_video_path, model_name=whisper_model_name)
            status.text(f"Transcription complete: {len(transcription)} segments found.")

            # Step 3: Find relevant segments
            relevant_segments = get_relevant_segments(
                transcription, user_query, groq_api_key, status_placeholder=status
            )

            if not relevant_segments:
                status.empty()
                progress_bar.empty()
                st.warning("No relevant segments were found for that description. Try rephrasing your query.")
            else:
                # Step 4: Create shorts
                status.text(f"Creating {len(relevant_segments)} short video(s)...")
                shorts = create_shorts(input_video_path, relevant_segments, shorts_dir, status_placeholder=status)

                status.empty()
                progress_bar.empty()

                if not shorts:
                    st.warning("Segments were found, but none produced a valid clip.")
                else:
                    st.success(f"Done! Created {len(shorts)} short video(s).")

                    for i, short in enumerate(shorts, start=1):
                        st.markdown(f"**Short {i}** — {short['start']:.1f}s to {short['end']:.1f}s "
                                    f"({short['duration']:.1f}s)")
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.video(short["path"])
                        with col2:
                            with open(short["path"], "rb") as f:
                                st.download_button(
                                    label="⬇️ Download",
                                    data=f.read(),
                                    file_name=f"short_{i}.mp4",
                                    mime="video/mp4",
                                    key=f"download_{i}",
                                    use_container_width=True
                                )
                        st.divider()

        except Exception as e:
            status.empty()
            progress_bar.empty()
            st.error(f"Something went wrong: {e}")

        finally:
            # Clean up the large source video, but keep shorts around for this session.
            if os.path.exists(input_video_path):
                try:
                    os.remove(input_video_path)
                except OSError:
                    pass
