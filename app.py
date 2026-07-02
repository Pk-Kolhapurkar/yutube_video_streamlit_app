import os
import re
import time
import json
import ast
import tempfile
import streamlit as st
import requests
from bs4 import BeautifulSoup
from moviepy import VideoFileClip, concatenate_videoclips
import whisper

# ---- Compatibility shim for moviepy/Pillow ----
from PIL import Image
if not hasattr(Image, 'ANTIALIAS'):
    Image.ANTIALIAS = Image.LANCZOS

# ==================== HARDCODED CONFIGURATION ====================
# WARNING: These are hardcoded for testing. Rotate/revoke before sharing.
APIFY_API_TOKEN = "apify_api_ddMBOcNe4LMijVOsErHSJDfvgUMlfE1Pi3LQ"
APIFY_ACTOR_ID = "streamers~youtube-video-downloader"
GROQ_API_KEY = "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"

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="YouTube Shorts Creator 🎬",
    page_icon="🎬",
    layout="wide"
)

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'transcript' not in st.session_state:
    st.session_state.transcript = None
if 'segments' not in st.session_state:
    st.session_state.segments = None
if 'shorts_created' not in st.session_state:
    st.session_state.shorts_created = []
if 'video_title' not in st.session_state:
    st.session_state.video_title = ""
if 'trending_videos' not in st.session_state:
    st.session_state.trending_videos = []

# ==================== VIDEO DOWNLOAD FUNCTIONS ====================

def clean_youtube_url(url):
    """Clean and validate YouTube URL."""
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
            video_id = match.group(1)
            return f"https://www.youtube.com/watch?v={video_id}"
    
    return url

def extract_video_id(url):
    """Extract video ID from a YouTube URL."""
    patterns = [
        r'v=([\w-]+)',
        r'youtu\.be/([\w-]+)',
        r'embed/([\w-]+)',
        r'shorts/([\w-]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def is_valid_youtube_url(url):
    """Check if a URL is a valid YouTube URL."""
    patterns = [
        r'^https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+',
        r'^https?://(?:www\.)?youtu\.be/[\w-]+',
        r'^https?://(?:www\.)?youtube\.com/embed/[\w-]+',
        r'^https?://(?:www\.)?youtube\.com/shorts/[\w-]+'
    ]
    
    for pattern in patterns:
        if re.match(pattern, url):
            return True
    return False

def get_top_videos():
    """Scrape top 10 most viewed videos from Kworb.net."""
    url = 'https://kworb.net/youtube/'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table', id='youtuberealtime')
        
        if not table:
            return []
        
        rows = table.find('tbody').find_all('tr')
        top_videos = []
        
        for row in rows[:10]:
            cols = row.find_all('td')
            if len(cols) < 5:
                continue
                
            rank = cols[0].get_text(strip=True)
            video_cell = cols[2]
            kworb_link_tag = video_cell.find('a')
            
            if not kworb_link_tag:
                continue
                
            kworb_link = kworb_link_tag['href']
            video_title = kworb_link_tag.get_text(strip=True)
            
            video_id = kworb_link.split('/')[-1].replace('.html', '')
            youtube_url = f'https://www.youtube.com/watch?v={video_id}'
            
            views = cols[3].get_text(strip=True)
            likes = cols[4].get_text(strip=True)
            
            top_videos.append({
                'rank': rank,
                'title': video_title,
                'video_id': video_id,
                'youtube_url': youtube_url,
                'views': views,
                'likes': likes
            })
        
        return top_videos
        
    except Exception as e:
        st.error(f"Error fetching top videos: {e}")
        return []

def _find_best_video_url(video_data):
    """Find the best video URL from Apify response."""
    # Prefer Apify-hosted stable copies
    stable_fields = ['downloadedFileUrl', 'fileUrl']
    for field in stable_fields:
        value = video_data.get(field)
        if isinstance(value, str) and value.startswith('http'):
            return value, True
    
    # Fall back to signed URLs
    priority_fields = [
        'muxedUrl', 'combinedUrl', 'videoUrl', 'hdUrl', 'sdUrl',
        'downloadUrl', 'url', 'download', 'link', 'videoOnlyUrl'
    ]
    for field in priority_fields:
        value = video_data.get(field)
        if isinstance(value, str) and value.startswith('http'):
            return value, False
    
    # Check nested structures
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
    
    return None, False

def download_video_apify(youtube_url, output_filename='input_video.mp4'):
    """Download a YouTube video using Apify's API."""
    try:
        cleaned_url = clean_youtube_url(youtube_url)
        
        if not is_valid_youtube_url(cleaned_url):
            return False, f"Invalid YouTube URL: {cleaned_url}"
        
        video_id = extract_video_id(cleaned_url)
        
        input_data = {"videos": [{"url": cleaned_url}]}
        
        api_base = "https://api.apify.com/v2"
        headers = {
            'Authorization': f'Bearer {APIFY_API_TOKEN}',
            'Content-Type': 'application/json'
        }
        
        start_url = f"{api_base}/actors/{APIFY_ACTOR_ID}/runs"
        response = requests.post(start_url, json=input_data, headers=headers)
        
        if response.status_code != 201:
            error_detail = response.text[:500]
            return False, f"Failed to start actor (status {response.status_code}): {error_detail}"
        
        run_data = response.json()
        run_id = run_data['data']['id']
        
        # Wait for completion
        max_attempts = 120
        attempts = 0
        status_data = None
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        while attempts < max_attempts:
            status_url = f"{api_base}/actor-runs/{run_id}"
            status_response = requests.get(status_url, headers=headers)
            
            if status_response.status_code != 200:
                time.sleep(5)
                attempts += 1
                continue
            
            status_data = status_response.json()
            current_status = status_data['data']['status']
            
            # Update progress
            progress = min(attempts / max_attempts, 0.9)
            progress_bar.progress(progress)
            status_text.text(f"Downloading video... Status: {current_status}")
            
            if current_status == 'SUCCEEDED':
                break
            elif current_status in ['FAILED', 'ABORTED', 'TIMED-OUT']:
                return False, f"Actor run {current_status}"
            
            attempts += 1
            time.sleep(5)
        
        progress_bar.progress(1.0)
        status_text.text("Download complete! Processing results...")
        
        if attempts >= max_attempts:
            return False, "Actor run timed out after 10 minutes"
        
        # Get results
        dataset_id = status_data['data']['defaultDatasetId']
        results_url = f"{api_base}/datasets/{dataset_id}/items"
        results_response = requests.get(results_url, headers=headers)
        
        if results_response.status_code != 200:
            return False, f"Failed to get results: {results_response.text[:200]}"
        
        results = results_response.json()
        
        if not results:
            # Try key-value store
            store_id = status_data['data']['defaultKeyValueStoreId']
            store_url = f"{api_base}/key-value-stores/{store_id}/records"
            store_response = requests.get(store_url, headers=headers)
            
            if store_response.status_code == 200:
                store_data = store_response.json()
                for key, value in store_data.items():
                    if key.endswith(('.mp4', '.webm', '.mkv')):
                        download_url = f"{api_base}/key-value-stores/{store_id}/records/{key}"
                        return download_file_from_url(download_url, output_filename)
            
            return False, "No results returned from actor"
        
        video_data = results[0]
        download_url, needs_apify_auth = _find_best_video_url(video_data)
        
        if not download_url:
            # Try key-value store as fallback
            store_id = status_data['data']['defaultKeyValueStoreId']
            store_url = f"{api_base}/key-value-stores/{store_id}/records"
            store_response = requests.get(store_url, headers=headers)
            
            if store_response.status_code == 200:
                store_data = store_response.json()
                for key, value in store_data.items():
                    if key.endswith(('.mp4', '.webm', '.mkv')):
                        download_url = f"{api_base}/key-value-stores/{store_id}/records/{key}?attachment=true"
                        needs_apify_auth = True
                        break
            
            if not download_url:
                return False, f"No download URL found. Available fields: {list(video_data.keys())}"
        
        auth_header = {'Authorization': f'Bearer {APIFY_API_TOKEN}'} if needs_apify_auth else None
        return download_file_from_url(download_url, output_filename, extra_headers=auth_header)
        
    except Exception as e:
        return False, f"Error: {str(e)}"

def download_file_from_url(download_url, output_filename, extra_headers=None):
    """Download a file from URL with progress tracking."""
    try:
        download_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'video/webm,video/ogg,video/*;q=0.9,*/*;q=0.8',
        }
        if extra_headers:
            download_headers.update(extra_headers)
        
        file_response = requests.get(download_url, stream=True, headers=download_headers, allow_redirects=True)
        file_response.raise_for_status()
        
        total_size = int(file_response.headers.get('content-length', 0))
        
        with open(output_filename, 'wb') as f:
            downloaded = 0
            for chunk in file_response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
        
        return True, output_filename
        
    except Exception as e:
        return False, f"Download error: {str(e)}"

def download_video(youtube_url, output_filename='input_video.mp4'):
    """Main download function."""
    return download_video_apify(youtube_url, output_filename)

# ==================== TRANSCRIPTION ====================

@st.cache_resource
def load_whisper_model(model_name="base"):
    with st.spinner(f"Loading Whisper {model_name} model..."):
        return whisper.load_model(model_name)

def transcribe_video(video_path, model_name="base"):
    with st.spinner("Transcribing video... This may take a while..."):
        model = load_whisper_model(model_name)
        
        # Extract audio using moviepy
        video = VideoFileClip(video_path)
        audio_path = os.path.join(os.path.dirname(video_path), "temp_audio.wav")
        video.audio.write_audiofile(audio_path, verbose=False, logger=None)
        video.close()
        
        # Transcribe
        result = model.transcribe(audio_path)
        transcription = []
        for segment in result['segments']:
            transcription.append({
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'].strip()
            })
        
        # Clean up
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        return transcription

# ==================== GROQ ANALYSIS ====================

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
    chunks = []
    current_chunk = []
    current_tokens = 0
    for seg in transcript:
        seg_text = json.dumps(seg)
        seg_tokens = _estimate_tokens(seg_text)
        if current_chunk and current_tokens + seg_tokens > max_tokens_per_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_tokens = 0
        current_chunk.append(seg)
        current_tokens += seg_tokens
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def call_groq(prompt_transcript, user_query, max_retries=4):
    prompt = f"""You are an expert video editor who can read video transcripts and perform video editing. Given a transcript with segments, your task is to identify all the conversations related to a user query. A group of continuous segments in the transcript is a conversation.

Guidelines:
1. The conversation should be relevant to the user query. The conversation should include more than one segment to provide context and continuity.
2. Include all the before and after segments needed in a conversation to make it complete.
3. The conversation should not cut off in the middle of a sentence or idea.
4. Choose multiple conversations from the transcript that are relevant to the user query.
5. Match the start and end time of the conversations using the segment timestamps from the transcript.
6. The conversations should be a direct part of the video and should not be out of context.
7. Each segment should be ideally between 15-59 seconds long for YouTube Shorts format.
8. This transcript may be a partial chunk of a longer video. Only use the segments given to you below — do not invent timestamps outside this range. If nothing in this chunk is relevant, return an empty list.

Output format: {{ "conversations": [{{"start": "s1", "end": "e1"}}, {{"start": "s2", "end": "e2"}}] }}

Important: respond with valid JSON only. Do not include any extra text, explanation, or markdown. If you cannot find any relevant conversation, return {{ "conversations": [] }}.

Transcript:
{prompt_transcript}

User query:
{user_query}"""

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }
    data = {
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Return only the JSON object described above. No explanation, no markdown, no extra text."}
        ],
        "model": "llama-3.1-8b-instant",
        "temperature": 0.2,
        "max_tokens": 1024,
        "top_p": 1,
        "stream": False,
        "response_format": {"type": "json_object"},
        "stop": None
    }

    for attempt in range(1, max_retries + 1):
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 429 or response.status_code == 413:
            wait = min(2 ** attempt, 30)
            time.sleep(wait)
            continue
        
        try:
            response_data = response.json()
        except ValueError:
            raise RuntimeError(f"Non-JSON response from API: {response.text}")
        
        if response.status_code != 200:
            raise RuntimeError(f"API request failed ({response.status_code}): {response_data}")
        
        if not response_data.get("choices"):
            raise RuntimeError(f"Unexpected API response structure: {response_data}")
        
        choice = response_data["choices"][0]
        message = choice.get("message") or choice
        raw_content = message.get("content") if isinstance(message, dict) else None
        if raw_content is None:
            raise RuntimeError(f"Missing message content in API response: {response_data}")
        
        raw_content = raw_content.strip()
        try:
            json_text = _extract_json_payload(raw_content)
        except ValueError as exc:
            raise RuntimeError(f"Could not extract JSON payload from API response: {raw_content}\nError: {exc}")
        
        try:
            conversations = json.loads(json_text)
        except ValueError:
            try:
                conversations = ast.literal_eval(json_text)
            except Exception as exc:
                raise RuntimeError(f"Could not parse extracted response content as JSON: {json_text}\nError: {exc}")
        
        if isinstance(conversations, list):
            return conversations
        
        if not isinstance(conversations, dict) or "conversations" not in conversations:
            raise RuntimeError(f"Parsed API response does not contain conversations: {conversations}")
        
        return conversations["conversations"]
    
    raise RuntimeError("Exceeded max retries due to repeated rate limiting.")

def get_relevant_segments(transcript, user_query, max_tokens_per_chunk=4000, delay_between_chunks=2.0):
    chunks = _chunk_transcript(transcript, max_tokens_per_chunk=max_tokens_per_chunk)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_conversations = []
    total_chunks = len(chunks)
    
    for i, chunk in enumerate(chunks, start=1):
        status_text.text(f"Processing chunk {i}/{total_chunks}...")
        try:
            conversations = call_groq(chunk, user_query)
            all_conversations.extend(conversations)
        except RuntimeError as exc:
            st.warning(f"Chunk {i} failed and will be skipped: {exc}")
        
        progress_bar.progress(i / total_chunks)
        if i < len(chunks):
            time.sleep(delay_between_chunks)
    
    status_text.empty()
    progress_bar.empty()
    return all_conversations

# ==================== CREATE SHORTS ====================

def create_shorts(original_video_path, segments, output_dir="shorts", fade_duration=0.4):
    """Create vertical shorts from video segments."""
    os.makedirs(output_dir, exist_ok=True)
    video = VideoFileClip(original_video_path)
    created_files = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, seg in enumerate(segments, start=1):
        status_text.text(f"Creating short {i}/{len(segments)}...")
        
        try:
            start = float(seg['start'])
            end = float(seg['end'])
        except (KeyError, ValueError, TypeError):
            st.warning(f"Skipping malformed segment: {seg}")
            continue
        
        if end <= start:
            st.warning(f"Skipping invalid segment: {seg}")
            continue
        
        # Cap at 59 seconds for YouTube Shorts
        if end - start > 59:
            end = start + 59
        
        try:
            # Extract and convert to vertical
            clip = video.subclip(start, end)
            
            # Convert to vertical (9:16)
            target_ratio = 1080 / 1920  # 0.5625
            clip_ratio = clip.w / clip.h
            
            if clip_ratio > target_ratio:
                new_width = int(clip.h * target_ratio)
                x_center = clip.w / 2
                clip = clip.crop(x1=x_center - new_width / 2, x2=x_center + new_width / 2)
            else:
                new_height = int(clip.w / target_ratio)
                y_center = clip.h / 2
                clip = clip.crop(y1=y_center - new_height / 2, y2=y_center + new_height / 2)
            
            clip = clip.resize((1080, 1920))
            clip = clip.fadein(fade_duration).fadeout(fade_duration)
            
            output_path = os.path.join(output_dir, f"short_{i}.mp4")
            clip.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=30, verbose=False, logger=None)
            created_files.append(output_path)
            clip.close()
            
        except Exception as e:
            st.warning(f"Failed to create short {i}: {str(e)}")
            continue
        
        progress_bar.progress(i / len(segments))
    
    video.close()
    status_text.empty()
    progress_bar.empty()
    
    return created_files

# ==================== STREAMLIT UI ====================

st.title("🎬 YouTube Shorts Creator")
st.markdown("Create engaging vertical shorts from YouTube videos using AI")

# Main content
tab1, tab2 = st.tabs(["🎯 Create Shorts", "📊 Trending Videos"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Video URL input
        video_url = st.text_input(
            "📹 YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Enter any YouTube video URL"
        )
        
        # User query
        user_query = st.text_area(
            "📝 What kind of moments do you want?",
            placeholder="e.g., 'Find the most engaging, funny, or educational moments that would work as shorts'",
            height=80
        )
        
        # Process button
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            process_button = st.button(
                "🚀 Create Shorts",
                type="primary",
                use_container_width=True
            )
        with col_btn2:
            use_trending = st.button(
                "🔥 Use Trending Video",
                use_container_width=True
            )
        
        # Show API status
        st.info("✅ API keys are pre-configured and ready to use!")
    
    with col2:
        st.markdown("### 📊 Status")
        status_placeholder = st.empty()
        status_placeholder.info("Ready to process")
        
        # Display results if available
        if st.session_state.processed and st.session_state.shorts_created:
            st.success(f"✅ Created {len(st.session_state.shorts_created)} shorts")
            
            # Preview first short
            if st.session_state.shorts_created:
                first_short = st.session_state.shorts_created[0]
                if os.path.exists(first_short):
                    st.video(first_short)
            
            # Download buttons for all shorts
            st.markdown("### 📥 Download Shorts")
            for i, short_path in enumerate(st.session_state.shorts_created, 1):
                if os.path.exists(short_path):
                    with open(short_path, "rb") as f:
                        video_data = f.read()
                    st.download_button(
                        label=f"📥 Short {i}",
                        data=video_data,
                        file_name=f"short_{i}.mp4",
                        mime="video/mp4",
                        use_container_width=True,
                        key=f"download_{i}"
                    )
    
    # Processing logic
    if use_trending:
        video_url = None
        status_placeholder.info("🔥 Fetching trending videos...")
        top_videos = get_top_videos()
        if top_videos:
            video_url = top_videos[0]['youtube_url']
            st.session_state.video_title = top_videos[0]['title']
            status_placeholder.info(f"Selected: {top_videos[0]['title']}")
            st.success(f"✅ Selected trending video: {top_videos[0]['title']}")
            # Auto-fill the URL
            st.rerun()
        else:
            status_placeholder.error("❌ Failed to fetch trending videos")
    
    if process_button:
        if not video_url:
            st.error("Please enter a YouTube URL or use a trending video")
        elif not user_query:
            st.error("Please describe what kind of moments you want")
        else:
            try:
                # Create temp directory
                with tempfile.TemporaryDirectory() as temp_dir:
                    status_placeholder.info("📥 Downloading video...")
                    
                    # Download video
                    input_video = os.path.join(temp_dir, "input_video.mp4")
                    success, result = download_video(video_url, input_video)
                    
                    if not success:
                        st.error(f"Download failed: {result}")
                        status_placeholder.error("❌ Download failed")
                        st.stop()
                    
                    video_path = result
                    status_placeholder.info("✅ Video downloaded successfully!")
                    
                    # Transcribe
                    status_placeholder.info("🎤 Transcribing video...")
                    transcription = transcribe_video(video_path, "base")
                    st.session_state.transcript = transcription
                    status_placeholder.info(f"✅ Transcription complete: {len(transcription)} segments")
                    
                    # Get relevant segments
                    status_placeholder.info("🧠 Analyzing transcript for relevant segments...")
                    relevant_segments = get_relevant_segments(
                        transcription,
                        user_query,
                        max_tokens_per_chunk=4000
                    )
                    st.session_state.segments = relevant_segments
                    status_placeholder.info(f"✅ Found {len(relevant_segments)} relevant segments")
                    
                    if not relevant_segments:
                        st.warning("No relevant segments found. Try a different query.")
                        status_placeholder.warning("No segments found")
                        st.stop()
                    
                    # Create shorts
                    status_placeholder.info("🎬 Creating vertical shorts...")
                    shorts_dir = os.path.join(temp_dir, "shorts")
                    shorts_files = create_shorts(
                        video_path,
                        relevant_segments,
                        output_dir=shorts_dir,
                        fade_duration=0.4
                    )
                    
                    if shorts_files:
                        # Copy shorts to current directory for download
                        import shutil
                        st.session_state.shorts_created = []
                        for i, short_path in enumerate(shorts_files, 1):
                            dest_path = f"short_{i}.mp4"
                            shutil.copy(short_path, dest_path)
                            st.session_state.shorts_created.append(dest_path)
                        
                        st.session_state.processed = True
                        status_placeholder.success(f"✅ Created {len(shorts_files)} shorts!")
                        st.balloons()
                        st.rerun()
                    else:
                        status_placeholder.error("❌ No shorts were created")
                        st.error("Failed to create shorts. Please check the segments and try again.")
            
            except Exception as e:
                status_placeholder.error(f"❌ Error: {str(e)}")
                st.exception(e)

with tab2:
    st.subheader("🔥 Trending YouTube Videos")
    st.markdown("Top 10 most viewed videos from Kworb.net")
    
    if st.button("🔄 Refresh Trending Videos"):
        with st.spinner("Fetching trending videos..."):
            trending_videos = get_top_videos()
            if trending_videos:
                st.session_state.trending_videos = trending_videos
                st.success(f"✅ Loaded {len(trending_videos)} trending videos!")
    
    if st.session_state.trending_videos:
        for video in st.session_state.trending_videos:
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.markdown(f"**#{video['rank']}** - {video['title']}")
                with col2:
                    st.caption(f"👁️ {video['views']} views")
                with col3:
                    if st.button("Select", key=f"select_{video['video_id']}"):
                        st.session_state.selected_url = video['youtube_url']
                        st.session_state.video_title = video['title']
                        st.success(f"✅ Selected: {video['title']}")
                        # Switch to first tab
                        st.rerun()

# Display transcript preview if available
if st.session_state.transcript:
    with st.expander("📄 View Transcript"):
        st.json(st.session_state.transcript[:10])
        if len(st.session_state.transcript) > 10:
            st.caption(f"... and {len(st.session_state.transcript) - 10} more segments")
