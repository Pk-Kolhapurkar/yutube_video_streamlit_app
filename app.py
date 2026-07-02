import os
import re
import time
import json
import ast
import tempfile
import streamlit as st
import requests
from moviepy import VideoFileClip, concatenate_videoclips
import whisper
from pytubefix import YouTube
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="AI Video Editor",
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

# Helper functions
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

@st.cache_resource
def load_whisper_model(model_name="base"):
    with st.spinner(f"Loading Whisper {model_name} model..."):
        return whisper.load_model(model_name)

def download_video(url, output_path):
    """Download video from YouTube or direct URL with validation"""
    video_path = os.path.join(output_path, "input_video.mp4")
    
    # 1. Try YouTube Download
    if "youtube.com" in url or "youtu.be" in url:
        try:
            yt = YouTube(url)
            # get_highest_resolution() is more reliable than progressive=True filters
            video = yt.streams.get_highest_resolution()
            if video:
                video.download(output_path=output_path, filename="input_video.mp4")
                
                # CRITICAL VALIDATION: Check if the file actually has data
                if os.path.exists(video_path) and os.path.getsize(video_path) > 1000:
                    return video_path
                else:
                    st.warning("YouTube download resulted in an empty file. Trying fallback direct download...")
        except Exception as e:
            st.warning(f"YouTube downloader failed: {str(e)}. Trying fallback direct download...")

    # 2. Try Direct Download (or fallback if YouTube scraped a dead link)
    try:
        # Use a real User-Agent so hosting servers don't block Streamlit
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(url, stream=True, headers=headers, timeout=30)
        
        if response.status_code == 200:
            with open(video_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            if os.path.exists(video_path) and os.path.getsize(video_path) > 1000:
                return video_path
    except Exception as e:
        pass
        
    raise ValueError("Could not download a valid video. Please verify the URL, or ensure the YouTube video is public and not age-restricted.")

def transcribe_video(video_path, model):
    with st.spinner("Transcribing video... This may take a while..."):
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

def call_groq(prompt_transcript, user_query, api_key, max_retries=4):
    prompt = f"""You are an expert video editor who can read video transcripts and perform video editing. Given a transcript with segments, your task is to identify all the conversations related to a user query. Follow these guidelines when choosing conversations. A group of continuous segments in the transcript is a conversation.

Guidelines:
1. The conversation should be relevant to the user query. The conversation should include more than one segment to provide context and continuity.
2. Include all the before and after segments needed in a conversation to make it complete.
3. The conversation should not cut off in the middle of a sentence or idea.
4. Choose multiple conversations from the transcript that are relevant to the user query.
5. Match the start and end time of the conversations using the segment timestamps from the transcript.
6. The conversations should be a direct part of the video and should not be out of context.
7. This transcript may be a partial chunk of a longer video. Only use the segments given to you below — do not invent timestamps outside this range. If nothing in this chunk is relevant, return an empty list.

Output format: {{ "conversations": [{{"start": "s1", "end": "e1"}}, {{"start": "s2", "end": "e2"}}] }}

Important: respond with valid JSON only. Do not include any extra text, explanation, or markdown. If you cannot find any relevant conversation, return {{ "conversations": [] }}.

Transcript:
{prompt_transcript}

User query:
{user_query}"""

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
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

def get_relevant_segments(transcript, user_query, api_key, max_tokens_per_chunk=4000, delay_between_chunks=2.0):
    chunks = _chunk_transcript(transcript, max_tokens_per_chunk=max_tokens_per_chunk)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_conversations = []
    total_chunks = len(chunks)
    
    for i, chunk in enumerate(chunks, start=1):
        status_text.text(f"Processing chunk {i}/{total_chunks}...")
        try:
            conversations = call_groq(chunk, user_query, api_key)
            all_conversations.extend(conversations)
        except RuntimeError as exc:
            st.warning(f"Chunk {i} failed and will be skipped: {exc}")
        
        progress_bar.progress(i / total_chunks)
        if i < len(chunks):
            time.sleep(delay_between_chunks)
    
    status_text.empty()
    progress_bar.empty()
    return all_conversations

def edit_video(original_video_path, segments, output_video_path, fade_duration=0.5):
    with st.spinner("Editing video..."):
        video = VideoFileClip(original_video_path)
        clips = []
        
        for i, seg in enumerate(segments, start=1):
            try:
                start = float(seg['start'])
                end = float(seg['end'])
            except (KeyError, ValueError, TypeError):
                st.warning(f"Skipping malformed segment: {seg}")
                continue
            
            if end <= start:
                st.warning(f"Skipping invalid segment (end <= start): {seg}")
                continue
            
            clip = video.subclip(start, end).fadein(fade_duration).fadeout(fade_duration)
            clips.append(clip)
        
        if clips:
            final_clip = concatenate_videoclips(clips, method="compose")
            final_clip.write_videofile(output_video_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)
            video.close()
            final_clip.close()
            for clip in clips:
                clip.close()
            return True
        else:
            st.error("No valid segments found to include in the edited video.")
            return False

# Streamlit UI
st.title("🎬 AI Video Editor")
st.markdown("Create short, focused videos from longer content using AI-powered transcript analysis")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key input
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Enter your Groq API key. Get one from https://console.groq.com"
    )
    
    # Model selection
    whisper_model = st.selectbox(
        "Whisper Model",
        ["base", "small", "medium", "large"],
        index=0,
        help="Larger models are more accurate but slower"
    )
    
    # Advanced options
    with st.expander("Advanced Options"):
        max_tokens = st.slider(
            "Max tokens per chunk",
            min_value=1000,
            max_value=8000,
            value=4000,
            step=500
        )
        fade_duration = st.slider(
            "Fade duration (seconds)",
            min_value=0.0,
            max_value=2.0,
            value=0.5,
            step=0.1
        )
    
    st.divider()
    st.markdown("### 📋 Instructions")
    st.markdown("""
    1. Enter your Groq API Key
    2. Paste a video URL (YouTube or direct link)
    3. Enter what you want to extract from the video
    4. Click 'Process Video'
    5. Wait for processing and download your edited video
    """)
    
    st.divider()
    st.markdown("### ℹ️ About")
    st.markdown("""
    This app uses:
    - **Whisper** for speech-to-text
    - **Groq's Llama** for smart segment selection
    - **MoviePy** for video editing
    """)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    # Video URL input
    video_url = st.text_input(
        "📹 Video URL",
        placeholder="https://www.youtube.com/watch?v=... or direct video URL",
        help="Enter a YouTube URL or direct link to a video file"
    )
    
    # User query
    user_query = st.text_area(
        "📝 What do you want to extract from the video?",
        placeholder="e.g., 'Create a 5-minute summary of all the key concepts explained in this video'",
        height=100
    )
    
    # Process button
    process_button = st.button(
        "🚀 Process Video",
        type="primary",
        use_container_width=True
    )

with col2:
    st.markdown("### 📊 Status")
    status_placeholder = st.empty()
    status_placeholder.info("Ready to process")
    
    # Display video info if available
    if st.session_state.processed and st.session_state.segments:
        st.success(f"✅ Found {len(st.session_state.segments)} relevant segments")
        
        # Download button
        if os.path.exists("edited_output.mp4"):
            with open("edited_output.mp4", "rb") as f:
                video_data = f.read()
            st.download_button(
                label="📥 Download Edited Video",
                data=video_data,
                file_name="edited_video.mp4",
                mime="video/mp4",
                use_container_width=True
            )

# Processing logic
if process_button:
    if not api_key:
        st.error("Please enter your Groq API key")
    elif not video_url:
        st.error("Please enter a video URL")
    elif not user_query:
        st.error("Please enter a user query")
    else:
        try:
            # Create temp directory
            with tempfile.TemporaryDirectory() as temp_dir:
                status_placeholder.info("📥 Downloading video...")
                
                # Download video
                video_path = download_video(video_url, temp_dir)
                
                # Load whisper model
                model = load_whisper_model(whisper_model)
                
                # Transcribe
                transcription = transcribe_video(video_path, model)
                st.session_state.transcript = transcription
                status_placeholder.info(f"✅ Transcription complete: {len(transcription)} segments")
                
                # Get relevant segments
                status_placeholder.info("🧠 Analyzing transcript for relevant segments...")
                relevant_segments = get_relevant_segments(
                    transcription, 
                    user_query, 
                    api_key,
                    max_tokens_per_chunk=max_tokens
                )
                st.session_state.segments = relevant_segments
                status_placeholder.info(f"✅ Found {len(relevant_segments)} relevant segments")
                
                # Edit video
                output_path = os.path.join(temp_dir, "edited_output.mp4")
                success = edit_video(
                    video_path,
                    relevant_segments,
                    output_path,
                    fade_duration=fade_duration
                )
                
                if success:
                    # Copy to current directory for download
                    import shutil
                    shutil.copy(output_path, "edited_output.mp4")
                    st.session_state.processed = True
                    status_placeholder.success("✅ Video processing complete!")
                    st.rerun()
                else:
                    status_placeholder.error("❌ Video editing failed")
        
        except Exception as e:
            status_placeholder.error(f"❌ Error: {str(e)}")
            st.exception(e)

# Display transcript preview if available
if st.session_state.transcript:
    with st.expander("📄 View Transcript"):
        st.json(st.session_state.transcript[:10])  # Show first 10 segments
        if len(st.session_state.transcript) > 10:
            st.caption(f"... and {len(st.session_state.transcript) - 10} more segments")
