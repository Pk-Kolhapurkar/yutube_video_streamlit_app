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

# Custom CSS for background image
st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("https://images.unsplash.com/photo-1516557070061-c3d1653fa646?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2070&q=80"); 
        background-attachment: fixed;
        background-size: cover;
    }}
    </style>
    """, unsafe_allow_html=True)

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

@st.cache_data
def get_video_info_and_streams(url):
    """Bypasses 403 restrictions and fetches progressive formats with explicit info arrays"""
    if "youtube.com" in url or "youtu.be" in url:
        yt = YouTube(url, client='web')
        streams = yt.streams.filter(progressive=True, type='video')
        details = {
            "is_youtube": True,
            "image": yt.thumbnail_url,
            "title": yt.title,
            "length": yt.length,
            "streams": streams
        }
        itag, resolutions, vformat, frate = ([] for _ in range(4))
        for stream in streams:
            res = re.search(r'(\d+)p', str(stream))
            typ = re.search(r'video/(\w+)', str(stream))
            fps = re.search(r'(\d+)fps', str(stream))
            tag = re.search(r'(\d+)', str(stream))
            
            itag.append(str(stream)[tag.start():tag.end()] if tag else "")
            resolutions.append(str(stream)[res.start():res.end()] if res else "Unknown")
            vformat.append(str(stream)[typ.start():typ.end()] if typ else "video/mp4")
            frate.append(str(stream)[fps.start():fps.end()] if fps else "30fps")
            
        details["resolutions"] = resolutions
        details["itag"] = itag
        details["fps"] = frate
        details["format"] = vformat
        return details
    else:
        # Fallback dictionary metadata signature for direct file URLs
        return {
            "is_youtube": False,
            "image": "https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe",
            "title": "Direct_Video_File",
            "length": "Unknown",
            "resolutions": ["Default Resolution"],
            "itag": ["0"],
            "fps": ["30fps"],
            "format": ["video/mp4"]
        }

def download_video_stream(url, output_path, v_info, selected_index=0):
    """Downloads chosen format streams directly into transient file pathways securely"""
    video_path = os.path.join(output_path, "input_video.mp4")
    
    if v_info.get("is_youtube"):
        try:
            chosen_itag = v_info['itag'][selected_index]
            yt = YouTube(url, client='web')
            ds = yt.streams.get_by_itag(chosen_itag)
            if ds:
                ds.download(output_path=output_path, filename="input_video.mp4")
                if os.path.exists(video_path) and os.path.getsize(video_path) > 1000:
                    return video_path
        except Exception as e:
            st.warning(f"Targeted stream pull failed: {str(e)}. Trying generic stream retrieval...")

    # Fallback/Direct HTTP fetch execution block
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(url, stream=True, headers=headers, timeout=45)
        if response.status_code == 200:
            with open(video_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            if os.path.exists(video_path) and os.path.getsize(video_path) > 1000:
                return video_path
    except Exception as e:
        pass
        
    raise ValueError("File stream generation failed. Ensure your link is accessible.")

def transcribe_video(video_path, model):
    with st.spinner("Transcribing video... This may take a while..."):
        video = VideoFileClip(video_path)
        audio_path = os.path.join(os.path.dirname(video_path), "temp_audio.wav")
        video.audio.write_audiofile(audio_path, verbose=False, logger=None)
        video.close()
        
        result = model.transcribe(audio_path)
        transcription = []
        for segment in result['segments']:
            transcription.append({
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'].strip()
            })
        
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
                # Update methods to match modern MoviePy v2.0 structures cleanly
                start = float(seg['start'])
                end = float(seg['end'])
            except (KeyError, ValueError, TypeError):
                st.warning(f"Skipping malformed segment: {seg}")
                continue
            
            if end <= start:
                st.warning(f"Skipping invalid segment (end <= start): {seg}")
                continue
            
            # Legacy compatibility check handling for .subclipped vs .subclip methods
            if hasattr(video, 'subclipped'):
                clip = video.subclipped(start, end).fadein(fade_duration).fadeout(fade_duration)
            else:
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

# Streamlit UI Setup
st.title("🎬 AI Video Editor")
st.markdown("Create short, focused videos from longer content using AI-powered transcript analysis")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Groq API Key", type="password", help="Enter your Groq API key.")
    whisper_model = st.selectbox("Whisper Model", ["base", "small", "medium", "large"], index=0)
    
    with st.expander("Advanced Options"):
        max_tokens = st.slider("Max tokens per chunk", min_value=1000, max_value=8000, value=4000, step=500)
        fade_duration = st.slider("Fade duration (seconds)", min_value=0.0, max_value=2.0, value=0.5, step=0.1)
    
    st.divider()
    st.markdown("### 📋 Instructions\n1. Provide your Key.\n2. Query configurations.\n3. Run automation pipelines.")

# Form Input Segments
col1, col2 = st.columns([2, 1])

with col1:
    video_url = st.text_input("📹 Video URL", placeholder="https://www.youtube.com/watch?v=...")
    user_query = st.text_area("📝 What do you want to extract from the video?", placeholder="e.g., 'Summarize key features'", height=100)
    process_button = st.button("🚀 Process Video", type="primary", use_container_width=True)

with col2:
    st.markdown("### 📊 Details & Configuration")
    status_placeholder = st.empty()
    status_placeholder.info("Provide URL link targets to start.")
    
    # Render Stream options dynamically as inputs flow
    if video_url:
        try:
            v_info = get_video_info_and_streams(video_url)
            st.image(v_info["image"])
            res_inp = st.selectbox('__Select Streaming Target Resolution__', v_info["resolutions"])
            selected_index = v_info["resolutions"].index(res_inp)
            
            st.write(f"**Title:** {v_info['title']}")
            if v_info['length'] != "Unknown":
                st.write(f"**Length:** {v_info['length']} sec")
                
            file_name_input = st.text_input('__Save as 🎯__', placeholder=v_info['title'])
            file_name = file_name_input if file_name_input else v_info['title']
            if not file_name.endswith(".mp4"):
                file_name += ".mp4"
            # Sanitize filename string constraints securely
            file_name = re.sub(r'[\\/*?:"<>|]', "", file_name)
        except Exception as e:
            st.error(f"Failed to pull streaming targets metadata profiles: {str(e)}")

    # Standard browser level persistence outputs download mapping
    if st.session_state.processed and st.session_state.segments:
        st.success(f"✅ Found {len(st.session_state.segments)} target elements.")
        if os.path.exists("edited_output.mp4"):
            with open("edited_output.mp4", "rb") as f:
                st.download_button(
                    label="📥 Download Processed Video File",
                    data=f.read(),
                    file_name=file_name if 'file_name' in locals() else "edited_video.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )

# Automation Controller Execution Workflow Pipeline Routing
if process_button:
    if not api_key:
        st.error("Missing valid API access profiles")
    elif not video_url:
        st.error("Missing validation link configuration parameter targets")
    elif not user_query:
        st.error("Query structural criteria configurations unassigned")
    else:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                status_placeholder.info("📥 Downloading targeted video payload stream profile...")
                
                # Use updated resolution target parameter download schema
                video_path = download_video_stream(video_url, temp_dir, v_info, selected_index)
                
                model = load_whisper_model(whisper_model)
                transcription = transcribe_video(video_path, model)
                st.session_state.transcript = transcription
                
                status_placeholder.info("🧠 Parsing structural indices using LLM parameters...")
                relevant_segments = get_relevant_segments(transcription, user_query, api_key, max_tokens_per_chunk=max_tokens)
                st.session_state.segments = relevant_segments
                
                status_placeholder.info("✂️ Stripping irrelevant segments...")
                output_path = os.path.join(temp_dir, "edited_output.mp4")
                success = edit_video(video_path, relevant_segments, output_path, fade_duration=fade_duration)
                
                if success:
                    import shutil
                    shutil.copy(output_path, "edited_output.mp4")
                    st.session_state.processed = True
                    status_placeholder.success("💥 Compilation structural configuration parameters set!")
                    st.rerun()
                else:
                    status_placeholder.error("❌ Video encoding error encountered.")
        except Exception as e:
            status_placeholder.error(f"❌ Error context failure sequence: {str(e)}")
            st.exception(e)

if st.session_state.transcript:
    with st.expander("📄 View Parsed Transcription Indexes"):
        st.json(st.session_state.transcript[:10])
