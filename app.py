import streamlit as st
import google.generativeai as genai
import dotenv
import os
from PIL import Image
from audio_recorder_streamlit import audio_recorder
import base64
from io import BytesIO
import random
from PyPDF2 import PdfReader
import requests
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
import logging
import atexit
import time

# ======================
# INITIAL SETUP
# ======================
# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

# File cleanup function
def cleanup():
    """Remove temporary files on exit"""
    for file in os.listdir():
        if file.startswith(("audio_", "video_", "temp_")) and os.path.exists(file):
            try:
                os.remove(file)
                logging.info(f"Cleaned up {file}")
            except Exception as e:
                logging.error(f"Error cleaning {file}: {str(e)}")

atexit.register(cleanup)

# ======================
# HELPER FUNCTIONS
# ======================
def get_image_base64(image_raw):
    """Convert image to base64"""
    try:
        buffered = BytesIO()
        image_raw.save(buffered, format=image_raw.format)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        logging.error(f"Image conversion error: {str(e)}")
        return None

def base64_to_image(base64_string):
    """Convert base64 to PIL image"""
    try:
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
        return Image.open(BytesIO(base64.b64decode(base64_string)))
    except Exception as e:
        logging.error(f"Base64 to image error: {str(e)}")
        return None

def extract_text_from_pdf(file, api_key):
    """Extract text from PDF with fallback to OCR"""
    text = ""
    try:
        # Try digital text extraction first
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        
        # Fallback to OCR if no text found
        if not text.strip():
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            file.seek(0)
            pdf_document = fitz.open(stream=file.read(), filetype="pdf")
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                for img_index, img in enumerate(page.get_images(full=True)):
                    xref = img[0]
                    base_image = pdf_document.extract_image(xref)
                    image = Image.open(BytesIO(base_image["image"]))
                    response = model.generate_content(["Extract text from this image:", image])
                    if response.text:
                        text += response.text + "\n"
    except Exception as e:
        logging.error(f"PDF extraction error: {str(e)}")
    return text if text.strip() else "Could not extract text from PDF"

def extract_text_from_url(url):
    """Extract main text content from URL"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        return soup.get_text(separator="\n", strip=True)
    except Exception as e:
        logging.error(f"URL extraction error: {str(e)}")
        return f"Failed to extract content from URL: {str(e)}"

def download_chat_transcript():
    """Generate downloadable chat history"""
    transcript = ""
    for message in st.session_state.messages:
        role = message["role"]
        for content in message["content"]:
            if content["type"] == "text":
                transcript += f"{role}: {content['text']}\n\n"
    return transcript

def messages_to_gemini(messages):
    """Convert message format for Gemini API"""
    gemini_messages = []
    prev_role = None
    
    for message in messages:
        if prev_role and (prev_role == message["role"]):
            gemini_message = gemini_messages[-1]
        else:
            gemini_message = {
                "role": "model" if message["role"] == "assistant" else "user",
                "parts": [],
            }

        for content in message["content"]:
            try:
                if content["type"] == "text":
                    gemini_message["parts"].append(content["text"])
                elif content["type"] == "image_url":
                    img = base64_to_image(content["image_url"]["url"])
                    if img:
                        gemini_message["parts"].append(img)
                elif content["type"] in ["video_file", "audio_file"]:
                    if os.path.exists(content[content["type"]]):
                        gemini_message["parts"].append(genai.upload_file(content[content["type"]]))
            except Exception as e:
                logging.error(f"Message conversion error: {str(e)}")
                continue

        if prev_role != message["role"]:
            gemini_messages.append(gemini_message)

        prev_role = message["role"]
        
    return gemini_messages

def stream_llm_response(model_params, api_key):
    """Stream response from Gemini"""
    response_message = ""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name=model_params["model"],
            generation_config={
                "temperature": model_params.get("temperature", 0.3),
                "max_output_tokens": model_params.get("max_tokens", 2048),
            }
        )
        
        gemini_messages = messages_to_gemini(st.session_state.messages)
        
        # Rate limiting
        if hasattr(st.session_state, "last_api_call"):
            elapsed = time.time() - st.session_state.last_api_call
            if elapsed < 1:  # 1 second between calls
                time.sleep(1 - elapsed)
        
        st.session_state.last_api_call = time.time()
        
        response = model.generate_content(gemini_messages, stream=True)
        
        for chunk in response:
            chunk_text = chunk.text or ""
            response_message += chunk_text
            yield chunk_text

        st.session_state.messages.append({
            "role": "assistant", 
            "content": [{"type": "text", "text": response_message}]
        })
        
    except Exception as e:
        logging.error(f"API Error: {str(e)}")
        yield f"⚠️ Error: {str(e)}"

# ======================
# MAIN APP FUNCTION
# ======================
def main():
    # App configuration
    st.set_page_config(
        page_title="The VyomniVerse",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    
    st.markdown("""
        <h1 style="text-align: center; color: #6ca395;">
            🤖 <i>The VyomniVerse</i> 💬
        </h1>
        """, unsafe_allow_html=True)
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "prev_speech_hash" not in st.session_state:
        st.session_state.prev_speech_hash = None
    
    # Model selection
    google_models = ["gemini-1.5-flash", "gemini-1.5-pro"]
    
    # Sidebar configuration
    with st.sidebar:
        # API Key input
        google_api_key = st.text_input(
            "Google API Key (https://aistudio.google.com/app/apikey)",
            type="password",
            value=os.getenv("GOOGLE_API_KEY", "")
        )
        
        st.divider()
        
        # Model settings
        model = st.selectbox("Select model:", google_models, index=0)
        
        with st.expander("⚙️ Model Parameters"):
            model_temp = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
            max_tokens = st.slider("Max Tokens", 512, 8192, 2048, 512)
        
        model_params = {
            "model": model,
            "temperature": model_temp,
            "max_tokens": max_tokens
        }
        
        # Conversation management
        if st.button("🗑️ Reset Conversation", use_container_width=True):
            st.session_state.messages = []
            cleanup()
            st.rerun()
        
        st.divider()
        
        # File upload sections
        if model in google_models:
            st.subheader("🖼️ Media Input")
            
            # Image/Video upload
            uploaded_img = st.file_uploader(
                "Upload image/video",
                type=["png", "jpg", "jpeg", "mp4"],
                accept_multiple_files=False,
                key="uploaded_img"
            )
            
            if uploaded_img:
                try:
                    if uploaded_img.type.startswith("video"):
                        video_path = f"temp_video_{random.randint(100000, 999999)}.mp4"
                        with open(video_path, "wb") as f:
                            f.write(uploaded_img.read())
                        st.session_state.messages.append({
                            "role": "user",
                            "content": [{"type": "video_file", "video_file": video_path}]
                        })
                        st.rerun()
                    else:
                        img = Image.open(uploaded_img)
                        img_base64 = get_image_base64(img)
                        if img_base64:
                            st.session_state.messages.append({
                                "role": "user",
                                "content": [{
                                    "type": "image_url",
                                    "image_url": {"url": f"data:{uploaded_img.type};base64,{img_base64}"}
                                }]
                            })
                            st.rerun()
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
                    logging.error(f"File processing error: {str(e)}")
        
            # Audio recording
            st.subheader("🎤 Audio Input")
            audio_bytes = audio_recorder(
                "Press to record",
                pause_threshold=5.0,
                sample_rate=44100,
            )
            
            if audio_bytes and st.session_state.prev_speech_hash != hash(audio_bytes):
                try:
                    audio_path = f"temp_audio_{random.randint(100000, 999999)}.wav"
                    with open(audio_path, "wb") as f:
                        f.write(audio_bytes)
                    st.session_state.messages.append({
                        "role": "user",
                        "content": [{"type": "audio_file", "audio_file": audio_path}]
                    })
                    st.session_state.prev_speech_hash = hash(audio_bytes)
                    st.rerun()
                except Exception as e:
                    st.error("Failed to process audio")
                    logging.error(f"Audio processing error: {str(e)}")
        
            # Document processing
            st.subheader("📄 Document Input")
            uploaded_pdf = st.file_uploader(
                "Upload PDF",
                type=["pdf"],
                accept_multiple_files=False,
                key="uploaded_pdf"
            )
            
            if uploaded_pdf:
                try:
                    pdf_text = extract_text_from_pdf(uploaded_pdf, google_api_key)
                    st.session_state.messages.append({
                        "role": "user",
                        "content": [{"type": "text", "text": f"PDF content:\n{pdf_text}"}]
                    })
                    st.rerun()
                except Exception as e:
                    st.error("Failed to process PDF")
                    logging.error(f"PDF processing error: {str(e)}")
            
            # URL processing
            st.subheader("🌐 Web Content")
            url = st.text_input("Enter URL to summarize", key="url_input")
            if url:
                try:
                    url_text = extract_text_from_url(url)
                    st.session_state.messages.append({
                        "role": "user",
                        "content": [{"type": "text", "text": f"URL content:\n{url_text}"}]
                    })
                    st.rerun()
                except Exception as e:
                    st.error("Failed to fetch URL")
                    logging.error(f"URL fetch error: {str(e)}")
        
        st.divider()
        
        # Chat export
        st.download_button(
            label="📥 Export Chat",
            data=download_chat_transcript(),
            file_name="vyomniverse_chat.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    # Main chat interface
    if not google_api_key:
        st.warning("⚠️ Please enter your Google API Key to continue")
        return
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            for content in message["content"]:
                if content["type"] == "text":
                    st.write(content["text"])
                elif content["type"] == "image_url":
                    st.image(content["image_url"]["url"])
                elif content["type"] == "video_file":
                    st.video(content["video_file"])
                elif content["type"] == "audio_file":
                    st.audio(content["audio_file"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        st.session_state.messages.append({
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        })
        
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            st.write_stream(stream_llm_response(model_params, google_api_key))

if __name__ == "__main__":
    main()