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
import fitz  # PyMuPDF for extracting images from PDF

# Load environment variables
dotenv.load_dotenv()

# Gemini Models
google_models = [
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

# Function to convert messages format from Streamlit to Gemini
def messages_to_gemini(messages):
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
            if content["type"] == "text":
                gemini_message["parts"].append(content["text"])
            elif content["type"] == "image_url":
                gemini_message["parts"].append(base64_to_image(content["image_url"]["url"]))
            elif content["type"] == "video_file":
                gemini_message["parts"].append(genai.upload_file(content["video_file"]))
            elif content["type"] == "audio_file":
                gemini_message["parts"].append(genai.upload_file(content["audio_file"]))

        if prev_role != message["role"]:
            gemini_messages.append(gemini_message)

        prev_role = message["role"]
        
    return gemini_messages

# Function to stream the response from Gemini
def stream_llm_response(model_params, api_key):
    response_message = ""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=model_params["model"],
        generation_config={
            "temperature": model_params["temperature"] if "temperature" in model_params else 0.3,
        }
    )
    gemini_messages = messages_to_gemini(st.session_state.messages)
    for chunk in model.generate_content(gemini_messages):
        chunk_text = chunk.text or ""
        response_message += chunk_text
        yield chunk_text

    st.session_state.messages.append({
        "role": "assistant", 
        "content": [
            {
                "type": "text",
                "text": response_message,
            }
        ]})

# Function to convert file to base64
def get_image_base64(image_raw):
    buffered = BytesIO()
    image_raw.save(buffered, format=image_raw.format)
    img_byte = buffered.getvalue()
    return base64.b64encode(img_byte).decode('utf-8')

def base64_to_image(base64_string):
    base64_string = base64_string.split(",")[1]
    return Image.open(BytesIO(base64.b64decode(base64_string)))

# Function to extract text from PDF (handles both digital and handwritten text)
def extract_text_from_pdf(file, api_key):
    text = ""
    try:
        # First, try extracting digital text using PyPDF2
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        
        # If no text is extracted, fall back to OCR using Gemini
        if not text.strip():
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")  # Use gemini-1.5-flash for OCR
            try:
                # Reopen the file for OCR
                file.seek(0)  # Reset file pointer
                pdf_document = fitz.open(stream=file.read(), filetype="pdf")
                for page_num in range(len(pdf_document)):
                    page = pdf_document.load_page(page_num)
                    image_list = page.get_images(full=True)
                    for img_index, img in enumerate(image_list):
                        xref = img[0]
                        base_image = pdf_document.extract_image(xref)
                        image_bytes = base_image["image"]
                        image = Image.open(BytesIO(image_bytes))
                        response = model.generate_content(["Extract text from this image:", image])
                        text += response.text + "\n"
            except Exception:
                # If OCR fails, return an empty string
                pass
    except Exception:
        # If PyPDF2 fails, fall back to OCR
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")  # Use gemini-1.5-flash for OCR
        try:
            # Reopen the file for OCR
            file.seek(0)  # Reset file pointer
            pdf_document = fitz.open(stream=file.read(), filetype="pdf")
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                image_list = page.get_images(full=True)
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = pdf_document.extract_image(xref)
                    image_bytes = base_image["image"]
                    image = Image.open(BytesIO(image_bytes))
                    response = model.generate_content(["Extract text from this image:", image])
                    text += response.text + "\n"
        except Exception:
            # If OCR fails, return an empty string
            pass
    return text

# Function to extract text from URL
def extract_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    return soup.get_text()

# Function to download chat transcript
def download_chat_transcript():
    transcript = ""
    for message in st.session_state.messages:
        role = message["role"]
        for content in message["content"]:
            if content["type"] == "text":
                transcript += f"{role}: {content['text']}\n"
    return transcript

# Main Function
def main():
    st.set_page_config(
        page_title="The VyomniVerse",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    st.html("""<h1 style="text-align: center; color: #6ca395;">🤖 <i>The VyomniVerse</i> 💬</h1>""")

    with st.sidebar:
        default_google_api_key = os.getenv("GOOGLE_API_KEY") if os.getenv("GOOGLE_API_KEY") is not None else ""
        google_api_key = st.text_input("Introduce your Google API Key (https://aistudio.google.com/app/apikey)", value=default_google_api_key, type="password")

    if google_api_key == "" or google_api_key is None:
        st.write("#")
        st.warning("⬅️ Please introduce an API Key to continue...")
    else:
        if "messages" not in st.session_state:
            st.session_state.messages = []

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

        with st.sidebar:
            st.divider()
            model = st.selectbox("Select a model:", google_models, index=0)
            with st.popover("⚙️ Model parameters"):
                model_temp = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.3, step=0.1)

            model_params = {
                "model": model,
                "temperature": model_temp,
            }

            def reset_conversation():
                if "messages" in st.session_state and len(st.session_state.messages) > 0:
                    st.session_state.pop("messages", None)

            st.button(
                "🗑️ Reset conversation", 
                on_click=reset_conversation,
            )

            st.divider()

            # Image and Video Upload
            if model in ["gemini-1.5-flash", "gemini-1.5-pro"]:
                st.write(f"### **🖼️ Add an image or a video file:**")

                def add_image_to_messages():
                    if st.session_state.uploaded_img or ("camera_img" in st.session_state and st.session_state.camera_img):
                        img_type = st.session_state.uploaded_img.type if st.session_state.uploaded_img else "image/jpeg"
                        if img_type == "video/mp4":
                            video_id = random.randint(100000, 999999)
                            with open(f"video_{video_id}.mp4", "wb") as f:
                                f.write(st.session_state.uploaded_img.read())
                            st.session_state.messages.append(
                                {
                                    "role": "user", 
                                    "content": [{
                                        "type": "video_file",
                                        "video_file": f"video_{video_id}.mp4",
                                    }]
                                }
                            )
                        else:
                            raw_img = Image.open(st.session_state.uploaded_img or st.session_state.camera_img)
                            img = get_image_base64(raw_img)
                            st.session_state.messages.append(
                                {
                                    "role": "user", 
                                    "content": [{
                                        "type": "image_url",
                                        "image_url": {"url": f"data:{img_type};base64,{img}"}
                                    }]
                                }
                            )

                cols_img = st.columns(2)
                with cols_img[0]:
                    with st.popover("📁 Upload"):
                        st.file_uploader(
                            "Upload an image or a video:", 
                            type=["png", "jpg", "jpeg", "mp4"], 
                            accept_multiple_files=False,
                            key="uploaded_img",
                            on_change=add_image_to_messages,
                        )

                with cols_img[1]:                    
                    with st.popover("📸 Camera"):
                        activate_camera = st.checkbox("Activate camera (only images)")
                        if activate_camera:
                            st.camera_input(
                                "Take a picture", 
                                key="camera_img",
                                on_change=add_image_to_messages,
                            )

            # Audio Upload
            st.write("#")
            st.write(f"### **🎤 Add an audio:**")

            audio_prompt = None
            audio_file_added = False
            if "prev_speech_hash" not in st.session_state:
                st.session_state.prev_speech_hash = None

            speech_input = audio_recorder("Press to talk:", icon_size="3x", neutral_color="#6ca395", )
            if speech_input and st.session_state.prev_speech_hash != hash(speech_input):
                st.session_state.prev_speech_hash = hash(speech_input)
                audio_id = random.randint(100000, 999999)
                with open(f"audio_{audio_id}.wav", "wb") as f:
                    f.write(speech_input)

                st.session_state.messages.append(
                    {
                        "role": "user", 
                        "content": [{
                            "type": "audio_file",
                            "audio_file": f"audio_{audio_id}.wav",
                        }]
                    }
                )

                audio_file_added = True

            # PDF Upload with Text Extraction
            st.write("### **📄 Add a PDF (supports both digital and handwritten text):**")
            def add_pdf_to_messages():
                if st.session_state.uploaded_pdf:
                    pdf_text = extract_text_from_pdf(st.session_state.uploaded_pdf, google_api_key)
                    st.session_state.messages.append(
                        {
                            "role": "user", 
                            "content": [{
                                "type": "text",
                                "text": f"Extracted text from PDF:\n{pdf_text}",
                            }]
                        }
                    )

            st.file_uploader(
                "Upload a PDF:", 
                type=["pdf"], 
                accept_multiple_files=False,
                key="uploaded_pdf",
                on_change=add_pdf_to_messages,
            )

            # URL Input
            st.write("### **🌐 Add a URL:**")
            def add_url_to_messages():
                if st.session_state.url_input:
                    url_text = extract_text_from_url(st.session_state.url_input)
                    st.session_state.messages.append(
                        {
                            "role": "user", 
                            "content": [{
                                "type": "text",
                                "text": f"Summarize this URL:\n{url_text}",
                            }]
                        }
                    )

            st.text_input("Enter a URL:", key="url_input", on_change=add_url_to_messages)

            
            # Download Chat Transcript
            st.download_button(
                label="📥 Download Chat Transcript",
                data=download_chat_transcript(),
                file_name="chat_transcript.txt",
                mime="text/plain",
            )

        # Chat Input
        if prompt := st.chat_input("Hi! Ask me anything..."):
            st.session_state.messages.append(
                {
                    "role": "user", 
                    "content": [{
                        "type": "text",
                        "text": prompt,
                    }]
                }
            )
                
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                st.write_stream(
                    stream_llm_response(
                        model_params=model_params, 
                        api_key=google_api_key)
                )

if __name__ == "__main__":
    main()