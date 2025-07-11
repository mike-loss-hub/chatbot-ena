import os
import time
import re
import streamlit as st
import boto3
import speech_recognition as sr
from app import load_environment_secrets, initialize_aws_clients, initialize_openai_client
from utils import ChatHandler, answer_query_talkie
import threading
from pygame import mixer
from audio_recorder_streamlit import audio_recorder
import tempfile

def text_to_speech(text, polly, voice_id='Ruth', speed=1.0):
    try:
        cleaned_text = text.replace('&', 'and')
        cleaned_text = cleaned_text.replace('<', '&lt;')
        cleaned_text = cleaned_text.replace('>', '&gt;')
        
        if speed != 1.0:
            ssml_text = f"""<speak><prosody rate="{int(speed*100)}%">{cleaned_text}</prosody></speak>"""
        else:
            ssml_text = f"""<speak>{cleaned_text}</speak>"""

        response = polly.synthesize_speech(
            Engine='neural',
            Text=ssml_text,
            TextType='ssml',
            OutputFormat='mp3',
            VoiceId=voice_id
        )

        if "AudioStream" in response:
            audio_data = response["AudioStream"].read()
            temp_file = "temp_audio.mp3"
            with open(temp_file, 'wb') as file:
                file.write(audio_data)
            return temp_file
        else:
            raise Exception("No AudioStream in response")
            
    except Exception as e:
        print(f"Error in text-to-speech conversion: {str(e)}")
        return None

def play_audio_with_stop(file_path, volume=1.0):
    try:
        mixer.init()
        mixer.music.load(file_path)
        mixer.music.set_volume(volume)
        mixer.music.play()
        is_speaking = True
        
        while mixer.music.get_busy() and is_speaking:
            time.sleep(0.1)
            
        mixer.music.stop()
        mixer.quit()
        try:
            os.remove(file_path)
        except:
            pass
    except Exception as e:
        print(f"Error playing audio: {str(e)}")

def speech_to_text(audio_bytes):
    """Convert speech to text using Google Speech Recognition"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f:
            f.write(audio_bytes)
            temp_filename = f.name

        r = sr.Recognizer()
        with sr.AudioFile(temp_filename) as source:
            audio = r.record(source)
        text = r.recognize_google(audio)
        os.unlink(temp_filename)
        return text
    except Exception as e:
        return f"Error in speech recognition: {str(e)}"
    
def extract_summary(response_text):
    """
    Extracts the 'Summary:' section from a chatbot response.
    Returns the summary as a string, or None if not found.
    """
    match = re.search(r"\*\*XXYZ:\*\*\s*(.+)", response_text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def main():
    # Set up the page layout with two columns
    st.set_page_config(layout="wide")
    
    # Create two columns: left panel and main content
    left_panel, main_content = st.columns([1, 4])
    
    with left_panel:
           # st.markdown("### Settings")
            
            # Logo placeholder
            try:
                logo_image = "WashSymbol.jpg"
                if os.path.exists(logo_image):
                    st.image(logo_image, use_container_width=True)
                else:
                    st.write("Logo not found")
            except Exception as e:
                st.write("Error loading logo")

            st.markdown("---")  # Add a separator line
            
            # Speech speed slider
            speed = st.slider(
                "Speech Speed",
                min_value=0.5,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="Adjust the speaking speed of the assistant"
            )
            
            # Volume slider
            volume = st.slider(
                "Volume",
                min_value=0.0,
                max_value=1.0,
                value=1.0,
                step=0.1,
                help="Adjust the volume of the assistant's voice"
            )

    # Main Content
    with main_content:
        st.title("WA-bot Services Assistant. Talk to me!")
        
        # Load environment variables and initialize clients
        load_environment_secrets()
        clients = (
            *initialize_aws_clients(),
            initialize_openai_client()
        )

        bedrock, bedrock_agent_runtime, s3, openai_client = clients
        polly = boto3.client('polly', region_name='us-west-2')

        # Voice mode toggle
        #voice_mode = st.toggle("Enable Voice Mode", value=True)

        # Initialize prompt variable
        prompt = None

        if True:
            st.write("Click the microphone button to record your question:")
            
            st.info("🎤 Click the button below and  speak...")
            
            audio_bytes = audio_recorder(
                pause_threshold=2.0,
                recording_color="#e74c3c",
                neutral_color="#2ecc71",
                icon_name="microphone",
                text="Click to record"
            )
            
            if audio_bytes:
                st.write("🎯 Processing your question...")
                prompt = speech_to_text(audio_bytes)
                
                if prompt and not prompt.startswith("Error"):
                    st.success(f"✅ You said: {prompt}")
                else:
                    st.error("Failed to recognize speech. Please try again.")
                    prompt = None
        else:
            prompt = "how do i book a campsite?"

        # Only proceed if we have a valid prompt
        if prompt:
            model_id = "us.amazon.nova-micro-v1:0"
            kb_id = "KIVB1LSZCN"
            mode = "KB-Website"
            report_mode = False

            with st.spinner("Generating response..."):
                response = answer_query_talkie(
                    prompt,
                    ChatHandler(),
                    bedrock,
                    bedrock_agent_runtime,
                    s3,
                    openai_client,
                    model_id,
                    kb_id,
                    mode,
                    report_mode,
                    cohort='user',
                    batch_mode=False
                )

            st.write("Assistant:", response)

            selected_voice = "Joanna"
            summary = extract_summary(response)
            with st.spinner("Converting to speech..."):
                audio_file = text_to_speech(summary, polly, voice_id=selected_voice, speed=speed)
                if audio_file:
                    audio_thread = threading.Thread(
                        target=play_audio_with_stop,
                        args=(audio_file, volume)
                    )
                    audio_thread.start()
                    audio_thread.join()

if __name__ == "__main__":
    main()
