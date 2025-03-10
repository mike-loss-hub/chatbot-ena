import streamlit as st
import boto3
from utils import ChatHandler, answer_query_nova_kb
import toml
from pathlib import Path
import os

#aadded this for the the class

def load_dotStreat_sl():
    try:
        is_streamlit_cloud = os.getenv('STREAMLIT_SHARING_MODE') is not None
        
        if is_streamlit_cloud:
            for key, value in st.secrets.items():
                if not key.startswith('_'):
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            full_key = f"{key}_{sub_key}".upper()
                            os.environ[full_key] = str(sub_value)
                    else:
                        os.environ[key.upper()] = str(value)
            return True
            
        else:
            secrets_path = Path('.streamlit/secrets.toml')
            
            if not secrets_path.exists():
                print(f"Warning: {secrets_path} not found")
                return False
                
            secrets = toml.load(secrets_path)
            
            for key, value in secrets.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        full_key = f"{key}_{sub_key}".upper()
                        os.environ[full_key] = str(sub_value)
                else:
                    os.environ[key.upper()] = str(value)
            
            return True
            
    except Exception as e:
        print(f"Error loading secrets: {str(e)}")
        return False

def initialize_aws_clients():
    session = boto3.Session(
        aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
        region_name=st.secrets["AWS_DEFAULT_REGION"]
    )

    bedrock = session.client('bedrock-runtime', 'us-east-1', 
                            endpoint_url='https://bedrock-runtime.us-east-1.amazonaws.com')
    bedrock_agent_runtime_client = boto3.client('bedrock-agent-runtime')
    
    return bedrock, bedrock_agent_runtime_client

def main():
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.chat_handler = ChatHandler()
        st.rerun()
        st.cache_data.clear()
        st.cache_resource.clear()

    # Initialize AWS clients
    load_dotStreat_sl()
    bedrock, bedrock_agent_runtime_client = initialize_aws_clients()

    # Sidebar setup
    with st.sidebar:
        st.image("AnitaMDorr.jpg", width=300, use_container_width=True)
        st.title("Hello! I'm ANITA - v3.1")

        def on_enafocus_change():
            st.session_state.chat_handler = ChatHandler()
            st.cache_data.clear()
            st.cache_resource.clear()

        enafocus = st.radio(
            "ENA Focus",
            ("Website","Old Website"),
            #("Position Statements", "Website"),#, "HR"),
            index=0,
            help="Select the ENA focus area",
            key="enafocus",
            on_change=on_enafocus_change
        )

        llm_model = st.radio(
            "LLM Model",
            ("Nova","Claude"),
            index=0,
            help="Select the LLM model"
        )

        clear_button = st.button("🧹", help="Clear conversation")
        if clear_button:
            st.session_state.chat_handler = ChatHandler()
            st.rerun()
            st.cache_data.clear()
            st.cache_resource.clear()

    # Set configurations based on selections
    if enafocus == "Position Statements":
        chat_input_prompt = "Ask me anything about ENA's position statements!"
        st.session_state["kb_id"] = st.secrets["knowledge_base_postions_id"]
        st.session_state["mode"] = "position statements"
    elif enafocus == "Old Website":
        chat_input_prompt = "Ask me anything about ENA's HR Documents!"
        st.session_state["kb_id"] = st.secrets["knowledge_base_hr_id"]
        st.session_state["mode"] = "human resources documents"
    elif enafocus == "Website":
        chat_input_prompt = "Ask me anything about ENA's Website!"
        st.session_state["kb_id"] = st.secrets["knowledge_base_website_id"]
        st.session_state["mode"] = "ENA's Website"

    if llm_model == "Claude":
        st.session_state["model_id"] = st.secrets["model_id_2"]
    elif llm_model == "Nova":
        st.session_state["model_id"] = st.secrets["model_id_1"]

    # Custom CSS
    st.markdown("""
        <style>
        .stButton>button {
            background-color: transparent;
            border: none;
            color: #4F8BF9;
            margin-top: 0px;
            padding: 5px 10px;
            border-radius: 5px;
        }
        .stButton>button:hover {
            background-color: #f0f2f6;
            color: #4F8BF9;
        }
        </style>
        """, unsafe_allow_html=True)

    # Chat interface
    for message in st.session_state.chat_handler.get_chat_history():
        with st.chat_message(message.type):
            st.write(message.content)

    prompt = st.chat_input(chat_input_prompt)
    if prompt:
        with st.chat_message("human"):
            st.write(prompt)

        with st.chat_message("ai"):
            with st.spinner("Thinking..."):
                response = answer_query_nova_kb(
                    prompt, 
                    st.session_state.chat_handler,
                    bedrock,
                    bedrock_agent_runtime_client,
                    st.session_state["model_id"],
                    st.session_state["kb_id"],
                    st.session_state["mode"]
                )
                st.write(response)

if __name__ == "__main__":
    main()
