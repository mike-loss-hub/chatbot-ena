"""
Main application module for the Wa-Bot chatbot interface.
This module handles the initialization and running of a Streamlit-based chat interface
that integrates with various AWS services and OpenAI.
"""

import streamlit as st
import boto3
from utils import ChatHandler, answer_query, assess_answer_query
import toml
from pathlib import Path
import os
import openai


def process_streamlit_cloud_secrets():
    """Process secrets when running on Streamlit Cloud."""
    for key, value in st.secrets.items():
        if not key.startswith('_'):
            set_environment_variable(key, value)


def process_local_secrets():
    """
    Process secrets from local secrets.toml file.
    
    Returns:
        bool: Success status of loading local secrets
    """
    secrets_path = Path('.streamlit/secrets.toml')
    if not secrets_path.exists():
        print(f"Warning: {secrets_path} not found")
        return False
        
    secrets = toml.load(secrets_path)
    for key, value in secrets.items():
        print(key)
        print(value)
        set_environment_variable(key, value)
    return True


def set_environment_variable(key, value):
    """
    Set an environment variable with proper formatting.
    
    Args:
        key (str): Environment variable key
        value (Any): Environment variable value
    """
    if isinstance(value, dict):
        for sub_key, sub_value in value.items():
            full_key = f"{key}_{sub_key}".upper()
            os.environ[full_key] = str(sub_value)
    else:
        os.environ[key.upper()] = str(value)


def load_environment_secrets():
    """
    Load environment variables from either Streamlit Cloud or local secrets.
    
    Returns:
        bool: True if secrets were loaded successfully, False otherwise
    """
    try:
        is_streamlit_cloud = os.getenv('STREAMLIT_SHARING_MODE') is not None
        
        if is_streamlit_cloud:
            process_streamlit_cloud_secrets()
            return True
        else:
            return process_local_secrets()
            
    except Exception as e:
        print(f"Error loading secrets: {str(e)}")
        return False


def initialize_aws_clients():
    """
    Initialize AWS service clients using credentials from Streamlit secrets.
    
    Returns:
        tuple: (bedrock_client, bedrock_agent_runtime_client, s3_client)
    """
    # Get AWS credentials from Streamlit secrets
    aws_access_key_id = st.secrets["AWS_ACCESS_KEY_ID"]
    aws_secret_access_key = st.secrets["AWS_SECRET_ACCESS_KEY"]
    region_name = st.secrets["AWS_DEFAULT_REGION"]
    endpoint_url = f"https://bedrock-runtime.{region_name}.amazonaws.com"
    
    # Create AWS session and clients
    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name
    )

    bedrock = session.client(
        'bedrock-runtime',
        region_name,
        endpoint_url=endpoint_url
    )
    
    bedrock_agent_runtime = boto3.client('bedrock-agent-runtime')
    
    s3 = boto3.client(
        's3',
        region_name=region_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

    return bedrock, bedrock_agent_runtime, s3


def initialize_openai_client():
    """
    Initialize OpenAI client using API key from Streamlit secrets.
    
    Returns:
        OpenAI: Initialized OpenAI client
    """
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    return openai.OpenAI(api_key=openai_api_key)

def initialize_session_state():
    """Initialize the session state and chat handler."""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.chat_handler = ChatHandler()
        st.rerun()
        st.cache_data.clear()
        st.cache_resource.clear()


def setup_sidebar():
    """
    Set up the sidebar with configuration options.
    
    Returns:
        tuple: (selected_mode, selected_model, report_mode)
    """
    with st.sidebar:
        st.image("WashSymbol.jpg", width=300, use_container_width=True)
        st.title("Hello! I'm Wa-Bot - v0.9")
        report_mode = st.checkbox("Report Mode", key="report_mode", value=True)

        selected_mode = st.radio(
            "Wa-Bot mode",
            ("KB-Website", "KB-Legal Assistant"),
            index=0,
            help="Select the ENA focus area"
        )

        selected_model = st.radio(
            "LLM Model",
            ("Nova Pro", "Nova Micro", "claude-3-5-haiku",
             "claude-3-5-sonnet", "gpt-4-turbo", "gpt-4o"),
            index=0,
            help="Select the LLM model"
        )

        if st.button("🧹", help="Clear conversation"):
            st.session_state.chat_handler = ChatHandler()
            st.rerun()
            st.cache_data.clear()
            st.cache_resource.clear()

    return selected_mode, selected_model, report_mode


def handle_mode_selection(mode):
    """
    Configure chat settings based on selected mode.
    
    Args:
        mode (str): Selected chat mode
    """
    mode_settings = {
        "Website": {
            "kb_id": "knowledge_base_hr_id",
            "mode": "Website"
        },
        "Website-Agencies": {
            "kb_id": "knowledge_base_website_id",
            "mode": "Website-Agencies"
        },
        "KB-Website": {
            "kb_id": "knowledge_base_website_id",
            "mode": "KB-Website"
        },
        "KB-Legal Assistant": {
            "kb_id": "knowledge_base_legal_id",
            "mode": "KB-Legal Assistant"
        }
    }
    
    if mode in mode_settings:
        settings = mode_settings[mode]
        st.session_state["kb_id"] = st.secrets[settings["kb_id"]]
        st.session_state["mode"] = settings["mode"]


def handle_model_selection(model):
    """
    Configure model settings based on selection.
    
    Args:
        model (str): Selected model name
    """
    model_mapping = {
        "claude-3-5-haiku": "model_id_3",
        "Nova Pro": "model_id_1",
        "Nova Micro": "model_id_2",
        "claude-3-5-sonnet": "model_id_4",
        "gpt-4-turbo": "model_id_5",
        "gpt-4o": "model_id_6",
        "Agent": "agent_id_1"
    }
    
    if model in model_mapping:
        st.session_state["model_id"] = st.secrets[model_mapping[model]]


def setup_custom_styling():
    """Apply custom CSS styling to the Streamlit interface."""
    st.markdown("""
        <h1 style='text-align: left; color: red; font-size: 24px; font-weight: bold;'>
            This Site Is For Demonstration Purposes Only
        </h1>
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
    

def display_chat_history():
    """Display the existing chat history."""
    for message in st.session_state.chat_handler.get_chat_history():
        with st.chat_message(message.type):
            st.write(message.content)


def process_evaluation_request(clients):
    """
    Process an evaluation request ('e' command).
    
    Args:
        clients (tuple): Tuple containing necessary client objects
    """
    bedrock, bedrock_agent_runtime, s3, openai_client = clients
    if "prompt_hold" in st.session_state:
        output = assess_answer_query(
            st.session_state["prompt_hold"],
            st.session_state["response_hold"],
            st.session_state["model_id_hold"],
            bedrock,
            bedrock_agent_runtime,
            s3,
            openai_client,
            st.session_state["model_id"]
        )
        st.write(output)
    else:
        st.write("Nothing to evaluate")


def generate_ai_response(prompt, clients, report_mode):
    """
    Generate and display AI response to user input.
    
    Args:
        prompt (str): User input
        clients (tuple): Tuple containing necessary client objects
        report_mode (bool): Whether report mode is enabled
    """
    bedrock, bedrock_agent_runtime, s3, openai_client = clients
    
    with st.chat_message("ai"):
        with st.spinner("Thinking..."):
            st.session_state.chat_handler = ChatHandler()
            st.cache_data.clear()
            st.cache_resource.clear()
            
            response = answer_query(
                prompt,
                st.session_state.chat_handler,
                bedrock,
                bedrock_agent_runtime,
                s3,
                openai_client,
                st.session_state["model_id"],
                st.session_state["kb_id"],
                st.session_state["mode"],
                report_mode,
                cohort='user',
                batch_mode=False
            )
            
            st.write(response)
            store_interaction(prompt, response)


def store_interaction(prompt, response):
    """
    Store the current interaction for potential later evaluation.
    
    Args:
        prompt (str): User input
        response (str): AI response
    """
    st.session_state["prompt_hold"] = prompt
    st.session_state["response_hold"] = response
    st.session_state["model_id_hold"] = st.session_state["model_id"]


def main():
    """
    Main function that sets up and runs the Streamlit interface.
    """
    initialize_session_state()

    # Load environment variables and initialize clients
    load_environment_secrets()
    clients = (
        *initialize_aws_clients(),
        initialize_openai_client()
    )
    
    # Setup sidebar and get user selections
    selected_mode, selected_model, report_mode = setup_sidebar()
    
    # Configure settings based on selections
    handle_mode_selection(selected_mode)
    handle_model_selection(selected_model)
    
    # Apply custom styling
    setup_custom_styling()
    
    # Display chat history
    display_chat_history()
    
    # Handle chat interactions
    prompt = st.chat_input("")
    if prompt:
        with st.chat_message("human"):
            st.write(prompt)
            
        if prompt == "e":
            process_evaluation_request(clients)
        else:
            generate_ai_response(prompt, clients, report_mode)


if __name__ == "__main__":
    main()
