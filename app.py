import streamlit as st
import boto3
from utils import ChatHandler, answer_query_nova_kb, load_csv_to_variable, load_simple_csv
import toml
from pathlib import Path
import os
import json
#test xxxx xcxcxc
#assaasa

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
                print(key)
                print(value)
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
    region_name=""

    aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"]
    aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"]
    region_name=st.secrets["AWS_DEFAULT_REGION"]
    epurl=f"""https://bedrock-runtime.{region_name}.amazonaws.com"""
    
    session = boto3.Session(
        aws_access_key_id=aws_access_key_id, #st.secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=aws_secret_access_key,#st.secrets["AWS_SECRET_ACCESS_KEY"],
        region_name=region_name #st.secrets["AWS_DEFAULT_REGION"]
    )

    bedrock = session.client('bedrock-runtime', region_name, 
                            #endpoint_url='https://bedrock-runtime.{us-east-1}.amazonaws.com')
                            endpoint_url=epurl)
    bedrock_agent_runtime_client = boto3.client('bedrock-agent-runtime')

    
    s3_client = boto3.client('s3',
        region_name=region_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key)

    
    return bedrock, bedrock_agent_runtime_client, s3_client

def do_batch():
    if st.session_state.get("batch"):
        st.write("Batch button was clicked.")
        if st.button("OK"):
            st.write("OK button clicked. Closing dialog.")
        elif st.button("Cancel"):
            st.write("Cancel button clicked. Closing dialog.")

def promptTest():
   
    bedrock_runtime = boto3.client('bedrock-runtime')

    try:
        # A minimal request just to test permissions
        response = bedrock_runtime.invoke_model(
            modelId='anthropic.claude-3-5-haiku-20241022-v1:0',
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "Test"}]
            })
        )
        print("Success! You have the required permissions.")
    except Exception as e:
        print(f"Error: {str(e)}")
        # If you see "AccessDeniedException", you don't have the required permissions

def main():
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.chat_handler = ChatHandler()
        st.rerun()
        st.cache_data.clear()
        st.cache_resource.clear()


    # Initialize AWS clients


    load_dotStreat_sl()
    bedrock, bedrock_agent_runtime_client, s3_client = initialize_aws_clients()
   
    # Sidebar setup
    with st.sidebar:
        st.image("WashSymbol.jpg", width=300, use_container_width=True)
        st.title("Hello! I'm Wa-Bot - v0.9")
        report_mode = st.checkbox("Report Mode", key="report_mode", value = True)

        def on_enafocus_change():
            st.session_state.chat_handler = ChatHandler()
            st.cache_data.clear()
            st.cache_resource.clear()

        enafocus = st.radio(
            "Wa-Bot mode",
            ("Website-Agencies", "Website"),
            #("Position Statements", "Website"),#, "HR"),
            index=0,
            help="Select the ENA focus area",
            key="enafocus",
            on_change=on_enafocus_change
        )

        llm_model = st.radio(
            "LLM Model",
            ("Nova Pro","Nova Micro","claude-3-5-haiku","claude-3-5-sonnet"),
            index=0,
            help="Select the LLM model",
            on_change=on_enafocus_change
        )

        clear_button = st.button("🧹", help="Clear conversation")
        if clear_button:
            st.session_state.chat_handler = ChatHandler()
            st.rerun()
            st.cache_data.clear()
            st.cache_resource.clear()  
        
        #batch_button = st.button("Batch", key="batch")
        #if batch_button:
         #   do_batch()

    # Set configurations based on selections
    if enafocus == "Website":
        chat_input_prompt = "Ask me anything about Washington Resident Services!"
        st.session_state["kb_id"] = st.secrets["knowledge_base_hr_id"]
        st.session_state["mode"] = "Website" 
    elif enafocus == "Website-Agencies":
        chat_input_prompt = "Washington Resident Services - Only Agency sites"
        st.session_state["kb_id"] = st.secrets["knowledge_base_website_id"]
        st.session_state["mode"] = "Website-Agencies"

    if llm_model == "claude-3-5-haiku":
        st.session_state["model_id"] = st.secrets["model_id_3"]
    elif llm_model == "Nova Pro":
        st.session_state["model_id"] = st.secrets["model_id_1"]
    elif llm_model == "Nova Micro":
        st.session_state["model_id"] = st.secrets["model_id_2"]
    elif llm_model == "claude-3-5-sonnet":
        st.session_state["model_id"] = st.secrets["model_id_4"]

    #Custom CSS
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
                    s3_client,
                    st.session_state["model_id"],
                    st.session_state["kb_id"],
                    st.session_state["mode"],
                    report_mode 
                )
                st.write(response)


if __name__ == "__main__":
    main()
