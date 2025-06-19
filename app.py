import streamlit as st
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from utils import ChatHandler, answer_query, assess_answer_query,create_analysis_csv, do_batch_assess, do_batch_prompts, load_csv_to_variable, load_simple_csv, LLM_Judge
import toml
from pathlib import Path
import os
import json
from urllib.parse import urlparse
import time

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
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name
    )

    bedrock = session.client('bedrock-runtime', region_name, endpoint_url=epurl)
    bedrock_agent_runtime_client = boto3.client('bedrock-agent-runtime')

    s3_client = boto3.client('s3',
        region_name=region_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key)

    return bedrock, bedrock_agent_runtime_client, s3_client

def main():
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.chat_handler = ChatHandler()
        st.rerun()
        st.cache_data.clear()
        st.cache_resource.clear()

    load_dotStreat_sl()
    bedrock, bedrock_agent_runtime_client, s3_client = initialize_aws_clients()

    #create_analysis_csv(s3_client)
    #LLM_Judge(bedrock, bedrock_agent_runtime_client,s3_client)
    #print(output)
    #do_batch_prompts(bedrock,bedrock_agent_runtime_client, s3_client, st.session_state.chat_handler, st.secrets["knowledge_base_hr_id"])
    
    with st.sidebar:
        st.image("WashSymbol.jpg", width=300, use_container_width=True)
        st.title("Hello! I'm Wa-Bot - v0.9")
        report_mode = st.checkbox("Report Mode", key="report_mode", value = True)
        #judge_mode = st.button("Judge", key="Judge")

        def on_enafocus_change():
            st.session_state.chat_handler = ChatHandler()
            st.cache_data.clear()
            st.cache_resource.clear()

        enafocus = st.radio(
            "Wa-Bot mode",
            ("Website", "Website-Agencies", "Knowledgebase"),
            index=0,
            help="Select the ENA focus area"
        )

        llm_model = st.radio(
            "LLM Model",
            ("Nova Pro","Nova Micro","claude-3-5-haiku","claude-3-5-sonnet", "Agent"),
            index=0,
            help="Select the LLM model"
        )

        clear_button = st.button("🧹", help="Clear conversation")
        if clear_button:
            st.session_state.chat_handler = ChatHandler()
            st.rerun()
            st.cache_data.clear()
            st.cache_resource.clear()  

        #st.button("Batch", key="batch")

        #s3_uri = st.text_input("Enter S3 URI (e.g., s3://my-bucket/path/to/file.txt)", key="batch_uri")
        #cohort_tag = st.text_input("Enter Batch Run Name", key="cohort")

        # if st.button("Batch"):
        #     if s3_uri.startswith("s3://"):
        #         try:
        #             parsed = urlparse(s3_uri)
        #             bucket = parsed.netloc
        #             key = parsed.path.lstrip("/")

        #             s3 = boto3.client("s3")
        #             response = s3.get_object(Bucket=bucket, Key=key)
        #             raw_bytes = response["Body"].read()

        #             try:
        #                 content = raw_bytes.decode("utf-8")
        #             except UnicodeDecodeError:
        #                 content = raw_bytes.decode("ISO-8859-1")
                    
        #             data_list = [item.strip() for item in content.splitlines() if item.strip()]

        #             st.write("### Parsed Elements:")
        #             total = len(data_list)
        #             for i, item in enumerate(data_list):  
        #                 batch_prompt = item.strip()
        #                 st.write(f"""Batching {batch_prompt}...""")
        #                 response = answer_query(
        #                     item.strip(), 
        #                     st.session_state.chat_handler,
        #                     bedrock,
        #                     bedrock_agent_runtime_client,
        #                     s3_client,
        #                     st.session_state["model_id"],
        #                     st.session_state["kb_id"],
        #                     st.session_state["mode"],
        #                     report_mode,
        #                     cohort=cohort_tag,
        #                     batch_mode=True)

        #         except Exception as e:
        #             st.error(f"Error reading file: {e}")
        #     else:
        #         st.warning("Please enter a valid S3 URI starting with s3://")


    # if judge_mode:
    #     with st.chat_message("human"):
    #         st.write("prompt")

    chat_input_prompt=""
    if enafocus == "Website":
        #chat_input_prompt = "Ask me anything about Washington Resident Services!"
        st.session_state["kb_id"] = st.secrets["knowledge_base_hr_id"]
        st.session_state["mode"] = "Website" 
    elif enafocus == "Website-Agencies":
        #chat_input_prompt = "Washington Resident Services - Only Agency sites"
        st.session_state["kb_id"] = st.secrets["knowledge_base_website_id"]
        st.session_state["mode"] = "Website-Agencies"
    elif enafocus == "Knowledgebase":
        #chat_input_prompt = "Washington Resident Services - Knowledgebase"
        st.session_state["kb_id"] = st.secrets["knowledge_base_website_id"]
        st.session_state["mode"] = "Knowledgebase"

    if llm_model == "claude-3-5-haiku":
        st.session_state["model_id"] = st.secrets["model_id_3"]
    elif llm_model == "Nova Pro":
        st.session_state["model_id"] = st.secrets["model_id_1"]
    elif llm_model == "Nova Micro":
        st.session_state["model_id"] = st.secrets["model_id_2"]
    elif llm_model == "claude-3-5-sonnet":
        st.session_state["model_id"] = st.secrets["model_id_4"]
    elif llm_model == "Agent":
        st.session_state["model_id"] = st.secrets["agent_id_1"]

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

    for message in st.session_state.chat_handler.get_chat_history():
        with st.chat_message(message.type):
            st.write(message.content)

    prompt = st.chat_input(chat_input_prompt)
    #prompt_hold = prompt
    if prompt:
        with st.chat_message("human"):
            st.write(prompt)
        if prompt == "e":
            if "prompt_hold" in st.session_state:
                user_query = st.session_state["prompt_hold"]
                ai_response = st.session_state["response_hold"]
                response_model = st.session_state["model_id_hold"]
                model_id = st.session_state["model_id"]
                output = assess_answer_query(user_query, ai_response, response_model, bedrock, bedrock_agent_runtime_client,s3_client, model_id)
                st.write(output)
            else:
                st.write("nothing to evaluate")
        else:
            with st.chat_message("ai"):
                with st.spinner("Thinking..."):
                    st.session_state.chat_handler = ChatHandler()
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    response = answer_query(
                        prompt, 
                        st.session_state.chat_handler,
                        bedrock,
                        bedrock_agent_runtime_client,
                        s3_client,
                        st.session_state["model_id"],
                        st.session_state["kb_id"],
                        st.session_state["mode"],
                        report_mode,
                        cohort ='user',
                        batch_mode=False 
                    )
                    st.write(response)
                    st.session_state["prompt_hold"] = prompt
                    st.session_state["response_hold"] = response
                    st.session_state["model_id_hold"] = st.session_state["model_id"]


if __name__ == "__main__":
    main()
