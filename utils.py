import boto3
import json
import os
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from langchain_community.chat_message_histories import ChatMessageHistory
from langdetect import detect
import time
import requests
from requests_aws4auth import AWS4Auth
import csv
import pandas as pd

import json

import uuid
from datetime import datetime

def generate_json_filename(tag):
    # Get current date and time
    current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Generate a UUID
    unique_id = str(uuid.uuid4())
    
    # Create the filename string
    filename = f"{current_datetime}_{unique_id}_{tag}.json"
    
    return filename

def build_json_string(**kwargs):
    """
    Function to dynamically build a JSON string with nested key-value pairs.
    
    Parameters:
    **kwargs: Arbitrary keyword arguments representing the key-value pairs to be included in the JSON string.

    Returns:
    str: JSON string representation of the key-value pairs.
    """
    data = kwargs
    json_string = json.dumps(data, indent=4)
    return json_string


def build_json_file(file_name, **kwargs):
    """
    Function to dynamically build a JSON file with nested key-value pairs.
    
    Parameters:
    file_name (str): The name of the JSON file to be created.
    **kwargs: Arbitrary keyword arguments representing the key-value pairs to be included in the JSON file.

    Example:
    build_json_file("example.json", name="John Doe", age=30, city="New York", skills=["Python", "Data Analysis"])

    """
    data = kwargs
    
    with open(file_name, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    
    print(f"JSON file '{file_name}' has been created successfully.")


def load_simple_csv(file_path):
    csv_reader=""
    with open('data.csv', 'r') as csvfile:
        # Create a reader object
        csv_reader = csv.reader(csvfile)
    return csv_reader


def load_csv_to_variable(file_path):
    """Loads a CSV file into a pandas DataFrame.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: A DataFrame containing the data from the CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


class ChatHandler:
    def __init__(self):
        self.memory = ChatMessageHistory()

    def add_message(self, role, content):
        if role == "human":
            self.memory.add_user_message(content)
        elif role == "ai":
            self.memory.add_ai_message(content)

    def get_chat_history(self):
        return self.memory.messages

    def get_conversation_string(self):
        return "\n".join([f"{msg.type}: {msg.content}" for msg in self.memory.messages])
    
    def save_message(self, user_input, ai_response):
        self.memory.chat_memory.add_user_message(user_input)
        self.memory.chat_memory.add_ai_message(ai_response)

def get_awsauth(region, service):
    credentials = boto3.Session().get_credentials()
    return AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        region,
        service,
        session_token=credentials.token
    )

def get_embedding(text, bedrock):
    request_body = {
        "inputText": text
    }
    
    response = bedrock.invoke_model(
        modelId='amazon.titan-embed-text-v1',
        body=json.dumps(request_body),
        contentType='application/json',
        accept='application/json'
    )
    
    response_body = json.loads(response.get('body').read())
    embedding = response_body.get('embedding')
    
    return embedding

def get_context(fbedrock_agent_runtime_client, foundation_model, kb_id_hierarchical, query, region='us-east-1'):
    try:
        context = fbedrock_agent_runtime_client.retrieve(
            knowledgeBaseId=kb_id_hierarchical, 
            nextToken='string',
            retrievalConfiguration={
                "vectorSearchConfiguration": {
                    "numberOfResults":15,
                    'overrideSearchType': 'HYBRID'
                } 
            },
            retrievalQuery={
                'text': query
            }
        )
        
        retrieval_results = context.get("retrievalResults", [])
        sorted_results = sorted(retrieval_results, key=lambda x: x.get("score", 0), reverse=True)
        
    except Exception as e:
        return f"An error occurred: {str(e)}"
    return sorted_results

def get_response(fbedrock_client, foundation_model, query, region='us-west-2'):
    system = [{
        "text": "You are a helpful AI assistant."
    }]

    messages = [{
        "role": "user",
        "content": [{"text": query}]
    }]

    inference_config = {
        "maxTokens": 2000,
        "temperature": 0.0,
        "topP": 1.0,
        "topK": 50
    }

    request_body = {
        "messages": messages,
        "system": system,
        "inferenceConfig": inference_config
    }

    response = fbedrock_client.invoke_model(
        modelId=foundation_model,
        body=json.dumps(request_body),
        contentType='application/json',
        accept='application/json'
    )
    
    response_body = json.loads(response['body'].read())
    output_text = response_body['output']['message']['content'][0]['text']

    return output_text

def get_response_claude(fbedrock_client, foundation_model, query, region='us-west-2'):
    system = "You are a helpful AI assistant."

    messages = [{
        "role": "user",
        "content": query
    }]

    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2000,
        "messages": messages,
        "system": system,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 50
    }

    response = fbedrock_client.invoke_model(
        modelId=foundation_model,
        body=json.dumps(request_body),
        contentType='application/json',
        accept='application/json'
    )
    
    response_body = json.loads(response['body'].read())
    output_text = response_body['content'][0]['text']

    return output_text

def answer_query_nova_kb(user_input, chat_handler, bedrock, bedrock_agent_runtime_client,s3_client, model_id, kb_id, mode, report_mode=False, tag="wabotpoc", bucket_name="watech-rppilot-bronze",object_key_path="evaluation_data/"):

    start_time = time.time()
    out_flag=True
    language_map = {
        "en": "English", "pl": "Polish", "es": "Spanish",
        # ... (rest of language map)
    }
    
    context="NONE"
    if mode == "Website-Agencies":
        context = load_csv_to_variable("AgencyList.csv")[['Website','Parent Domain','Domain']]
        context.reset_index(drop=True)

    userQuery = user_input
    chat_history = chat_handler.get_conversation_string()  
    
    #detected_language_code = detect(userQuery)    
    #detected_language_name = language_map.get(detected_language_code, "Unknown")
    detected_language_name = 'English'
 
    prompt_data = f"""You are a knowledgeable and trustworthy virtual assistant for Washington State residents. 
    Your role is to provide accurate, up-to-date information and direct links to official state agency services, forms, and resources.
    The response should be in {detected_language_name}.

    Instructions:
    1. Provide clear, concise, and actionable answers.
    2. Use only the URLs in the Context section as your primary sources.
    3. Include direct links to relevant forms or service pages whenever possible.
    4. List the names and official websites of any state agencies mentioned.
    5. Offer the option to connect with a human representative when appropriate.
    6. Use a confident, reassuring tone that reflects official guidance.
    7. If no reliable answer is available, state that clearly.
    8. End each response with a helpful follow-up question to guide the user.

    URLs must be valid:
    1. Do not fabricate or guess URLs.
    2. Format links as: URL. Do not invent or modify URLs.
    3. Only include links that are explicitly present in the Context section or are childrens of those URLS. 

    Quote from a Resident about what they expect:
    "I need to have confidence in the chat-bot helping me go through 
    the steps... link me to the right places because once I start 
    googling, I don't know if I'm in the right place... I want it to link me in the furthest it can take 
    me before doing the process of services like: filing unemployment.”

    Conversation History (for reference to clarify intent, NOT as a source for answers):
    {chat_history}

    Context (url sources for answers):
    {context}

    Question:
    {userQuery}
                
    Answer:
        """


    #if model_id == 'us.amazon.nova-pro-v1:0':
    if model_id.find("nova")!=-1:
        output_text = get_response(bedrock, model_id, prompt_data)
    else:
        output_text = get_response_claude(bedrock, model_id, prompt_data)

    chat_handler.add_message("human", userQuery)
    chat_handler.add_message("ai", output_text)

    end_time = time.time()
    elapsed_time = end_time - start_time
    runTime = f"Elapsed time: {elapsed_time:.4f} seconds"
    output_text = f"{output_text}\n\nModel used: {model_id}\n\nbot type: {mode}\n\nTime to run: {runTime}\n\n"
    
    if report_mode:
        filename = generate_json_filename(tag)
        object_key=f"{object_key_path}{filename}"
        #build_json_file(filename, name="John Doe", age=30, city="New York", skills=["Python", "Data Analysis"])
        content= build_json_string(question = userQuery, prompt=prompt_data, response=output_text, timetorun=runTime, model=model_id, bot_type = mode)
        s3_client.put_object(Bucket=bucket_name, Key=object_key, Body=content)
    
    return output_text
