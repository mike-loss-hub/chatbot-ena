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

def answer_query_nova_kb(user_input, chat_handler, bedrock, bedrock_agent_runtime_client, model_id, kb_id, mode):
    
    start_time = time.time()
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
    
    detected_language_code = detect(userQuery)    
    detected_language_name = language_map.get(detected_language_code, "Unknown")
    #context = get_context(bedrock_agent_runtime_client, model_id, kb_id, userQuery)
    #context=""

    prompt_data = f"""
        Assistant: You are a helpful chatbot designed to assist residents of the State of Washington to get answers to questions 
        about things like proceedures, benefits and regulations. The Agency websites in the Context below as a primary source to answer the Question below.
        At the bottom list the names of the agencies referenced, along with their parent domain and domain as specified in the Context. 

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
    return output_text
