import boto3
import json
import os
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from langchain_community.chat_message_histories import ChatMessageHistory
from langdetect import detect

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

def get_response(fbedrock_client, foundation_model, query, region='us-east-1'):
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

def get_response_claude(fbedrock_client, foundation_model, query, region='us-east-1'):
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
    language_map = {
        "en": "English", "pl": "Polish", "es": "Spanish",
        # ... (rest of language map)
    }
    
    userQuery = user_input
    chat_history = chat_handler.get_conversation_string()  
    
    detected_language_code = detect(userQuery)    
    detected_language_name = language_map.get(detected_language_code, "Unknown")
    context = get_context(bedrock_agent_runtime_client, model_id, kb_id, userQuery)

    prompt_data = f"""
        Assistant: You are an AI assistant designed to provide factual and accurate answers to user questions based on the Context provided.
        Language Consistency: The user's question is in {detected_language_name}. Respond in {detected_language_name}.
        
        Conversation History (for reference to clarify intent, NOT as a source for answers):
        {chat_history}

        Context (primary and authoritative source for answers):
        {context}

        Question:
        {userQuery}

        Instructions:
        1. Always use the Context as the primary and authoritative source for your answers.
        2. Use the Conversation History ONLY to:
        - Clarify the user's intent
        - Maintain continuity in the conversation.
        3. Do NOT generate answers based on the Conversation History alone.
        4. Be concise and professional in your responses.
        5. Include specific details from the Context when applicable.
        6. If the user references a previous answer, verify its accuracy against the Context.
        7. Please mention the sources by referring to specific ENA documents.
        8. Sources URLs may be derived from information outside of the context.
        
        Answer:
    """

    if model_id == 'amazon.nova-pro-v1:0':
        output_text = get_response(bedrock, model_id, prompt_data)
    else:
        output_text = get_response_claude(bedrock, model_id, prompt_data)

    chat_handler.add_message("human", userQuery)
    chat_handler.add_message("ai", output_text)
   
    output_text = f"{output_text}\n\nModel used: {model_id}\n\nbot type: {mode}\n\n"
    return output_text
