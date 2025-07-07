import boto3
import json
import os

from langchain_community.chat_message_histories import ChatMessageHistory
from langdetect import detect
import time

from requests_aws4auth import AWS4Auth
import csv
import pandas as pd

import json
import re

import uuid
from datetime import datetime
import csv
import re
from typing import List, Dict, Any
import io

from concurrent.futures import ThreadPoolExecutor

def generate_json_filename(tag):
    # Get current date and time
    current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Generate a UUID
    unique_id = str(uuid.uuid4())
    
    # Create the filename string
    filename = f"{current_datetime}_{unique_id}_{tag}.json"
    
    return filename

def build_json_string(**kwargs):
    data = {}
    for key, value in kwargs.items():
        try:
            # Try to parse the value as JSON
            parsed = json.loads(value)
            data[key] = parsed
        except (TypeError, json.JSONDecodeError):
            # If it's not a JSON string, keep the original value
            data[key] = value
    return json.dumps(data, indent=4)


def build_json_string_(**kwargs):
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
    data = {}
    for key, value in kwargs.items():
        try:
            parsed = json.loads(value)
            data[key] = parsed
        except (TypeError, json.JSONDecodeError):
            data[key] = value

    with open(file_name, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    
    print(f"JSON file '{file_name}' has been created successfully.")


def build_json_file_(file_name, **kwargs):
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


def list_json_files_in_s3_folder(s3_uri):
    # Parse the S3 URI
    match = re.match(r's3://([^/]+)/(.+)', s3_uri)
    if not match:
        raise ValueError("Invalid S3 URI format. Expected format: s3://bucket-name/folder-path/")
    
    bucket_name, prefix = match.groups()
    
    # Ensure prefix ends with a slash
    if not prefix.endswith('/'):
        prefix += '/'

    # Initialize S3 client
    s3 = boto3.client('s3')

    # List objects in the specified S3 folder
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    
    # Extract .json files
    json_files = [
        f"s3://{bucket_name}/{obj['Key']}"
        for obj in response.get('Contents', [])
        if obj['Key'].endswith('.json')
    ]
    
    return json_files


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

def get_context(fbedrock_agent_runtime_client, foundation_model, kb_id_hierarchical, query, region='us-west-2'):
    try:
        context = fbedrock_agent_runtime_client.retrieve(
            knowledgeBaseId=kb_id_hierarchical, 
            #nextToken='string',
            retrievalConfiguration={
                "vectorSearchConfiguration": {
                    "numberOfResults":50,
                    'overrideSearchType': 'HYBRID'
                } 
            },
            retrievalQuery={
                'text': query
            }
        )

        retrieval_results = context.get("retrievalResults", [])
        sorted_results = sorted(retrieval_results, key=lambda x: x.get("score", 0), reverse=True)[:5]
        
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

def do_batch_assess():
    reponse_files=[]
    s3_uri="s3://watech-rppilot-bronze/evaluation_data/batch/"
    reponse_files = list_json_files_in_s3_folder(s3_uri)
    for response_file in reponse_files:
        print(response_file)
  
def send_prompt_to_agent(client, agent_id,agent_alias_id, prompt):

    # Initialize the Bedrock Agent Runtime client with a specific region
    #client = boto3.client('bedrock-agent-runtime', region_name='us-west-2')  # Change to your region

    # Define the agent ID and alias ID
    #agent_id = 'WYNNZUBAH3'
    #agent_alias_id = 'JIFVQV4MZK'  # You need to provide the correct alias ID
    #prompt = "What are the benefits of using generative AI in manufacturing?"

    try:
        # Generate a session ID
        session_id = 'session-' + str(uuid.uuid4())[:8]
        print(f"Using session ID: {session_id}")
        
        # Send the prompt to the Bedrock Agent
        response = client.invoke_agent(
            agentId=agent_id,
            agentAliasId=agent_alias_id,
            sessionId=session_id,
            inputText=prompt
        )
        
        # Handle the streaming response
        event_stream = response['completion']
        
        print("\nAgent response:")
        full_response = ""
        
        # Process each event in the stream
        for event in event_stream:
            # Check if the event contains a chunk
            if 'chunk' in event:
                # The chunk is plain text, not JSON
                chunk_text = event['chunk']['bytes'].decode('utf-8')
                full_response += chunk_text
                print(chunk_text, end='', flush=True)  # Print incrementally
        
        print("\n\n--- End of response ---")
        return full_response
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()



def get_response_agent__(fbedrock_client, agent, query, region='us-west-2'):
    agent_id = "RQ6LBHRUT1"
    alias_id = "EVJVNO80DK"
    session_id="VERSION9"

    response = fbedrock_client.invoke_agent(
        agentId=agent_id,
        agentAliasId=alias_id,
        sessionId=session_id,
        inputText= query
    )

    # Read and decode the response
    #response_body = json.loads(response['completion']['text'])
    return response


def write_prompt_to_audit_file(prompt_data, filename="audit.txt"):
    with open(filename, "w", encoding="utf-8") as file:
        file.write(prompt_data)

def get_response_openai(openai_client, model_id, prompt_data):
    response = openai_client.chat.completions.create(
        model=model_id, #"gpt-4",
        messages=[
            {"role": "user", "content": prompt_data}
        ],
        temperature=0.0,
        top_p=1.0,
        max_tokens=500,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    output_text = response.choices[0].message.content
    return output_text

def get_response_agent_(fbedrock_client, foundation_model, query, region='us-west-2'):
    
    agent_id="RQ6LBHRUT1"
    session_id="VUBSQOHTG5"
    
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

    response = fbedrock_client.invoke_agent(
        #modelId=foundation_model,
        agentId=agent_id,
        sessionId='example-session-id-001',
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

def answer_query(user_input, chat_handler, bedrock, bedrock_agent_runtime_client,s3_client, openai_client,model_id, kb_id, mode,report_mode=False, tag="wabotpoc", bucket_name="watech-rppilot-bronze",object_key_path="evaluation_data/users/", cohort = "user", batch_mode=False):

    start_time = time.time()
    cohort_name=str(cohort).strip().lower() 
    language_map = {
        "en": "English", "pl": "Polish", "es": "Spanish",
        # ... (rest of language map)
    }
    
    userQuery = user_input

    context="NONE"
    if mode == "Website-Agencies":
        context = load_csv_to_variable("AgencyList.csv")[['Website','Parent Domain','Domain']]
        context.reset_index(drop=True)
    elif mode == "KB-Website":
        context = get_context(bedrock_agent_runtime_client, model_id, kb_id, userQuery )
        context = f"""{context} Only return URLS present in this context"""
    elif mode == "KB-Legal Assistant":
        context = get_context(bedrock_agent_runtime_client, model_id, kb_id, userQuery )
        #context = f"""{context} Only return URLS present in this context"""
    elif mode == "KB-Website":
        context = get_context(bedrock_agent_runtime_client, model_id, kb_id, userQuery )
        context = f"""{context} Only return URLS present in this context"""

    if batch_mode:
        chat_history = "NONE" #chat_handler.get_conversation_string()
    else:
        chat_history= chat_handler.get_conversation_string()
    
    detected_language_code = detect(userQuery)    
    detected_language_name = language_map.get(detected_language_code, "Unknown")
    #detected_language_name = 'English'
 
    if mode == "KB-Legal Assistant":
        prompt_data = f""" You are a responsible, transparent, and equitable AI assistant designed to help Washington state residents understand 
        and navigate the 2025 Washington Session Laws. Your responses must be accurate, accessible, and aligned with Washington State’s Executive Order 24-01 on Artificial Intelligence, including its principles of fairness, privacy, accountability, and public benefit.
        Your role is to:
        •	Provide clear, plain-language explanations of legal provisions from the 2025 Washington Session Laws.
        •	Help users understand how specific laws may apply to their situation, without offering legal advice.
        •	Ensure that your responses are inclusive, culturally sensitive, and accessible to people of all backgrounds, including those 
            with limited English proficiency or disabilities.
        •	Flag and explain any limitations or uncertainties in the information you provide.
        •	Always disclose when content is AI-generated and encourage users to verify critical information with official or human legal sources.
        •   If you can't find and answer from the RAG content, say "I don't know."
        * Where possible provide helpful url links with the response.
        You must:
        •	Avoid generating or reinforcing bias, discrimination, or stereotypes.
        •	Never make decisions or recommendations that could impact a person’s rights, benefits, or legal standing.
        •	Be transparent about your sources and limitations.
        •	Support human oversight and encourage users to consult legal professionals for complex or high-risk issues.
        You are grounded in the following principles:
        •	Equity and Inclusion: Prioritize equitable outcomes and avoid harm to vulnerable communities.
        •	Transparency and Explainability: Clearly explain how you arrived at your answers and what sources were used.
        •	Privacy and Security: Do not collect or store personal data. Do not process sensitive information unless explicitly permitted and necessary.
        •	Accountability: Your outputs must be auditable, and you must defer to human judgment in all high-risk or ambiguous scenarios.
        You may provide legal advice, but must emphasize thar you are not a lawyer and all decisions should be guided by a qualified human lawyer. 
        You are a public service tool designed to increase understanding of Washington state laws and support informed civic engagement.

        The response should be in {detected_language_name}.

        Conversation History (for reference only; do not use as a source of truth):
        {chat_history}

        Context (retrieved from official sources; translate to English if needed):
        {context}

        User Question:
        {userQuery}

        Answer:
        """

    else:
        prompt_data = f"""
        You are a knowledgeable and trustworthy virtual assistant for Washington State residents.
        Your role is to provide accurate, up-to-date information and direct links to official Washington State government services, forms, and resources.

        The response should be in {detected_language_name}.

        Instructions:
        - Provide clear, step-by-step, and actionable answers that are understandable by someone with a 5th-grade reading level.
        - Use a respectful, formal tone
        - Always provide a helpful response, even if the question is vague—do not ask the user to rephrase.
        - If no reliable or official answer is available, say so clearly and suggest how the user can get help (e.g., contact info or live chat).
        - Include direct URLs to official Washington State websites, forms, or service pages whenever possible.
        - Only include URLs that are valid, relevant, and lead directly to the service or form—not to generic landing pages.
        - Never repeat the same URL in a response.
        - When appropriate, offer the option to connect with a human representative, including phone numbers, email, or live chat links.
        - Use a confident, friendly, and reassuring tone that reflects official guidance.
        - End each response with a helpful follow-up question to guide the user to their next step.

        Resident Expectation:
        "I need to have confidence in the chatbot helping me go through the steps... link me to the right places because once I start googling, I don't know if I'm in the right place... I want it to link me in the furthest it can take me before doing the process of services like: filing unemployment.”

        Conversation History (for reference only; do not use as a source of truth):
        {chat_history}

        Context (retrieved from official sources; translate to English if needed):
        {context}

        User Question:
        {userQuery}

        Answer:
        """

    #if model_id == 'us.amazon.nova-pro-v1:0':
    if model_id.find("nova")!=-1:
        output_text = get_response(bedrock, model_id, prompt_data)
    elif model_id.find("claude")!=-1:
        output_text = get_response_claude(bedrock, model_id, prompt_data)
    elif model_id.find("gpt")!=-1:
        output_text = get_response_openai(openai_client, model_id, prompt_data)
    else:
        #output_text = get_response_agent(bedrock_agent_runtime_client, model_id, prompt_data)
        #send_prompt_to_agent(prompt, agent_id, agent_alias_id, bedrock_agent, region_name='us-west-2')
        agent_id = "WYNNZUBAH3"
        agent_alias_id = "JIFVQV4MZK"
        #send_prompt_to_agent(client, agent_id,agent_alias_id, prompt):
        output_text= send_prompt_to_agent(bedrock_agent_runtime_client,agent_id, agent_alias_id, userQuery)
    if not batch_mode:
        chat_handler.add_message("human", userQuery)
        chat_handler.add_message("ai", output_text)

    end_time = time.time()
    elapsed_time = end_time - start_time
    runTime = f"Elapsed time: {elapsed_time:.4f} seconds"
    output_text = f"{output_text}\n\nModel used: {model_id}\n\nbot type: {mode}\n\nTime to run: {runTime}\n\n"
    
    if report_mode:
        filename = generate_json_filename(tag)
        #object_key=f"{object_key_path}{filename}_{cohort_name}"
        object_key=f"{object_key_path}{cohort_name}_{filename}"
        #content= build_json_string(question = userQuery, prompt=prompt_data, response=output_text, timetorun=runTime, model=model_id, bot_type = mode, cohort_tag=cohort_name)
        content= build_json_string(question = userQuery, response=output_text, timetorun=runTime, model=model_id, bot_type = mode, cohort_tag=cohort_name)
        s3_client.put_object(Bucket=bucket_name, Key=object_key, Body=content)
        
    return output_text

def answer_query_txt(user_input, chat_handler, bedrock, bedrock_agent_runtime_client, s3_client, model_id, kb_id, mode,
                 report_mode=False, tag="wabotpoc", bucket_name="watech-rppilot-bronze",
                 object_key_path="evaluation_data/users/", cohort="user", batch_mode=False):

    import time

    start_time = time.time()
    cohort_name = str(cohort).strip().lower()
    language_map = {
        "en": "English", "pl": "Polish", "es": "Spanish",
        # ... (rest of language map)
    }

    userQuery = user_input

    context = "NONE"
    if mode == "Website-Agencies":
        context = load_csv_to_variable("AgencyList.csv")[['Website', 'Parent Domain', 'Domain']]
        context.reset_index(drop=True)
    elif mode == "Knowledgebase":
        context = get_context(bedrock_agent_runtime_client, model_id, kb_id, userQuery)
        context = f"""{context} Only return URLS present in this context"""

    chat_history = "NONE" if batch_mode else chat_handler.get_conversation_string()

    detected_language_code = detect(userQuery)
    detected_language_name = language_map.get(detected_language_code, "Unknown")

    # Load prompt template from external file
    with open("prompt_data.txt", "r", encoding="utf-8") as file:
        prompt_template = file.read()

    # Format the prompt with dynamic values
    prompt_data = prompt_template.format(
        detected_language_name=detected_language_name,
        chat_history=chat_history,
        context=context,
        userQuery=userQuery
    )

    write_prompt_to_audit_file(prompt_data)

    if "nova" in model_id:
        output_text = get_response(bedrock, model_id, prompt_data)
    elif "claude" in model_id:
        output_text = get_response_claude(bedrock, model_id, prompt_data)
    else:
        agent_id = "WYNNZUBAH3"
        agent_alias_id = "JIFVQV4MZK"
        output_text = send_prompt_to_agent(bedrock_agent_runtime_client, agent_id, agent_alias_id, userQuery)

    if not batch_mode:
        chat_handler.add_message("human", userQuery)
        chat_handler.add_message("ai", output_text)

    elapsed_time = time.time() - start_time
    runTime = f"Elapsed time: {elapsed_time:.4f} seconds"
    output_text = f"{output_text}\n\nModel used: {model_id}\n\nbot type: {mode}\n\nTime to run: {runTime}\n\n"

    if report_mode:
        filename = generate_json_filename(tag)
        object_key = f"{object_key_path}{cohort_name}_{filename}"
        content = build_json_string(
            question=userQuery,
            response=output_text,
            timetorun=runTime,
            model=model_id,
            bot_type=mode,
            cohort_tag=cohort_name
        )
        s3_client.put_object(Bucket=bucket_name, Key=object_key, Body=content)

    return output_text



# def answer_query_nova_batch(user_input, resonse, chat_handler, bedrock, bedrock_agent_runtime_client,s3_client, model_id, kb_id, mode,report_mode=False, tag="wabotpoc", bucket_name="watech-rppilot-bronze",object_key_path="evaluation_data/knowledgebase/Agent/", cohort = "user"):

#     start_time = time.time()
#     # cohort_name=str(cohort).strip().lower() 
#     # language_map = {
#     #     "en": "English", "pl": "Polish", "es": "Spanish",
#     #     # ... (rest of language map)
#     # }
    
#     userQuery = user_input

#     # context="NONE"
#     # if mode == "Website-Agencies":
#     #     context = load_csv_to_variable("AgencyList.csv")[['Website','Parent Domain','Domain']]
#     #     context.reset_index(drop=True)
#     # elif mode == "Knowledgebase":
#     #     context = get_context(bedrock_agent_runtime_client, model_id, kb_id, userQuery )
#     #     context = f"""{context} Only return URLS present in this context"""

    
#     # chat_history = "NONE" #chat_handler.get_conversation_string()  
    
#     #detected_language_code = detect(userQuery)    
#     #detected_language_name = language_map.get(detected_language_code, "Unknown")
#     #detected_language_name = 'English'
 
#     prompt_data = f"""You are a knowledgeable and trustworthy virtual assistant for Washington State residents. 
#     Your role is to provide accurate, up-to-date information and direct links to official state agency services, forms, and resources.
#     The response should be in {detected_language_name}.

#     Instructions:
#     - Provide clear, detailed, and actionable answers that would be understandable by a 5th grader.
#     - Always provide an answer and don't ask for more info.
#     - If no reliable answer is available, state that clearly.
#     - Include direct URLs to relevant forms or service pages whenever possible.
#     - Use only valid URLs with helpful content, relevant to the user question.
#     - Offer the option to connect with a human representative when appropriate along with contact information.
#     - Use a confident, reassuring tone that reflects official guidance.
#     - End each response with a helpful follow-up question to guide the user.

#     Quote from a Resident about what they expect:
#     "I need to have confidence in the chat-bot helping me go through 
#     the steps... link me to the right places because once I start 
#     googling, I don't know if I'm in the right place... I want it to link me in the furthest it can take 
#     me before doing the process of services like: filing unemployment.”

#     Context (url sources for answers. Always translate all contexts into english):
#     {context}

#     Question:
#     {userQuery}
                
#     Answer:
#         """
    
#     #if model_id == 'us.amazon.nova-pro-v1:0':
#     if model_id.find("nova")!=-1:
#         output_text = get_response(bedrock, model_id, prompt_data)
#     elif model_id.find("claude")!=-1:
#         output_text = get_response_claude(bedrock, model_id, prompt_data)
#     else:
#         #output_text = get_response_agent(bedrock_agent_runtime_client, model_id, prompt_data)
#         #send_prompt_to_agent(prompt, agent_id, agent_alias_id, bedrock_agent, region_name='us-west-2')
#         agent_id = "WYNNZUBAH3"
#         agent_alias_id = "JIFVQV4MZK"
#         #send_prompt_to_agent(client, agent_id,agent_alias_id, prompt):
#         output_text= send_prompt_to_agent(bedrock_agent_runtime_client,agent_id, agent_alias_id, userQuery)
#         prompt_data=userQuery

#     #chat_handler.add_message("human", userQuery)
#     #chat_handler.add_message("ai", output_text)

#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     runTime = f"Elapsed time: {elapsed_time:.4f} seconds"
#     output_text = f"{output_text}\n\nModel used: {model_id}\n\nbot type: {mode}\n\nTime to run: {runTime}\n\n"
    
#     if report_mode:
#         filename = generate_json_filename(tag)
#         object_key=f"{object_key_path}{cohort_name}_{filename}"
#         content= build_json_string(question = userQuery, prompt=prompt_data, response=output_text, timetorun=runTime, model=model_id, bot_type = mode, cohort_tag=cohort_name)
#         s3_client.put_object(Bucket=bucket_name, Key=object_key, Body=content)
        
#     return output_text

def assess_answer_query(user_query, response, response_model, bedrock, bedrock_agent_runtime_client,s3_client,openai_client, model_id, batch_mode=False): #,report_mode=False, tag="wabotpoc", bucket_name="watech-rppilot-bronze",object_key_path="evaluation_data/users/", cohort = "user", batch_mode=False):

    start_time = time.time()
    if batch_mode:
        report_style="Return the answer as a .json readable object"
    else:
        report_style="Return the answer in readable report."
    #cohort_name=str(cohort).strip().lower() 
    # language_map = {
    #     "en": "English", "pl": "Polish", "es": "Spanish",
    #     # ... (rest of language map)
    # }
    
    #userQuery = user_input

    # context="NONE"
    # if mode == "Website-Agencies":
    #     context = load_csv_to_variable("AgencyList.csv")[['Website','Parent Domain','Domain']]
    #     context.reset_index(drop=True)
    # elif mode == "Knowledgebase":

    #     context = get_context(bedrock_agent_runtime_client, model_id, kb_id, userQuery )
    #     context = f"""{context} Only return URLS present in this context"""

    # if batch_mode:
    #     chat_history = "NONE" #chat_handler.get_conversation_string()
    # else:
    #     chat_handler.get_conversation_string()
    
    # detected_language_code = detect(userQuery)    
    # detected_language_name = language_map.get(detected_language_code, "Unknown")
    #detected_language_name = 'English'
 
    prompt_data = f"""
        You are a extremely critical, detail oriented expert evaluator of chatbot responses to help residents of the State of Washinghton. 
        Given a user prompt below and the chatbot response, evaluate on a scale of 1 to 5 for the following criteria: 
        helpfulness, accuracy, clarity, tone, and conciseness. Confirm that all URLS are valid and factor that into 
        the assessment scores. Also add a field for total URLS and Num of Valid URLS. Provide a brief overall assessment at the end. 
        Be extremely picky. 5's should be rare.
        {report_style}

        User Prompt:
        {user_query}
        
        Response:
        {response}

        Output Format:

        Helpfulness:  
        Accuracy:   
        Clarity:  
        Tone:   
        Conciseness:
        NmuberOfURLs:  
        Useful URL Links: 

        Assesment:    
        """

    #if model_id == 'us.amazon.nova-pro-v1:0':
    if model_id.find("nova")!=-1:
        output_text = get_response(bedrock, model_id, prompt_data)
    elif model_id.find("claude")!=-1:
        output_text = get_response_claude(bedrock, model_id, prompt_data)
    elif model_id.find("gpt")!=-1:
        output_text = get_response_openai(openai_client, model_id, prompt_data)
    else:
        #output_text = get_response_agent(bedrock_agent_runtime_client, model_id, prompt_data)
        #send_prompt_to_agent(prompt, agent_id, agent_alias_id, bedrock_agent, region_name='us-west-2')
        agent_id = "WYNNZUBAH3"
        agent_alias_id = "JIFVQV4MZK"
        #send_prompt_to_agent(client, agent_id,agent_alias_id, prompt):
        output_text= send_prompt_to_agent(bedrock_agent_runtime_client,agent_id, agent_alias_id, user_query)
    # if not batch_mode:
    #     chat_handler.add_message("human", userQuery)
    #     chat_handler.add_message("ai", output_text)

    end_time = time.time()
    elapsed_time = end_time - start_time
    runTime = f"Elapsed time: {elapsed_time:.4f} seconds"
    if batch_mode:
        output_text=output_text
    else:
        output_text = f"{output_text}\n\nModel Assessed:{response_model}n/n/Judge Model: {model_id}\n\nTime to run: {runTime}\n\n"
    # if report_mode:
    #     filename = generate_json_filename(tag)
    #     #object_key=f"{object_key_path}{filename}_{cohort_name}"
    #     object_key=f"{object_key_path}{cohort_name}_{filename}"
    #     #content= build_json_string(question = userQuery, prompt=prompt_data, response=output_text, timetorun=runTime, model=model_id, bot_type = mode, cohort_tag=cohort_name)
    #     content= build_json_string(question = userQuery, response=output_text, timetorun=runTime, model=model_id, bot_type = mode, cohort_tag=cohort_name)
    #     s3_client.put_object(Bucket=bucket_name, Key=object_key, Body=content)
        
    return output_text


def extract_nested_value(data: Dict[str, Any], path: str):
    """Extract value from nested dictionary using dot-separated path."""
    keys = path.split('.')
    for key in keys:
        if isinstance(data, dict):
            data = data.get(key, None)
        else:
            return None
    return data

def list_json_files_in_s3_folder(s3_client, bucket_name: str, prefix: str):
    """List all JSON files in the specified S3 folder."""
    paginator = s3_client.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    json_keys = []
    for page in page_iterator:
        for obj in page.get('Contents', []):
            if obj['Key'].endswith('.json'):
                json_keys.append(obj['Key'])

    return json_keys

def build_csv_from_json_s3_folder(s3_client, bucket_name: str, prefix: str, s3_input_uri: str, field_paths: List[str], s3_output_uri: str):
    """
    Build a CSV file from JSON files in an S3 folder and upload it to a specified S3 URI.
    """
    # Parse output S3 URI
    output_match = re.match(r's3://([^/]+)/(.+)', s3_output_uri)
    if not output_match:
        raise ValueError("Invalid output S3 URI format. Expected format: s3://bucket-name/path/to/output.csv")
    output_bucket, output_key = output_match.groups()

    # Get input files
    json_files = list_json_files_in_s3_folder(s3_client, bucket_name, prefix)

    # Create CSV in memory
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(field_paths)  # Use field paths as headers

    for key in json_files:
        obj = s3_client.get_object(Bucket=bucket_name, Key=key)
        content = obj['Body'].read().decode('utf-8')
        data = json.loads(content)
        row = [extract_nested_value(data, path) for path in field_paths]
        writer.writerow(row)

    # Upload CSV to S3
    s3_client.put_object(Bucket=output_bucket, Key=output_key, Body=csv_buffer.getvalue().encode('utf-8'))

    print(f"CSV file has been uploaded to s3://{output_bucket}/{output_key}")




def LLM_Judge(bedrock, bedrock_agent_runtime_client,s3_client):

    # AWS S3 configuration
    bucket_name_out = "watech-rppilot-silver"
    prefix = "evaluation_data/batch/"  # Optional: folder path inside the bucket
    keys_to_extract = ['question', 'response']  # Replace with the actual keys you want to extract
    user_query=""
    response=""
    model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0"
    object_key_path=""
    mode="assess"
    tag="wabotpoc" 
    bucket_name="watech-rppilot-bronze"
    object_key_path="evaluation_data/users/"
    object_key_path_out="evaluation_data/assessments/big/"
    # Initialize S3 client (ensure your AWS credentials are configured)
    #s3 = boto3.client('s3')
    cohort_tag_target =  "simple_prompts_big.csv_fullloop05"
    # List all JSON files in the bucket
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    json_files = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.json')]

    
    # Loop through each file and extract specified keys
    for file_key in json_files:
        obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        content = obj['Body'].read().decode('utf-8')
        try:
            data = json.loads(content)
            #extracted = {key: data.get(key, None) for key in keys_to_extract}
            #print(f"\nFile: {file_key}")
            #print("Extracted Data:", extracted)
            user_query= data.get('question', None)
            response=data.get('response', None)
            response_model=data.get('model', None)
            cohort_tag=data.get('cohort_tag', None)
            run_time = data.get('timetorun', None)
            cohort_tag_assess=f"""{cohort_tag_target}_assess"""
            if cohort_tag_target==cohort_tag:
                print(f"""assessing {file_key}""")
                output = assess_answer_query(user_query, response, response_model, bedrock, bedrock_agent_runtime_client,s3_client, model_id, batch_mode=True)
                #assess_answer_query(user_query, response, response_model, bedrock, bedrock_agent_runtime_client,s3_client, model_id)
                #print(output)
                filename = generate_json_filename(tag)
                #object_key=f"{object_key_path}{filename}_{cohort_name}"
                object_key=f"{object_key_path_out}{cohort_tag}_{filename}"
                #content= build_json_string(question = userQuery, prompt=prompt_data, response=output_text, timetorun=runTime, model=model_id, bot_type = mode, cohort_tag=cohort_name)
                content= build_json_string(response=output, assessed_response = response, response_model=response_model, assess_model=model_id, runttime=run_time,bot_type = mode, cohort_tag=cohort_tag_assess)
                s3_client.put_object(Bucket=bucket_name_out, Key=object_key, Body=content)
        except json.JSONDecodeError:
            print(f"Error decoding JSON in file: {file_key}")



    
    