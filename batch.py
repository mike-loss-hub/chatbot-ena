    
import streamlit as st
import boto3
from urllib.parse import urlparse
from app import load_environment_secrets, initialize_aws_clients, initialize_openai_client
from utils import ChatHandler, answer_query, assess_answer_query,build_csv_from_json_s3_folder, generate_json_filename, build_json_string
from datetime import datetime
import json
from concurrent.futures import ThreadPoolExecutor

# def do_batch_prompts_threads_k(max_threads=30, **kwargs):
#     # Default values for kwargs
#     defaults = {
#         'report_mode': True,
#         'promptlist': "simple_prompts_small.csv",
#         's3_uri_base': "s3://watech-rppilot-bronze/evaluation_data/prompt_lists",
#         'cohort_tag_suffix': "demo_threads_01",
#         's3_out_batch': "evaluation_data/batch/small/",
#         'model_ids': ["us.amazon.nova-pro-v1:0", "us.amazon.nova-micro-v1:0", 
#                      "us.anthropic.claude-3-5-haiku-20241022-v1:0", 
#                      "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
#                      "gpt-4-turbo", "gpt-4o"],
#         'mode_names': ["KB-Website"],
#         'kb_id': '4BFLETNCSZ'  # website
#     }

#     # Required client parameters
#     required_clients = ['bedrock', 'bedrock_agent_runtime_client', 's3_client', 
#                        'openai_client', 'chat_handler']
    
#     # Verify required client parameters are present
#     for param in required_clients:
#         if param not in kwargs:
#             raise ValueError(f"Missing required client parameter: {param}")

#     # Update defaults with provided kwargs
#     config = {**defaults, **kwargs}

#     # Extract client parameters
#     bedrock = config['bedrock']
#     bedrock_agent_runtime_client = config['bedrock_agent_runtime_client']
#     s3_client = config['s3_client']
#     openai_client = config['openai_client']
#     chat_handler = config['chat_handler']

#     # Extract other parameters with defaults
#     report_mode = config['report_mode']
#     promptlist = config['promptlist']
#     s3_uri = f"{config['s3_uri_base']}/{promptlist}"
#     cohort_tag = f"{promptlist}_{config['cohort_tag_suffix']}"
#     s3_out_batch = config['s3_out_batch']
#     model_ids = config['model_ids']
#     mode_names = config['mode_names']
#     kb_id = config['kb_id']

#     def process_item_k(item):
#         try:
#             results = []
#             for model_id in model_ids:
#                 for mode_name in mode_names:
#                     print(f"Processing {model_id} - {mode_name} - {item}")
                    
#                     # Modified to use handle_chat instead of process_prompt
#                     result = chat_handler.handle_chat(
#                         user_input=item,
#                         model_id=model_id,
#                         mode_name=mode_name,
#                         kb_id=kb_id,
#                         report_mode=report_mode
#                     )
                    
#                     results.append({
#                         'prompt': item,
#                         'model_id': model_id,
#                         'mode_name': mode_name,
#                         'result': result
#                     })
            
#             return results
        
#         except Exception as e:
#             print(f"Error processing item {item}: {str(e)}")
#             return [{
#                 'prompt': item,
#                 'error': str(e)
#             }]

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

#             # Process items using ThreadPoolExecutor
#             with ThreadPoolExecutor(max_workers=max_threads) as executor:
#                 all_results = list(executor.map(process_item_k, data_list))

#             # Flatten results
#             flat_results = [item for sublist in all_results for item in sublist]

#             # Save results to S3
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#             result_filename = f"{cohort_tag}_{timestamp}.json"
            
#             s3_client.put_object(
#                 Bucket=bucket,
#                 Key=f"{s3_out_batch}{result_filename}",
#                 Body=json.dumps(flat_results, indent=2)
#             )

#             print(f"Results saved to s3://{bucket}/{s3_out_batch}{result_filename}")
#             return flat_results

#         except Exception as e:
#             print(f"Error reading file: {e}")
#             return
#     else:
#         print("Please enter a valid S3 URI starting with s3://")
#         return


def do_batch_prompts_threads(bedrock, bedrock_agent_runtime_client, s3_client, openai_client,chat_handler, kb_id, max_threads=30):
   # ********* INPUTS *********"
   
    report_mode = True
    promptlist = "simple_prompts_small.csv"
    s3_uri = f"s3://watech-rppilot-bronze/evaluation_data/prompt_lists/{promptlist}"
    cohort_tag = f"{promptlist}_demo_threads_01"
    s3_out_batch = "evaluation_data/batch/small/"
    model_ids = ["us.amazon.nova-pro-v1:0", "us.amazon.nova-micro-v1:0", "us.anthropic.claude-3-5-haiku-20241022-v1:0", "us.anthropic.claude-3-5-sonnet-20241022-v2:0","gpt-4-turbo","gpt-4o"]
    #mode_names = ["KB-Legal Assistant"]
    mode_names = ["KB-Website"]

    kb_id='4BFLETNCSZ'#website
    #kb_id='4BFLETNCSZ'#legalaid

    if s3_uri.startswith("s3://"):
        try:
            parsed = urlparse(s3_uri)
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")

            s3 = boto3.client("s3")
            response = s3.get_object(Bucket=bucket, Key=key)
            raw_bytes = response["Body"].read()

            try:
                content = raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                content = raw_bytes.decode("ISO-8859-1")

            data_list = [item.strip() for item in content.splitlines() if item.strip()]

        except Exception as e:
            print(f"Error reading file: {e}")
            return
    else:
        print("Please enter a valid S3 URI starting with s3://")
        return

    def process_item(item, model_id, mode,kb_id, s3_path = "evaluation_data/batch/"):
        return answer_query(
            item.strip(),
            chat_handler,
            bedrock,
            bedrock_agent_runtime_client,
            s3_client,
            openai_client,
            model_id,
            kb_id,
            mode,
            report_mode,
            cohort=cohort_tag,
            batch_mode=True,
            object_key_path="evaluation_data/batch/demo/"
        )
    for model_id in model_ids:
        for mode in mode_names:
            print(f"Processing {model_id} in {mode} using {promptlist} with up to {max_threads} threads")
            with ThreadPoolExecutor(max_workers=max_threads) as executor:
                executor.map(lambda item: process_item(item, model_id, mode,kb_id, s3_out_batch), data_list)

def LLM_Judge_threads(bedrock, bedrock_agent_runtime_client, s3_client, openai_client, max_threads=30):
    # AWS S3 configuration
    
    prefix = "evaluation_data/batch/demo/"  
    response = ""
    model_id = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
    mode = "assess"
    tag = "wabotpoc"
    bucket_name = "watech-rppilot-bronze"
    
    bucket_name_out = "watech-rppilot-silver"
    object_key_path_out = "evaluation_data/assessments/demo/"
    cohort_tag_target = "simple_prompts_mid.csv__demo_threads_01"
    # List all JSON files in the bucket
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    json_files = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.json')]

    def process_file(file_key):
        obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        content = obj['Body'].read().decode('utf-8')
        try:
            data = json.loads(content)
            user_query = data.get('question', None)
            response = data.get('response', None)
            response_model = data.get('model', None)
            response_mode = data.get('bot_type',None)
            cohort_tag = data.get('cohort_tag', None)
            run_time = data.get('timetorun', None)
            cohort_tag_assess = f"{cohort_tag_target}_assess"

            if cohort_tag_target == cohort_tag:
                print(f"Assessing {file_key}")
                output = assess_answer_query(
                    user_query, response, response_model,
                    bedrock, bedrock_agent_runtime_client, s3_client,openai_client,
                    model_id, batch_mode=True
                )
                filename = generate_json_filename(tag)
                object_key = f"{object_key_path_out}{cohort_tag}_{filename}"
                content = build_json_string(
                    response=output,
                    assessed_response=response,
                    response_model=response_model,
                    response_mode=response_mode,
                    assess_model=model_id,
                    runttime=run_time,
                    bot_type=mode,
                    cohort_tag=cohort_tag_assess
                )
                s3_client.put_object(Bucket=bucket_name_out, Key=object_key, Body=content)
        except json.JSONDecodeError:
            print(f"Error decoding JSON in file: {file_key}")

    # Use ThreadPoolExecutor to process files concurrently
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        executor.map(process_file, json_files)

def create_analysis_csv(s3_client):
    s3_input_uri = "s3://your-input-bucket/your-folder/"
    s3_output_uri = "s3://watech-rppilot-silver/evaluation_data/reports/legalhelper_prompts_big.csv_bedgpt_threads02.csv"
    bucket_name = "watech-rppilot-silver"
    prefix = "evaluation_data/assessments/big/legalhelper/"
    field_paths = [
        "response.scores.helpfulness",
        "response.scores.accuracy",
        "response.scores.clarity",
        "response.scores.tone",
        "response.scores.conciseness",
        "response.urls.totalURLs",
        "response.urls.validURLs",
        "response.urls.list",
        "response.assessment",
        "assessed_response",
        "response_model",
        "response_mode",
        "assess_model",
        "runttime",
        "bot_type",
        "cohort_tag"
    ]
    build_csv_from_json_s3_folder(s3_client, bucket_name, prefix, s3_input_uri, field_paths, s3_output_uri)

def main():
    
    # Load environment variables and initialize clients
    load_environment_secrets()
    clients = (
        *initialize_aws_clients(),
        initialize_openai_client()
    )
    #bedrock, bedrock_agent_runtime_client, s3_client = initialize_aws_clients()
    #openai_client = initialize_openai_client()

    bedrock, bedrock_agent_runtime, s3, openai_client = clients

        # 'report_mode': True,
        # 'promptlist': "simple_prompts_small.csv",
        # 's3_uri_base': "s3://watech-rppilot-bronze/evaluation_data/prompt_lists",
        # 'cohort_tag_suffix': "demo_threads_01",
        # 's3_out_batch': "evaluation_data/batch/small/",
        # 'model_ids': ["us.amazon.nova-pro-v1:0", "us.amazon.nova-micro-v1:0", 
        #              "us.anthropic.claude-3-5-haiku-20241022-v1:0", 
        #              "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        #              "gpt-4-turbo", "gpt-4o"],
        # 'mode_names': ["KB-Website"],
        # 'kb_id': '4BFLETNCSZ'  # website


    report_mode = True
    promptlist = "simple_prompts_small.csv"
    s3_uri = f"s3://watech-rppilot-bronze/evaluation_data/prompt_lists/{promptlist}"
    cohort_tag = f"{promptlist}_demo_threads_01"
    s3_out_batch = "evaluation_data/batch/small/"
    model_ids = ["us.amazon.nova-pro-v1:0", "us.amazon.nova-micro-v1:0", "us.anthropic.claude-3-5-haiku-20241022-v1:0", "us.anthropic.claude-3-5-sonnet-20241022-v2:0","gpt-4-turbo","gpt-4o"]
    #mode_names = ["KB-Legal Assistant"]
    mode_names = ["KB-Website"]

    kb_id='4BFLETNCSZ'#website
    #kb_id='4BFLETNCSZ'#legalaid

    do_batch_prompts_threads(bedrock,bedrock_agent_runtime, s3,openai_client, ChatHandler(), kb_id)
    
    #LLM_Judge_threads(bedrock, bedrock_agent_runtime,s3,openai_client)
    #create_analysis_csv(s3)

if __name__ == "__main__":
    main()