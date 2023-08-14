import requests
import time
from time import time
from datetime import datetime
from uuid import uuid4
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, Range, MatchValue
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import re
        
        
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()
  
        
def timestamp_to_datetime(unix_time):
    datetime_obj = datetime.fromtimestamp(unix_time)
    datetime_str = datetime_obj.strftime("%A, %B %d, %Y at %I:%M%p %Z")
    return datetime_str


# Connect to Oobabooga Api
# For local streaming, the websockets are hosted without ssl - http://
HOST = 'localhost:5000'
URI = f'http://{HOST}/api/v1/chat'

# For reverse-proxied streaming, the remote will likely host with ssl - https://
# URI = 'https://your-uri-here.trycloudflare.com/api/v1/generate'

model = SentenceTransformer('all-mpnet-base-v2')


def check_local_server_running():
    try:
        response = requests.get("http://localhost:6333/dashboard/")
        return response.status_code == 200
    except requests.ConnectionError:
        return False


# Check if local server is running
if check_local_server_running():
    client = QdrantClient(url="http://localhost:6333")
    print("Connected to local Qdrant server.")
else:
    url = open_file('./qdrant_url.txt')
    api_key = open_file('./qdrant_api_key.txt')
    try:
        client = QdrantClient(url=url, api_key=api_key)
        print("Connected to cloud Qdrant server.")
    except Exception as e:
        print(f"Failed to Connect to Qdrant Server: {e}")
    


def oobabooga(instruction, prompt):
    history = {'internal': [], 'visible': []}
    request = {
        'user_input': prompt,
        'max_new_tokens': 800,
        'history': history,
        'mode': 'instruct',  # Valid options: 'chat', 'chat-instruct', 'instruct'
        'instruction_template': 'Llama-v2',  # Will get autodetected if unset
        'context_instruct': f"{instruction}",  # Optional
        'your_name': f'USER',
        'regenerate': False,
        '_continue': False,
        'stop_at_newline': False,
        'chat_generation_attempts': 1,
        # Generation params. If 'preset' is set to different than 'None', the values
        # in presets/preset-name.yaml are used instead of the individual numbers.
        'preset': 'None',  
        'do_sample': True,
        'temperature': 0.85,
        'top_p': 0.2,
        'typical_p': 1,
        'epsilon_cutoff': 0,  # In units of 1e-4
        'eta_cutoff': 0,  # In units of 1e-4
        'tfs': 1,
        'top_a': 0,
        'repetition_penalty': 1.18,
        'top_k': 40,
        'min_length': 100,
        'no_repeat_ngram_size': 0,
        'num_beams': 1,
        'penalty_alpha': 0,
        'length_penalty': 1,
        'early_stopping': False,
        'mirostat_mode': 0,
        'mirostat_tau': 5,
        'mirostat_eta': 0.1,

        'seed': -1,
        'add_bos_token': True,
        'truncation_length': 4096,
        'ban_eos_token': False,
        'skip_special_tokens': True,
        'stopping_strings': []
    }

    response = requests.post(URI, json=request)

    if response.status_code == 200:
        result = response.json()['results'][0]['history']
    #    print(json.dumps(result, indent=4))
        print()
    #    print(result['visible'][-1][1])
        return result['visible'][-1][1]
        
        
def Qdrant_Upload(bot_name, query):
    bot_name = 'ASSISTANT'
    while True:
        try:
            payload = list()       
            timestamp = time()
            timestring = timestamp_to_datetime(timestamp)
            # Define the collection name, make sure to change search query collection name too.
            collection_name = f"ENTER COLLECTION NAME HERE"
            try:
                collection_info = client.get_collection(collection_name=collection_name)
            except:
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(size=model.get_sentence_embedding_dimension(), distance=Distance.COSINE),
                )
            embedding = model.encode([query])[0].tolist()
            unique_id = str(uuid4())
            metadata = {
                'bot': bot_name,
                'time': timestamp,
                'message': query,
                'timestring': timestring,
                'uuid': unique_id,
                'memory_type': 'Long_Term_Memory'
            }
            client.upsert(collection_name=collection_name,
                                 points=[PointStruct(id=unique_id, payload=metadata, vector=embedding)])
            return
        except Exception as e:
            print(f"ERROR: {e}")
            return
                  

if __name__ == '__main__':
    conversation = list()
    summary = list()
    bot_name = open_file('./Prompts/bot_name.txt')
    botnameupper = bot_name.upper()
    main_prompt = open_file(f'./Prompts/prompt_main.txt').replace('<<NAME>>', bot_name)
    greeting_msg = open_file(f'./Prompts/prompt_greeting.txt').replace('<<NAME>>', bot_name)
    conv_length = 12
    collection_name = f"ENTER COLLECTION NAME HERE"
    while True:
        try:
            instruction = "[INST] <<SYS>>\nYou are in the middle of a conversation with a user, generate a natural sounding response to their message.\n<</SYS>>"
            user_input = input(f'\n\nUSER: ')
            conversation.append({'content': f"{main_prompt}[/INST]\n{botnameupper}: {greeting_msg}"})
            db_result = None
            try:
                vector = model.encode([user_input])[0].tolist()
                hits = client.search(
                    collection_name=collection_name,
                    query_vector=vector, 
                    query_filter=Filter(
                        must=[
                            FieldCondition(
                                key="memory_type",
                                match=MatchValue(value="Long_Term_Memory")
                            )
                        ]
                    ),
                    limit=20
                )
                results = [hit.payload['message'] for hit in hits]
                # Sort results by most recent time    
                sorted_results = sorted(hits, key=lambda hit: hit.payload['time'], reverse=False)

                # Extract the 'message' field for the top 10 results
                db_result = [entry.payload['message'] for entry in sorted_results[:10]]
                print(f"{db_result}\n\n")
            except Exception as e:
                if "Not found: Collection" in str(e):
                    print("Collection has no memories.")
                else:
                    print(f"An unexpected error occurred: {str(e)}")
        #    print(db_result)
            conversation.append({'content': f"CHATBOT MEMORIES: {db_result}"})
            conversation.append({'content': f"[INST] USER INPUT: {user_input} [/INST]"})    
            conversation.append({'content': f"{botnameupper}: "})    
            prompt = ''.join([message_dict['content'] for message_dict in conversation])
            output = oobabooga(instruction, prompt)
            print(f"{botnameupper}: {output}")
            instruction = f"[INST] <<SYS>>\nExtract short and concise memories based on {bot_name}'s final response for upload to a memory database.  These should be executive summaries and will serve as {bot_name}'s memories.  Use the bullet point format: •<Executive Summary>\n<</SYS>>"
            summary.append({'content': f"LOG: {output}[/INST][INST]SYSTEM: Use the log to extract the salient points about the user and {bot_name}'s conversation. These points should be used to create concise executive summaries in bullet point format to serve as {bot_name}'s memories. Each bullet point should be considered a separate memory and contain full context.  Use the bullet point format: •<Executive Summary>[/INST]{botnameupper}: Sure! Here are some memories based on {bot_name}'s response:"})
            prompt = ''.join([message_dict['content'] for message_dict in summary])
            output_sum = oobabooga(instruction, prompt)
            print(output_sum)
            mem_check = input(f'\n\nUpload Memories? Y or N?: ')
            if 'y' in mem_check.lower():
                # Split on bullet point or double linebreak.
                segments = re.split(r'•|\n\s*\n', output_sum)
                for segment in segments:
                    if segment.strip() == '':
                        continue
                    else:
                        Qdrant_Upload(bot_name, segment)
                print('\nUpload Successful')
            else:
                pass
            conversation.clear()
            summary.clear()
        except Exception as e:
            print(e)