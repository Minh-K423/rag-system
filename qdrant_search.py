import string

import json
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

import numpy as np
import matplotlib
matplotlib.use('QtAgg') 
import matplotlib.pyplot as plt
from PIL import Image
import pathlib

import torch
import torch.nn.functional as F

from transformers import CLIPProcessor, CLIPModel
from openai import OpenAI


client_openai = OpenAI(
    base_url= "http://127.0.0.1:1234/v1",
    api_key = "qwen"
)


client_qdrant = QdrantClient(host="localhost", port=6333)
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32") 
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def build_user_payload(user_input: str) -> str:
    sanitized_input = user_input.replace("</raw_query>", "") #This is to prevent the user from injecting malicious content that could break the XML structure. For example, if the user input contains "</raw_query>", it could prematurely close the XML tag and potentially allow for injection of arbitrary content. By removing this substring, we ensure that the user input cannot interfere with the XML structure.
    return f"<raw_query>\n{sanitized_input}\n</raw_query>"

# Few-shot demonstrations covering each branch of the rules: Vietnamese -> English translation,
# English passthrough with typo correction, fluff removal, slang/dialect handling, and pure JSON output shape.
# Each pair is (messy user input, expected raw JSON reply) and is expanded into real user/assistant turns below.
FEW_SHOT_EXAMPLES = [
    (
        "Xin chao, ban co the giup toi tim hieu ve tri tue nhan tao khong?",
        {"cleaned_query": "artificial intelligence"},
    ),
    (
        "cho minh xem video ve pho bo Ha Noi di",
        {"cleaned_query": "Hanoi beef pho"},
    ),
    (
        "Heloo!! plz show me some vidz of a cat playin with a ball",
        {"cleaned_query": "cat playing with a ball"},
    ),
    (
        "what is the weather like in paris tomorrw?",
        {"cleaned_query": "weather in Paris tomorrow"},
    ),
    (
        "Hôm nay tôi rất là mệt, đánh xe đi chơi đâu",
        {"cleaned_query": "driving while tired"},
    ),
    (
        "machine learning",
        {"cleaned_query": "machine learning"},
    ),
]


def build_few_shot_messages():
    """Expand FEW_SHOT_EXAMPLES into alternating user/assistant message turns.
    The user turns go through build_user_payload so demonstrations match the exact
    message format the model will receive at inference time."""
    messages = []
    for raw_input, expected_output in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": build_user_payload(raw_input)})
        messages.append({"role": "assistant", "content": json.dumps(expected_output)})
    return messages


def translate_query(query: str):

    prompt_instructions = f'''You are a precise query-preprocessing component for a RAG retrieval system. Your sole job is to clean, correct, and standardize user search queries. Strict Operational Rules:
    1. Detect the language of the user input.
    2. If the query is in Vietnamese (including slang, regional dialects, or informal text), correct any typos/grammar and TRANSLATE it to clear, searchable English.
    3. If the query is already in English, correct any typos or spelling mistakes, but KEEP it in English.
    4. Normalize the text: remove conversational fluff (e.g., "Hello", "can you help me find", "please show me"), greeting phrases, and punctuation.
    5. Keep the query short and keyword-like. Prefer 3-8 content words. Do not invent detail that is not implied by the input.
    6. Output your final answer ONLY as a raw JSON object matching this schema: {{"cleaned_query": "string"}}. Do not include any introductory or concluding text, markdown, or code fences.
    Follow the examples exactly in both the transformation applied and the JSON output format.
    '''

    messages = [{"role": "system", "content": prompt_instructions}]
    messages.extend(build_few_shot_messages())
    messages.append({"role": "user", "content": build_user_payload(query)})

    completion = client_openai.chat.completions.create(
        model="qwen/qwen3-4b-2507",
        messages=messages,
        temperature=0.4,  
    )

    try: 
        response_content = json.loads(completion.choices[0].message.content)
        return response_content.get("cleaned_query", query)  # Fallback to original query if key is missing
    except json.JSONDecodeError:
        return query  # Fallback to original query if JSON parsing fails


def process_query(query: str):
    processed_query = processor(text=query, return_tensors='pt')
    
    with torch.no_grad():
        output = model.get_text_features(
            input_ids=processed_query.input_ids,
            attention_mask=processed_query.attention_mask,
        )
        unnormed_query_vector = output.pooler_output[0].numpy() 
        normalized_query_vector = unnormed_query_vector / np.linalg.norm(unnormed_query_vector)  # Normalize the query vector
        query_vector = normalized_query_vector.tolist()  # Convert to list for Qdrant  

    return query_vector 

def display_images(image_paths: list):
    plt.figure(figsize=(15, 15))
    for i, image_path in enumerate(image_paths):
        img = Image.open(image_path)
        plt.subplot(3, 2, i + 1)
        plt.imshow(img)
        plt.axis('off')
    plt.show()


def main():

    while True:

        user_query = input("Enter your search query: (Type 'exit' or 'quit' to stop)")

        if user_query.lower() in ["exit", "quit"]:
            print("Exiting the search.")
            break

        user_query = translate_query(user_query)
        print(f"Processed Query: {user_query}")

        query_vector = process_query(user_query)
            
        search_results = client_qdrant.query_points(
            collection_name="video_features_collection",
            query=query_vector,
            limit=6,
            with_payload=True,
            with_vectors=True
        ).points

        image_paths = []

        print("\nTop 6 search results:")

        for i, result in enumerate(search_results):

            assert result.payload is not None
            
            video_name = result.payload.get("video_name", "Unknown")
            frame_index = result.payload.get("frame_index", "Unknown")
            title = result.payload.get("title", "No available key")
            description = result.payload.get("description", "No available key")
            score = result.score if hasattr(result, 'score') else "N/A"
            print(f"Result {i+1}: Video Name: {video_name}, Frame Index: {frame_index}, Score: {score}")

            if video_name[0:3] == "L26":
                category = int(video_name[-3:]) // 100
                image_path = pathlib.Path("/mnt/Shared_data/Data") / f"Keyframes_{video_name[0:3]}_{chr(ord('a') + category)}"  / "keyframes" / video_name / f"{frame_index:03d}.jpg"
            else: 
                image_path = pathlib.Path("/mnt/Shared_data/Data") / f"Keyframes_{video_name[0:3]}" / "keyframes" / video_name / f"{frame_index:03d}.jpg" #The / operator has higher level of priority compared to + operator so image_path = pathlib.Path("/mnt/Shared_data/Data") / "Keyframes" + video_name[0:3] / "keyframes" / video_name / f"{frame_index:03d}" + ".jpg"  would be wrong, so either change to f"..." or wrap them into (...)
        
            image_paths.append(image_path)

        
        display_images(image_paths)



if __name__ == "__main__":
    main()