
from flask import Flask, render_template, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
import torch.nn.functional as F
import requests
import time
import json
import hashlib
from pinecone import Pinecone, ServerlessSpec
from keyword_extraction import extract_entities_using_llm
from related_question_generation import generate_related_question

# Initialize Pinecone client
api_key = 'XXX'

index_name = "XXX"
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )

pinecone_index = pc.Index(index_name)
app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
sentiment_analyzer = pipeline("sentiment-analysis")
Name = "Ben"

# Mean pooling function
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Generate embeddings
def generate_embeddings(text):
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return F.normalize(sentence_embeddings, p=2, dim=1).squeeze().tolist()

# Initialize variables to store the last conversation
last_user_input = ""
last_chatbot_response = ""

def chatbot_query(user_input):
    global last_user_input, last_chatbot_response

    print(f"Previous User Input: {last_user_input}")
    print(f"Previous Chatbot Response: {last_chatbot_response}")

    end_conversation_keywords = ["thank you", "goodbye", "thanks", "exit"]
    if any(keyword in user_input.lower() for keyword in end_conversation_keywords):
        return "You're welcome! If you have more questions later, feel free to ask. 😊", None

    sentiment = sentiment_analyzer(user_input)[0]['label'].lower()
    vector = generate_embeddings(user_input)

    past_context = ""

    try:
        query_response = pinecone_index.query(
            vector=vector,
            top_k=5,
            include_values=True,
            include_metadata=True
        )

        # Retrieve the nearest chat entry
        for match in query_response['matches']:
            combined = match.get('metadata', {}).get('combined_entry', '')
            past_context += f"combined: {combined}"

    except Exception as e:
        print(f"Error querying Pinecone: {e}")

    # Call the keyword extraction function and pass the user input
    extracted_entities = extract_entities_using_llm(user_input,last_user_input="", last_chatbot_response="")
    print(f"Extracted entities: {extracted_entities}")  

    related_question = None
    
    print(past_context)

    prompt = (
    f"You are {Name}, a specialized AI real estate assistant. Answer real estate-related queries concisely and with relevant emojis where appropriate. "
    f"Avoid providing unnecessary details, especially if not directly asked. Keep responses clear and between 10 to 20 words. "
    f"Redirect unrelated questions back to real estate topics, but respond appropriately to the user's input without over-explaining. "
    f"Do not provide property details. If asked about properties, Gather information like Location, Budget, Bedroom and then simply say, 'Here are some properties for you' without extra explanations or apologies. "
    f"If any query is about about ben respond briefly with pride."
    f"You must respond in the same language used by the user in every case, without providing translations, explanations, or interpretations of terms in other languages."
    f"Respond like a human, balancing professionalism with friendliness. For example, if asked 'How do you remember my name?', respond with: 'I don’t actually remember names, but I’m here to help you with real estate matters.' "
    f"Handle user_input exactly as needed in a natural, conversational way."
    f"Refer to this past conversation context to remember previous interactions but do not include it directly in responses: {past_context}.\n"
    f"Last User Input: {last_user_input}. Last chatbot response:{last_chatbot_response}\n"
    f"If the user input is related to their last query (e.g., asking 'Tell me more' or 'What else?' after a previous list of points), continue the response from where it left off. "
    f"Sentiment Context: {sentiment}. Respond by remembering user emotions.\n"
    f"User: {user_input}\n"
)

    try:
        response = requests.post(
            'https://api.together.xyz/v1/chat/completions',
            json={
                "model": "mistralai/Mistral-7B-Instruct-v0.3",
                "max_tokens": 1024,
                "temperature": 0.1,
                "top_p": 0.7,
                "top_k": 50,
                "messages": [{"content": prompt, "role": "user"}]
            },
            headers={"Authorization": "Bearer XXX"}
        )
        if response.status_code == 200:
            last_chatbot_response = "\n".join(response.json()['choices'][0]['message']['content'].strip().split('. '))
            unique_id = hashlib.md5(user_input.encode('utf-8')).hexdigest() + f"-{int(time.time())}"
            combined_entry = f"User: {user_input} | Response: {last_chatbot_response}"
            pinecone_index.upsert([{
                "id": unique_id,
                "values": vector,
                "metadata": {
                    "combined_entry": combined_entry,
                    "user_message": user_input,
                    "bot_response": last_chatbot_response
                }
            }])
        else:
            last_chatbot_response = "Sorry, I'm unable to respond at the moment."
    except requests.exceptions.RequestException as e:
        last_chatbot_response = "Sorry, I'm unable to respond at the moment."

    # Update last_user_input at the end
    last_user_input = user_input  
    print(f"Updated Last User Input: {last_user_input}")
    print(f"Updated Last Chatbot Response: {last_chatbot_response}")

    if (extracted_entities and 'Type' in extracted_entities and extracted_entities['Type'] == "Search Intent") or\
       (extracted_entities and 'Ben' in extracted_entities and extracted_entities['Ben'] == "True"):
        related_question = generate_related_question(user_input, past_context)
        print("Related question generated:", related_question)
    if (related_question == user_input):
        related_question = generate_related_question(user_input,past_context)

    return last_chatbot_response, extracted_entities, related_question

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    response, extracted_entities, related_question = chatbot_query(user_input)
    return jsonify({'response': response, 'extracted_entities': extracted_entities, 'related_question': related_question})

if __name__ == '__main__':
    app.run(debug=True)