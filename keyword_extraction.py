# keywordextract.py
import requests
import json

def extract_entities_using_llm(user_input, last_user_input="", last_chatbot_response=""):
    model = "mistralai/Mistral-7B-Instruct-v0.3"    
    prompt = f"""
    Analyze the user's query to extract key details about property search. Focus on identifying information like location, maximum price (numerical), number of bedrooms, number of bathrooms, and property type.
    User Input: "{user_input}"
    Last User Input: "{last_user_input}"
    - If the user input is like 'show me some properties' in one chat 'In miami in one chat'
    - If they are looking to find properties, set "Type" as "Search Intent."
    - If the query is general or unrelated to finding properties, set "Type" as "General Intent."
    - If the query is related to Ben like' What does ben do', then set 'Ben' is 'True' else 'false'.
    Base "Type" only on the overall purpose of the query.

    Return the extracted details in the following JSON format:

    {{
        "Location": "<Extracted location>",
        "MaximumPrice": "<Extracted price>",
        "Bedrooms": "<Extracted bedrooms>",
        "Bathrooms": "<Extracted bathrooms>",
        "PropertyType": "<Extracted property type>",
        "Type": "<Search Intent or General Intent>",
        "Ben": "True, False"
    }}
    """
    try:
        response = requests.post(
            'https://api.together.xyz/v1/chat/completions',
            json={
                "model": model,
                "max_tokens": 500,
                "temperature": 0.5,
                "top_p": 0.9,
                "messages": [{"content": prompt, "role": "user"}]
            },
            headers={"Authorization": "Bearer XXX"}  # Replace with your API key
        )

        if response.status_code == 200:
            content = response.json().get('choices', [{}])[0].get('message', {}).get('content', '')
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                print("Failed to decode JSON from content:", content)  # Debugging log
                return None
            
        else:
            print(f"Error: {response.status_code}")
            return None
        
    except requests.exceptions.RequestException as e:
        print(f"API request error: {e}")
        return None
