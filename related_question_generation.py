import requests

def generate_related_question(user_input, past_context):
    model = "meta-llama/Llama-3.2-3B-Instruct-Turbo"
    # model = "mistralai/Mistral-7B-Instruct-v0.3"
    prompt = (
        f"Dont have to generate question like user input. You have to generate question for user input. "
        f"Dont translate the questions which you are generating in other language. You have to generate related question in user language if it is tamil you have to generate in tamil or what ever the language is."
        f"Generte one question at a time."
        f"You dont have to ask question to user for eg'What are your ideal amenities when looking for a neighborhood near public transportation options?' dont generate question like this. you are helping user to ask question. generate like this'What are the ideal amenities when looking for a neighborhood near public transportation options?'. You have to generate what user ask in next question. These questions are sent to a chatbot remember"
        f"Focus on topics like neighborhood, amenities, nearby places, budget, etc. Make your question within 10 to 20 words, ensuring it's detailed and meaningful:\n"
        f"Dont generate big "
        f"Dont ask question like this'What properties are available for sale in Miami?' and 'What properties are currently listed for sale in Miami?' instead generate like ''How does Ben help me navigate the home-buying process without a traditional agent?"
        f"Generate different questions every time to encourage user what ever the language is be consice in question. It has to be unique and epic.."
        f"Just return the question thats enough dont give like this'Here's a generated question based on the user's input:'"
        f"If the query is about ben or bens support then generate question in style like 'Can Ben assist with understanding the different steps involved in the home-buying process?','What kind of support does Ben provide during the negotiation phase of buying a home?'"
        f"Just provide response. No need of mentioning it as Assistant"
        f"{past_context} use past context of user input if needed"
        f"User Input: {user_input}\n"
    )


    try:
        response = requests.post(
            'https://api.together.xyz/v1/chat/completions',
            json={
                "model": model,
                "max_tokens": 1024,
                "temperature": 0.5,
                "messages": [{"content": prompt, "role": "user"}]
            },
            headers={"Authorization": "Bearer XXX"}  
        )

        if response.status_code == 200:
            response_json = response.json()
            related_question = response_json['choices'][0]['message']['content'].strip()
            print("Generated related question:", related_question)  
            return related_question
        else:
            print("Error generating related question:", response.status_code, response.text)
            return None
    except requests.exceptions.RequestException as e:
        print(f"API request error: {e}")
        print("Response JSON for related question:", response_json)

        return None