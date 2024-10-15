# app.py
from flask import Flask, request, jsonify
from transformers import MT5ForConditionalGeneration, T5Tokenizer
import torch
from google.cloud import translate_v2 as translate
import requests
import os
import re

app = Flask(__name__)

# Set up Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials/google_api_key.json"

# Load the tokenizer and model
model_dir = "Model"
tokenizer = T5Tokenizer.from_pretrained(model_dir, legacy=True)
model = MT5ForConditionalGeneration.from_pretrained(model_dir)

# Initialize Google Translate client
translate_client = translate.Client()


def translate_with_confidence(text, source_lang, target_lang):
    input_text = f"{source_lang} to {target_lang}: {text}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Generate translation with output scores
    outputs = model.generate(
        input_ids=input_ids,
        max_length=512,
        num_beams=4,
        early_stopping=True,
        return_dict_in_generate=True,
        output_scores=True
    )

    # Decode the generated tokens to get the translated text
    translated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True,
                                       clean_up_tokenization_spaces=True)

    # Get the sequence score (log probability of the sequence)
    sequence_score = outputs.sequences_scores[0].item()

    # Get the length of the generated sequence (number of tokens)
    output_ids = outputs.sequences[0]
    sequence_length = len(output_ids)

    # Compute average log probability per token
    avg_log_prob_per_token = sequence_score / sequence_length

    # Exponentiate to get average per-token probability
    avg_prob_per_token = torch.exp(torch.tensor(avg_log_prob_per_token)).item()

    return translated_text, avg_prob_per_token


def translate_with_api(text, source_lang, target_lang):
    result = translate_client.translate(
        text,
        source_language=source_lang,
        target_language=target_lang
    )
    return result['translatedText']

def clean_text(text):
    # Remove any non-printable characters
    text = ''.join(filter(lambda x: x.isprintable(), text))
    # Remove any extra whitespace
    text = ' '.join(text.split())
    # Remove any special tokens or placeholders
    text = re.sub(r'<.*?>', '', text)
    return text

@app.route('/')
def home():
    return "Welcome to the Toddler App API! Use the /process endpoint to interact with the service."


@app.route('/process', methods=['POST'])
def process_request():
    data = request.json
    question = data.get("question")
    source_lang = data.get("source_lang", "si")
    target_lang = data.get("target_lang", "en")

    # First translation attempt using the MT5 model
    translated_question, avg_prob_per_token = translate_with_confidence(
        question, source_lang, target_lang
    )

    # Print the translated question
    print("Translated Question (Sinhala to English):", translated_question)

    # Set a threshold for confidence
    threshold = 0.2  # Adjust this threshold as needed

    # Check confidence and decide on using fallback API if confidence is low
    if avg_prob_per_token < threshold:
        print(f"Low confidence ({avg_prob_per_token:.4f}). Using Google Translate.")
        translated_question = translate_with_api(question, source_lang, target_lang)
        print("Translated Question after Google Translate:", translated_question)
    else:
        print(f"Translation confidence is acceptable ({avg_prob_per_token:.4f}).")

    # Send translated question to the LLM
    llm_response = requests.post(
        "http://127.0.0.1:11434/api/generate",
        headers={"Content-Type": "application/json"},
        json={"model": "blackalpha/todlymist", "prompt": translated_question, "stream": False}
    )

    if llm_response.status_code == 200:
        english_answer = llm_response.json().get("response")
    else:
        return jsonify({"error": "Failed to get a response from LLM"}), 500

    # Print the LLM's response
    print("LLM's Response (in English):", english_answer)

    # Clean the LLM's response
    english_answer = clean_text(english_answer)
    print("Cleaned LLM's Response (in English):", english_answer)

    # Translate the LLM's answer back to Sinhala
    final_answer, avg_prob_per_token = translate_with_confidence(english_answer, "en", "si")

    # Optionally, check the confidence again for the final translation
    if avg_prob_per_token < threshold:
        print(f"Low confidence in final translation ({avg_prob_per_token:.4f}). Using Google Translate.")
        final_answer = translate_with_api(english_answer, "en", "si")
        print("Final Answer after Google Translate:", final_answer)
    else:
        print(f"Final translation confidence is acceptable ({avg_prob_per_token:.4f}).")

    # Print the final answer
    print("Final Answer (English to Sinhala):", final_answer)

    return jsonify({"answer": final_answer})



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
