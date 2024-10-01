from flask import Flask, request, jsonify
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import requests
import os

app = Flask(__name__)

# Path to the directory containing the downloaded files
model_dir = "Model"

# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained(model_dir)
model = T5ForConditionalGeneration.from_pretrained(model_dir)


def translate(text, source_lang, target_lang):
    # Prepend source and target language tokens to input text
    input_text = f"{source_lang} to {target_lang}: {text}"

    # Tokenize input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Generate translation
    translated_ids = model.generate(input_ids=input_ids,
                                    max_length=256,
                                    num_beams=4,
                                    early_stopping=True)

    # Decode translated text
    translated_text = tokenizer.decode(translated_ids[0], skip_special_tokens=True)

    return translated_text


def generate_response_with_llm(prompt):
    # Send the translated question to the LLM using Ollama API
    llm_response = requests.post(
        "http://34.67.106.101:11434/api/generate",
        headers={"Content-Type": "application/json"},
        json={"model": "blackalpha/todlymist", "prompt": prompt, "stream": False}  # Replace with your model name
    )

    # Check if the LLM responded successfully
    if llm_response.status_code == 200:
        response = llm_response.json().get("response", "")
        return response
    else:
        return None


@app.route('/translate-and-respond', methods=['POST'])
def translate_and_respond():
    data = request.json
    sinhala_text = data.get("sinhala_text")

    # Step 1: Translate Sinhala to English
    english_translation = translate(sinhala_text, source_lang="si", target_lang="en")

    # Step 2: Generate response using LLM
    llm_response = generate_response_with_llm(english_translation)

    if llm_response is None:
        return jsonify({"error": "Failed to get a response from LLM"}), 500

    # Step 3: Translate LLM response from English to Sinhala
    sinhala_response = translate(llm_response, source_lang="en", target_lang="si")

    # Return the Sinhala response
    return jsonify({"sinhala_response": sinhala_response})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
