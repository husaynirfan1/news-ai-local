from flask import Flask, request, jsonify, render_template
import torch
import ollama
import os
from openai import OpenAI
import json

app = Flask(__name__)

# ANSI escape codes for colors (not needed in web UI, but kept for consistency)
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

# Initialize OpenAI client
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='qwen2.5:1.5b'
)

# Load vault content and generate embeddings
vault_content = []
if os.path.exists("vault.txt"):
    with open("vault.txt", "r", encoding='utf-8') as vault_file:
        vault_content = vault_file.readlines()

vault_embeddings = []
for content in vault_content:
    response = ollama.embeddings(model='mxbai-embed-large', prompt=content)
    vault_embeddings.append(response["embedding"])

vault_embeddings_tensor = torch.tensor(vault_embeddings)

# Conversation history
conversation_history = []
system_message = "You are a helpful assistant that is an expert at extracting the most useful information from a given text. Also bring in extra relevant information to the user query from outside the given context."

# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('user_input')
    if user_input.lower() == 'quit':
        return jsonify({"response": "Goodbye!"})

    # Call your existing Ollama chat function
    response = ollama_chat(user_input, system_message, vault_embeddings_tensor, vault_content, "qwen2.5:1.5b", conversation_history)
    return jsonify({"response": response})

# Your existing functions (rewrite_query, get_relevant_context, ollama_chat) go here

if __name__ == '__main__':
    app.run(debug=True)