import os
import sys
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
from atproto import Client

# Add the correct path for local import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'bluesky-user-classifier'))
from classifier import BlueskyClassifier

load_dotenv()

MODEL_PATH = os.environ.get('MODEL_PATH', 'output/user-classifier')
BLUESKY_USER = os.environ.get('BLUESKY_USERNAME')
BLUESKY_PASS = os.environ.get('BLUESKY_PASSWORD')

# Use absolute path for static_folder
static_folder = os.path.join(os.path.dirname(__file__))
app = Flask(__name__, static_folder=static_folder, static_url_path='')

@app.route('/')
def index():
    return send_from_directory(static_folder, 'index.html')

@app.route('/<path:path>')
def static_proxy(path):
    return send_from_directory(static_folder, path)

@app.route('/api/classify', methods=['POST'])
def classify():
    data = request.get_json()
    username = data.get('username')
    if not username:
        return jsonify({'error': 'No username provided'}), 400
    if not BLUESKY_USER or not BLUESKY_PASS:
        return jsonify({'error': 'Bluesky credentials not set in environment (.env)'}), 500
    try:
        classifier = BlueskyClassifier(model_name=MODEL_PATH)
        classifier.load_finetuned_model(MODEL_PATH)
        client = Client()
        client.login(BLUESKY_USER, BLUESKY_PASS)
        classifier.client = client
        # Fetch posts (sync wrapper for async)
        import asyncio
        posts = asyncio.run(classifier.fetch_user_posts(username))
        result = classifier.classify_user(posts)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 