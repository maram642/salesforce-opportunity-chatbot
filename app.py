from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

import salesforce_connector
from chatbot_core import Chatbot
from salesforce_connector import load_data

STATIC_DIR = Path(__file__).parent / "static"

app = Flask(__name__, static_folder=None)
CORS(app)
app.json.ensure_ascii = False  # keep emojis readable in JSON

# Loads Salesforce data if credentials are set, otherwise data/opportunities.csv
chatbot = Chatbot(load_data())


@app.route("/")
def index():
    """The chat web page."""
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/health", methods=["GET"])
def health():
    names = [n.rstrip("_") for n in chatbot.opportunity_names[:3]]
    return jsonify({
        "status": "ok",
        "source": salesforce_connector.LAST_SOURCE,
        "opportunities": len(chatbot.opportunity_names),
        "examples": names,
    })


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = data.get("message")
    if not isinstance(message, str) or not message.strip():
        return jsonify({"error": "Send JSON like {\"message\": \"who owns AM003?\"}"}), 400
    return jsonify({"reply": chatbot.respond(message)})


if __name__ == "__main__":
    app.run(debug=True)
