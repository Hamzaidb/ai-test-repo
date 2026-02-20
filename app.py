"""
app.py – API Flask exposant un endpoint /chat.
Utilise le SDK OpenAI v1+ pour générer des réponses via gpt-4o.
"""

import os

from flask import Flask, jsonify, request
from openai import OpenAI

app = Flask(__name__)

# ---------- OpenAI client ----------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- Configuration ----------
DEFAULT_MODEL = "gpt-4o"
DEFAULT_SYSTEM_PROMPT = (
    "Tu es un assistant utile et concis. "
    "Réponds toujours en français sauf indication contraire."
)


@app.route("/chat", methods=["POST"])
def chat():
    """Endpoint principal : reçoit un message utilisateur et renvoie la réponse du modèle."""
    data = request.get_json(force=True)
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"error": "Le champ 'message' est requis."}), 400

    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.7,
        max_tokens=1024,
    )

    assistant_reply = response.choices[0].message.content
    usage = response.usage

    return jsonify(
        {
            "reply": assistant_reply,
            "model": response.model,
            "usage": {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            },
        }
    )


@app.route("/health", methods=["GET"])
def health():
    """Health-check basique."""
    return jsonify({"status": "ok", "model": DEFAULT_MODEL})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
