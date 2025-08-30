# app.py
from flask import Blueprint, request, jsonify
from app.models.models import ChatSession, ChatMessage
from datetime import datetime
from app import db

# Create a Blueprint instance with a unique name
bp = Blueprint('chatbot', __name__, url_prefix='/chat')

# Fake chatbot response function (replace with OpenAI/HF API later)
def generate_bot_reply(user_message: str) -> str:
    return f"Echo: {user_message}"  # placeholder bot

# ------------------- Routes -------------------

@bp.route("/chat", methods=["POST"])
def chat():
    data = request.json
    session_id = data.get("session_id")
    user_message = data.get("message")

    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    # Create new chat session if not provided
    if not session_id:
        chat_session = ChatSession()
        db.add(chat_session)
        db.commit()
        db.refresh(chat_session)
        session_id = chat_session.id
    else:
        chat_session = db.query(ChatSession).get(session_id)
        if not chat_session:
            return jsonify({"error": "Session not found"}), 404

    # Save user message
    user_msg = ChatMessage(
        session_id=session_id,
        sender="user",
        message=user_message
    )
    db.add(user_msg)

    # Generate bot response
    bot_reply = generate_bot_reply(user_message)

    bot_msg = ChatMessage(
        session_id=session_id,
        sender="bot",
        message=bot_reply
    )
    db.add(bot_msg)

    db.commit()

    return jsonify({
        "session_id": session_id,
        "user_message": user_message,
        "bot_reply": bot_reply
    })



@bp.route("/<int:session_id>", methods=["GET"])
def get_chat_history(session_id):
    chat_session = db.query(ChatSession).get(session_id)
    if not chat_session:
        return jsonify({"error": "Session not found"}), 404

    history = [
        {
            "id": msg.id,
            "sender": msg.sender,
            "message": msg.message,
            "timestamp": msg.timestamp.isoformat()
        }
        for msg in chat_session.messages
    ]

    return jsonify({"session_id": session_id, "messages": history})

