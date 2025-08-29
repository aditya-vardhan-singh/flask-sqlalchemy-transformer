from flask import Blueprint, render_template, request, redirect, url_for, jsonify
from app import db
from app.models.models import User # Import your User model
from transformers import AutoModel, AutoTokenizer
import torch

# Create a Blueprint instance
bp = Blueprint('main', __name__)

# Load model + tokenizer once at startup
model_name = "nomic-ai/nomic-embed-text-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype="auto")

@bp.route('/')
def index():
    users = User.query.all()
    usernames = [user.username for user in users]
    print(users)
    return f"<h1>Hello, Modular Flask!</h1><p>Users: {usernames}</p>"

@bp.route('/add_user', methods=['GET', 'POST'])
def add_user():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        new_user = User(username=username, email=email)
        try:
            db.session.add(new_user)
            db.session.commit()
            return redirect(url_for('main.index')) # Note the 'main.' prefix due to Blueprint
        except Exception as e:
            db.session.rollback()
            return f"Error adding user: {e}", 500
    return '''
        <form method="POST">
            <input type="text" name="username" placeholder="Username">
            <input type="email" name="email" placeholder="Email">
            <input type="submit" value="Add User">
        </form>
    '''
    
    
@bp.route("/embed", methods=["POST"])
def embed_text():
    try:
        data = request.get_json()
        text = data.get("text", None)

        if not text:
            return jsonify({"error": "Missing 'text' field in request"}), 400

        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

        # Get embeddings (last_hidden_state pooled)
        with torch.no_grad():
            outputs = model(**inputs)
            # Example: mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

        return jsonify({"embedding": embeddings})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
