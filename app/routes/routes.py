from flask import Blueprint, render_template, request, redirect, url_for, jsonify
from app import db
from app.models.models import User # Import your User model
from transformers import AutoModel, AutoTokenizer
import torch
# from models.gen_pipeline import NextStepPipeline

# Create a Blueprint instance
bp = Blueprint('main', __name__)

### Gemini API to detect and verify user's claim about items

import os
import json
from flask import Flask, request, jsonify
import google.generativeai as genai

app = Flask(__name__)

# Configure Gemini with your API key (set it in environment)
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Initialize the multimodal model
model = genai.GenerativeModel("gemini-1.5-flash")

@bp.route("/detect", methods=["POST"])
def detect_items():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    if "claim" not in request.form:
        return jsonify({"error": "No claim provided"}), 400

    image_file = request.files["image"]
    user_claim = request.form["claim"]

    # Read the file as bytes
    image_bytes = image_file.read()

    # Gemini expects dict with mime_type + data
    image_part = {
        "mime_type": image_file.mimetype,  # e.g. "image/jpeg"
        "data": image_bytes
    }

    prompt = f"""
    You are a survival item detection assistant.
    Task:
    1. Analyze the uploaded image for survival items.
    2. Respond in strict JSON with these keys:
       - food_grains (true/false)
       - water (true/false)
       - shelter (true/false)
       - oxygen_cylinder (true/false)
       - detected_items (list of strings naming visible items)
    3. The user claims: "{user_claim}"
       - Add "claim_correct" (true/false) depending on whether the claim matches the image.
       - Add "reason" explaining why the claim is correct or incorrect.

    Example response:
    {{
      "food_grains": true,
      "water": false,
      "shelter": true,
      "oxygen_cylinder": false,
      "detected_items": ["rice sack", "tent"],
      "claim_correct": false,
      "reason": "The image shows rice sacks and a tent, but no oxygen cylinder as claimed."
    }}
    """

    response = model.generate_content(
        [prompt, image_part],
        generation_config={"response_mime_type": "application/json"}
    )

    try:
        result = json.loads(response.text)
    except Exception:
        result = {"error": "Model did not return valid JSON", "raw": response.text}

    return jsonify(result)



# @bp.route("/detect", methods=["POST"])
# def detect_items():
#     if "image" not in request.files:
#         return jsonify({"error": "No image uploaded"}), 400

#     image_file = request.files["image"]

#     # Build the prompt for survival items
#     prompt = """
#     You are a survival item detection assistant.
#     Given the image, respond in strict JSON with these keys:
#     - food_grains (true/false)
#     - water (true/false)
#     - shelter (true/false)
#     - oxygen_cylinder (true/false)

#     Example response:
#     {"food_grains": true, "water": false, "shelter": true, "oxygen_cylinder": false}
#     """

#     response = model.generate_content(
#         [prompt, image_file],
#         generation_config={"response_mime_type": "application/json"}
#     )

#     # The response.text should already be JSON
#     try:
#         result = json.loads(response.text)
#     except Exception:
#         result = {"error": "Model did not return valid JSON", "raw": response.text}

#     return jsonify(result)





# Use a pipeline as a high-level helper
# from transformers import pipeline

# pipe = pipeline("object-detection", model="vosstalane/object-detection")

# # Load model + tokenizer once at startup
# model_name = "nomic-ai/nomic-embed-text-v1.5"
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# model = AutoModel.from_pretrained(
#     "nomic-ai/nomic-embed-text-v1.5",
#     trust_remote_code=True,
#     torch_dtype=torch.float32  # or torch.float16 if you want FP16
# )


# HF_HUB = "stepfun-ai/NextStep-1-Large"

# # load model and tokenizer
# tokenizer = AutoTokenizer.from_pretrained(HF_HUB, local_files_only=True, trust_remote_code=True)
# model = AutoModel.from_pretrained(HF_HUB, local_files_only=True, trust_remote_code=True)
# pipeline = NextStepPipeline(tokenizer=tokenizer, model=model).to(device="cuda", dtype=torch.bfloat16)

# # set prompts
# positive_prompt = "masterpiece, film grained, best quality."
# negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry."
# example_prompt = "A realistic photograph of a wall with \"NextStep-1.1 is coming\" prominently displayed"

# # generate image from text
# IMG_SIZE = 512
# image = pipeline.generate_image(
#     example_prompt,
#     hw=(IMG_SIZE, IMG_SIZE),
#     num_images_per_caption=1,
#     positive_prompt=positive_prompt,
#     negative_prompt=negative_prompt,
#     cfg=7.5,
#     cfg_img=1.0,
#     cfg_schedule="constant",
#     use_norm=False,
#     num_sampling_steps=28,
#     timesteps_shift=1.0,
#     seed=3407,
# )[0]
# image.save("./assets/output.jpg")


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



#### YOLO final
# import os
# from flask import Flask, request, jsonify
# from ultralytics import YOLO

# app = Flask(__name__)
# model = YOLO("yolov8n.pt")

# UPLOAD_FOLDER = "./uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # ✅ create if not exists

# @bp.route("/verify", methods=["POST"])
# def verify():
#     file = request.files["image"]
#     file_path = os.path.join(UPLOAD_FOLDER, file.filename)
#     file.save(file_path)   # now it won't crash

#     claim = request.form.get("claim")

#     results = model(file_path)

#     detected_objects = []
#     for r in results:
#         for box in r.boxes:
#             cls = model.names[int(box.cls[0])]
#             conf = float(box.conf[0])
#             detected_objects.append({"object": cls, "confidence": conf})

#     claim_found = any(
#         claim.lower() in obj["object"].lower()
#         for obj in detected_objects
#     )

#     return jsonify({
#         "claim": claim,
#         "claim_verified": claim_found,
#         "detections": detected_objects
#     })



## YOLO

# from flask import Flask, request, jsonify
# from ultralytics import YOLO

# app = Flask(__name__)

# Load pretrained YOLO model (COCO dataset, 80 classes)
# model = YOLO("yolov8n.pt")

# @app.route("/predict", methods=["POST"])
# def predict():
#     file = request.files["image"]
#     file_path = f"./uploads/{file.filename}"
#     file.save(file_path)

#     results = model(file_path)  # Run detection
#     objects = []
#     for r in results:
#         for box in r.boxes:
#             cls = model.names[int(box.cls[0])]
#             conf = float(box.conf[0])
#             objects.append({"object": cls, "confidence": conf})

#     return jsonify(objects)

# if __name__ == "__main__":
#     app.run(debug=True)

    
# @bp.route("/embed", methods=["POST"])
# def embed_text():
#     try:
#         data = request.get_json()
#         text = data.get("text", None)

#         if not text:
#             return jsonify({"error": "Missing 'text' field in request"}), 400

#         # Tokenize
#         inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

#         # Get embeddings (last_hidden_state pooled)
#         with torch.no_grad():
#             outputs = model(**inputs)
#             # Example: mean pooling
#             embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

#         return jsonify({"embedding": embeddings})

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

def create_survival_prompt(user_message):
    return f"""You are a survival expert and emergency medical assistance AI. 
    Your goal is to provide immediate, practical guidance for survival and emergency situations.
    
    Important guidelines:
    1. Always prioritize life-threatening situations
    2. Provide clear, step-by-step instructions
    3. If medical attention is needed, always recommend seeking professional help
    4. Focus on immediate actions that can be taken
    5. Be concise but thorough
    
    User situation: {user_message}
    
    Provide guidance in this format:
    1. Severity assessment
    2. Immediate actions needed
    3. Step-by-step instructions
    4. Additional precautions
    5. When to seek professional help
    """

@bp.route("/chat", methods=["POST"])
def survival_chat():
    try:
        data = request.get_json()
        if not data or "message" not in data:
            return jsonify({"error": "No message provided"}), 400

        user_message = data["message"]
        prompt = create_survival_prompt(user_message)

        # Generate response using Gemini
        response = model.generate_content(prompt)
        
        return jsonify({
            "response": response.text,
            "status": "success"
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

