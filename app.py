from flask import Flask, request, jsonify
import requests
import os
import uuid

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Hugging Face Inference API Configuration
HF_API_URL = "https://api-inference.huggingface.co/models/speechbrain/emotion-recognition-wav2vec2-IEMOCAP"
HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    print("WARNING: HF_TOKEN environment variable not set. Requests may be rate-limited or rejected.")


@app.route("/")
def home():
    return {
        "message": "Speech Emotion Detection API is running",
        "backend": "Hugging Face Inference API"
    }


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save uploaded file temporarily
    filename = f"{uuid.uuid4()}.wav"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        # Read audio bytes to send to HF API
        with open(filepath, "rb") as f:
            audio_bytes = f.read()

        # Prepare headers
        headers = {}
        if HF_TOKEN:
            headers["Authorization"] = f"Bearer {HF_TOKEN}"

        # Call Hugging Face Inference API
        response = requests.post(
            HF_API_URL,
            headers=headers,
            data=audio_bytes,
            timeout=60  # allow time for model cold-start
        )

        # Handle non-200 responses (rate limits, model loading, etc.)
        if response.status_code != 200:
            return jsonify({
                "error": "Inference API request failed",
                "status_code": response.status_code,
                "details": response.text
            }), 502

        hf_response = response.json()

        # Parse HF API response
        # Expected format: [{"label": "neu", "score": 0.95}, {"label": "ang", "score": 0.03}, ...]
        if isinstance(hf_response, list) and len(hf_response) > 0:
            # Extract top prediction by score
            top_prediction = max(hf_response, key=lambda x: x.get("score", 0))
            emotion = top_prediction.get("label")
            score = top_prediction.get("score")
        elif isinstance(hf_response, dict):
            emotion = hf_response.get("label") or hf_response.get("emotion")
            score = hf_response.get("score")
        else:
            return jsonify({
                "error": "Unexpected API response format",
                "raw_response": hf_response
            }), 500

        result = {
            "emotion": emotion,
            "score": float(score) if score is not None else None,
            "all_predictions": hf_response  # optional: expose full breakdown
        }

        return jsonify(result)

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Request to inference API failed: {str(e)}"}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Always clean up the temp file
        if os.path.exists(filepath):
            os.remove(filepath)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
