import os
import numpy as np
import logging
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)


model = load_model("fake_real_model.h5")

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)



@app.route("/")
def home():
    return render_template("index.html")



@app.route("/info")
def info():
    return jsonify({
        "model": "MobileNetV2",
        "task": "Fake vs Real Image Classification",
        "classes": ["Real", "Fake"]
    })



@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"})

        file = request.files["file"]

        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return jsonify({"error": "Invalid file type"})

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0][0]

        # Improved logic
        if prediction > 0.7:
            result = "Fake"
            confidence = prediction
        elif prediction < 0.3:
            result = "Real"
            confidence = 1 - prediction
        else:
            result = "Uncertain"
            confidence = max(prediction, 1 - prediction)

        confidence = round(float(confidence * 100), 2)

        logging.info(f"Prediction: {result}, Confidence: {confidence}%")

        return jsonify({
            "prediction": result,
            "confidence": confidence,
            "status": "success"
        })

    except Exception as e:
        logging.error(str(e))
        return jsonify({
            "error": str(e),
            "status": "failed"
        })



if __name__ == "__main__":
    app.run(debug=False)