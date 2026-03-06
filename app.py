from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model path
MODEL_PATH = os.path.join(BASE_DIR, "models", "fruit_classifier.h5")

# Load model
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully")
except Exception as e:
    model = None
    print("Error loading model:", e)

# Classes
class_names = [
    'freshapples','freshbanana','freshcapsicum','freshcucumber',
    'freshokra','freshoranges','freshpotato','freshtomato',
    'rottenapples','rottenbanana','rottencapsicum','rottencucumber',
    'rottenokra','rottenoranges','rottenpotato','rottentomato'
]

# Static upload folder
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Image preprocessing
        img = image.load_img(filepath, target_size=(128,128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        if model is None:
            return jsonify({"error":"Model not loaded"}),500

        prediction = model.predict(img_array)
        index = np.argmax(prediction[0])

        predicted_class = class_names[index]
        confidence = float(prediction[0][index])

        result = {
            "predicted_class": predicted_class.replace("fresh","Fresh ").replace("rotten","Rotten "),
            "confidence": round(confidence*100,2)
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0", port=port)
