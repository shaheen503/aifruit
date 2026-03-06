from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import json

# Initialize Flask app
app = Flask(__name__)

# Dynamically locate the model path
base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_path, 'models', 'fruit_classifier.h5')

# Load the trained model
try:
    model = load_model(model_path)
    print("✅ Model loaded successfully.")
except Exception as e:
    model = None
    print(f"❌ Error loading model: {e}")

# Class labels
class_names = [
    'freshapples', 'freshbanana', 'freshcapsicum', 'freshcucumber',
    'freshokra', 'freshoranges', 'freshpotato', 'freshtomato',
    'rottenapples', 'rottenbanana', 'rottencapsicum', 'rottencucumber',
    'rottenokra', 'rottenoranges', 'rottenpotato', 'rottentomato'
]

# Ensure 'static' folder exists
static_dir = os.path.join(base_path, 'static')
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        try:
            # Save uploaded file
            image_filename = os.path.join(static_dir, file.filename)
            file.save(image_filename)
            print(f"🖼️ Image saved: {image_filename}")

            # Preprocess image
            img = image.load_img(image_filename, target_size=(128, 128))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0
            print("✅ Image preprocessed")

            if model is None:
                return jsonify({'error': 'Model not loaded'}), 500

            # Predict
            predictions = model.predict(img_array)
            predicted_index = np.argmax(predictions[0])
            predicted_class = class_names[predicted_index]
            confidence = predictions[0][predicted_index]

            result = {
                'predicted_class': predicted_class.replace('fresh', 'Fresh ').replace('rotten', 'Rotten '),
                'confidence': float(confidence)
            }
            print(f"🧠 Prediction: {result}")
            return jsonify(result)
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
