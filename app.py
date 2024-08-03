import tensorflow as tf
from flask import Flask, request, render_template, jsonify
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)
app.debug = True  # Enable debug mode

# Print current working directory and list files
print("Current working directory:", os.getcwd())
print("Files in the current directory:", os.listdir())

# Custom objects dictionary
custom_objects = {
    'mse': tf.keras.losses.MeanSquaredError(),
    'binary_crossentropy': tf.keras.losses.BinaryCrossentropy()
}

# Load the model with custom objects
try:
    model = tf.keras.models.load_model('age_gender_model.h5', custom_objects=custom_objects)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')

def home():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file and model:
        try:
            # Read and preprocess the image
            img = Image.open(io.BytesIO(file.read()))
            img_resized = img.resize((64, 64))  # Resize for model input
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Make predictions
            age_pred, gender_pred = model.predict(img_array)
            age = int(age_pred[0][0])
            gender = "Male" if gender_pred[0][0] > 0.5 else "Female"
            
            return jsonify({'age': age, 'gender': gender})
        except Exception as e:
            return jsonify({'error': str(e)})
    return jsonify({'error': 'Model not loaded or file processing failed'})

if __name__ == '__main__':
    app.run(debug=True)