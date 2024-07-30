from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = Flask(__name__)
model = load_model('C:/Users/Bhindu Suma/Downloads/air_pollution_model.keras')

def preprocess_image(img):
    img = img.resize((150, 150))  # Change the size to match your model's input
    img = img.convert('RGB')
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    try:
        img = Image.open(io.BytesIO(file.read()))
        processed_img = preprocess_image(img)
        
        prediction = model.predict(processed_img)
        damage_level = np.argmax(prediction)
        weather_conditions = ['Sunny', 'Rainy', 'Cloudy']  # Update based on your categories
        return jsonify({'weather_condition': weather_conditions[int(damage_level)]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)






