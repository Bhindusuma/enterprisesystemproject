from flask import Flask, request, jsonify, render_template
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = Flask(__name__)


def preprocess_image(image):
    if not isinstance(image, Image.Image):
        raise ValueError("Input is not a valid PIL image")
    
    img = image.resize((224, 224))  # Resize image
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)  # Apply any required preprocessing
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route('/predict', methods=['POST'])

def analyze_sentiment():
    try:
        file = request.files['image']
        image = Image.open(io.BytesIO(file.read()))
        input_image = preprocess_image(image)

        # Load your pre-trained model
        model = load_model(r'C:\Users\Prave\Downloads\enterprisesystemproject\weather_prediction_model.keras')

        # Predict using the model
        prediction = model.predict(input_image)
       # return jsonify(prediction.tolist())
        damage_level = np.argmax(prediction)
        weather_conditions = ['Sunny', 'Rainy', 'Cloudy']  # Update based on your categories
        return jsonify({'weather_condition': weather_conditions[prediction.tolist()]})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)






