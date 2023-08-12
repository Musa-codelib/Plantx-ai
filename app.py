import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the pre-trained model
model = load_model('crop.h5')  # Replace with the actual path to your model file

app = Flask(__name__)

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    x = image.img_to_array(img)
    x = x / 255
    return np.expand_dims(x, axis=0)

@app.route('/upload', methods=['POST'])
def upload():
    # Check if the 'image' file is part of the request
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})

    file = request.files['image']

    # Save the uploaded file to a temporary location
    temp_img_path = 'temp_image.jpg'
    file.save(temp_img_path)

    # Prepare the image for prediction
    x = prepare_image(temp_img_path)

    # Perform the prediction
    result = model.predict(x)

    # Get the predicted class
    predicted_class = np.argmax(result)

    # Define your Classes list here (replace with your actual classes)
    Classes = ["Tomato___Bacterial_spot","Tomato___Early_blight","Tomato___Late_blight","Tomato___Leaf_Mold","Tomato___Septoria_leaf_spot","Tomato___Spider_mites Two-spotted_spider_mite","Tomato___Target_Spot","Tomato___Tomato_Yellow_Leaf_Curl_Virus","Tomato___Tomato_mosaic_virus","Tomato___healthy"]

    # Get the disease class label
    disease = Classes[predicted_class]

    # Delete the temporary image file
    os.remove(temp_img_path)

    return jsonify({'disease': disease})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
