import base64
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image
from io import BytesIO

# Initialize app
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from frontend

# Load your Keras model
model = load_model("best_model1.keras")  # Ensure this file is in the same directory

# Load face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("📥 /predict endpoint hit")
        data = request.get_json()

        if 'image' not in data:
            return jsonify({'error': 'No image received'}), 400

        # Decode the base64 image
        # Decode the base64 image (safe for both raw base64 or data URL)
        image_base64 = data['image']
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]

        image_data = base64.b64decode(image_base64)

        image = Image.open(BytesIO(image_data)).convert('RGB')
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        print("🔍 Faces detected:", len(faces))

        if len(faces) == 0:
            return jsonify({'emotion': 'No face detected'})

        # Pick the first face detected
        x, y, w, h = faces[0]
        roi_color = img_array[y:y+h, x:x+w]  # RGB face region
        roi_color = cv2.resize(roi_color, (224, 224))
        roi = img_to_array(roi_color)
        roi = np.expand_dims(roi, axis=0) / 255.0


        # Make prediction
        prediction = model.predict(roi)
        emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        predicted_emotion = emotions[np.argmax(prediction[0])]

        print("✅ Predicted:", predicted_emotion)
        return jsonify({'emotion': predicted_emotion})

    except Exception as e:
        print("❌ Error during prediction:", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

