from flask import Flask, render_template, request, jsonify
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import pyttsx3
import base64
import tensorflow as tf

app = Flask(__name__)

# Load your hand tracking and classification models
cap = cv2.VideoCapture(0)  # Use 0 for the default camera
detector = HandDetector(maxHands=1)

# Load TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="model/model_unquant.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Dictionary mapping class labels to images
class_images = {
    "A": cv2.imread("static/images/A.jpg"),
    "B": cv2.imread("static/images/B.jpg"),
    "C": cv2.imread("static/images/C.jpg"),
    "D": cv2.imread("static/images/D.jpg"),
    "E": cv2.imread("static/images/E.jpg"),
    "F": cv2.imread("static/images/F.jpg"),
    "G": cv2.imread("static/images/G.jpg"),
    "H": cv2.imread("static/images/H.jpg"),
    "I": cv2.imread("static/images/I.jpg"),
    "J": cv2.imread("static/images/J.jpg"),
    "YES": cv2.imread("static/images/YES.jpg"),
    "NO": cv2.imread("static/images/NO.jpg"),
    "HELLO": cv2.imread("static/images/HELLO.jpg"),
    "PEACE": cv2.imread("static/images/PEACE.jpg"),
    "I LOVE YOU": cv2.imread("static/images/I_LOVE_YOU.jpg"),
    "THUMBS UP": cv2.imread("static/images/THUMBS_UP.jpg"),
    "THUMBS DOWN": cv2.imread("static/images/THUMBS_DOWN.jpg"),
    "1": cv2.imread("static/images/1.jpg"),
    "2": cv2.imread("static/images/2.jpg"),
    "3": cv2.imread("static/images/3.jpg"),
    "4": cv2.imread("static/images/4.jpg"),
    "5": cv2.imread("static/images/5.jpg"),
    "6": cv2.imread("static/images/6.jpg"),
    "7": cv2.imread("static/images/7.jpg"),
}

offset = 20
imgSize = 300
labels = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", 
    "YES", "NO", "HELLO", "PEACE", "I LOVE YOU", 
    "THUMBS UP", "THUMBS DOWN", "1", "2", "3", 
    "4", "5", "6", "7"
]

# Initialize initial height and width with default values
initial_height, initial_width = 480, 640

# Initialize the text-to-speech engine once outside the function
engine = pyttsx3.init()

# Function to perform text-to-speech
def speak_output(output_text):
    engine.say(output_text)
    engine.runAndWait()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/alp')
def alp():
    return render_template('alp.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        # Capture a frame
        ret, frame = cap.read()

        if not ret:
            return jsonify({"error": "Unable to capture the frame."})

        hands, frame = detector.findHands(frame)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = frame[y - offset:y + h + offset, x - offset:x + w + offset]

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # Resize imgWhite to 224x224 as expected by the model
            imgWhite_resized = cv2.resize(imgWhite, (224, 224))

            # Preprocess the image for the model
            imgWhite_resized = np.expand_dims(imgWhite_resized, axis=0).astype(np.float32)

            # Set the tensor to the model
            interpreter.set_tensor(input_details[0]['index'], imgWhite_resized)

            # Run the interpreter
            interpreter.invoke()

            # Get the prediction
            prediction = interpreter.get_tensor(output_details[0]['index'])
            index = np.argmax(prediction)
            predicted_label = labels[index]

            if predicted_label in class_images:
                output_img = cv2.resize(class_images[predicted_label], (initial_width, initial_height))
                _, img_encoded = cv2.imencode('.png', output_img)
                output_img_base64 = base64.b64encode(img_encoded).decode('utf-8')
                
                return jsonify({"result": predicted_label, "output_img": output_img_base64})
            else:
                return jsonify({"error": f"No image defined for class {predicted_label}"})
        else:
            return jsonify({"error": "No hands detected in the frame."})

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e), "result": None, "output_img": None})

if __name__ == '__main__':
    try:
        app.run(debug=True)
    finally:
        # Release the video capture resource when the application exits
        cap.release()
        cv2.destroyAllWindows()