from flask import Flask, render_template, request, jsonify
import cv2
import base64
import speech_recognition as sr
from fuzzywuzzy import fuzz

app = Flask(__name__)

# Load the classification model and labels
class_images = {
    "A": cv2.imread("static/images/A.jpg"),
    "B": cv2.imread("static/images/B.jpg"),
    "C": cv2.imread("static/images/C.jpg"),
    # Add more alphabet images as needed
}
labels = ["A", "B", "C"]
alphabets = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Initialize the result and output image variables
result = ""
output_img_base64 = ""

# Function to recognize speech from the microphone
def listen_to_microphone():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say an alphabet:")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        user_input = recognizer.recognize_google(audio)
        return user_input.upper()
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html', result=result, output_img_base64=output_img_base64)

@app.route('/alp2')
def alp2():
    return render_template('alp2.html')

@app.route('/process_voice', methods=['POST'])
def process_voice():
    spoken_letter = listen_to_microphone()
    print(spoken_letter)

    if spoken_letter is None or spoken_letter.upper() == 'EXIT':
        return jsonify({'result': '', 'output_img_base64': ''})
    
    try:
        spoken_letter_first_letter = spoken_letter[0].upper()
        print(f"Spoken letter first letter: {spoken_letter_first_letter}")

        if spoken_letter_first_letter in class_images:
            output_img = class_images[spoken_letter_first_letter]

            # Resize the image to fit the output window
            output_img = cv2.resize(output_img, (640, 480))

            # Convert the output image to bytes
            _, img_encoded = cv2.imencode('.png', output_img)
            global output_img_base64
            output_img_base64 = base64.b64encode(img_encoded).decode('utf-8')

            global result
            result = f"The recognized letter is {spoken_letter_first_letter}"
        else:
            result = f"No image defined for label {spoken_letter_first_letter}"

        # Return the result and output image as JSON
        return jsonify({'result': result, 'output_img_base64': output_img_base64})

    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
        return jsonify({'result': '', 'output_img_base64': ''})
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return jsonify({'result': '', 'output_img_base64': ''})
    except Exception as e:
        print(f"Error processing voice: {str(e)}")
        return jsonify({'result': '', 'output_img_base64': ''})

if __name__ == '__main__':
    app.run(debug=True)
