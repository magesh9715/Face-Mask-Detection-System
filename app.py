import os
from werkzeug.utils import secure_filename
from urllib.request import Request
from flask import Flask, render_template, Response, request, redirect, flash
from Myfunctions import *
import urllib
import secrets
import cv2
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Try to load TensorFlow components
try:
    from tensorflow.keras.preprocessing.image import img_to_array
    from tensorflow.keras.models import load_model
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    HAS_TENSORFLOW = True
except ImportError:
    print("Warning: TensorFlow not available")
    HAS_TENSORFLOW = False
    img_to_array = None
    load_model = None
    preprocess_input = None

secret = secrets.token_urlsafe(32)

app = Flask(__name__)
app.secret_key = secret
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model and cascade classifier for video stream
model = None
if HAS_TENSORFLOW and load_model is not None:
    try:
        model = load_model('mask_detector.h5')
    except Exception as e:
        print(f"Warning: Could not load model - {e}")
        model = None

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
camera = cv2.VideoCapture(0)


@app.route('/')
def index():
    """my main page"""
    return render_template('index.html')


@app.route('/ImageStream', methods=['GET', 'POST'])
def ImageStream():
    """the live page"""
    return render_template('RealtimeImage.html')


def gen_frames():
    """Generate frames with mask detection"""
    if model is None or not HAS_TENSORFLOW:
        # If model not loaded, just stream raw video with face detection
        while True:
            success, frame = camera.read()
            if not success:
                break
            
            # Just detect faces without classification
            faces_detected = face_cascade.detectMultiScale(
                frame, 1.2, 7, minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            for (x, y, w, h) in faces_detected:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Face Detected", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        return
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Detect faces
        faces_detected = face_cascade.detectMultiScale(
            frame, 1.2, 7, minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces_detected) > 0:
            faces_images = []
            for (x, y, w, h) in faces_detected:
                cropped_face = frame[y:y + h, x:x + w]
                face_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
                face_resized = cv2.resize(face_rgb, (224, 224))
                face_array = img_to_array(face_resized)
                faces_images.append(face_array)
            
            faces_array = np.array(faces_images)
            faces_preprocessed = preprocess_input(faces_array)
            predictions = model.predict(faces_array, verbose=0)
            
            # Draw predictions
            for idx, (x, y, w, h) in enumerate(faces_detected):
                (WithoutMask, CorrectMask, InCorrectMask) = predictions[idx]
                
                if max(predictions[idx]) == CorrectMask:
                    label = "Correct Mask"
                    color = (0, 255, 0)
                elif max(predictions[idx]) == InCorrectMask:
                    label = "Incorrect Mask"
                    color = (0, 165, 255)
                else:
                    label = "No Mask"
                    color = (0, 0, 255)
                
                confidence = max(WithoutMask, CorrectMask, InCorrectMask) * 100
                label_text = f"{label}: {confidence:.2f}%"
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label_text, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/takeimage', methods=['POST'])
def takeimage():
    # Open webcam fresh each time
    camera_temp = cv2.VideoCapture(0)
    ret, frame = camera_temp.read()

    if not ret or frame is None:
        camera_temp.release()
        flash("Error: Could not capture image from webcam. Please retry.")
        return render_template('Error.html')

    filename = "capture.png"
    save_path = os.path.join("static", filename)
    cv2.imwrite(save_path, frame)
    camera_temp.release()  # ✅ release before continuing

    # Now process image
    results = image_preprocessing(filename)
    if results is None:
        return render_template('Error.html')
    else:
        img_preds, frame, faces_detected = results
        results2 = predictions_results(img_preds, frame, faces_detected, filename)
        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        return render_template('PictureResult.html',
                               user_image=full_filename,
                               number_of_face=f"Number of faces detected: {results2[0]}",
                               no_mask_face=f"No face mask count: {results2[1]}",
                               correct_mask_face=f"Correct face mask count: {results2[2]}",
                               incorrect_mask_face=f"Incorrect face mask count: {results2[3]}")


@app.route('/UploadImage', methods=['POST'])
def UploadImage():
    """the upload image page"""
    return render_template('UploadPicture.html')


@app.route('/UploadImageFunction', methods=['POST'])
def UploadImageFunction():
    if request.method == 'POST':

        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']

        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        # If user uploads the correct Image File
        if file and allowed_file(file.filename):

            # Pass it a filename and it will return a secure version of it.
            # The filename returned is an ASCII only string for maximum portability.
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            results = image_preprocessing(filename)
            if results is None:
                return render_template('Error.html')
            else:

                img_preds = results[0]
                frame = results[1]
                faces_detected = results[2]

                results2 = predictions_results(img_preds, frame, faces_detected, filename)
                full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)

                return render_template('UploadPicture.html', user_image=full_filename,
                                       number_of_face="Number of faces detected: {}".format(results2[0]),
                                       no_mask_face="No face mask count: {}".format(results2[1]),
                                       correct_mask_face="Correct face mask count: {}".format(results2[2]),
                                       incorrect_mask_face="Incorrect face mask count: {}".format(results2[3]))


@app.route('/UploadURLImage', methods=['POST'])
def UploadURLImage():
    """the upload an image URL page"""
    return render_template('UploadURLImage.html')


@app.route('/ImageUrl', methods=['POST'])
def ImageUrl():
    """the upload an image URL page"""

    # Fetch the Image from the Provided URL
    url = request.form['url']
    filename = url.split('/')[-1]
    if allowed_file(filename):
        try:
            urllib.request.urlretrieve(url, f"static/{filename}")
        except Exception as e:
            flash(f"Error downloading image: {str(e)}")
            return render_template('Error.html')

        results = image_preprocessing(filename)
        if results is None:
            return render_template('Error.html')
        else:

            img_preds = results[0]
            frame = results[1]
            faces_detected = results[2]

            results2 = predictions_results(img_preds, frame, faces_detected, filename)
            full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            return render_template('UploadURLImage.html', user_image=full_filename,
                                   number_of_face="Number of faces detected: {}".format(results2[0]),
                                   no_mask_face="No face mask count: {}".format(results2[1]),
                                   correct_mask_face="Correct face mask count: {}".format(results2[2]),
                                   incorrect_mask_face="Incorrect face mask count: {}".format(results2[3]))


if __name__ == '__main__':
    app.run(debug=True)
