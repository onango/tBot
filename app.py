import cv2
import os
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import numpy as np
import base64
import encode
import face_recognition
import cvzone

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'
socketio = SocketIO(app)


def process_img(image_array):
    encodeList, user_ids = encode.fetch_all()

    modeType = 0
    counter = 0
    id = -1
    visaImg = []
    imgModeList = []

    folderPath = 'Images'

    imgF = image_array

    imgS = cv2.resize(imgF, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    if faceCurFrame:
        for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
            matches = face_recognition.compare_faces(
                encodeList, encodeFace)
            faceDis = face_recognition.face_distance(
                encodeList, encodeFace)
            
            if faceDis.size > 0:
                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    emit('message', {'text': 'Stop streaming... Match found'})
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    bbox = x1, y1, x2 - x1, y2 - y1
                    imgF = cvzone.cornerRect(
                        imgF, bbox, rt=0)
                    id = user_ids[matchIndex]
                    if counter == 0:
                        # cvzone.putTextRect(
                        #     imgF, "Loading", (275, 400))
                        counter = 1
                        modeType = 1
            else:
                counter = 0
                modeType = 0
                emit('message', {'text': '11 Stop streaming... Match found'})

    if counter != 0:
        if counter == 1:
            visaInfo = encode.fetch_visa(id)
            visaImg = cv2.imread(os.path.join(
                folderPath, visaInfo['path']), cv2.IMREAD_COLOR)
            modeType = 2

            resized_visaImg = cv2.resize(visaImg, (216, 216))
            #imgF = resized_visaImg
            # Send recognized image as SSE event
            ret, buffer = cv2.imencode('.jpg', imgF)
            frame = buffer.tobytes()
            #frame = frame + b'\nRecognized: yes\n'

            k = cv2.waitKey(1)
            if k == ord('o'):
                counter = 0
                modeType = 0
    else:
        modeType = 3
        counter = 0


# Path to the directory to save frames
SAVE_DIR = 'frames'

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def test_connect():
    emit('message', {'match': False, 'step': 1})
    print('Client connected')

@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected')

@socketio.on('stream')
def process_frame(data):
    frame_data = data['image']
    frame_number = data['frame_number']
    
    image_bytes = base64.b64decode(frame_data .split(',')[1])
    
    # Convert the image bytes to a numpy array
    image_array = np.frombuffer(image_bytes, np.uint8)
    
    # Decode the image array using OpenCV
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    # Save the image using OpenCV's imwrite() method
    frame_number = 1
    filename = f'frame_{frame_number}.jpg'
    cv2.imwrite('frames/'+filename, image)
    
    # Process the frame
    # Example: Check if frame number is 100
    if frame_number != 100:
        encodeList, user_ids = encode.fetch_all()

        modeType = 0
        counter = 0

        imgF = image
        

        imgS = cv2.resize(image, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        faceCurFrame = face_recognition.face_locations(imgS)
        encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

        if faceCurFrame:
            for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
                matches = face_recognition.compare_faces(encodeList, encodeFace)
                faceDis = face_recognition.face_distance(encodeList, encodeFace)
                
                if faceDis.size > 0:
                    matchIndex = np.argmin(faceDis)

                    if matches[matchIndex]:
                       
                        y1, x2, y2, x1 = faceLoc
                        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                        bbox = x1, y1, x2 - x1, y2 - y1
                        imgF = cvzone.cornerRect(imgF, bbox, rt=0)
                        _, image_encoded = cv2.imencode('.jpg', imgF)
                        image_base64 = base64.b64encode(image_encoded).decode('utf-8')
                        emit('message', {'match': True, 'step': 3,'frame': image_base64})
                        id = user_ids[matchIndex]
                        if counter == 0:
                            counter = 1
                            modeType = 1
                else:
                    counter = 0
                    modeType = 0
                    emit('message', {'match': False, 'step': 4})
        else:
            emit('message', {'match': False, 'step': 1})
        
        # Save the frame to a directory
        # filename = f'frame_{frame_number}.jpg'
        # cv2.imwrite('output.jpg', imgF)

    else:
        # Continue streaming
        emit('message', {'text': 'Continuing streaming...'})

if __name__ == '__main__':
    socketio.run(app)
