from flask import Flask, jsonify, request
import os
import cv2
import face_recognition as fr
import face_recognition
import numpy as np
import matplotlib.pyplot as plt
from flask_cors import CORS



app = Flask(__name__)
CORS(app)

root_path = 'faces'
dir_names = os.listdir(root_path)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB


def get_encoded_faces():
    encoded = {}
    for fnames in dir_names:
        #print(fnames)
        persons_dir = os.path.join(root_path, fnames)
        encodings = []
        for f in os.listdir(persons_dir):
            #print(f)
            if f.endswith(".jpg") or f.endswith(".png"):
                face = fr.load_image_file('faces/' + fnames + '/' + f)
                encoding = fr.face_encodings(face)
                #encoded[fnames] = encoding
                if encoding: 
                    encodings.append(encoding[0])
        if encodings:  
            encoded[fnames] = encodings
    print (encoded)

    return encoded


def unknown_image_encoded(img):
    face = fr.load_image_file("faces/" + img)
    encoding = fr.face_encodings(face)[0]

    return encoding

def classify_face(im):
    faces_encoded = []
    faces = get_encoded_faces()
    faces_encode = list(faces.values())
    for face in range(len(faces_encode)):
        for f in faces_encode[face]:
            # print("faces: ")
            # print(f)
            faces_encoded.append(f)
    # print('faces encode : ')
    # print(faces_encoded)
    known_face_names = list(faces.keys())

    img = cv2.imread(im, 1)
 
    face_locations = face_recognition.face_locations(img)
    unknown_face_encodings = face_recognition.face_encodings(img, face_locations)

    face_names = []
    for face_encoding in unknown_face_encodings:
        matches = face_recognition.compare_faces(faces_encoded, face_encoding)
        name = "Unknown"
        print('Matches: ')
        print(matches)

        face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
        print('Distances: ')
        print(face_distances)
        best_match_index = np.argmin(face_distances)
        print("index: ")
        print(best_match_index)
        if matches[best_match_index]:
            best_match_index = best_match_index // 2
            name = known_face_names[best_match_index]

        face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw a box around the face
            cv2.rectangle(img, (left-25, top-25), (right+25, bottom+25), (255, 0, 0), 1)

            # Draw a label with a name below the face
            cv2.rectangle(img, (left-20, bottom -15), (right+20, bottom+20), (255, 0, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, name, (left -20, bottom + 15), font, 1.0, (255, 255, 255), 1)


    while True:
        cv2.imshow('Video', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return face_names 


@app.route('/recognize', methods=['POST'])
def recognize_face():
    try:
        # Print request headers and body for debugging
        # print("Request Headers:", request.headers)
        # print("Request Body:", request.get_data(as_text=True))

        # Get the image file from the request
        image_file = request.files.get('image')

        if image_file:
            print(image_file)
            # Save the image temporarily
            image_path = 'temp.jpg'
            image_file.save(image_path)

            # Display the created image using matplotlib for debugging
            image = cv2.imread(image_path)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            # plt.show()

            # print(request.files)

            print('test')

            # Call your face recognition function
            result = classify_face(image_path)
            print("RESULTAT: ",result)

            # Return the result as JSON
            return jsonify({'result': result})
        else:
            return jsonify({'error': 'No image file received'})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
