from flask import Flask, jsonify, request
import os
import face_recognition as fr
import face_recognition
import numpy as np
import json


root_path = 'faces'
dir_names = os.listdir(root_path)

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

encoded_faces = get_encoded_faces()

data_json_serializable = {}

for key, value in encoded_faces.items():
    data_json_serializable[key] = []
    for arr in value:
        data_json_serializable[key].append(arr.tolist())
with open('Train.json', 'w') as json_file:
    json.dump(data_json_serializable, json_file)