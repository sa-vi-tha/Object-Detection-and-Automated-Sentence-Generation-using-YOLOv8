#Question1
#perform detection of all objects in any given image.
#List the objects names detected.
#Generate a meaningful sentences Which includes all the objects names detected.

import cv2
from ultralytics import YOLO
from collections import Counter
import matplotlib.pyplot as plt

model = YOLO('yolov8n.pt') 

def det_obj(image_path):
    
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (640, 480))
    results = model(image_resized)
 
    object_names = [model.names[int(box.cls)] for box in results[0].boxes] #where keys or indices correspond to class IDs
                                                                           #and values are the human-readable object names 
    object_counts = Counter(object_names)  

    detected_objects = list(object_counts.keys())
    print("Detected Objects:", detected_objects)

    return detected_objects
    #return results
    
def generate_sentence(detected_objects):
    if detected_objects:
        sentence = "In this image, I can see " + ", ".join(detected_objects[:-1]) + " and " + detected_objects[-1] + "."
    return sentence

image_path = "Object2.jpg" 
detected_objects = det_obj(image_path)
sentence = generate_sentence(detected_objects)
print("Generated Sentence:", sentence)
