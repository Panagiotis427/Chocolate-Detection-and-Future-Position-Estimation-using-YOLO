"""Module to detect objects on an image."""
import os
from ultralytics import YOLO
import cv2
import yaml

# Set current working directory and load control parameters
cwd = os.getcwd()
with open(os.path.join(cwd, 'control_panel.yaml'),'r', encoding="utf-8") as file:
    panel = yaml.safe_load(file)

# Load the YOLOv8 model
model = YOLO(os.path.join(os.path.dirname(cwd), panel["MODEL_PATH"]))

# Predict objects
results = model.predict(os.path.join(os.path.dirname(cwd), panel["IMAGE_PATH"]),
                        conf=panel["CONFIDENCE_THRESHOLD"], iou=panel["IOU_THRESHOLD"],
                        verbose=False)
ACCURACY = 0

# Show results
while True:
    for pointer, result in enumerate(results):

        # Print the objects detected in the image
        for box in result.boxes:
            conf = round(box.conf[0].item(), 2)*100
            ACCURACY += conf
            if conf:
                class_id = result.names[box.cls[0].item()]
                cords = box.xyxy[0].tolist()
                cords = [round(x) for x in cords]
                print("Probability: " + str(conf)[:-2] + "%")
                print("Object type:", class_id)
                print("Coordinates:", cords)
                print("-------------------------------------------------\n")

        print(f"\nDetected: {len(result.boxes)}.\nAverage accuracy: "
              f"{ACCURACY/len(result.boxes):.2f}%.")
        result_nparray = result.plot()
        aspect_ratio = len(result_nparray[0])/len(result_nparray)
        if aspect_ratio > 1:
            result_nparray = cv2.resize(result_nparray, (int(720*aspect_ratio), 720))
        else:
            result_nparray = cv2.resize(result_nparray, (int(720*aspect_ratio), 720))

        cv2.imshow('Object detection', result_nparray)
        cv2.waitKey(0)

        key = cv2.waitKey(1)
        # Break the loop if 'q' or 'Q' or Esc character is pressed
        if key in [ord("q"), ord("Q"), ord(";"), 27]:
            break
    else:
        break
