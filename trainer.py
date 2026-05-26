import cv2
import os
import numpy as np
from PIL import Image

# Create LBPH recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load face detector
detector = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

# Path to dataset
path = "dataset"

face_samples = []
ids = []

# Read all images from dataset
image_paths = [
    os.path.join(path, f)
    for f in os.listdir(path)
]

for image_path in image_paths:

    # Convert image to grayscale
    pil_image = Image.open(image_path).convert("L")

    image_numpy = np.array(
        pil_image,
        "uint8"
    )

    # Get ID from filename
    id = int(
        os.path.split(image_path)[-1].split(".")[1]
    )

    faces = detector.detectMultiScale(image_numpy)

    for (x, y, w, h) in faces:

        face_samples.append(
            image_numpy[y:y+h, x:x+w]
        )

        ids.append(id)

print("Training faces...")

# Train recognizer
recognizer.train(
    face_samples,
    np.array(ids)
)

# Save trained model
recognizer.write(
    "trainer/trainer.yml"
)

print("Model trained successfully")