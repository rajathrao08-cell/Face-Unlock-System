import cv2
import os

# Create recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load trained model
recognizer.read("trainer/trainer.yml")

# Load face detector
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

# Open webcam (Mac compatible)
camera = cv2.VideoCapture(
    0,
    cv2.CAP_AVFOUNDATION
)

# Give camera time to initialize
cv2.waitKey(1000)

# Check camera access
if not camera.isOpened():
    print("Unable to access camera")
    exit()

print("Face Unlock System Started")

while True:

    # Read webcam frame
    ret, frame = camera.read()

    # Retry if frame not received
    if not ret or frame is None:
        print("Waiting for camera...")
        continue

    # Convert to grayscale
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5
    )

    # Loop through detected faces
    for (x, y, w, h) in faces:

        # Predict face
        id, confidence = recognizer.predict(
            gray[y:y+h, x:x+w]
        )

        # Draw rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Recognized face
        if confidence < 60:

            # Access granted text
            cv2.putText(
                frame,
                "ACCESS GRANTED",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # Confidence value
            cv2.putText(
                frame,
                f"Confidence: {round(confidence, 2)}",
                (x, y + h + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            print("Face Recognized")

            # Show frame
            cv2.imshow(
                "Face Unlock System",
                frame
            )

            # Wait before unlocking
            cv2.waitKey(2000)

            # Open protected folder
            os.system("open protected_folder")

            # Release resources
            camera.release()
            cv2.destroyAllWindows()

            exit()

        else:

            # Unknown face text
            cv2.putText(
                frame,
                "UNKNOWN FACE",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

            # Confidence value
            cv2.putText(
                frame,
                f"Confidence: {round(confidence, 2)}",
                (x, y + h + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

    # Show webcam
    cv2.imshow(
        "Face Unlock System",
        frame
    )

    # ESC to exit
    if cv2.waitKey(1) == 27:
        break

# Release camera
camera.release()

# Close windows
cv2.destroyAllWindows()