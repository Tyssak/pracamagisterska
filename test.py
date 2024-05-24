import cv2
import numpy as np
import dlib


detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")

# Video capture from webcam
cap = cv2.VideoCapture(0)

def resize_frame(frame, target_height, target_width=None):
    if target_width is None:
        aspect_ratio = 1 if frame.shape[0] == 0 else frame.shape[1] / frame.shape[0]

        target_width = int(target_height * aspect_ratio)

    try:
        resized_frame = cv2.resize(frame, (target_width, target_height))
        scale_width = target_width / frame.shape[1]
        scale_height = target_height / frame.shape[0]
    except Exception as e:
        print(f"Error resizing frame: {e}")

        return frame, 1, 1

    return resized_frame, scale_width, scale_height


def cnn_face_detection(frame):
    resized_frame, _, _ = resize_frame(frame, target_height=300, target_width=300)
    blob = cv2.dnn.blobFromImage(resized_frame, 1.0, (300, 300), (104, 177, 123))
    net.setInput(blob)
    detections = net.forward()

    return detections

def process_detection(detections, frame):
        """
        Returns the list of detected faces (one for now) with max confidence.

        Args:
            detections: Detections from the face detection model.
            frame: The input frame as a NumPy array.

        Returns:
            List: List of faces as tuples (startX, startY, width, height).
        """

        # Find the face with the highest confidence
        max_confidence_face = max(range(detections.shape[2]), key=lambda i: detections[0, 0, i, 2])

        # Get the confidence value of the highest-confidence detection
        confidence = detections[0, 0, max_confidence_face, 2]

        if confidence > 0.9:
            startX, startY, endX, endY = (detections[0, 0, max_confidence_face, 3:7] *
                                          np.array([frame.shape[1], frame.shape[0],
                                                    frame.shape[1], frame.shape[0]])).astype("int")
            face = (startX, startY, endX - startX, endY - startY)
            return face
        else:
            return None

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detections = cnn_face_detection(frame)

    face = process_detection(detections, frame)

    # for face in faces:
    #     # Detect landmarks (i.e., facial feature points)
    #     shape = predictor(gray, face)
    #     shape = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])
    #
    #     # Extract left and right eye landmarks
    #     left_eye = shape[36:42]
    #     right_eye = shape[42:48]
    #
    #     # Draw circles around the eyes
    #     for (x, y) in left_eye:
    #         cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
    #     for (x, y) in right_eye:
    #         cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    # Display the frame
    cv2.imshow("Frame", face)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
