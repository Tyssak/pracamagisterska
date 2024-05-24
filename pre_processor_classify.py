import os
import random
import string
import sys
import time
import cv2
import dlib
import numpy as np
from filters import FiltersOption, Filters
from clasyficate import Clasyficator
import concurrent.futures


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

def crop_around_face(frame, face, scale=1.0):
    (x, y, w, h) = face

    center = (x + w // 2, y + h // 2)

    crop_x, crop_y = max(0, int(center[0] - w * scale / 2)), \
            max(0, int(center[1] - h * scale / 2))
    crop_width, crop_height = min(frame.shape[1], int(w * scale)), \
            min(frame.shape[0], int(h * scale))

    if crop_width > 0 and crop_height > 0:
        # Crop the frame around the face
        cropped_frame = frame[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]
    else:
        cropped_frame = frame

    return cropped_frame


def rotate_frame(frame, rot, center):
    #rot = rot % 180

    # Get the frame dimensions
    height, width = frame.shape[:2]

    # Define the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, rot, 1.0)
    # Perform the rotation
    rotated_frame = cv2.warpAffine(frame, rotation_matrix, (width, height))


    return rotated_frame, rotation_matrix

def apply_filter(frame, my_filter, selected_filter_option):

    filter_options_mapping = {
        FiltersOption.NO_FILTER: lambda my_frame: my_frame,
        FiltersOption.MEDIAN_FILTER: lambda my_frame: my_filter.median_filter(my_frame, kernel_size=3),
        FiltersOption.GAUSSIAN_FILTER: lambda my_frame: my_filter.gaussian_filter(my_frame, sigma=1.0,
                                                                                  kernel_size=3),
        FiltersOption.BILATERAL_FILTER: lambda my_frame: my_filter.bilateral_filter(my_frame, diameter=9,
                                                                                    sigma_color=75,
                                                                                    sigma_space=75),
        FiltersOption.SOBEL_EDGE_DETECTOR: lambda my_frame: my_filter.sobel_filter(my_frame),
        FiltersOption.CANNY_EDGE_DETECTOR: lambda my_frame: my_filter.canny_detector(my_frame),
        FiltersOption.LAPLACIAN_OF_GAUSSIAN: lambda my_frame: my_filter.laplacian_of_gaussian(my_frame,
                                                                                              kernel_size=3,
                                                                                              sigma=1.0),
        FiltersOption.CLAHE: lambda my_frame: my_filter.apply_clahe(my_frame),
    }

    output_frame = filter_options_mapping.get(selected_filter_option, lambda my_frame: my_frame)(
        frame)

    return output_frame


def put_label_on_frame(frame, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    color = (255, 255, 255)  # White color in BGR
    thickness = 2

    if ", " in text:
        emotion, probability = text.split(", ")
        # Calculate text position
        text_x = (frame.shape[1]) // 20
        text_x2 = frame.shape[1] // 2 + frame.shape[1] // 20
        text_y = (frame.shape[0]) // 20

        # Put the text on the frame
        cv2.putText(frame, emotion, (text_x, text_y), font, font_scale, color, thickness)
        cv2.putText(frame, probability, (text_x2, text_y), font, font_scale, color, thickness)

class PreProcessor:
    def __init__(self):
        self.first = True
        self.visualisation = True
        self.frame_substraction = False
        self.filter_option = 0
        self.active_mode = 0
        self.CAMERA_MODE, self.VIDEO_MODE, self.PHOTO_MODE = 0, 1, 2

        self.desired_width = 227
        self.desired_height = 227

        self.net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")

        self.tracker = None
        self.prev_face = None

        self.error_threshold = 20

        # load landmark prediction model
        #self.predictor_path = "shape_predictor_68_face_landmarks.dat"
        self.predictor_path = "shape_predictor_5_face_landmarks.dat"
        self.predictor = dlib.shape_predictor(self.predictor_path)


        self.scaleFactor, self.minNeighbors, self.minSize = 1.01, 5, [40, 40]

        self.prev_rotations = []

        self.prev_frame1 = None
        self.prev_frame2 = None
        self.prev_frames = []

        self.detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")

        self.clasyficator = None

        self.dataset_praparation = False

        self.text = ""


    def run_face_detection(self, selected_option, selected_filter_option, file_path,
                           model_path = 'model_eq.h5', frame_substraction = False):
        self.filter_option = selected_filter_option
        self.frame_substraction = frame_substraction

        if selected_option == "Kamera":
            self.active_mode = self.CAMERA_MODE
        elif selected_option == "Nagranie wideo":
            self.active_mode = self.VIDEO_MODE
        elif selected_option == "Zdjęcia":
            self.active_mode = self.PHOTO_MODE
        else:
            print("Unsupported mode")
            sys.exit()

        cap = self.start_capture(file_path)

        # Set the desired frame rate (e.g., 60 fps)
        #desired_fps = 30
        # fps = cap.get(cv2.CAP_PROP_FPS)
        # print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
        #frame_interval = 1.0 / desired_fps

        self.clasyficator = Clasyficator(model_path)

        my_filter = Filters()

        first = True
        with concurrent.futures.ThreadPoolExecutor() as executor:
            while True:
                start_time = time.time()

                if self.active_mode != self.PHOTO_MODE:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        print("End of video or error in reading frame.")
                        break
                else:
                    frame, _, _ = resize_frame(cap, self.desired_height)

                if first or detect_thread.done():
                    first = False
                    detect_thread = executor.submit(self.pre_process, frame, my_filter)


                # Calculate the time elapsed for processing the current frame
                elapsed_time = time.time() - start_time

                # Calculate the time to wait before processing the next frame
                #remaining_wait_time = max(0, frame_interval - elapsed_time)

                # Wait for the specified time before processing the next frame
                #time.sleep(remaining_wait_time)

                if frame.shape[0] > 0 and frame.shape[1] > 0:
                    frame, _, _ = resize_frame(frame, 384)
                    put_label_on_frame(frame, self.text)
                    cv2.imshow('Detected Faces', frame)

                if (cv2.waitKey(1) & 0xFF == ord('q')) or self.active_mode == self.PHOTO_MODE:
                    break


        cv2.waitKey(0)
        if selected_option != "Zdjęcia":
            cap.release()

        cv2.destroyAllWindows()

    def pre_process(self, frame, my_filter):
        frame_interval = 0.17
        start_time = time.time()
        just_face, frame_face = self.extract_face(frame)
        if just_face is not None and frame_face is not None:
            angle, left_eye, right_eye, nose = self.detect_features_from_5_dlib(just_face)
            angle = self.average_angle(angle)
            left_eye, right_eye, nose, center = self.change_coordinates(just_face, frame_face, left_eye, right_eye,
                                                                        nose)

            if self.first:
                if self.filter_option == FiltersOption.CANNY_EDGE_DETECTOR:
                    my_filter.init_canny(frame, sigma=0.33)

            self.first = False

            # Normalize the size and angle
            filered_frame = apply_filter(frame_face, my_filter, selected_filter_option=self.filter_option)
            output_frame, rotaion_matrix = rotate_frame(filered_frame, angle, center)
            output_frame = self.apply_mask(output_frame, left_eye, right_eye, nose, center, rotaion_matrix)

            if self.active_mode != self.PHOTO_MODE:
                output_frame = self.three_frames_into_three_channels(output_frame)
            self.text = self.clasyficator.classify(output_frame)

        # Calculate the time elapsed for processing the current frame
        elapsed_time = time.time() - start_time

        # Calculate the time to wait before processing the next frame
        remaining_wait_time = max(0, frame_interval - elapsed_time)

        # Wait for the specified time before processing the next frame
        time.sleep(remaining_wait_time)
        return 1

    def start_capture(self, file_path):
        """
        Starts the capture based on the selected mode.

        Args:
            file_path: Path to the input source.

        Returns:
            cv2.VideoCapture: Video capture object.
        """

        if self.active_mode == self.CAMERA_MODE:
            return cv2.VideoCapture(0)
        elif self.active_mode == self.VIDEO_MODE:
            return cv2.VideoCapture(file_path)
        elif self.active_mode == self.PHOTO_MODE:
            frame = cv2.imread(file_path)
            return cv2.resize(frame, None, fx=10, fy=10)
        else:
            print("Unsupported mode")
            sys.exit()

    def extract_face(self, frame):

        if self.first:
            face = self.detect_face(frame)
        else:
            face = self.track_face(frame)
        #face = self.detect_face(frame)

        if face is not None:
            just_face = crop_around_face(frame, face, scale=1.0)
            frame_face = crop_around_face(frame, face, scale=1.5)

            return just_face, frame_face
        else:
            return None, None

    def detect_face(self, frame):
        """
        Performs face detection and initialization on the first frame.

        Args:
            frame: The input frame as a NumPy array.

        Returns:
            Tuple: Cropped frame around the detected face and the new center position.
        """

        if self.active_mode != self.PHOTO_MODE:
            detections = self.cnn_face_detection(frame)

            face = self.process_detection(detections, frame)

        else:
            #face = self.haar_detection(frame)
            face = self.dlib_face_detection(frame)

        if face is not None:
            if self.active_mode != self.PHOTO_MODE:
                self.start_face_tracking(face, frame)
                self.prev_face = face
        else:
            if self.visualisation:
                print("Nie wykryto twarzy")
            else:
                pass

        return face

    def cnn_face_detection(self, frame):
        resized_frame, _, _ = resize_frame(frame, target_height=300, target_width=300)
        blob = cv2.dnn.blobFromImage(resized_frame, 1.0, (300, 300), (104, 177, 123))
        self.net.setInput(blob)
        detections = self.net.forward()

        return detections

    def haar_detection(self, gray):
        """
        Performs face detection using the Haar cascade classifier (for gray-scaled photos).

        Args:
            gray: Grayscale version of the input frame.

        Returns:
            List: List of faces as tuples (x, y, width, height).
        """
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=self.scaleFactor,
                                               minNeighbors=self.minNeighbors, minSize=self.minSize)

        if len(faces) > 0:  # Check if there are any detected faces
            biggest_face = max(faces, key=lambda rect: rect[2] * rect[3])  # Use rect[2] * rect[3] for area

            return biggest_face
        else:
            return None

    def dlib_face_detection(self, frame):
        """
        Performs face detection using the dlib face detector (for gray-scaled photos).

        Args:
            gray: Grayscale version of the input frame.

        Returns:
            List: List of faces as tuples (x, y, width, height).
        """
        faces = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets = self.detector(gray, 1)
        for det in dets:
            x, y, width, height = det.rect.left(), det.rect.top(), det.rect.width(), det.rect.height()
            faces.append((x, y, width, height))

        if faces:  # Check if there are any detected faces
            biggest_face = max(faces, key=lambda rect: rect[2] * rect[3])  # Use rect[2] * rect[3] for area

            return biggest_face
        else:
            return None


    def process_detection(self, detections, frame):
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

    def start_face_tracking(self, face, gray):
        """
        Initializes the dlib face tracker.

        Args:
            face: List of faces as tuples (startX, startY, width, height).
            gray: Grayscale version of the input frame.

        Returns:
            None
        """
        self.tracker = dlib.correlation_tracker()
        (x, y, w, h) = face
        rect = dlib.rectangle(x, y, x + w, y + h)
        self.tracker.start_track(gray, rect)


    def track_face(self, frame):
        self.tracker.update(frame)
        pos = self.tracker.get_position()

        (tracked_startX, tracked_startY, tracked_endX, tracked_endY) = \
                map(int, [pos.left(), pos.top(), pos.right(), pos.bottom()])

        (x, y, w, h) = self.prev_face
        is_error_small = self.check_tracking_error(tracked_startX, tracked_startY, np.array([x, y]))

        if not is_error_small:
            face = self.detect_face(frame)
        else:
            face = (tracked_startX, tracked_startY, tracked_endX - tracked_startX, tracked_endY - tracked_startY)

        return face


    def check_tracking_error(self, tracked_startX, tracked_startY, prev_points):

        curr_points = np.array([tracked_startX, tracked_startY])
        error = np.linalg.norm(curr_points - prev_points)

        return error < self.error_threshold

    def detect_features_from_68_dlib(self, frame):
        # Use Dlib to find facial landmarks
        frame_height, frame_width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #gray = cv2.GaussianBlur(gray, (3, 3), 1.0)
        shape = self.predictor(gray, dlib.rectangle(0, 0, frame_width, frame_height))

        # Extract eye landmarks (assuming a typical 68-face landmarks model)
        left_eye = shape.parts()[36:42]
        right_eye = shape.parts()[42:48]

        left_eye_points, right_eye_points = np.array([(point.x, point.y) for point in left_eye]), np.array(
            [(point.x, point.y) for point in right_eye])
        left_eye_avg, right_eye_avg = np.mean(left_eye_points, axis=0).astype(int), np.mean(right_eye_points,
                                                                                            axis=0).astype(int)

        # cv2.circle(frame, (left_eye_avg[0], left_eye_avg[1]), 4, (255, 0, 0), -1)
        # cv2.circle(frame, (right_eye_avg[0], right_eye_avg[1]), 4, (255, 0, 0), -1)

        angle_rad = np.arctan2(right_eye_avg[1] - left_eye_avg[1],
                               right_eye_avg[0] - left_eye_avg[0])

        angle_deg = np.degrees(angle_rad)

        return frame, angle_deg

    def detect_features_from_5_dlib(self, frame):
        """
        Detects eye landmarks using Dlib's shape predictor with 5 face landmarks.

        Args:
            frame: The input frame as a NumPy array.

        Returns:
            Tuple: A tuple containing the frame with drawn landmarks and the calculated angle in degrees.
        """
        # Use Dlib to find facial landmarks
        frame_height, frame_width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        shape = self.predictor(gray, dlib.rectangle(0, 0, frame_width, frame_height))

        # Extract eye landmarks (using 5 face landmarks)
        left_eye = shape.parts()[2:4]  # Adjust index for left eye
        right_eye = shape.parts()[0:2]  # Adjust index for right eye

        nose = np.array([shape.part(4).x, shape.part(4).y]).astype(int)

        # Calculate average positions of left and right eyes
        left_eye_points, right_eye_points = np.array([(point.x, point.y) for point in left_eye]), np.array(
            [(point.x, point.y) for point in right_eye])
        left_eye_avg, right_eye_avg = np.mean(left_eye_points, axis=0), np.mean(right_eye_points,
                                                                                            axis=0).astype(int)

        # Draw circles at the positions of the eyes
        # cv2.circle(frame, (left_eye_avg[0], left_eye_avg[1]), 4, (255, 0, 0), -1)
        # cv2.circle(frame, (right_eye_avg[0], right_eye_avg[1]), 4, (255, 0, 0), -1)

        # Calculate the angle between the eyes
        angle_rad = np.arctan2(right_eye_avg[1] - left_eye_avg[1], right_eye_avg[0] - left_eye_avg[0])
        angle_deg = np.degrees(angle_rad)


        return angle_deg, left_eye_avg, right_eye_avg, nose

    def apply_mask(self, frame, left_eye, right_eye, nose, center, rotation_matrix):
        scale_x = 1.2
        scale_y = 2.0

        new_left_eye_pos = np.dot(rotation_matrix, np.array([left_eye[0], left_eye[1], 1]))
        new_right_eye_pos = np.dot(rotation_matrix, np.array([right_eye[0], right_eye[1], 1]))

        new_nose_pos = np.dot(rotation_matrix, np.array([nose[0], nose[1], 1]))

        eyes_center = np.dot(rotation_matrix, np.array([center[0], center[1], 1]))

        distance_between_eyes = np.sqrt(
            (new_right_eye_pos[0] - new_left_eye_pos[0]) ** 2 + (new_right_eye_pos[1] - new_left_eye_pos[1]) ** 2)
        distance_to_nose =  np.sqrt(
            (eyes_center[0] - new_nose_pos[0]) ** 2 + (eyes_center[1] - new_nose_pos[1]) ** 2)

        new_center = int(eyes_center[0] + new_nose_pos[0]) // 2, int(eyes_center[1] + new_nose_pos[1]) // 2

        width = int(scale_x * distance_between_eyes)
        height = int(scale_y * distance_to_nose)

        mask11 = np.zeros_like(frame)
        mask_eyes = cv2.ellipse(mask11, new_center, (width, height), 0,
                                0, 360,
                                (255, 255, 255), -1)

        masked_frame = cv2.bitwise_and(frame, mask_eyes)

        crop_height = height * 2
        crop_width = width * 2

        # Crop the frame
        crop_x = max(0, new_center[0] - width)
        crop_y = max(0, new_center[1] - height)
        cropped_frame = masked_frame[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]
        #cropped_frame = frame
        #cropped_frame = frame[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]
        resized_frame, scale_width, scale_height = resize_frame(cropped_frame, target_height=self.desired_height,
                                                                target_width=self.desired_width)

        return resized_frame

    def change_coordinates(self, just_face, frame_face, left_eye, right_eye, nose):
        cord_start = ((frame_face.shape[1] - just_face.shape[1]) // 2, (frame_face.shape[0] - just_face.shape[0]) // 2)
        left_eye = (cord_start[0] + left_eye[0], cord_start[1] + left_eye[1])
        right_eye = (cord_start[0] + right_eye[0], cord_start[1] + right_eye[1])
        nose = (cord_start[0] + nose[0], cord_start[1] + nose[1])

        center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

        return left_eye, right_eye, nose, center

    def average_angle(self, angle):
        if self.active_mode != self.PHOTO_MODE:
            max_prev_frames = 3
            self.prev_rotations = self.prev_rotations[-max_prev_frames:]
            if self.prev_rotations:
                angle = int((angle + sum(self.prev_rotations)) / (len(self.prev_rotations) + 1))
            self.prev_rotations.append(angle)

        return angle

    def show_frame(self, output_frame):
        if output_frame.shape[0] > 0 and output_frame.shape[1] > 0:
            frame, _, _ = resize_frame(output_frame, 384)
            cv2.imshow('Detected Faces', frame)

    def three_frames_into_three_channels(self, output_frame):
        combined_frame = output_frame
        if len(output_frame.shape) == 3 and output_frame.shape[2] == 3:
            output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2GRAY)
        frame, _, _ = resize_frame(output_frame, self.desired_height, self.desired_width)
        if len(self.prev_frames) > 1:
            combined_frame = np.stack((frame, self.prev_frames[0], self.prev_frames[1]), axis=-1)
        max_prev_frames = 2
        self.prev_frames.append(frame)
        self.prev_frames = self.prev_frames[-max_prev_frames:]

        return combined_frame
