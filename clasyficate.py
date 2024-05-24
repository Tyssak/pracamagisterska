import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from collections import Counter


class Clasyficator:
    def __init__(self, model_path):
        self.prev_recognitions = []
        self.emotion_model = load_model(model_path)
        #self.emotion_labels = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise'] #CK+
        #self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'] #FER2013
        self.emotion_labels = ['Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised'] #RAVDESS
        #self.emotion_labels = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']  # RAVDESS 8
        # self.emotion_labels = ['Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Surprised']  # RAVDESS 6

        # # Check the input shape using summary()
        # self.emotion_model.summary()
        #
        # # Alternatively, you can directly access the input shape attribute
        # input_shape = self.emotion_model.input_shape
        # print("Input shape:", input_shape)

    def get_most_common_index(self, predictions):
        #print(predictions)
        max_prev_recognitions = 5
        self.prev_recognitions.append(predictions)
        self.prev_recognitions = self.prev_recognitions[-max_prev_recognitions:]
        if self.prev_recognitions:
            predictions = np.mean(self.prev_recognitions, axis=0)
        return predictions
    def classify(self, frame):
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #
        # # Resize and normalize the grayscale face
        # gray = cv2.resize(gray, (48, 48))
        gray = cv2.resize(frame, (227, 227))
        gray = np.expand_dims(gray, axis=0)
        gray = gray / 255.0  # Normalize the image

        # Make predictions using the emotion recognition model
        predictions = self.emotion_model.predict(gray)
        #text = f'Emotion: {predicted_emotion}, Probability: {probability:.2f}%'
        predictions = self.get_most_common_index(predictions)
        emotion_index = np.argmax(predictions)
        predicted_emotion = self.emotion_labels[emotion_index]
        probability = predictions[0][emotion_index] * 100

        # if probability < 50.0:
        #     predicted_emotion = 'Neutral'

        x, y = frame.shape[0], frame.shape[1]
        # Display the predicted emotion and probability on the frame
        text = f'Emotion: {predicted_emotion}, Probability: {probability:.2f}%'
        print(text)
        #cv2.putText(frame, text, (y // 2, x // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return text
