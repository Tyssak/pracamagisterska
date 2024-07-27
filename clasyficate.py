import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np


class Clasyficator:
    def __init__(self, model_path):
        """
        Initialize the Clasyficator class.

        Parameters:
        - model_path: str path to the pre-trained emotion recognition model
        """
        self.prev_recognitions = []
        self.emotion_model = load_model(model_path)
        #self.emotion_labels = ['Zlosc', 'Pogarda', 'Wstret', 'Zlosc', 'Szczescie', 'Smutek', 'Zaskoczenie'] # CK+
        #self.emotion_labels = ['Zlosc', 'Wstret', 'Zlosc', 'Szczescie', 'Spokoj', 'Smutek', 'Zaskoczenie'] # FER2013
        self.emotion_labels = ['Spokoj', 'Szczescie', 'Smutek', 'Zlosc', 'Strach', 'Wstret', 'Zaskoczenie']  # RAVDESS

    def get_most_common_index(self, predictions):
        """
         Compute the most common index from the previous recognitions.

         Parameters:
         - predictions: array, current predictions from the model

         Returns:
         - predictions: array, averaged predictions considering the previous recognitions
         """
        max_prev_recognitions = 5
        self.prev_recognitions.append(predictions)
        self.prev_recognitions = self.prev_recognitions[-max_prev_recognitions:]
        if self.prev_recognitions:
            predictions = np.mean(self.prev_recognitions, axis=0)
        return predictions

    def classify(self, frame):
        """
        Classify the emotion in the given frame.

        Parameters:
        - frame: array, the input image/frame to classify

        Returns:
        - text: str, the predicted emotion and its probability
        """
        gray = cv2.resize(frame, (227, 227))
        gray = np.expand_dims(gray, axis=0)
        gray = gray / 255.0
        predictions = self.emotion_model.predict(gray)
        predictions = self.get_most_common_index(predictions)
        emotion_index = np.argmax(predictions)
        predicted_emotion = self.emotion_labels[emotion_index]
        probability = predictions[0][emotion_index] * 100
        text = f'emocja: {predicted_emotion}, p.qq {probability:.2f}%'
        print(text)

        return text
