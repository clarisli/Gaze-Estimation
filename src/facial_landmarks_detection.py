import sys
import logging as log
import cv2
import numpy as np
import time

from inference import Network

class FacialLandmarksDetector:

    def __init__(self, model_name, device='CPU', extensions=None):
        self.network = Network(model_name, device, extensions)

    def load_model(self):
        self.network.load_model()

    def predict(self, face_image):
        input_image, preprocess_input_time = self._preprocess_input(face_image)
        self.network.exec_net(0, input_image)
        status = self.network.wait(0)
        if status == 0:
            outputs = self.network.get_output(0)
            eye_boxes, eye_centers, preprocess_output_time = self._preprocess_output(outputs, face_image)
            self.preprocess_time = preprocess_input_time + preprocess_output_time
            return eye_boxes, eye_centers

    def _preprocess_input(self, image):
        start_preprocess_time = time.time()
        n, c, h, w = self.network.get_input_shape()
        input_image = cv2.resize(image, (w,h), interpolation = cv2.INTER_AREA)
        input_image = input_image.transpose((2, 0, 1))
        input_image = input_image.reshape((n, c, h, w))
        total_preprocess_time = time.time() - start_preprocess_time
        return input_image, total_preprocess_time


    def _preprocess_output(self, outputs, image):
        start_preprocess_time = time.time()
        normalized_landmarks = np.squeeze(outputs).reshape((5,2))
        h, w, _ = image.shape
        color = (255,255,255)
        length_offset = int(w * 0.15) 
        eye_boxes, eye_centers = [], []
        for i in range(2):
            normalized_x, normalized_y = normalized_landmarks[i]
            x = int(normalized_x*w)
            y = int(normalized_y*h)
            eye_centers.append([x, y])
            xmin, xmax = max(0, x - length_offset), min(w, x + length_offset)
            ymin, ymax = max(0, y - length_offset), min(h, y + length_offset)
            eye_boxes.append([xmin, ymin, xmax, ymax])
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1)
        total_preprocess_time = time.time() - start_preprocess_time
        return eye_boxes, eye_centers, total_preprocess_time

        