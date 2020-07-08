
import os
import sys
import logging as log
import cv2
import time

from inference import Network

class FaceDetector:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None, threshold=0.60):
        self.threshold = threshold
        self.network = Network(model_name, device, extensions)

    def load_model(self):
        self.network.load_model()

    def predict(self, image):
        input_image, preprocess_input_time = self._preprocess_input(image)
        self.network.exec_net(0, input_image)
        status = self.network.wait(0)
        if status == 0:
            outputs = self.network.get_output(0)
            face_boxes, preprocess_output_time = self._preprocess_output(outputs, image)
            self.preprocess_time = preprocess_input_time + preprocess_output_time
            return face_boxes

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
        face_boxes = []
        h, w, _ = image.shape
        color = (255,0,0)
        for obj in outputs[0][0]:
            if obj[2] > self.threshold:
                xmin = int(obj[3] * w)
                ymin = int(obj[4] * h)
                xmax = int(obj[5] * w)
                ymax = int(obj[6] * h)
                face_boxes.append([xmin, ymin, xmax, ymax])
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1)
        total_preprocess_time = time.time() - start_preprocess_time
        return face_boxes, total_preprocess_time
