import os
import sys
import logging as log
import math

from openvino.inference_engine import IENetwork, IECore
import numpy as np
import cv2
import time

from inference import Network

class HeadPoseEstimator:
    '''
    Class for the Head Pose Estimation Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        self.network = Network(model_name, device, extensions)

    def load_model(self):
        self.network.load_model()

    def predict(self, image):
        input_image, preprocess_input_time = self._preprocess_input(image)
        self.network.exec_net(0, input_image)
        status = self.network.wait(0)
        if status == 0:
            outputs = self.network.get_outputs(0)
            head_pose_angles, preprocess_output_time = self._preprocess_output(outputs, image)
            self.preprocess_time = preprocess_input_time + preprocess_output_time
            return head_pose_angles

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
        yaw = outputs['angle_y_fc'][0][0]
        pitch = outputs['angle_p_fc'][0][0]
        roll = outputs['angle_r_fc'][0][0]
        total_preprocess_time = time.time() - start_preprocess_time
        return [yaw, pitch, roll], total_preprocess_time
