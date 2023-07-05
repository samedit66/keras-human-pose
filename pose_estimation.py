# -*- coding: utf-8 -*-
from pprint import pprint as pp

import cv2
import matplotlib.pyplot as plt

from config_reader import config_reader
from pose_estimation_model import get_pose_estimation_model
from processing import get_coordinates


model = get_pose_estimation_model('model.h5')

test_image = 'test.jpg'
input_image = cv2.imread(test_image) # B,G,R order
plt.imshow(input_image)

params, model_params = config_reader()
coordinates = get_coordinates(input_image, params, model, model_params)

pp(coordinates)