import cv2
import glob
import numpy as np
from matplotlib import pyplot as plt
import glob
import time

start = time.time()

for image_path in glob.glob(f'C:/Users/user/Desktop/OceanDebris/dataset/instance_version/val/*.jpg'):
    # Read in the image and format it
    file_path_components = image_path.split('/')
    filename = file_path_components[-1].split('\\')
    filename = filename[-1]
    print(filename)
    img = cv2.imread(image_path, 1)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]

    # Create and apply clahe, then reformat
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    v = clahe.apply(v)
    hsv_img = np.dstack((h,s,v))
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

    # cv2.imshow('example_image', img)
    # cv2.waitKey(0)

    cv2.imwrite('C:/Users/user/Desktop/OceanDebris/pre_processed_imgs/type_1/val/{}'.format(filename), img)


end = time.time()

print(end-start)
