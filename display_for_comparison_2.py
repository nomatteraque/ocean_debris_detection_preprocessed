import cv2
import glob

# Change image source directory accordingly
for image_path in glob.glob(f'C:/Users/user/Desktop/OceanDebris/dataset/instance_version/val/*.jpg'):
    img = cv2.imread(image_path, 1)

    cv2.imshow('2nd_method', img)

    cv2.waitKey(0)