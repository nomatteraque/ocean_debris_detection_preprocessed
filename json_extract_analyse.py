import json
import cv2
import glob
import numpy as np
from matplotlib import pyplot as plt
import glob
import time
from ultralytics import YOLO
from collections import defaultdict

# Load the JSON data
with open('C:/Users/user/Desktop/OceanDebris/pre_processed_imgs/type_1/instances_val_trashcan.json') as json_file:
    data = json.load(json_file)

with open('C:/Users/user/Desktop/OceanDebris/pre_processed_imgs/type_4/_annotations.coco.json') as json_file_fun:
    data_fun = json.load(json_file_fun)

# Creating a dictionary to store labels by image ID
labels_by_image_id = defaultdict(list)
labels_by_image_id_fun = defaultdict(list)

# Populate the dictionary with labels associated with each image ID
for annotation in data['annotations']:
    image_id = annotation['image_id']
    labels_by_image_id[image_id].append(annotation['category_id'])

for annotation in data_fun['annotations']:
    image_id = annotation['image_id']
    labels_by_image_id_fun[image_id].append(annotation['category_id'])

# Load in the models
model_og = YOLO('C:/Users/user/Desktop/OceanDebris/SegV1(RegDat)/Nano480/DebsegmentorV1nano480.pt')
model_clahe = YOLO('C:/Users/user/Desktop/OceanDebris/pre_processed_imgs/type_1_RES/model_clahe.pt')
model_ucm = YOLO('C:/Users/user/Desktop/OceanDebris/pre_processed_imgs/type_2_RES/model_ucm.pt')
model_ibla = YOLO('C:/Users/user/Desktop/OceanDebris/pre_processed_imgs/type_3_RES/model_IBLA.pt')
model_fun = YOLO('C:/Users/user/Desktop/OceanDebris/pre_processed_imgs/type_4_RES/model_funieGAN.pt')

# Keep accuracy score
score_og = 0
score_clahe = 0
score_ucm = 0
score_ibla = 0
score_fun = 0

# Keep precision
precision_og = 0
precision_clahe = 0
precision_ucm = 0
precision_ibla = 0
precision_fun = 0

# Keep recall
recall_og = 0
recall_clahe = 0
recall_ucm = 0
recall_ibla = 0
recall_fun = 0

# Save predictions in corresponding arrays
pred_og = []
pred_clahe = []
pred_ucm = []
pred_ibla = []
pred_fun = []

# Keep scores, recalls, precisions
scores = []
precisions = []
recalls = []

# Image filename to find labels for
for image_path in glob.glob(f'C:/Users/user/Desktop/OceanDebris/pre_processed_imgs/type_1/val/*.jpg'):
    # Read in the image and format it

    file_path_components = image_path.split('/')
    filename = file_path_components[-1].split('\\')
    filename = filename[-1]

    img = cv2.imread(image_path, 1)
    img = cv2.resize(img, (480, 480))

    # Find the image data by filename
    image_data = next((image for image in data['images'] if image['file_name'] == filename), None)

    labels = []
    if image_data:
        image_id = image_data['id']
        # Get labels associated with the image ID from the dictionary
        labels = labels_by_image_id.get(image_id)
        # print(labels)

    # model_ucm.predict(img, imgsz=480, conf=0.5, show=True)
    results = model_clahe(img)
    for result in results:
        pred_clahe.append(result.boxes.cls.numpy().tolist())

        if len(result.boxes.conf) == 0:
            if len(labels) == 0:
                score_clahe += 1
        elif any(int(item) in result.boxes.cls.numpy() for item in labels):
            # print(result.boxes.conf.numpy()[0])

            false_n = [elem for elem in labels if elem not in result.boxes.cls.numpy()]
            false_p = [elem for elem in result.boxes.cls.numpy() if elem not in labels]
            true_p = [elem for elem in result.boxes.cls.numpy() if elem in labels]

            precision_clahe += len(true_p) / (len(true_p) + len(false_p))
            recall_clahe += len(true_p) / (len(true_p) + len(false_n))

            score_clahe += 1
        # if len(result.boxes.conf) != 0:
            # print(result.boxes.cls.numpy())
        # print(score_clahe)

# print(len(pred_clahe))
score_clahe = score_clahe/1147
precision_clahe = precision_clahe/1147
recall_clahe = recall_clahe/1147
scores.append(score_clahe)
precisions.append(precision_clahe)
recalls.append(recall_clahe)
# print(pred_clahe)



# Image filename to find labels for
for image_path in glob.glob(f'C:/Users/user/Desktop/OceanDebris/pre_processed_imgs/type_2/val/*.jpg'):
    # Read in the image and format it

    file_path_components = image_path.split('/')
    filename = file_path_components[-1].split('\\')
    filename = filename[-1]

    img = cv2.imread(image_path, 1)
    img = cv2.resize(img, (480, 480))

    # Find the image data by filename
    image_data = next((image for image in data['images'] if image['file_name'] == filename), None)

    labels = []
    if image_data:
        image_id = image_data['id']
        # Get labels associated with the image ID from the dictionary
        labels = labels_by_image_id.get(image_id)
        # print(labels)

    # model_ucm.predict(img, imgsz=480, conf=0.5, show=True)
    results = model_ucm(img)
    for result in results:
        pred_ucm.append(result.boxes.cls.numpy().tolist())

        if len(result.boxes.conf) == 0:
            if len(labels) == 0:
                score_ucm += 1
        elif any(int(item) in result.boxes.cls.numpy() for item in labels):
            # print(result.boxes.conf.numpy()[0])

            false_n = [elem for elem in labels if elem not in result.boxes.cls.numpy()]
            false_p = [elem for elem in result.boxes.cls.numpy() if elem not in labels]
            true_p = [elem for elem in result.boxes.cls.numpy() if elem in labels]

            precision_ucm += len(true_p) / (len(true_p) + len(false_p))
            recall_ucm += len(true_p) / (len(true_p) + len(false_n))

            score_ucm += 1
        # if len(result.boxes.conf) != 0:
            # print(result.boxes.cls.numpy())
        # print(score_ucm)

score_ucm = score_ucm/1147
precision_ucm = precision_ucm/1147
recall_ucm = recall_ucm/1147
scores.append(score_ucm)
precisions.append(precision_ucm)
recalls.append(recall_ucm)
# print(pred_ucm)


# Image filename to find labels for
for image_path in glob.glob(f'C:/Users/user/Desktop/OceanDebris/pre_processed_imgs/type_3/val/*.jpg'):
    # Read in the image and format it

    file_path_components = image_path.split('/')
    filename = file_path_components[-1].split('\\')
    filename = filename[-1]

    img = cv2.imread(image_path, 1)
    img = cv2.resize(img, (480, 480))

    # Find the image data by filename
    image_data = next((image for image in data['images'] if image['file_name'] == filename), None)

    labels = []
    if image_data:
        image_id = image_data['id']
        # Get labels associated with the image ID from the dictionary
        labels = labels_by_image_id.get(image_id)
        # print(labels)

    # model_ucm.predict(img, imgsz=480, conf=0.5, show=True)
    results = model_ibla(img)
    for result in results:
        pred_ibla.append(result.boxes.cls.numpy().tolist())

        if len(result.boxes.conf) == 0:
            if len(labels) == 0:
                score_ibla += 1
        elif any(int(item) in result.boxes.cls.numpy() for item in labels):
            # print(result.boxes.conf.numpy()[0])

            false_n = [elem for elem in labels if elem not in result.boxes.cls.numpy()]
            false_p = [elem for elem in result.boxes.cls.numpy() if elem not in labels]
            true_p = [elem for elem in result.boxes.cls.numpy() if elem in labels]

            precision_ibla += len(true_p) / (len(true_p) + len(false_p))
            recall_ibla += len(true_p) / (len(true_p) + len(false_n))

            score_ibla += 1
        # if len(result.boxes.conf) != 0:
            # print(result.boxes.cls.numpy())
        # print(score_ibla)

# print(len(pred_ibla))
score_ibla = score_ibla/1147
precision_ibla = precision_ibla/1147
recall_ibla = recall_ibla/1147
scores.append(score_ibla)
precisions.append(precision_ibla)
recalls.append(recall_ibla)
# print(pred_ibla)

# Image filename to find labels for
for image_path in glob.glob(f'C:/Users/user/Desktop/OceanDebris/dataset/instance_version/val/*.jpg'):
    # Read in the image and format it

    file_path_components = image_path.split('/')
    filename = file_path_components[-1].split('\\')
    filename = filename[-1]

    img = cv2.imread(image_path, 1)
    img = cv2.resize(img, (480, 480))

    # Find the image data by filename
    image_data = next((image for image in data['images'] if image['file_name'] == filename), None)

    labels = []
    if image_data:
        image_id = image_data['id']
        # Get labels associated with the image ID from the dictionary
        labels = labels_by_image_id.get(image_id)
        # print(labels)

    # model_ucm.predict(img, imgsz=480, conf=0.5, show=True)
    results = model_og(img)
    for result in results:
        pred_og.append(result.boxes.cls.numpy().tolist())

        if len(result.boxes.conf) == 0:
            if len(labels) == 0:
                score_og += 1
        elif any(int(item) in result.boxes.cls.numpy() for item in labels):
            # print(result.boxes.conf.numpy())

            false_n = [elem for elem in labels if elem not in result.boxes.cls.numpy()]
            false_p = [elem for elem in result.boxes.cls.numpy() if elem not in labels]
            true_p = [elem for elem in result.boxes.cls.numpy() if elem in labels]

            precision_og += len(true_p) / (len(true_p) + len(false_p))
            recall_og += len(true_p) / (len(true_p) + len(false_n))

            score_og += 1
        # if len(result.boxes.conf) != 0:
            # print(result.boxes.cls.numpy())
        # print(score_og)


score_og = score_og/1147
precision_og = precision_og/1147
recall_og = recall_og/1147
scores.append(score_og)
precisions.append(precision_og)
recalls.append(recall_og)



for image_path in glob.glob(f'C:/Users/user/Desktop/OceanDebris/pre_processed_imgs/type_2/val/*.jpg'):
    # Read in the image and format it

    file_path_components = image_path.split('/')
    filename = file_path_components[-1].split('\\')
    filename = filename[-1]

    img = cv2.imread(image_path, 1)
    img = cv2.resize(img, (480, 480))

    # Find the image data by filename
    image_data = next((image for image in data_fun['images'] if image['file_name'] == filename), None)

    labels = []
    if image_data:
        image_id = image_data['id']
        # Get labels associated with the image ID from the dictionary
        labels = labels_by_image_id_fun.get(image_id)
        # print(labels)

    # model_ucm.predict(img, imgsz=480, conf=0.5, show=True)
    results = model_fun(img)
    for result in results:
        pred_fun.append(result.boxes.cls.numpy().tolist())

        if len(result.boxes.conf) == 0:
            if len(labels) == 0:
                score_fun += 1
        elif any(int(item) in result.boxes.cls.numpy() for item in labels):
            # print(result.boxes.conf.numpy()[0])

            false_n = [elem for elem in labels if elem not in result.boxes.cls.numpy()]
            false_p = [elem for elem in result.boxes.cls.numpy() if elem not in labels]
            true_p = [elem for elem in result.boxes.cls.numpy() if elem in labels]

            precision_fun += len(true_p) / (len(true_p) + len(false_p))
            recall_fun += len(true_p) / (len(true_p) + len(false_n))

            score_fun += 1
        # if len(result.boxes.conf) != 0:
            # print(result.boxes.cls.numpy())
        # print(score_ucm)

score_fun = score_fun/1147
precision_fun = precision_fun/1147
recall_fun = recall_fun/1147
scores.append(score_fun)
precisions.append(precision_fun)
recalls.append(recall_fun)
# print(pred_fun)

pred_orcla = []
pred_combined = []
for i in range(1147):
    pred_combined.append(pred_og[i] + pred_clahe[i] + pred_ucm[i] + pred_ibla[i] + pred_fun[i])

for i in range(1147):
    pred_orcla.append(pred_og[i] + pred_clahe[i])

print(pred_combined)

score_combined = 0
precision_combined = 0
recall_combined = 0
idx = 0

score_orcla = 0
precision_orcla = 0
recall_orcla = 0
# Image filename to find labels for
for image_path in glob.glob(f'C:/Users/user/Desktop/OceanDebris/pre_processed_imgs/type_1/val/*.jpg'):
    # Read in the image and format it

    file_path_components = image_path.split('/')
    filename = file_path_components[-1].split('\\')
    filename = filename[-1]

    # Find the image data by filename
    image_data = next((image for image in data['images'] if image['file_name'] == filename), None)

    labels = []
    if image_data:
        image_id = image_data['id']
        # Get labels associated with the image ID from the dictionary
        labels = labels_by_image_id.get(image_id)
        # print(labels)

    # Find matches in combined predictions
    if len(pred_combined[idx]) == 0:
        if len(labels) == 0:
            score_combined += 1
    elif any(int(item) in pred_combined[idx] for item in labels):

        false_n = [elem for elem in labels if elem not in pred_combined[idx]]
        false_p = [elem for elem in pred_combined[idx] if elem not in labels]
        true_p = [elem for elem in pred_combined[idx] if elem in labels]

        precision_combined += len(true_p) / (len(true_p) + len(false_p))
        recall_combined += len(true_p) / (len(true_p) + len(false_n))
        score_combined += 1

    idx += 1

idx = 0
for image_path in glob.glob(f'C:/Users/user/Desktop/OceanDebris/pre_processed_imgs/type_1/val/*.jpg'):
    # Read in the image and format it

    file_path_components = image_path.split('/')
    filename = file_path_components[-1].split('\\')
    filename = filename[-1]

    # Find the image data by filename
    image_data = next((image for image in data['images'] if image['file_name'] == filename), None)

    labels = []
    if image_data:
        image_id = image_data['id']
        # Get labels associated with the image ID from the dictionary
        labels = labels_by_image_id.get(image_id)
        # print(labels)

    # Find matches in combined predictions
    if len(pred_orcla[idx]) == 0:
        if len(labels) == 0:
            score_orcla += 1
    elif any(int(item) in pred_orcla[idx] for item in labels):

        false_n = [elem for elem in labels if elem not in pred_orcla[idx]]
        false_p = [elem for elem in pred_orcla[idx] if elem not in labels]
        true_p = [elem for elem in pred_orcla[idx] if elem in labels]

        precision_orcla += len(true_p) / (len(true_p) + len(false_p))
        recall_orcla += len(true_p) / (len(true_p) + len(false_n))
        score_orcla += 1

    idx += 1

score_combined = score_combined/1147
precision_combined = precision_combined/1147
recall_combined = recall_combined/1147
scores.append(score_combined)
precisions.append(precision_combined)
recalls.append(recall_combined)

score_orcla = score_orcla/1147
precision_orcla = precision_orcla/1147
recall_orcla = recall_orcla/1147
scores.append(score_orcla)
precisions.append(precision_orcla)
recalls.append(recall_orcla)

print(scores)
print(precisions)
print(recalls)


"""        
    results = model_clahe(img)
    for result in results:
        if len(result.boxes.conf) == 0:
            if len(labels) == 0:
                score_og += 1
        elif any(int(item) in result.boxes.cls.numpy() for item in labels):
            # print(result.boxes.conf.numpy()[0])

            score_og += 1
        if len(result.boxes.conf) != 0:
            print(result.boxes.cls.numpy())
        print(score_og)
"""


