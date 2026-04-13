import os
import cv2
import numpy as np

def load_dataset(path, img_size=224):

    images = []
    ages = []

    for file in os.listdir(path):

        try:
            age = int(file.split('_')[0])

            img_path = os.path.join(path,file)
            img = cv2.imread(img_path)

            if img is None:
                continue

            img = cv2.resize(img,(img_size,img_size))
            img = img / 255.0

            images.append(img)
            ages.append(age)

        except:
            continue

    X = np.array(images)

    # normalize age
    y = np.array(ages) / 100.0

    return X,y