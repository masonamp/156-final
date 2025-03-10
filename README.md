# 156-final
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


df = pd.read_csv("/content/Data_Entry_2017_v2020.csv")  #path of data
image_files = df["Image Index"].tolist()
diagnosis_dict = dict(zip(df["Image Index"], df["Finding Labels"]))

IMAGE_SIZE = (224, 224)  #resize images
k = 15  

def svd(img, k=15):
    U, S, VT = np.linalg.svd(img)
    V_reduced = VT[:k, :].T
    U_reduced = U[:, :k]
    S_reduced = np.diag(S[:k])
    Z = S_reduced @ V_reduced.T
    A_0 = np.dot(U_reduced, Z)
    return A_0  

dataset = []  #name / new compressed image / diagnosis

for img_name in image_files:
    img_path = os.path.join(df, img_name)

    #load and preprocess
    img = cv2.imread(img_path, 0)
    img = cv2.resize(img, IMAGE_SIZE)
    img = img / 255.0  

    compressed_img = svd(img, k=k)
    diagnosis = diagnosis_dict.get(img_name)
    
    dataset.append((img_name, compressed_img, diagnosis))