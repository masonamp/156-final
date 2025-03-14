# 156-final
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils.extmath import randomized_svd as rsvd
import keras
from sklearn.metrics import confusion_matrix
from keras import layers, Sequential, Input
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import data as tfd
from sklearn.utils import resample

label_path = "/Users/SavitaSahay/Downloads/Data_Entry_2017_v2020.csv" # path of csv file
image_path = "/Users/SavitaSahay/Downloads/chest-xrays" # path of folder comtaining images


EPOCHS = 25
BATCH_SIZE = 100
N_files = 5000
# N_files = len(dirlist)
IMAGE_SIZE = 256  #resize images
k = 100  
sample_size = 1000
sample_matrix = np.zeros((sample_size, IMAGE_SIZE**2))
df = pd.read_csv(label_path)  #path of data
image_files = df["Image Index"].tolist()
diagnosis_dict = dict(zip(df["Image Index"], df["Finding Labels"]))
labelnames = [
    "No Finding",
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]
n_labels = len(labelnames)



def svd(img, k):
    U, S, VT = rsvd(img, n_components = k)
    V_reduced = VT[:k, :].T
    # U_reduced = U[:, :k]
    # S_reduced = np.diag(S[:k])
    # Z = S_reduced @ V_reduced.T
    # A_0 = np.dot(U_reduced, Z)
    return V_reduced


dirlist = os.listdir(image_path)

compressed_images = np.empty((N_files, k))
labels = np.empty((N_files, n_labels))

i = 0
print("Processing first image")
for img_name in dirlist[:N_files]:
    img_path = os.path.join(image_path, img_name)
    #load and preprocess
    img = cv2.imread(img_path, 0)
    if img is not None:
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)).ravel() / 255.0 
        if i < sample_size:
            sample_matrix[i] = img
            
        else:
            if i == sample_size:
                print("calculating basis")
                basis = svd(sample_matrix - np.mean(sample_matrix, axis=0), k=k)
                print(basis.shape)
                compressed_images[:sample_size] = sample_matrix @ basis
            compressed_images[i] = img @ basis
            if i % sample_size == 0:
                print(f"{i} files processed out of {N_files}")
            # recovered_img = basis @ compressed_img
        diagnosis = diagnosis_dict.get(img_name).split('|')[0]
        labels[i] = [int(x in diagnosis) for x in labelnames]
        i += 1

keras.backend.set_image_data_format("channels_last")

# Define model
model1 = Sequential(
    [
        layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),
        layers.Conv2D(32, (3, 3), activation="relu", padding="valid"),
        layers.MaxPooling2D((2, 2), padding="valid"),
        layers.Dropout(0.3),  # dropout to reduce overfitting
        layers.Conv2D(32, (3, 3), activation="relu", padding="valid"),
        layers.MaxPooling2D((2, 2), padding="valid"),
        layers.Dropout(0.3),
        layers.Conv2D(32, (3, 3), activation="relu", padding="valid"),
        layers.MaxPooling2D((2, 2), padding="valid"),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(n_labels, activation="softmax"),
    ]
)
print(model1.summary())

# Split data into train, validation, and test sets
print("Basis: ", np.shape(basis))
print("Compressed images: ", np.shape(compressed_images))
X_train, X_temp, y_train, y_temp = train_test_split(
    np.reshape((compressed_images @ basis.T), (-1, IMAGE_SIZE, IMAGE_SIZE, 1)),
    labels,
    test_size=0.3,
)

# Resample minority conditions for balance
# X_resample = X_train.copy()
# y_resample = y_train.copy()


# for i in range(n_labels):
#     pos_ind = np.where(y_resample[:, i] == 1)[0]
#     num_pos = len(pos_ind)
#     neg_ind = np.where(y_resample[:, i] == 0)[0]
#     num_neg = len(neg_ind)
#     print(num_pos, num_neg)
#     if num_pos == 0 or num_neg == 0:
#         print("Skipping this condition: no matches found.")
#         continue
#     if num_pos < num_neg:
#         resample_ind = resample(pos_ind, n_samples = num_neg - num_pos)
#     else:
#         resample_ind = resample(neg_ind, n_samples = num_pos - num_neg)
#     X_train = np.vstack([X_train, X_resample[resample_ind]])
#     y_train = np.vstack([y_train, y_resample[resample_ind]])
#     print(i)


print(np.shape(X_train))
print(np.shape(y_train))
print(np.sum(y_train, axis=0))

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

train_ds = tfd.Dataset.from_tensor_slices((X_train, y_train)).batch(BATCH_SIZE)
val_ds = tfd.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE)
test_ds = tfd.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)

print("Made datasets")

# Custom callback for confusion matrix
class ConfusionMatrixCallback(keras.callbacks.Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        cm = np.zeros((n_labels, n_labels))
        for batch in self.validation_data:
            val_data, val_labels = batch
            predictions = self.model.predict(val_data)
            predicted_labels = np.argmax(predictions, axis=1)
            true_labels = np.argmax(val_labels, axis=1)
            cm += confusion_matrix(true_labels, predicted_labels, labels = range(n_labels))
        print(f"\nConfusion Matrix after epoch {epoch + 1}:\n{cm}\n")



# Compile and train
callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
    ConfusionMatrixCallback(val_ds),
    ]

model1.compile(
    optimizer=keras.optimizers.Adam(1e-4),  # reduced for inbalance
    loss=keras.losses.CategoricalCrossentropy(),
    # Log metrics (accuracy, precision, recall, f1)
    metrics=[
        keras.metrics.CategoricalAccuracy(name="acc"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.F1Score(name="f1"),
        # keras.metrics.TruePositives(name="tp"),
        # keras.metrics.FalsePositives(name="fp"),
        # keras.metrics.TrueNegatives(name="tn"),
        # keras.metrics.FalseNegatives(name="fn")
    ],
)

# Include label weights to prevent bias toward "No Finding"
label_counts = np.sum(y_train, axis=0)
weights = {
    i: (np.shape(y_train)[0] / label_count) for i, label_count in enumerate(label_counts)
}


model1.fit(
    train_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    validation_data=val_ds,
    class_weight=weights,
)

# Evaluate trained model on test set
model1.evaluate(test_ds)
