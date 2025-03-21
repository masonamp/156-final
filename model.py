# 156-final
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils.extmath import randomized_svd as rsvd
import keras
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras import layers, Sequential, Input
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import data as tfd
from sklearn.utils import resample

label_path = "/Users/SavitaSahay/Downloads/Data_Entry_2017_v2020.csv" # path of csv file
image_path = "/Users/SavitaSahay/Downloads/chest-xrays" # path of folder comtaining images


EPOCHS = 10
BATCH_SIZE = 32
N_files = 10000
# N_files = len(dirlist)
IMAGE_SIZE = 256  #resize images
k = 100  
sample_size = 2000
sample_matrix = np.zeros((sample_size, IMAGE_SIZE**2))
df = pd.read_csv(label_path)  #path of data
image_files = df["Image Index"].tolist()
diagnosis_dict = dict(zip(df["Image Index"], df["Finding Labels"]))
labelnames = [
    "No Finding",
    # "Atelectasis",
    # "Cardiomegaly",
    # "Effusion",
    # "Infiltration",
    # "Mass",
    # "Nodule",
    # "Pneumonia",
    # "Pneumothorax",
    # "Consolidation",
    # "Edema",
    # "Emphysema",
    # "Fibrosis",
    # "Pleural_Thickening",
    # "Hernia",
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
        layers.Dropout(0.4),  # dropout to reduce overfitting
        layers.Conv2D(32, (3, 3), activation="relu", padding="valid"),
        layers.MaxPooling2D((2, 2), padding="valid"),
        layers.Dropout(0.4),
        layers.Conv2D(32, (3, 3), activation="relu", padding="valid"),
        layers.MaxPooling2D((2, 2), padding="valid"),
        layers.Dropout(0.4),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(n_labels, activation="sigmoid"),
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
        cm = np.zeros((2, 2))
        for batch in self.validation_data:
            val_data, val_labels = batch
            predictions = self.model.predict(val_data)
            predicted_labels = [int(x[0] > 0.5) for x in predictions]
            true_labels = val_labels
            cm += confusion_matrix(true_labels, predicted_labels, labels = [0,1])
        print(f"\nConfusion Matrix after epoch {epoch + 1}:\n{cm}\n")



# Compile and train
callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
    ConfusionMatrixCallback(val_ds),
    ]

model1.compile(
    optimizer=keras.optimizers.Adam(1e-4),  # reduced for inbalance
    loss=keras.losses.BinaryCrossentropy(),
    # Log metrics (accuracy, precision, recall, f1)
    metrics=[
        keras.metrics.BinaryAccuracy(name="acc"),
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


history = model1.fit(
    train_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    validation_data=val_ds,
    # class_weight=weights,
)

# Evaluate trained model on test set
print(y_test)
model1.evaluate(test_ds)

def plot_history(history):
    # Plot loss
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Plot Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['acc'], label='Train Accuracy')
    plt.plot(history.history['val_acc'], label='Val Accuracy')
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    # Plot Precision
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['precision'], label='Train Precision')
    plt.plot(history.history['val_precision'], label='Val Precision')
    plt.title("Training and Validation Precision")
    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.legend()
    plt.show()

    # Plot Recall
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['recall'], label='Train Recall')
    plt.plot(history.history['val_recall'], label='Val Recall')
    plt.title("Training and Validation Recall")
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.legend()
    plt.show()


plot_history(history)

#-----------------------------------------------------------------------
from sklearn.metrics import confusion_matrix

# Get predictions on the test set
y_pred_prob = model1.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)  # thresholding at 0.5
y_true = y_test

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ["No Finding", "Condition"], rotation=45)
plt.yticks(tick_marks, ["No Finding", "Condition"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

#----------------------------------------------------------------------
import random
num_samples = 6  # number of images to display
indices = random.sample(range(X_test.shape[0]), num_samples)
sample_images = X_test[indices]
sample_true = y_test[indices]
sample_pred_prob = model1.predict(sample_images)
sample_pred = (sample_pred_prob > 0.5).astype(int)

plt.figure(figsize=(15, 8))
for i in range(num_samples):
    plt.subplot(2, 3, i+1)
    plt.imshow(sample_images[i].reshape(IMAGE_SIZE, IMAGE_SIZE), cmap='gray')
    plt.title(f"True: {sample_true[i][0]}, Pred: {sample_pred[i][0]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

#-------------------------------------------------------------------------
# Count samples per label
label_counts = np.sum(labels, axis=0)
plt.figure(figsize=(6, 4))
plt.bar(labelnames, label_counts)
plt.title("Label Distribution")
plt.xlabel("Labels")
plt.ylabel("Count")
plt.show()