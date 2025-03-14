from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import os
import csv
import sqlite3
import keras
from keras import layers, Sequential, Input
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import data as tfd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

EPOCHS = 25
BATCH_SIZE = 32
N = 1000


# Load images
def load_png_to_numpy(image_path):
    try:
        img = (
            Image.open(image_path).convert("L").resize((256, 256))
        )  # set to greyscale and reduce size
        numpy_array = np.array(img) / 255.0  # normalize
        return numpy_array
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


labelnames = [
    "No Finding",  # when no disease detected
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
labeldict = {labelnames[i]: i for i in range(15)}

# Assumes file is in ./data
tablepath = "data/Data_Entry_2017_v2020.csv"


images = np.zeros((N, 256, 256))

dirname = "data/images"
filename_list = sorted(os.listdir(dirname)[:N])
for i, filename in enumerate(filename_list):
    if filename.endswith(".png"):
        image_array = load_png_to_numpy(f"{dirname}/{filename}")
        images[i] = image_array


with open(tablepath, "r") as fin:
    dr = csv.DictReader(fin)
    # When multiple diseases are present, they are separated by '|'
    to_db = np.array(
        [
            [i["Image Index"]]
            + [
                1 if name in i["Finding Labels"].split("|") else 0
                for name in labelnames
            ]
            for i in dr
        ]
    )

with sqlite3.connect("data/xrays.db") as conn:

    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS data")
    cursor.execute(
        f"CREATE TABLE IF NOT EXISTS data (filename, {', '.join([f'[{name}]' for name in labelnames])})"
    )

    cursor.executemany(
        f"INSERT INTO data (filename, {', '.join([f'[{name}]' for name in labelnames])}) VALUES ({', '.join(16*['?'])})",
        to_db,
    )

    cursor.execute(
        f"SELECT {', '.join([f'[{name}]' for name in labelnames])} FROM data WHERE filename IN ({', '.join(['?']*len(filename_list))}) ORDER BY filename",
        filename_list,
    )
    labels = np.array(cursor.fetchall(), dtype=np.float32)
    conn.commit()

keras.backend.set_image_data_format("channels_last")

# Define model
model1 = Sequential(
    [
        layers.Input(shape=(256, 256, 1)),
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
        layers.Dense(15, activation="sigmoid"),
    ]
)
print(model1.summary())

# Split data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(
    np.expand_dims(images, axis=-1),
    labels[:N],
    test_size=0.3,
)

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

train_ds = tfd.Dataset.from_tensor_slices((X_train, y_train)).batch(BATCH_SIZE)
val_ds = tfd.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE)
test_ds = tfd.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)

# Compile and train
callbacks = [keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras")]

model1.compile(
    optimizer=keras.optimizers.Adam(1e-4),  # reduced for inbalance
    loss=keras.losses.BinaryFocalCrossentropy(
        alpha=0.5, gamma=2.0
    ),  # may help with label imbalance
    # Log metrics (accuracy, precision, recall, f1)
    metrics=[
        keras.metrics.BinaryAccuracy(name="acc"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.AUC(name="auc"),
    ],
)

# Include label weights to prevent bias toward "No Finding"
label_counts = np.sum(labels, axis=0)
weights = {
    i: (labels.shape[0] / label_count) for i, label_count in enumerate(label_counts)
}


history = model1.fit(
    train_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    validation_data=val_ds,
    # class_weight=weights,
)

# Evaluate trained model on test set
model1.evaluate(test_ds)

# Plot Loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.show()

# Plot Accuracy 
plt.figure(figsize=(10, 5))
plt.plot(history.history['acc'], label='Training Accuracy')
plt.plot(history.history['val_acc'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.show()

# Get predictions on test set
y_pred = model1.predict(test_ds)
y_pred_binary = (y_pred > 0.5).astype(int)  # thresholding predictions

# Flatten test labels and predictions
y_true = y_test

cm = confusion_matrix(y_true.argmax(axis=1), y_pred_binary.argmax(axis=1))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()


# 1. Get predictions for the validation data
y_pred_probs = model1.predict(val_ds)
y_pred = (y_pred_probs > 0.5).astype(int)  # Convert probabilities to binary predictions

# 2. Convert true labels from the validation dataset into an array
y_true = np.concatenate([y for x, y in val_ds], axis=0)

# 3. Count the number of true and predicted samples for each class
true_counts = np.sum(y_true, axis=0)
pred_counts = np.sum(y_pred, axis=0)

# 4. Plot the distributions side by side
x = np.arange(len(labelnames))  # Label locations
width = 0.35  # Width of the bars

plt.figure(figsize=(12, 6))
plt.bar(x - width/2, true_counts, width, label='True Count')
plt.bar(x + width/2, pred_counts, width, label='Predicted Count')

plt.xlabel('Classes')
plt.ylabel('Count')
plt.title('True vs Predicted Label Distribution (Validation Set)')
plt.xticks(ticks=x, labels=labelnames, rotation=45)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()