from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import os
import csv
import sqlite3
import keras
from keras import layers, Sequential, Input
from sklearn.model_selection import train_test_split
from tensorflow import data as tfd


def load_png_to_numpy(image_path):
    try:
        img = Image.open(image_path)
        numpy_array = np.array(img)
        return numpy_array
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

labelnames = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
              'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
              'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
              'Pleural_Thickening','Hernia']
labeldict = {labelnames[i] : i for i in range(14)}

tablepath = "/Users/SavitaSahay/Downloads/Data_Entry_2017_v2020.csv"

N = 150

images = np.zeros((N, 1024, 1024))
    
dirname = "/Users/SavitaSahay/Downloads/images"
filename_list = sorted(os.listdir(dirname)[:N])
for (i, filename) in enumerate(filename_list):
    if filename.endswith('.png'):
        image_array = load_png_to_numpy(f'{dirname}/{filename}')
        if image_array.shape[-1] == 4:
            image_array = image_array[:,:,0]
        images[i] = image_array
        

with open(tablepath,'r') as fin: 
    dr = csv.DictReader(fin)
    to_db = [[[i['Image Index']] + [1 if name in i['Finding Labels'] else 0 for name in labelnames]] for i in dr]
    to_db = np.array(to_db)[:,0,:]

with sqlite3.connect('/Users/SavitaSahay/Downloads/xrays.db') as conn:

    cursor = conn.cursor()
    cursor.execute("DROP TABLE data")
    cursor.execute(f"CREATE TABLE IF NOT EXISTS data (filename, {', '.join(labelnames)})")

    cursor.executemany(
        f"INSERT INTO data (filename, {', '.join(labelnames)}) VALUES ({', '.join(15*['?'])})",
        to_db
    )
    
    cursor.execute(f"SELECT {', '.join(labelnames)} FROM data WHERE filename IN ({', '.join(['?']*len(filename_list))}) ORDER BY filename", filename_list)
    labels = cursor.fetchall()
    conn.commit()

keras.backend.set_image_data_format('channels_last')

model1 = Sequential([
    layers.Input(shape=(1024, 1024, 1)),
    layers.Conv2D(32, (3, 3), activation="relu", padding="valid"),
    layers.MaxPooling2D((2, 2), padding="valid"),
    layers.Conv2D(32, (3, 3), activation="relu", padding="valid"),
    layers.MaxPooling2D((2, 2), padding="valid"),
    layers.Conv2D(32, (3, 3), activation="relu", padding="valid"),
    layers.MaxPooling2D((2, 2), padding="valid"),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(14, activation="sigmoid")
])
print(model1.summary())

X_train, X_val, y_train, y_val = train_test_split(
    np.expand_dims(images, axis = -1), 
    [[int(a[0]) for a in b] for b in labels],
    test_size = 0.2)

train_ds = tfd.Dataset.from_tensor_slices((X_train, y_train))
val_ds = tfd.Dataset.from_tensor_slices((X_val, y_val))

print(X_train.shape)
print(np.array(y_train).shape)
print(y_train[0])


epochs = 25

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras")
]
model1.compile(
    optimizer=keras.optimizers.Adam(3e-4),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy(name="acc")],
)
model1.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds,
)