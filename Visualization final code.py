

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