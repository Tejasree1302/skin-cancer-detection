import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.optimizers import Adam

# ==========================
# Paths
# ==========================
train_dir = "dataset/train"
val_dir = "dataset/validation"
model_path = "model/model_resnet.h5"

# ==========================
# Image preprocessing & augmentation
# ==========================
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224,224),
    batch_size=16,
    class_mode='categorical'
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224,224),
    batch_size=16,
    class_mode='categorical'
)

# ==========================
# Load ResNet50 base model
# ==========================
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

# ==========================
# Add custom classification layers
# ==========================
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(5, activation='softmax')(x)  # 5 classes

model = Model(inputs=base_model.input, outputs=predictions)

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ==========================
# Train the model
# ==========================
history = model.fit(
    train_gen,
    epochs=30,
    validation_data=val_gen
)

# ==========================
# Save the trained model
# ==========================
if not os.path.exists("model"):
    os.makedirs("model")

model.save(model_path)
print("Model saved at:", model_path)
# -----------------------------------------------
# 📊 Evaluate Model and Generate Confusion Matrix
# -----------------------------------------------
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import numpy as np

# Evaluate model on validation data
val_loss, val_acc = model.evaluate(val_gen)
print(f"\n✅ Validation Accuracy: {val_acc*100:.2f}%")
print(f"✅ Validation Loss: {val_loss:.4f}")

# Predict classes
Y_pred = model.predict(val_gen)
y_pred = np.argmax(Y_pred, axis=1)
y_true = val_gen.classes

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
class_labels = list(val_gen.class_indices.keys())

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - ResNet50")
plt.show()

# Classification Report
print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))

# Plot Accuracy
plt.figure(figsize=(8,4))
plt.plot(model.history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(model.history.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# Plot Loss
plt.figure(figsize=(8,4))
plt.plot(model.history.history['loss'], label='Train Loss', marker='o')
plt.plot(model.history.history['val_loss'], label='Validation Loss', marker='o')
plt.title("Training vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
