
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

SRC = r"C:\Users\sneha sidhwani\Downloads\Real vs Fake Faces\Real vs Fake Faces\6040"

# MobileNetV2 preprocessing
datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

train_data = datagen.flow_from_directory(
    SRC + r"\train",
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    classes=["fake", "real"],
    shuffle=True
)

val_data = datagen.flow_from_directory(
    SRC + r"\valid",
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    classes=["fake", "real"],
    shuffle=False
)

base = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base.trainable = False

model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_data,
    validation_data=val_data,
    epochs=3
)

model.save("backend/fake_real_model_v2.keras")
print("NEW MODEL SAVED SUCCESSFULLY!")
print("Class mapping:", train_data.class_indices)
