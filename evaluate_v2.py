
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

SRC = r"C:\Users\sneha sidhwani\Downloads\Real vs Fake Faces\Real vs Fake Faces\6040"

model = tf.keras.models.load_model(
    "backend/fake_real_model_v2.keras",
    compile=False
)

datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

test_data = datagen.flow_from_directory(
    SRC + r"\test",
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    classes=["fake", "real"],
    shuffle=False
)

probabilities = model.predict(test_data, verbose=1).ravel()
predictions = (probabilities >= 0.5).astype(int)
actual = test_data.classes

print("\nTest accuracy:", round(accuracy_score(actual, predictions) * 100, 2), "%")
print("\nConfusion matrix:\n", confusion_matrix(actual, predictions))
print("\nClassification report:\n",
      classification_report(
          actual, predictions,
          labels=[0, 1],
          target_names=["fake", "real"],
          zero_division=0
      ))
