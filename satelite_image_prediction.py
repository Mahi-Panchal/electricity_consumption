import streamlit as st
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
import tempfile
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import zipfile

# GPU check and config
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Optional: set memory growth to prevent OOM
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        st.sidebar.success("✅ GPU detected and configured.")
    except RuntimeError as e:
        st.sidebar.error(f"GPU config error: {e}")
else:
    st.sidebar.warning("⚠️ No GPU found. Training will run on CPU.")

# Set Streamlit page config
st.set_page_config(
    page_title="CNN Satellite Image Classifier (GPU)",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Class labels
class_names = ['Cloudy', 'Desert', 'Green_Area', 'Water']

# Session state init
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'dataset_loaded' not in st.session_state:
    st.session_state.dataset_loaded = False
if 'data' not in st.session_state:
    st.session_state.data = None

# CNN model creation
def create_cnn_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(255, 255, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Dataset loader
def load_dataset_from_zip(zip_file):
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            temp_dir = tempfile.mkdtemp()
            zip_ref.extractall(temp_dir)
            dataset_path = None
            for root, dirs, files in os.walk(temp_dir):
                if set(dirs) >= {'cloudy', 'desert', 'green_area', 'water'}:
                    dataset_path = root
                    break
            if not dataset_path:
                st.error("Expected folders: cloudy, desert, green_area, water")
                return None
            data = pd.DataFrame(columns=['image_path', 'label'])
            labels = {
                os.path.join(dataset_path, "cloudy"): "Cloudy",
                os.path.join(dataset_path, "desert"): "Desert",
                os.path.join(dataset_path, "green_area"): "Green_Area",
                os.path.join(dataset_path, "water"): "Water",
            }
            for folder, label in labels.items():
                for image_name in os.listdir(folder):
                    path = os.path.join(folder, image_name)
                    if os.path.isfile(path):
                        data = pd.concat([data, pd.DataFrame({'image_path': [path], 'label': [label]})])
            return data
    except Exception as e:
        st.error(f"Error loading zip: {e}")
        return None

# Image prediction
def predict_image(model, img_path):
    try:
        img = image.load_img(img_path, target_size=(255, 255))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array)[0]
        predicted_idx = np.argmax(pred)
        return class_names[predicted_idx], np.max(pred), pred
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None, None

# Confusion matrix plot
def plot_confusion_matrix(cm):
    return px.imshow(cm, 
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=class_names, y=class_names,
        text_auto=True, color_continuous_scale="Blues"
    ).update_layout(title="Confusion Matrix", width=600, height=500)

# Training history plots
def plot_training_history(history):
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Accuracy", "Loss"))
    fig.add_trace(go.Scatter(y=history.history['accuracy'], name='Train Acc'), row=1, col=1)
    fig.add_trace(go.Scatter(y=history.history['val_accuracy'], name='Val Acc'), row=1, col=1)
    fig.add_trace(go.Scatter(y=history.history['loss'], name='Train Loss'), row=1, col=2)
    fig.add_trace(go.Scatter(y=history.history['val_loss'], name='Val Loss'), row=1, col=2)
    return fig.update_layout(height=400)

# Main UI
def main():
    st.title("🛰️ CNN Satellite Image Classifier (GPU)")

    page = st.sidebar.radio("Select Page", 
        ["📁 Data Loading", "🔧 Model Training", "📊 Evaluation", "🔮 Predict"])

    if page == "📁 Data Loading":
        st.header("Upload Dataset (ZIP with class folders)")
        uploaded_file = st.file_uploader("Upload ZIP", type=["zip"])
        if uploaded_file:
            data = load_dataset_from_zip(uploaded_file)
            if data is not None:
                st.session_state.data = data
                st.session_state.dataset_loaded = True
                st.success(f"{len(data)} images loaded.")
                st.dataframe(data.head())
                fig = px.histogram(data, x="label", title="Class Distribution")
                st.plotly_chart(fig, use_container_width=True)

    elif page == "🔧 Model Training":
        if not st.session_state.dataset_loaded:
            st.warning("Upload dataset first.")
            return
        st.header("Model Training")
        epochs = st.slider("Epochs", 1, 50, 10)
        batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)
        test_size = st.slider("Test Size", 0.1, 0.4, 0.2)

        if st.button("Train Model"):
            train_df, test_df = train_test_split(st.session_state.data, test_size=test_size,
                                                 stratify=st.session_state.data['label'])
            train_gen = ImageDataGenerator(rescale=1./255, zoom_range=0.2, shear_range=0.2,
                                           horizontal_flip=True).flow_from_dataframe(
                dataframe=train_df, x_col="image_path", y_col="label",
                target_size=(255, 255), batch_size=batch_size, class_mode="categorical")
            test_gen = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
                dataframe=test_df, x_col="image_path", y_col="label",
                target_size=(255, 255), batch_size=batch_size, class_mode="categorical")

            model = create_cnn_model()
            with st.spinner("Training..."):
                history = model.fit(train_gen, validation_data=test_gen, epochs=epochs)
            st.session_state.model = model
            st.session_state.model_trained = True
            st.session_state.test_gen = test_gen
            st.session_state.history = history
            st.success("Model trained.")
            st.plotly_chart(plot_training_history(history), use_container_width=True)

    elif page == "📊 Evaluation":
        if not st.session_state.model_trained:
            st.warning("Train the model first.")
            return
        st.header("Evaluate Model")
        model = st.session_state.model
        test_gen = st.session_state.test_gen
        y_true = test_gen.classes
        y_pred = np.argmax(model.predict(test_gen), axis=1)
        st.metric("Accuracy", f"{np.mean(y_true == y_pred):.4f}")
        st.plotly_chart(plot_confusion_matrix(confusion_matrix(y_true, y_pred)), use_container_width=True)
        st.text("Classification Report:")
        st.text(classification_report(y_true, y_pred, target_names=class_names))

    elif page == "🔮 Predict":
        st.header("Upload Image for Prediction")
        img_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        if img_file:
            img = Image.open(img_file)
            st.image(img, caption="Uploaded Image", use_column_width=True)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                img.save(tmp.name)
                pred_class, confidence, prob = predict_image(st.session_state.model, tmp.name)
                if pred_class:
                    st.success(f"Prediction: {pred_class} ({confidence:.2%})")
                    fig = px.bar(x=class_names, y=prob, labels={'x': 'Class', 'y': 'Probability'})
                    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
