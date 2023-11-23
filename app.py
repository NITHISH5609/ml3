import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load MobileNetV2 model
mobilenet_model = MobileNetV2(weights="imagenet")
mobilenet_model = Model(inputs=mobilenet_model.inputs, outputs=mobilenet_model.layers[-2].output)

# Load your trained model
model = tf.keras.models.load_model('mymodel.h5')

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# Set custom web page title
st.set_page_config(page_title="Caption Generator App", page_icon="📷")

# Streamlit app
st.title("Image Caption Generator")
st.markdown(
    "Upload an image, and this app will generate a caption for it using a trained LSTM model."
)

# Upload image
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# Process uploaded images in batch
if uploaded_image is not None:
    st.subheader("Uploaded Image")
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    st.subheader("Generated Caption")
    # Display loading spinner while processing
    with st.spinner("Generating caption..."):
        # Load and preprocess image
        image = load_img(uploaded_image, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)

        # Extract features using MobileNetV2
        image_features = mobilenet_model.predict(image, verbose=0)

        # Repeat the image features for batch processing
        batch_size = 5  # Adjust batch size based on system capabilities
        image_features = np.repeat(image_features, batch_size, axis=0)

        # Repeat the image for batch processing
        images = np.repeat(image, batch_size, axis=0)

        # Generate captions using the model
        captions = ["startseq" for _ in range(batch_size)]
        for _ in range(max_caption_length):
            sequence = tokenizer.texts_to_sequences(captions)
            sequence = pad_sequences(sequence, maxlen=max_caption_length)
            yhat = model.predict([image_features, sequence], verbose=0)
            predicted_indices = np.argmax(yhat, axis=2)
            captions = [captions[i] + " " + get_word_from_index(predicted_indices[i][-1], tokenizer) for i in range(batch_size)]
            if all(predicted_word is None or predicted_word == "endseq" for predicted_word in predicted_words):
                break

        # Extract the final generated caption for each image
        generated_captions = [caption.replace("startseq", "").replace("endseq", "") for caption in captions]

    # Display the generated captions with custom styling
    for i, generated_caption in enumerate(generated_captions):
        st.markdown(
            f'<div style="border-left: 6px solid #ccc; padding: 5px 20px; margin-top: 20px;">'
            f'<p style="font-style: italic;">“{generated_caption}”</p>'
            f'</div>',
            unsafe_allow_html=True
        )
