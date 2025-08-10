# app.py
import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Page config (must be before other Streamlit calls)
st.set_page_config(page_title="Caption Generator App", page_icon="📷")

st.title("Image Caption Generator")
st.markdown("Upload an image, and this app will generate a caption for it using a trained LSTM model.")

# --- CACHED LOADING FUNCTIONS ---
@st.cache_resource
def load_feature_extractor():
    """
    Load MobileNetV2 with top removed and global average pooling.
    Returns a model that outputs a feature vector for an input image.
    """
    base = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
    # base.summary()  # debug if needed
    return base

@st.cache_resource
def load_caption_model(path="mymodel.h5"):
    """
    Load the trained captioning model (your LSTM model).
    """
    return tf.keras.models.load_model(path)

@st.cache_resource
def load_tokenizer(path="tokenizer.pkl"):
    """
    Load tokenizer and build index->word mapping for quick lookup.
    """
    with open(path, "rb") as f:
        tok = pickle.load(f)
    # Try to use tokenizer.index_word if present (keras tokenizer), otherwise build it from word_index
    if hasattr(tok, "index_word") and isinstance(tok.index_word, dict) and tok.index_word:
        index_to_word = tok.index_word
    else:
        index_to_word = {idx: word for word, idx in getattr(tok, "word_index", {}).items()}
    return tok, index_to_word

# Load resources
try:
    feature_extractor = load_feature_extractor()
    caption_model = load_caption_model("mymodel.h5")
    tokenizer, index_to_word = load_tokenizer("tokenizer.pkl")
except Exception as e:
    st.error(f"Error loading models/tokenizer: {e}")
    st.stop()

# --- UTILITIES ---
def preprocess_image_for_model(uploaded_file, target_size=(224, 224)):
    """
    Read an uploaded file (BytesIO) and return preprocessed numpy array ready for MobileNetV2.
    """
    img = load_img(uploaded_file, target_size=target_size)
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)  # shape (1, H, W, C)
    arr = preprocess_input(arr)
    return arr

def sequence_to_word(index, index_to_word_map):
    """
    Safely convert an index to a word. Returns None if not found.
    """
    return index_to_word_map.get(index, None)

def predict_caption(model, image_features, tokenizer, index_to_word_map, max_length=34):
    """
    Greedy decoding: generate caption token-by-token using argmax.
    """
    in_text = "startseq"
    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_length)
        # model expects [image_features, sequence] — adjust if your model expects other ordering
        yhat = model.predict([image_features, seq], verbose=0)
        # if model outputs probabilities across vocab shape (1, vocab_size)
        predicted_index = np.argmax(yhat, axis=-1)[0]
        word = sequence_to_word(predicted_index, index_to_word_map)
        if word is None:
            # stop if index not found
            break
        in_text += " " + word
        if word == "endseq":
            break
    # remove special tokens if present
    output = in_text.replace("startseq", "").replace("endseq", "").strip()
    return output

# Upload image
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    st.subheader("Uploaded Image")
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Generating caption..."):
        try:
            # Preprocess and extract features
            image_arr = preprocess_image_for_model(uploaded_image, target_size=(224, 224))
            features = feature_extractor.predict(image_arr, verbose=0)  # shape (1, D)

            # If your LSTM model expects features in a different shape, adapt here.
            # Example: if your model was trained with shape (1, 1, D), reshape:
            # features_for_model = features.reshape((features.shape[0], 1, features.shape[1]))
            features_for_model = features

            # Determine max_caption_length:
            # If you saved it elsewhere, use that; otherwise use a safe default (e.g., 34).
            max_caption_length = 34

            generated_caption = predict_caption(
                caption_model,
                features_for_model,
                tokenizer,
                index_to_word,
                max_length=max_caption_length,
            )

            if not generated_caption:
                generated_caption = "(couldn't generate a caption — check model/tokenizer compatibility)"

        except Exception as e:
            st.error(f"Error while generating caption: {e}")
            generated_caption = None

    if generated_caption:
        st.subheader("Generated Caption")
        st.markdown(
            f'<div style="border-left: 6px solid #ccc; padding: 5px 20px; margin-top: 20px;">'
            f'<p style="font-style: italic;">“{generated_caption}”</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
