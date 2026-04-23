import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
import pickle
from pathlib import Path
from mlm_model import build_causal_mlm

BASE_DIR = Path(__file__).resolve().parent

def train():
    print("Loading generative synthetic data...")
    with open(BASE_DIR / "generative_data.json", "r") as f:
        queries = json.load(f)

    # Text Vectorization
    MAX_LEN = 48
    VOCAB_SIZE = 10000
    
    import re
    def custom_standardization(input_data):
        lowercase = tf.strings.lower(input_data)
        return tf.strings.regex_replace(lowercase, r"[^a-z0-9\s\[\]/:]", "")

    vectorizer = TextVectorization(
        max_tokens=VOCAB_SIZE, 
        output_sequence_length=MAX_LEN + 1,
        standardize=custom_standardization
    )
    vectorizer.adapt(queries)
    
    vocab = vectorizer.get_vocabulary()
    with open(BASE_DIR / "mlm_vocab.pkl", "wb") as f:
        pickle.dump({"vocab": vocab}, f)

    bot_token_id = vocab.index("[bot]")
    print(f"Vocabulary size: {len(vocab)} (bot_token_id: {bot_token_id})")

    # Vectorize all sequences
    seqs = vectorizer(queries).numpy()
    
    # Next-Token Prediction setup
    X = seqs[:, :-1]
    y = seqs[:, 1:]

    # Create Loss Mask (Sample Weights)
    # We only want to train the model to predict tokens AFTER the [bot] tag.
    # Predicting the user's query or the [bot] tag itself often leads to prompt-looping.
    weights = np.zeros(y.shape)
    for i in range(len(seqs)):
        # Find where [bot] occurs in X
        indices = np.where(X[i] == bot_token_id)[0]
        if len(indices) > 0:
            bot_idx = indices[0]
            # Set weights to 1 for all tokens after the [bot] tag
            weights[i, bot_idx:] = 1.0

    # Build and Compile Model
    print("Building Generative Decoder-Only Causal MLM...")
    model = build_causal_mlm(
        vocab_size=len(vocab),
        max_len=MAX_LEN
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        weighted_metrics=["accuracy"]
    )

    print("Training Causal MLM with Loss Masking (Focusing on Responses)...")
    model.fit(
        X, y,
        sample_weight=weights,
        epochs=8, # Increased epochs since we are masking half the sequence
        batch_size=64,
        validation_split=0.1
    )

    # Save weights
    model.save_weights(str(BASE_DIR / "mlm_weights.weights.h5"))
    print("Training complete! Saved weights to mlm_weights.weights.h5")

if __name__ == "__main__":
    train()
