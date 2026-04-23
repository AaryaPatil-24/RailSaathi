import tensorflow as tf
import pickle
import numpy as np
from pathlib import Path

# Paths
BASE_DIR = Path("/Users/karan/Desktop/Desktop - Computer/Projects/railsaathi/RailSaathi")
VOCAB_PATH = BASE_DIR / "mlm_vocab.pkl"

# Load Vocab
with open(VOCAB_PATH, "rb") as f:
    meta = pickle.load(f)
vocab = meta["vocab"]

# Standardization
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    return tf.strings.regex_replace(lowercase, r"[^a-z0-9\s\[\]/:]", "")

# Setup Vectorizer
vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=len(vocab), 
    output_sequence_length=48,
    standardize=custom_standardization
)
vectorizer.set_vocabulary(vocab)

# Test Queries
queries = [
    "[user] timing of train from mumbai to pune [bot]",
    "[user] trains from mumbai to pune [bot]",
    "[user] hello [bot]"
]

for q in queries:
    tokens = vectorizer([q])[0].numpy()
    words = [vocab[t] for t in tokens if t > 0]
    print(f"Query: {q}")
    print(f"Tokens: {words}\n")
