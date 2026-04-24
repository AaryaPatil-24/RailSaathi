import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
import pickle
import sys
from pathlib import Path
from mlm_model import build_causal_mlm

BASE_DIR = Path(__file__).resolve().parent

query = sys.argv[1] if len(sys.argv) > 1 else "baggage limit"

with open(BASE_DIR / "mlm_vocab.pkl", "rb") as f:
    meta = pickle.load(f)
vocab = meta["vocab"]
id_to_word = {i: w for i, w in enumerate(vocab)}

def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    return tf.strings.regex_replace(lowercase, r"[^a-z0-9\s\[\]/:]", "")

vectorizer = TextVectorization(max_tokens=len(vocab), output_sequence_length=48, standardize=custom_standardization)
vectorizer.set_vocabulary(vocab)

model = build_causal_mlm(vocab_size=len(vocab), max_len=48)
model(tf.zeros((1, 48), dtype=tf.int32))
model.load_weights(str(BASE_DIR / "mlm_weights.weights.h5"))

prompt = f"[user] {query} [bot]"
input_tensor_full = vectorizer([prompt])[0].numpy()
input_ids = [tid for tid in input_tensor_full if tid > 1]

input_tensor = np.zeros(48, dtype=np.int32)
for i, tid in enumerate(input_ids[:48]):
    input_tensor[i] = tid
current_len = len(input_ids)
if current_len > 48: current_len = 48

generated_tokens = []
for step in range(48 - current_len):
    preds = model.predict(np.array([input_tensor]), verbose=0)
    next_token_id = np.argmax(preds[0, current_len - 1, :])
    word = id_to_word.get(next_token_id, "")
    
    if word == "[end]" or word == "" or word == "[(":
        print(f"Stopped at step {step+1}: '{word}'")
        break
    generated_tokens.append(word)
    print(f"Step {step+1}: '{word}' (id={next_token_id})")
    input_tensor[current_len] = next_token_id
    current_len += 1

print(f"Query: {query}")
print(f"Bot: {' '.join(generated_tokens)}")