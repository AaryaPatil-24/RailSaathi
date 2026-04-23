import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
import pandas as pd
import pickle
import numpy as np
import re
from pathlib import Path
from difflib import get_close_matches

# Import our generative causal MLM architecture
from mlm_model import build_causal_mlm

BASE_DIR = Path(__file__).resolve().parent

# =============================
# Load Assets & Generative MLM
# =============================
def load_assets():
    print("Loading assets...")
    # 1. Load Data
    df = pd.read_csv(BASE_DIR / "train_info.csv")
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    df.drop_duplicates(inplace=True)
    df["days"] = df["days"].fillna("Not Available")
    
    def normalize_station(text):
        text = str(text).lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        text = text.replace("jn", "").replace("junction", "")
        return re.sub(r"\s+", " ", text).strip()
        
    df["src_norm"] = df["Source_Station_Name"].apply(normalize_station)
    df["dest_norm"] = df["Destination_Station_Name"].apply(normalize_station)
    
    # 2. Load MLM Vocab
    with open(BASE_DIR / "mlm_vocab.pkl", "rb") as f:
        meta = pickle.load(f)
        
    vocab = meta["vocab"]
    id_to_word = {i: w for i, w in enumerate(vocab)}
    
    # 3. Setup Vectorizer
    def custom_standardization(input_data):
        lowercase = tf.strings.lower(input_data)
        return tf.strings.regex_replace(lowercase, r"[^a-z0-9\s\[\]/:]", "")

    vectorizer = TextVectorization(
        max_tokens=len(vocab), 
        output_sequence_length=48,
        standardize=custom_standardization
    )
    vectorizer.set_vocabulary(vocab)
    
    # 4. Load Generative Causal MLM Weights
    model = build_causal_mlm(vocab_size=len(vocab), max_len=48)
    model(tf.zeros((1, 48), dtype=tf.int32)) # Build graph
    model.load_weights(str(BASE_DIR / "mlm_weights.weights.h5"))
    
    return df, vectorizer, model, id_to_word

# =============================
# Autoregressive Generation
# =============================
def generate_response(query, vectorizer, mlm_model, id_to_word, temperature=0.7, top_k=5):
    prompt = f"[user] {query} [bot]"
    input_tensor_full = vectorizer([prompt])[0].numpy()
    
    # Filter out [UNK] tokens (ID 1) to make the model more robust to unknown words
    input_ids = [tid for tid in input_tensor_full if tid > 1]
    
    # Re-pad to MAX_LEN
    input_tensor = np.zeros(48, dtype=np.int32)
    for i, tid in enumerate(input_ids[:48]):
        input_tensor[i] = tid
        
    current_len = len(input_ids)
    if current_len > 48: current_len = 48
            
    generated_tokens = []
    repetition_penalty = 1.2
    
    for _ in range(48 - current_len):
        preds = mlm_model.predict(np.array([input_tensor]), verbose=0)
        logits = np.log(preds[0, current_len - 1, :] + 1e-10) / temperature
        
        # Apply Repetition Penalty
        for token_id in set(input_tensor[:current_len]):
            if token_id > 0: # Don't penalize padding
                if logits[token_id] > 0:
                    logits[token_id] /= repetition_penalty
                else:
                    logits[token_id] *= repetition_penalty
        
        # Never generate [user] or [bot] tags in the response
        # (This prevents prompt-looping)
        for tag in ["[user]", "[bot]"]:
            tag_id = [k for k, v in id_to_word.items() if v == tag]
            if tag_id: logits[tag_id[0]] = -1e10

        top_indices = np.argsort(logits)[-top_k:]
        exp_logits = np.exp(logits[top_indices] - np.max(logits[top_indices]))
        probs = exp_logits / np.sum(exp_logits)
        
        next_token_id = np.random.choice(top_indices, p=probs)
        word = id_to_word.get(next_token_id, "")
        
        if word == "[end]": break
        generated_tokens.append(word)
        input_tensor[current_len] = next_token_id
        current_len += 1
        
    return " ".join(generated_tokens)

# =============================
# DB Processing
# =============================
def fuzzy_match_station(user_station, station_list):
    matches = get_close_matches(user_station, station_list, n=1, cutoff=0.5)
    return matches[0] if matches else None

def process_slm_output(generated_text, df):
    route_match = re.search(r"\[call_route\](.*?)\[/call_route\]", generated_text)
    if route_match:
        content = route_match.group(1)
        src = re.search(r"src:\s*([a-z0-9\s]+?)\s*dest:", content).group(1).strip()
        dest = re.search(r"dest:\s*([a-z0-9\s]+)", content).group(1).strip()
        
        s_match = fuzzy_match_station(src, df["src_norm"].unique())
        d_match = fuzzy_match_station(dest, df["dest_norm"].unique())
        
        if not s_match or not d_match: return f"Could not find stations {src} or {dest}."
        matches = df[(df["src_norm"] == s_match) & (df["dest_norm"] == d_match)]
        if matches.empty: return f"No trains found from {src.title()} to {dest.title()}."
        
        res = f"\n--- Trains from {src.title()} to {dest.title()} ---\n"
        for _, r in matches.head(5).iterrows():
            res += f"[{r['Train_No']}] {r['Train_Name']} | Days: {r['days']}\n"
        return res

    info_match = re.search(r"\[call_info\](.*?)\[/call_info\]", generated_text)
    if info_match:
        tno = re.search(r"tno:\s*([0-9]+)", info_match.group(1)).group(1)
        row = df[df["Train_No"].astype(str) == tno]
        if not row.empty:
            r = row.iloc[0]
            return f"\n--- Train {r['Train_No']} Details ---\nName: {r['Train_Name']}\nRoute: {r['Source_Station_Name']} to {r['Destination_Station_Name']}\nDays: {r['days']}"
        return f"Train {tno} not found."

    return generated_text.capitalize()

def main():
    print("="*30)
    print("🚆 RailSaathi CLI (Generative SLM)")
    print("="*30)
    
    df, vectorizer, model, id_to_word = load_assets()
    print("\nModel ready! Type 'exit' to quit.\n")
    
    while True:
        try:
            query = input("You: ").strip()
            if query.lower() in ["exit", "quit"]: break
            if not query: continue
            
            raw_gen = generate_response(query, vectorizer, model, id_to_word)
            response = process_slm_output(raw_gen, df)
            
            print(f"RailSaathi: {response}\n")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
