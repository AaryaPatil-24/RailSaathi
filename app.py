import streamlit as st
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

# =============================
# Streamlit config
# =============================
st.set_page_config(page_title="Railway Query Assistant", page_icon="🚆", layout="centered")

BASE_DIR = Path(__file__).resolve().parent
STOPWORDS = {
    "what", "is", "the", "a", "an", "for", "to", "of", "in", "on", "with", "and",
    "can", "i", "me", "my", "about", "policy", "rule", "please", "tell", "how",
    "do", "does", "if", "who", "from", "by", "as", "per",
}
DISPLAY_PREFIXES = [
    "railway guideline states that ",
    "according to railway policy ",
    "as per indian railways rules ",
    "under current railway norms ",
    "in indian railways operations ",
    "official railway instructions mention that ",
    "for passenger guidance railway rules say ",
    "as per reservation manual ",
    "as per ticketing policy ",
    "railway administration clarifies that ",
    "for train travel compliance ",
    "under railway passenger charter ",
    "as per official railway advisory ",
    "under standard operating railway rules ",
]

# =============================
# Load Assets & Generative MLM
# =============================
@st.cache_resource
def load_assets():
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
    max_len = int(meta.get("max_len", 48))
    id_to_word = {i: w for i, w in enumerate(vocab)}
    
    # 3. Setup Vectorizer
    def custom_standardization(input_data):
        lowercase = tf.strings.lower(input_data)
        return tf.strings.regex_replace(lowercase, r"[^a-z0-9\s\[\]/:]", "")

    vectorizer = TextVectorization(
        max_tokens=len(vocab), 
        output_sequence_length=max_len,
        standardize=custom_standardization
    )
    vectorizer.set_vocabulary(vocab)
    
    # 4. Load Generative Causal MLM Weights
    model = build_causal_mlm(vocab_size=len(vocab), max_len=max_len)
    model(tf.zeros((1, max_len), dtype=tf.int32)) # Build graph
    model.load_weights(str(BASE_DIR / "mlm_weights.weights.h5"))
    
    rules = [line.strip().lower() for line in (BASE_DIR / "railway_data.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    rule_index = []
    for rule in rules:
        tokens = set(re.findall(r"[a-z0-9]+", rule)) - STOPWORDS
        if tokens:
            rule_index.append((rule, tokens))

    return df, vectorizer, model, id_to_word, max_len, rule_index

try:
    df, vectorizer, mlm_model, id_to_word, max_len, rule_index = load_assets()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading Generative MLM. Please run `python generate_data.py` then `python train_mlm.py`. Error: {e}")
    model_loaded = False

# =============================
# Autoregressive Generation
# =============================
def retrieve_rule(query):
    q_tokens = set(re.findall(r"[a-z0-9]+", query.lower())) - STOPWORDS
    if not q_tokens:
        return None

    best_rule = None
    best_score = 0.0
    for rule, tokens in rule_index:
        overlap = len(q_tokens & tokens)
        if overlap == 0:
            continue
        precision = overlap / len(q_tokens)
        recall = overlap / len(tokens)
        score = 0.75 * precision + 0.25 * recall
        if q_tokens.issubset(tokens):
            score += 0.1
        if score > best_score:
            best_score = score
            best_rule = rule
    if best_score >= 0.50:
        return best_rule
    return None


def clean_rule_for_output(rule: str) -> str:
    text = rule.strip()
    changed = True
    while changed:
        changed = False
        for prefix in DISPLAY_PREFIXES:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
                changed = True
    return text


def format_policy_text(text: str) -> str:
    body = text.strip().lower()
    if body.startswith("railway guideline:"):
        body = body.split(":", 1)[1].strip()
    body = clean_rule_for_output(body)
    body = re.sub(r"\s+", " ", body).strip()
    if not body:
        return "please share your railway question in detail"
    return f"railway guideline: {body}"


def generate_response(query, temperature=0.3, top_k=1):
    """Feeds prompt to the Causal LM and generates token-by-token until [end].
    Uses Temperature and Top-K sampling for natural variance."""
    retrieved = retrieve_rule(query)
    if retrieved:
        return format_policy_text(retrieved)

    prompt = f"[user] {query} [bot]"
    
    # Convert prompt to token IDs
    input_tensor_full = vectorizer([prompt])[0].numpy()
    
    # Filter out [UNK] tokens (ID 1) to make the model more robust to unknown words
    input_ids = [tid for tid in input_tensor_full if tid > 1]
    
    # Re-pad to MAX_LEN
    input_tensor = np.zeros(max_len, dtype=np.int32)
    for i, tid in enumerate(input_ids[:max_len]):
        input_tensor[i] = tid
        
    current_len = len(input_ids)
    if current_len > max_len:
        current_len = max_len
    if current_len == 0:
        return "please share your railway question in detail"
            
    generated_tokens = []
    repetition_penalty = 1.2
    blocked_tokens = {0, 1}
    for tag in ["[user]", "[bot]"]:
        tag_id = [k for k, v in id_to_word.items() if v == tag]
        if tag_id:
            blocked_tokens.add(tag_id[0])
    
    # Autoregressive loop
    for _ in range(max_len - current_len):
        # Predict next token probabilities
        preds = mlm_model.predict(np.array([input_tensor]), verbose=0)
        logits = np.log(preds[0, current_len - 1, :] + 1e-10) # convert to logits
        
        # Apply temperature
        logits = logits / temperature
        
        # Apply Repetition Penalty
        for token_id in set(input_tensor[:current_len]):
            if token_id > 0: # Don't penalize padding
                if logits[token_id] > 0:
                    logits[token_id] /= repetition_penalty
                else:
                    logits[token_id] *= repetition_penalty
        
        # Never generate control or unknown tokens.
        for token_id in blocked_tokens:
            logits[token_id] = -1e10

        # Top-K Sampling
        top_indices = np.argsort(logits)[-top_k:]
        exp_logits = np.exp(logits[top_indices] - np.max(logits[top_indices]))
        probs = exp_logits / np.sum(exp_logits)
        
        # Sample
        next_token_id = np.random.choice(top_indices, p=probs)
        word = id_to_word.get(next_token_id, "")
        
        if word == "[end]":
            break
            
        generated_tokens.append(word)
        input_tensor[current_len] = next_token_id
        current_len += 1
        
    raw = " ".join(generated_tokens).strip()
    if not raw:
        return "please share your railway question in detail"

    lowered = raw.lower()
    if lowered.startswith("railway guideline:") or any(lowered.startswith(prefix) for prefix in DISPLAY_PREFIXES):
        return format_policy_text(raw)
    return raw

# =============================
# Action Execution (RAG / DB Lookup)
# =============================
def fuzzy_match_station(user_station, station_list):
    matches = get_close_matches(user_station, station_list, n=1, cutoff=0.5)
    return matches[0] if matches else None

def format_trains(matches, title):
    res = f"**{title}**\n\n"
    for _, r in matches.head(10).iterrows():
        res += f"- 🚂 **{r['Train_No']}** {r['Train_Name']} | {r['Source_Station_Name']} ➡️ {r['Destination_Station_Name']} | 📅 {r['days']}\n"
    return res

def process_slm_output(generated_text):
    """
    Parses the generated text for structured API/DB calls.
    E.g. "I can help with that. [call_route] src: mumbai dest: pune [/call_route]"
    """
    # Look for route call
    route_match = re.search(r"\[call_route\](.*?)\[/call_route\]", generated_text)
    if route_match:
        content = route_match.group(1)
        src_match = re.search(r"src:\s*([a-z0-9\s]+?)\s*dest:", content)
        dest_match = re.search(r"dest:\s*([a-z0-9\s]+)", content)
        
        if src_match and dest_match:
            src = src_match.group(1).strip()
            dest = dest_match.group(1).strip()
            
            s_match = fuzzy_match_station(src, df["src_norm"].unique())
            d_match = fuzzy_match_station(dest, df["dest_norm"].unique())
            
            if not s_match or not d_match:
                return f"I understood you want to go from {src} to {dest}, but I couldn't find those stations in the database."
                
            matches = df[(df["src_norm"] == s_match) & (df["dest_norm"] == d_match)]
            if matches.empty:
                return f"No trains found from {src.title()} to {dest.title()}."
                
            return format_trains(matches, f"Trains from {src.title()} to {dest.title()}:")

    # Look for day call
    day_match = re.search(r"\[call_day\](.*?)\[/call_day\]", generated_text)
    if day_match:
        content = day_match.group(1)
        src_match = re.search(r"src:\s*([a-z0-9\s]+?)\s*dest:", content)
        dest_match = re.search(r"dest:\s*([a-z0-9\s]+)", content)
        
        if src_match and dest_match:
            src = src_match.group(1).strip()
            dest = dest_match.group(1).strip()
            
            s_match = fuzzy_match_station(src, df["src_norm"].unique())
            d_match = fuzzy_match_station(dest, df["dest_norm"].unique())
            
            if not s_match or not d_match:
                return f"I couldn't find stations matching {src} or {dest}."
                
            matches = df[(df["src_norm"] == s_match) & (df["dest_norm"] == d_match)]
            if matches.empty:
                return f"No trains found from {src.title()} to {dest.title()}."
                
            return format_trains(matches, f"Running days from {src.title()} to {dest.title()}:")

    # Look for info call
    info_match = re.search(r"\[call_info\](.*?)\[/call_info\]", generated_text)
    if info_match:
        content = info_match.group(1)
        tno_match = re.search(r"tno:\s*([0-9]+)", content)
        if tno_match:
            tno = tno_match.group(1)
            row = df[df["Train_No"].astype(str) == tno]
            if not row.empty:
                r = row.iloc[0]
                return f"🚂 **Train {r['Train_No']}** — {r['Train_Name']}\n📍 Route: {r['Source_Station_Name']} → {r['Destination_Station_Name']}\n📅 Runs on: {r['days']}"
            return f"I couldn't find a train with number {tno}."

    # If no structured tags, just return the SLM's raw conversational text
    # (Capitalize the first letter and make it look clean)
    clean_text = generated_text.replace("[", "").replace("]", "")
    return clean_text.capitalize()

# =============================
# App UI
# =============================
st.title("🚆 RailSaathi (Generative SLM)")
st.caption("Powered by a custom-built Generative Causal Transformer Decoder (Mini-GPT).")

if model_loaded:
    with st.sidebar:
        st.header("🧠 SLM Internals")
        st.write("This app uses a custom Autoregressive Decoder built from scratch.")
        st.write("It generates text token-by-token natively.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("E.g. trains from delhi to mumbai")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"): st.markdown(user_input)
        
        # Generate raw response from SLM
        with st.spinner("Generating..."):
            raw_generated = generate_response(user_input)
            
        # Optional: Show what the SLM literally predicted before formatting
        # st.sidebar.info(f"Raw Generation: `{raw_generated}`")
        
        # Process tags and render response
        final_response = process_slm_output(raw_generated)
        
        st.session_state.chat_history.append({"role": "assistant", "content": final_response})
        with st.chat_message("assistant"): st.markdown(final_response)
