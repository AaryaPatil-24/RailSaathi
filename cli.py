import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
import pickle
import numpy as np
import re
import math
from collections import Counter
from pathlib import Path
from mlm_model import build_causal_mlm

BASE_DIR = Path(__file__).resolve().parent
STOPWORDS = {
    "what", "is", "the", "a", "an", "for", "to", "of", "in", "on", "with", "and",
    "can", "i", "me", "my", "about", "policy", "rule", "please", "tell", "how",
    "do", "does", "if", "who", "from", "by", "as", "per", "train", "railway",
    "railways", "passenger", "passengers", "travel", "travelling", "allowed",
    "allow", "get", "happen", "happens", "someone",
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
DISPLAY_SUFFIXES = [
    " as per notified railway policy",
    " under railway act provisions",
    " as per applicable circulars",
    " subject to operational conditions",
    " unless revised by competent authority",
    " according to reservation and ticketing manual",
    " as per commercial department instructions",
    " as per coaching tariff norms",
    " as per current railway notification",
    " under passenger amenity standards",
    " as per onboard ticket checking protocol",
    " under standard customer service procedures",
]
TOKEN_SYNONYMS = {
    "explosive": {"hazardous", "inflammable", "material"},
    "smoke": {"smoking"},
    "lost": {"lose", "missing"},
    "without": {"ticketless"},
    "men": {"male"},
    "lady": {"women", "female", "ladies"},
    "compartment": {"coach"},
    "complain": {"complaint", "helpline", "madad"},
    "refund": {"reimbursement", "tdr"},
    "baggage": {"luggage"},
    "rac": {"reservation", "berth"},
}
GENERIC_QUERY_TOKENS = {
    "refund", "ticket", "policy", "rule", "class", "status", "board", "berth",
    "validity", "check", "file", "carry",
}


def strip_display_prefixes(text: str) -> str:
    cleaned = text.strip()
    changed = True
    while changed:
        changed = False
        for prefix in DISPLAY_PREFIXES:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                changed = True
    return cleaned


def strip_display_suffixes(text: str) -> str:
    cleaned = text.strip()
    changed = True
    while changed:
        changed = False
        for suffix in DISPLAY_SUFFIXES:
            if cleaned.endswith(suffix):
                cleaned = cleaned[: -len(suffix)].strip()
                changed = True
    return cleaned


def normalize_token(token: str) -> str:
    token = token.lower().strip()
    if token == "ladies":
        return "lady"
    if token.endswith("ies") and len(token) > 4:
        token = token[:-3] + "y"
    elif token.endswith("s") and len(token) > 4 and not token.endswith("ss"):
        token = token[:-1]
    return token


def tokenize_for_retrieval(text: str) -> set[str]:
    raw_tokens = re.findall(r"[a-z0-9]+", text.lower())
    normalized = {normalize_token(tok) for tok in raw_tokens}
    return {tok for tok in normalized if tok and tok not in STOPWORDS}


def expand_query_tokens(tokens: set[str]) -> set[str]:
    expanded = set(tokens)
    for tok in list(tokens):
        if tok in TOKEN_SYNONYMS:
            expanded.update(TOKEN_SYNONYMS[tok])
    return expanded

def load_assets():
    print("Loading RailSaathi micro language model assets...")
    # 1. Load MLM Vocab
    with open(BASE_DIR / "mlm_vocab.pkl", "rb") as f:
        meta = pickle.load(f)
        
    vocab = meta["vocab"]
    max_len = int(meta.get("max_len", 48))
    id_to_word = {i: w for i, w in enumerate(vocab)}
    
    # 2. Setup Vectorizer
    def custom_standardization(input_data):
        lowercase = tf.strings.lower(input_data)
        return tf.strings.regex_replace(lowercase, r"[^a-z0-9\s\[\]/:]", "")

    vectorizer = TextVectorization(
        max_tokens=len(vocab), 
        output_sequence_length=max_len,
        standardize=custom_standardization
    )
    vectorizer.set_vocabulary(vocab)
    
    # 3. Load trained micro LM weights
    model = build_causal_mlm(vocab_size=len(vocab), max_len=max_len)
    model(tf.zeros((1, max_len), dtype=tf.int32)) # Build graph
    try:
        model.load_weights(str(BASE_DIR / "mlm_weights.weights.h5"))
        print("Model weights loaded successfully.")
    except:
        print("Warning: Weights file not found. Please wait for training to finish.")
    
    rules = [line.strip().lower() for line in (BASE_DIR / "railway_data.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    tokenized_rules = []
    token_df = Counter()

    for rule in rules:
        core_rule = strip_display_prefixes(rule)
        tokens = tokenize_for_retrieval(core_rule)
        if not tokens:
            continue
        tokenized_rules.append((rule, tokens))
        token_df.update(tokens)

    total_rules = max(1, len(tokenized_rules))
    token_idf = {tok: math.log((1 + total_rules) / (1 + df)) + 1.0 for tok, df in token_df.items()}

    rule_index = []
    for rule, tokens in tokenized_rules:
        token_weight_sum = sum(token_idf.get(tok, 1.0) for tok in tokens)
        rule_index.append((rule, tokens, token_weight_sum))

    return vectorizer, model, id_to_word, max_len, rule_index, token_idf


def retrieve_rule(query, rule_index, token_idf):
    base_query_tokens = tokenize_for_retrieval(query)
    q_tokens = expand_query_tokens(base_query_tokens)
    if not q_tokens:
        return None
    anchor_tokens = {tok for tok in base_query_tokens if tok not in GENERIC_QUERY_TOKENS}

    best_rule = None
    best_score = 0.0
    query_weight_sum = sum(token_idf.get(tok, 1.5) for tok in q_tokens)
    for rule, tokens, token_weight_sum in rule_index:
        if anchor_tokens and not (anchor_tokens & tokens):
            continue
        overlap_tokens = q_tokens & tokens
        if not overlap_tokens:
            continue

        overlap_weight = sum(token_idf.get(tok, 1.5) for tok in overlap_tokens)
        precision = overlap_weight / max(query_weight_sum, 1e-9)
        recall = overlap_weight / max(token_weight_sum, 1e-9)
        score = 0.78 * precision + 0.22 * recall

        rare_hits = sum(1 for tok in overlap_tokens if token_idf.get(tok, 1.0) >= 2.0)
        score += min(0.12, 0.04 * rare_hits)

        if len(overlap_tokens) == 1 and next(iter(overlap_tokens)) in {"ticket", "coach", "berth"}:
            score -= 0.15

        if score > best_score:
            best_score = score
            best_rule = rule

    threshold = 0.38 if len(q_tokens) >= 3 else 0.44
    if best_score >= threshold:
        return best_rule
    return None


def clean_rule_for_output(rule: str) -> str:
    return strip_display_suffixes(strip_display_prefixes(rule))


def format_policy_text(text: str) -> str:
    body = text.strip().lower()
    if body.startswith("railway guideline:"):
        body = body.split(":", 1)[1].strip()
    body = clean_rule_for_output(body)
    body = re.sub(r"\s+", " ", body).strip()
    if not body:
        return "Please share your railway question in detail."
    return f"Railway guideline: {body}"

def generate_response(query, vectorizer, mlm_model, id_to_word, max_len, rule_index, token_idf, temperature=0.3, top_k=1):
    retrieved = retrieve_rule(query, rule_index, token_idf)
    if retrieved:
        return format_policy_text(retrieved)

    prompt = f"[user] {query} [bot]"
    input_tensor_full = vectorizer([prompt])[0].numpy()
    input_ids = [tid for tid in input_tensor_full if tid > 1]
    
    input_tensor = np.zeros(max_len, dtype=np.int32)
    for i, tid in enumerate(input_ids[:max_len]):
        input_tensor[i] = tid
        
    current_len = len(input_ids)
    if current_len > max_len:
        current_len = max_len
    if current_len == 0:
        return "Please share your railway question in detail."
            
    generated_tokens = []
    repetition_penalty = 1.2
    blocked_tokens = {0, 1}
    for tag in ["[user]", "[bot]"]:
        tag_id = [k for k, v in id_to_word.items() if v == tag]
        if tag_id:
            blocked_tokens.add(tag_id[0])
    
    for _ in range(max_len - current_len):
        preds = mlm_model.predict(np.array([input_tensor]), verbose=0)
        logits = np.log(preds[0, current_len - 1, :] + 1e-10) / temperature
        
        # Penalize repetition
        for token_id in set(input_tensor[:current_len]):
            if token_id > 0:
                if logits[token_id] > 0: logits[token_id] /= repetition_penalty
                else: logits[token_id] *= repetition_penalty
        
        # Suppress tags and unknown token IDs.
        for token_id in blocked_tokens:
            logits[token_id] = -1e10

        top_indices = np.argsort(logits)[-top_k:]
        exp_logits = np.exp(logits[top_indices] - np.max(logits[top_indices]))
        probs = exp_logits / np.sum(exp_logits)
        
        next_token_id = np.random.choice(top_indices, p=probs)
        word = id_to_word.get(next_token_id, "")
        
        if word == "[end]": break
        generated_tokens.append(word)
        input_tensor[current_len] = next_token_id
        current_len += 1
        
    raw = " ".join(generated_tokens).strip()
    if not raw:
        return "Please share your railway question in detail."

    lowered = raw.lower()
    if lowered.startswith("railway guideline:") or any(lowered.startswith(prefix) for prefix in DISPLAY_PREFIXES):
        return format_policy_text(raw)
    return raw.capitalize()

def main():
    print("="*40)
    print("🚆 RailSaathi CLI (Micro Language Model)")
    print("="*40)
    
    vectorizer, model, id_to_word, max_len, rule_index, token_idf = load_assets()
    print("\nRailSaathi micro LM ready! Ask me about railway policies, refunds, baggage, fines, or berth rules.\n")
    
    while True:
        try:
            query = input("You: ").strip()
            if query.lower() in ["exit", "quit"]: break
            if not query: continue
            
            response = generate_response(query, vectorizer, model, id_to_word, max_len, rule_index, token_idf)
            print(f"RailSaathi: {response}\n")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
