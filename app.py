import streamlit as st
import tensorflow as tf
import pickle
import pandas as pd
import re
from difflib import get_close_matches

# =============================
# Streamlit config
# =============================
st.set_page_config(
    page_title="Railway Query Assistant",
    page_icon="🚆",
    layout="centered"
)

st.title("🚆 Railway Query Assistant")
st.caption("Intent-based Micro Language Model (Interactive Chat)")

# =============================
# Build model architecture
# =============================
def build_model(input_dim, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation="relu", input_shape=(input_dim,)),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])
    return model

# =============================
# Load assets (cached)
# =============================
@st.cache_resource
def load_assets():
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    with open("intent_map.pkl", "rb") as f:
        intent_map = pickle.load(f)

    reverse_intent_map = {v: k for k, v in intent_map.items()}

    model = build_model(
        input_dim=len(vectorizer.get_feature_names_out()),
        num_classes=len(reverse_intent_map)
    )
    model.load_weights("railway_query_tf.h5")

    df = pd.read_csv("train_info.csv")

    return model, vectorizer, reverse_intent_map, df

model, vectorizer, reverse_intent_map, df = load_assets()

# =============================
# Text utilities
# =============================
def normalize_station(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = text.replace("jn", "").replace("junction", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["src_norm"] = df["Source_Station_Name"].apply(normalize_station)
df["dest_norm"] = df["Destination_Station_Name"].apply(normalize_station)

def fuzzy_match_station(user_station, station_list):
    matches = get_close_matches(user_station, station_list, n=1, cutoff=0.6)
    return matches[0] if matches else None

def predict_intent(query):
    vec = vectorizer.transform([query.lower()]).toarray()
    pred = model.predict(vec, verbose=0)
    return reverse_intent_map[pred.argmax()]

# =============================
# Core logic (NO hallucination)
# =============================
def get_answer(query):
    query = query.lower().strip()
    intent = predict_intent(query)

    # Route + Day
    if "day" in query and "from" in query and "to" in query:
        try:
            src = query.split("from")[1].split("to")[0].strip()
            dest = query.split("to")[1].strip()
        except:
            return "Please specify source and destination clearly."

        src_match = fuzzy_match_station(normalize_station(src), df["src_norm"].unique())
        dest_match = fuzzy_match_station(normalize_station(dest), df["dest_norm"].unique())

        if not src_match or not dest_match:
            return f"No trains found from {src} to {dest}."

        matches = df[
            (df["src_norm"] == src_match) &
            (df["dest_norm"] == dest_match)
        ]

        if matches.empty:
            return f"No trains found from {src} to {dest}."

        return " | ".join(
            [f"Train {r['Train_No']} runs on {r['days']}" for _, r in matches.iterrows()]
        )

    # Train info
    if intent == "TRAIN_INFO":
        nums = [w for w in query.split() if w.isdigit()]
        if not nums:
            return "Please provide a train number."

        train_no = int(nums[0])
        row = df[df["Train_No"] == train_no]

        if row.empty:
            return "Train number not found."

        r = row.iloc[0]
        return (
            f"Train {train_no} ({r['Train_Name']}) runs from "
            f"{r['Source_Station_Name']} to {r['Destination_Station_Name']} "
            f"on {r['days']}."
        )

    # Route query
    if intent == "ROUTE_QUERY":
        try:
            src = query.split("from")[1].split("to")[0].strip()
            dest = query.split("to")[1].strip()
        except:
            return "Please specify source and destination clearly."

        src_match = fuzzy_match_station(normalize_station(src), df["src_norm"].unique())
        dest_match = fuzzy_match_station(normalize_station(dest), df["dest_norm"].unique())

        if not src_match or not dest_match:
            return f"No trains found from {src} to {dest}."

        matches = df[
            (df["src_norm"] == src_match) &
            (df["dest_norm"] == dest_match)
        ]

        if matches.empty:
            return f"No trains found from {src} to {dest}."

        trains = matches["Train_No"].astype(str).tolist()
        return f"Trains from {src} to {dest}: {', '.join(trains)}"

    return "Query understood, but this query type is not supported."

# =============================
# Conversation memory
# =============================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =============================
# Chat input
# =============================
user_input = st.chat_input("Ask a railway-related question...")

if user_input:
    # User message
    st.session_state.chat_history.append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    # Assistant response
    response = get_answer(user_input)
    st.session_state.chat_history.append(
        {"role": "assistant", "content": response}
    )
    with st.chat_message("assistant"):
        st.markdown(response)