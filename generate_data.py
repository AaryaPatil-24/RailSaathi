import pandas as pd
import random
import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def normalize(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = text.replace("jn", "").replace("junction", "")
    return re.sub(r"\s+", " ", text).strip()

def create_dataset():
    print("Loading train_info.csv...")
    df = pd.read_csv(BASE_DIR / "train_info.csv")
    
    stations = list(set([normalize(s) for s in df["Source_Station_Name"].dropna()] + 
                        [normalize(s) for s in df["Destination_Station_Name"].dropna()]))
    stations = [s for s in stations if s]
    
    train_names = list(set([normalize(t) for t in df["Train_Name"].dropna()]))
    train_names = [t for t in train_names if t]
    
    train_nos = list(set(df["Train_No"].dropna().astype(str)))

    dataset = []

    # MASSIVELY EXPANDED QUERY VARIETY
    route_queries = [
        "trains from {src} to {dest}",
        "i want to go from {src} to {dest}",
        "show me trains from {src} to {dest}",
        "is there a train from {src} to {dest}",
        "{src} to {dest}",
        "trains to {dest} from {src}",
        "find trains from {src} to {dest}",
        "how to travel from {src} to {dest}",
        "need a train traveling from {src} to {dest}",
        "can you look up trains between {src} and {dest}",
        "timing of train from {src} to {dest}",
        "schedule for {src} to {dest}",
        "when is the next train from {src} to {dest}",
        "list all trains connecting {src} and {dest}",
        "route between {src} and {dest}",
        "how can i reach {dest} from {src}",
        "train timings from {src} to {dest}",
        "pune to mumbai", # hardcoded examples
        "delhi to chennai",
        "give me the list of trains from {src} to {dest}",
        "any trains available between {src} and {dest}?"
    ]
    
    route_responses = [
        "I can help you find trains from {src} to {dest}. [CALL_ROUTE] SRC: {src} DEST: {dest} [/CALL_ROUTE] [END]",
        "Sure, let me check trains traveling from {src} to {dest}. [CALL_ROUTE] SRC: {src} DEST: {dest} [/CALL_ROUTE] [END]",
        "Searching the database for trains from {src} to {dest}... [CALL_ROUTE] SRC: {src} DEST: {dest} [/CALL_ROUTE] [END]",
        "Absolutely. Let's look up the available trains departing from {src} towards {dest}. [CALL_ROUTE] SRC: {src} DEST: {dest} [/CALL_ROUTE] [END]",
        "I'd be happy to check the route from {src} to {dest} for you. [CALL_ROUTE] SRC: {src} DEST: {dest} [/CALL_ROUTE] [END]",
        "Sure thing! Checking the schedule for trains connecting {src} and {dest}. [CALL_ROUTE] SRC: {src} DEST: {dest} [/CALL_ROUTE] [END]",
        "Please wait a moment while I pull up trains going from {src} to {dest}. [CALL_ROUTE] SRC: {src} DEST: {dest} [/CALL_ROUTE] [END]",
        "Checking routes for your journey from {src} to {dest}... [CALL_ROUTE] SRC: {src} DEST: {dest} [/CALL_ROUTE] [END]",
        "Here are the options I found for traveling from {src} to {dest}. [CALL_ROUTE] SRC: {src} DEST: {dest} [/CALL_ROUTE] [END]",
        "Certainly, searching for railway connections from {src} to {dest}. [CALL_ROUTE] SRC: {src} DEST: {dest} [/CALL_ROUTE] [END]"
    ]
    
    route_missing_src = [
        "trains to {dest}",
        "going to {dest}",
        "how to reach {dest}",
        "i need to travel to {dest}",
        "what trains go to {dest}",
        "timing for {dest}",
        "schedule for {dest}"
    ]
    
    route_missing_src_responses = [
        "Got it, you want to go to {dest}. Where are you traveling from? [END]",
        "To find trains to {dest}, please tell me your starting station. [END]",
        "I can help with trains to {dest}, but I need to know your departure city first. [END]",
        "Sure, traveling to {dest}. Where will you be boarding the train? [END]",
        "I need a source station to look up trains to {dest}. Where are you leaving from? [END]"
    ]

    day_queries = [
        "what days do trains run from {src} to {dest}",
        "days of operation from {src} to {dest}",
        "when does the train run from {src} to {dest}",
        "which days to travel between {src} and {dest}",
        "tell me the schedule days from {src} to {dest}",
        "running days for {src} to {dest}"
    ]

    day_responses = [
        "Let me find the running days for trains from {src} to {dest}. [CALL_DAY] SRC: {src} DEST: {dest} [/CALL_DAY] [END]",
        "I will check which days trains operate between {src} and {dest}. [CALL_DAY] SRC: {src} DEST: {dest} [/CALL_DAY] [END]",
        "Pulling up the schedule to see which days trains run from {src} to {dest}. [CALL_DAY] SRC: {src} DEST: {dest} [/CALL_DAY] [END]",
        "Checking the operational days for the {src} to {dest} route. [CALL_DAY] SRC: {src} DEST: {dest} [/CALL_DAY] [END]",
        "Sure, let me verify the days trains depart from {src} heading to {dest}. [CALL_DAY] SRC: {src} DEST: {dest} [/CALL_DAY] [END]"
    ]

    info_no_queries = [
        "tell me about train {tno}",
        "info for train {tno}",
        "train {tno} details",
        "search train number {tno}",
        "what is train {tno}",
        "schedule of {tno}",
        "route of {tno}"
    ]
    
    info_no_responses = [
        "Here is the information for train {tno}. [CALL_INFO] TNO: {tno} [/CALL_INFO] [END]",
        "Checking the details for train number {tno}... [CALL_INFO] TNO: {tno} [/CALL_INFO] [END]",
        "I'll look up the schedule and route for train {tno}. [CALL_INFO] TNO: {tno} [/CALL_INFO] [END]",
        "Searching the railway database for train {tno}. [CALL_INFO] TNO: {tno} [/CALL_INFO] [END]",
        "Let me pull up the specifics for train number {tno}. [CALL_INFO] TNO: {tno} [/CALL_INFO] [END]"
    ]

    greetings = ["hi", "hello", "hey", "good morning", "good evening", "namaste", "howdy", "hi there"]
    greeting_responses = [
        "Hello! I am RailSaathi, a generative language model. How can I assist you with your railway queries? [END]",
        "Hi there! Where would you like to travel today? [END]",
        "Welcome to RailSaathi! I can help you find trains and check schedules. What do you need? [END]",
        "Greetings! Feel free to ask me about any train routes or schedules. [END]",
        "Hello! I'm here to help you plan your train journey. Where are you heading? [END]"
    ]

    print("Generating generative conversational data...")

    def append_example(q, a):
        seq = f"[USER] {q} [BOT] {a}"
        dataset.append(seq)

    # Generate 100,000 examples for maximum robustness
    for _ in range(40000):
        src, dest = random.sample(stations, 2)
        q = random.choice(route_queries).replace("{src}", src).replace("{dest}", dest)
        a = random.choice(route_responses).replace("{src}", src).replace("{dest}", dest)
        append_example(q, a)

    for _ in range(15000):
        dest = random.choice(stations)
        q = random.choice(route_missing_src).replace("{dest}", dest)
        a = random.choice(route_missing_src_responses).replace("{dest}", dest)
        append_example(q, a)

    for _ in range(25000):
        src, dest = random.sample(stations, 2)
        q = random.choice(day_queries).replace("{src}", src).replace("{dest}", dest)
        a = random.choice(day_responses).replace("{src}", src).replace("{dest}", dest)
        append_example(q, a)

    for _ in range(15000):
        tno = random.choice(train_nos)
        q = random.choice(info_no_queries).replace("{tno}", tno)
        a = random.choice(info_no_responses).replace("{tno}", tno)
        append_example(q, a)

    for _ in range(5000):
        q = random.choice(greetings)
        a = random.choice(greeting_responses)
        append_example(q, a)

    # Shuffle dataset
    random.shuffle(dataset)

    print(f"Generated {len(dataset)} highly diverse examples.")
    
    out_path = BASE_DIR / "generative_data.json"
    with open(out_path, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    create_dataset()
