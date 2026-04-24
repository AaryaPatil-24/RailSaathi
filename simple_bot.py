import re
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def load_rules():
    rules = {}
    with open(BASE_DIR / "railway_data.txt", "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Extract key words for matching
            words = line.lower().split()
            key = " ".join(words[:4])  # first 4 words as key
            if key not in rules:
                rules[key] = line
    return rules

RULES = load_rules()
print(f"Loaded {len(RULES)} rules")

def find_rule(query):
    query = query.lower()
    query_words = set(query.split())
    
    # Score each rule
    best_score = 0
    best_rule = None
    
    for key, rule in RULES.items():
        rule_words = set(rule.lower().split())
        # Check overlap
        score = len(query_words & rule_words)
        if score > best_score:
            best_score = score
            best_rule = rule
    
    if best_rule and best_score >= 1:
        return best_rule
    
    # Fallback: random rule
    return random.choice(list(RULES.values()))

# Chat loop
print("\nRailSaathi Rule Bot Ready!")
print("Ask about railway rules...\n")

while True:
    query = input("You: ").strip()
    if not query:
        continue
    if query.lower() in ['exit', 'quit']:
        break
    
    response = find_rule(query)
    print(f"Bot: {response}\n")