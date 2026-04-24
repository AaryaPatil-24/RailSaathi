# RailSaathi

RailSaathi is a **micro language model for Indian Railways policy and rules**.  
It trains a compact causal decoder (Mini-GPT style) on a synthetic railway-policy corpus and serves answers via CLI or Streamlit.

## Why This Project

- Build a lightweight model that can train on CPU in minutes.
- Keep responses focused on railway policy domains.
- Reduce gibberish by combining a compact causal LM with retrieval guardrails over `railway_data.txt`.

## Current Status

- `railway_data.txt`: 10,000 unique synthetic policy lines.
- `generative_data.json`: conversational training examples generated from those rules.
- Micro model: 2-layer causal decoder (`embed_dim=128`, `num_heads=4`).
- Typical training time on CPU-only setup: ~4-6 minutes for 4 epochs.
- CLI retrieval layer includes token normalization, synonym expansion, weighted scoring, and output cleanup for stronger policy matching.

## Tech Stack

- Python 3.11+
- TensorFlow / Keras
- NumPy, Pandas
- Streamlit

## Project Structure

```text
app.py                    # Streamlit app (chat UI + policy retrieval + optional train lookup flow)
cli.py                    # Terminal chat interface for policy Q&A
augment_railway_data.py   # Builds realistic synthetic railway policy corpus (10k+ lines)
generate_data.py          # Converts rules into conversational [USER]/[BOT] samples
train_mlm.py              # Training pipeline (vectorization, masking, callbacks, saving artifacts)
mlm_model.py              # Micro causal decoder architecture
railway_data.txt          # Policy corpus used as base knowledge + retrieval index
generative_data.json      # Training dataset generated from railway_data.txt
mlm_vocab.pkl             # Saved vocabulary + model metadata
mlm_weights.weights.h5    # Trained model weights
requirements.txt          # Dependencies
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training Pipeline

Run end-to-end:

```bash
python3 augment_railway_data.py --target 10000
python3 generate_data.py
python3 train_mlm.py --epochs 4 --batch-size 128 --vocab-size 6000 --max-len 48
```

Artifacts produced/updated:

- `railway_data.txt`
- `generative_data.json`
- `mlm_vocab.pkl`
- `mlm_weights.weights.h5`

## Run

CLI:

```bash
python3 cli.py
```

Streamlit UI:

```bash
streamlit run app.py
```

## Notes on Accuracy and Stability

- The model is strongest for in-domain policy questions: waiting list, tatkal, refunds, baggage, berth, fines, complaint channels.
- `cli.py` uses an enhanced retrieval-first matcher (weighted token scoring + normalization), which is the most stable mode for policy Q&A.
- `app.py` also uses retrieval-first matching, but its scoring logic is currently simpler than CLI.
- For ambiguous or multi-intent questions, ask users to provide one clear query at a time for best results.

## Optional Data Requirement for App Train Lookup Mode

`app.py` also contains optional train route/day/info parsing logic that expects `train_info.csv`.  
If you only need policy-mode micro-LM behavior, `cli.py` is sufficient.

## Roadmap

- Add offline evaluation script for exact-match / semantic-match policy QA scoring.
- Add confusion and failure-case reports per topic.
- Add incremental fine-tuning on real user queries.

## License

Add your license here (MIT/Apache-2.0/etc.).
