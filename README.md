# 🚆 RailSaathi — Custom Generative SLM

A custom-built **Generative Small Language Model (SLM)** for Indian Railways queries. Instead of an intent classifier, RailSaathi is now powered by a **Causal Transformer Decoder (Mini-GPT)** built from scratch using TensorFlow/Keras. It autoregressively generates responses word-by-word and dynamically intercepts database calls to answer your queries naturally.

## Features

- **Generative NLU**: No hardcoded if/else routing. The model understands context and speaks naturally.
- **Top-K & Temperature Sampling**: Ensures creative, varied, and human-like conversational responses.
- **Dialogue State Tracking (Memory)**: The bot remembers context (like your source station) across multiple messages.
- **Train Info** — Look up any train by number (e.g. *tell me about train 12301*)
- **Route Search** — Find trains between two stations (e.g. *trains from Mumbai to Pune*)
- **Running Days** — Check which days a train operates (e.g. *what days from Delhi to Chennai*)

## Setup (Using `uv`)

We use [`uv`](https://github.com/astral-sh/uv) because it is incredibly fast for creating virtual environments and installing Python packages.

### 1. Install `uv` (if you don't have it)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Setup Project & Install Dependencies
Make sure you are in the `RailSaathi` directory:
```bash
cd RailSaathi
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 3. Generate Data & Train the SLM
Since this is a custom model built from scratch, you need to generate the synthetic training data and train the weights locally.

```bash
uv run generate_data.py
uv run train_mlm.py
```
*(Note: Training the 4-layer generative model on 62,000 sequences will take a few minutes).*

### 4. Run the App
```bash
uv run streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Architecture

```
app.py                  # Streamlit app + Autoregressive Generation Loop
mlm_model.py            # Causal Transformer Decoder Architecture (Mini-GPT)
generate_data.py        # Generates 62k+ diverse conversational training sequences
train_mlm.py            # Next-Token Prediction Training pipeline
requirements.txt        # Dependencies
train_info.csv          # Railway dataset (~11K train entries)
```

## License

This project is open source. See the repository for details.
