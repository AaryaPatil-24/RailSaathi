import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import TextVectorization

from mlm_model import build_causal_mlm

BASE_DIR = Path(__file__).resolve().parent


def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    return tf.strings.regex_replace(lowercase, r"[^a-z0-9\s\[\]/:\-]", "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RailSaathi micro language model")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-len", type=int, default=48)
    parser.add_argument("--vocab-size", type=int, default=6000)
    parser.add_argument("--val-split", type=float, default=0.1)
    return parser.parse_args()


def build_loss_mask(X: np.ndarray, bot_token_id: int) -> np.ndarray:
    bot_hits = X == bot_token_id
    has_bot = bot_hits.any(axis=1)

    first_bot_pos = np.zeros(X.shape[0], dtype=np.int32)
    first_bot_pos[has_bot] = np.argmax(bot_hits[has_bot], axis=1)

    token_positions = np.arange(X.shape[1])[None, :]
    weights = (token_positions >= first_bot_pos[:, None]).astype(np.float32)

    # If [bot] is missing for any row, avoid training on it.
    weights[~has_bot] = 0.0
    return weights


def train() -> None:
    args = parse_args()
    start = time.time()

    print("Loading conversational training data...")
    with open(BASE_DIR / "generative_data.json", "r", encoding="utf-8") as f:
        samples = json.load(f)

    print(f"Samples: {len(samples)}")

    vectorizer = TextVectorization(
        max_tokens=args.vocab_size,
        output_sequence_length=args.max_len + 1,
        standardize=custom_standardization,
    )
    vectorizer.adapt(samples)

    vocab = vectorizer.get_vocabulary()
    if "[bot]" not in vocab:
        raise RuntimeError("[bot] token missing from vocabulary. Training data format is invalid.")

    with open(BASE_DIR / "mlm_vocab.pkl", "wb") as f:
        pickle.dump(
            {
                "vocab": vocab,
                "max_len": args.max_len,
                "vocab_size": len(vocab),
                "model_type": "railsaathi_micro_decoder_v2",
            },
            f,
        )

    bot_token_id = vocab.index("[bot]")
    print(f"Vocabulary size: {len(vocab)} (bot_token_id={bot_token_id})")

    seqs = vectorizer(samples).numpy()
    X = seqs[:, :-1]
    y = seqs[:, 1:]
    weights = build_loss_mask(X, bot_token_id)

    val_count = max(1, int(len(X) * args.val_split))
    train_count = len(X) - val_count

    X_train, X_val = X[:train_count], X[train_count:]
    y_train, y_val = y[:train_count], y[train_count:]
    w_train, w_val = weights[:train_count], weights[train_count:]

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train, w_train)).shuffle(min(len(X_train), 20000), seed=42).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val, w_val)).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    model = build_causal_mlm(vocab_size=len(vocab), max_len=args.max_len)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss="sparse_categorical_crossentropy",
        weighted_metrics=["accuracy"],
    )

    callbacks = [
        ModelCheckpoint(
            filepath=str(BASE_DIR / "mlm_weights.weights.h5"),
            save_weights_only=True,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=1, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=2, mode="max", restore_best_weights=True, verbose=1),
    ]

    print("Training RailSaathi micro LM...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks, verbose=1)

    best_val_acc = max(history.history.get("val_accuracy", [0.0]))
    total_time = time.time() - start

    print(f"Best val_accuracy: {best_val_acc:.4f}")
    print(f"Total training time: {total_time/60:.2f} minutes")
    print("Saved weights to mlm_weights.weights.h5")


if __name__ == "__main__":
    train()
