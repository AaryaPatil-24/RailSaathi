import tensorflow as tf
from tensorflow.keras import layers


class PositionalEmbedding(layers.Layer):
    def __init__(self, vocab_size, d_model, max_len=128, dropout=0.1):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=d_model)
        self.pos_emb = layers.Embedding(input_dim=max_len, output_dim=d_model)
        self.dropout = layers.Dropout(dropout)

    def call(self, x):
        seq_len = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        x = self.token_emb(x) + self.pos_emb(positions)
        return self.dropout(x)


class TransformerDecoderBlock(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout,
        )
        self.dense_proj = tf.keras.Sequential(
            [
                layers.Dense(dense_dim, activation="relu"),
                layers.Dropout(dropout),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm_2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(dropout)

    def call(self, inputs):
        attention_output = self.attention(
            query=inputs,
            value=inputs,
            key=inputs,
            use_causal_mask=True,
        )
        x = self.layernorm_1(inputs + self.dropout(attention_output))
        proj_output = self.dense_proj(x)
        return self.layernorm_2(x + self.dropout(proj_output))


def build_causal_mlm(
    vocab_size,
    max_len=48,
    embed_dim=128,
    dense_dim=256,
    num_heads=4,
    num_layers=2,
    dropout=0.1,
):
    """Builds a lightweight causal decoder suitable for CPU training."""
    inputs = layers.Input(shape=(max_len,), dtype=tf.int32, name="input_ids")

    x = PositionalEmbedding(vocab_size, embed_dim, max_len=max_len, dropout=dropout)(inputs)

    for _ in range(num_layers):
        x = TransformerDecoderBlock(
            embed_dim=embed_dim,
            dense_dim=dense_dim,
            num_heads=num_heads,
            dropout=dropout,
        )(x)

    outputs = layers.Dense(vocab_size, activation="softmax", name="next_token")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)
