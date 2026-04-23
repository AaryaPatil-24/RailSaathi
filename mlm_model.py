import tensorflow as tf
from tensorflow.keras import layers

class PositionalEmbedding(layers.Layer):
    def __init__(self, vocab_size, d_model, max_len=128):
        super().__init__()
        self.d_model = d_model
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=d_model)
        self.pos_emb = layers.Embedding(input_dim=max_len, output_dim=d_model)
        self.dropout = layers.Dropout(0.1)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        x = x + positions
        return self.dropout(x)

class TransformerDecoderBlock(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = tf.keras.Sequential([
            layers.Dense(dense_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.dropout_1 = layers.Dropout(0.1)
        self.dropout_2 = layers.Dropout(0.1)

    def call(self, inputs):
        # Create a causal mask for the current batch
        # This prevents the attention mechanism from looking at future tokens
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        
        # Lower triangular matrix
        causal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len), dtype=tf.bool), -1, 0)
        # Reshape to (batch_size, num_heads, seq_len, seq_len) expected by MultiHeadAttention
        # In Keras, we can just pass a (seq_len, seq_len) mask and it broadcasts
        causal_mask = tf.cast(causal_mask, dtype=tf.int32)

        attention_output = self.attention(
            query=inputs, value=inputs, key=inputs, use_causal_mask=True
        )
        attention_output = self.dropout_1(attention_output)
        
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        proj_output = self.dropout_2(proj_output)
        
        return self.layernorm_2(proj_input + proj_output)

def build_causal_mlm(vocab_size, max_len=64):
    """
    Builds a custom Causal Transformer Decoder (Miniature GPT) 
    for autoregressive language modeling.
    """
    embed_dim = 256
    dense_dim = 512
    num_heads = 4
    
    inputs = layers.Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    
    # 1. Embeddings
    x = PositionalEmbedding(vocab_size, embed_dim, max_len=max_len)(inputs)
    
    # 2. Transformer Decoder Blocks (with causal masking)
    x = TransformerDecoderBlock(embed_dim, dense_dim, num_heads)(x)
    x = TransformerDecoderBlock(embed_dim, dense_dim, num_heads)(x)
    x = TransformerDecoderBlock(embed_dim, dense_dim, num_heads)(x)
    x = TransformerDecoderBlock(embed_dim, dense_dim, num_heads)(x)
    
    # 3. Next-Token Output Head
    # Outputs probabilities over the entire vocabulary for every time step
    outputs = layers.Dense(vocab_size, activation="softmax", name="next_token")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
