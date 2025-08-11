#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adapted from "Neural machine translation with a Transformer and Keras"
https://www.tensorflow.org/text/tutorials/transformer
"""
import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # keep before keras/tf imports

import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras.layers import (
    Dense, Dropout, MultiHeadAttention, Add, LayerNormalization,
    TextVectorization, Embedding
)
from packaging import version

# Keep version check mild unless you truly need newer TF
MIN_TF = "2.15.0"
assert version.parse(tf.__version__) >= version.parse(MIN_TF), (
    f"Requires TensorFlow >= {MIN_TF}, found {tf.__version__}"
)

# -----------------------------
# Positional Encoding
# -----------------------------
def positional_encoding(seq_len, depth):
    d = depth / 2
    positions = np.arange(seq_len)[:, np.newaxis]     # (seq_len, 1)
    d = np.arange(d)[np.newaxis, :] / d               # (1, d)
    angle_rates = 1 / (10000 ** d)                    # (1, d)
    angle_rads = positions * angle_rates              # (seq_len, d)
    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1
    )  # (seq_len, 2*d)
    return tf.cast(pos_encoding, dtype=tf.float32)    # (seq_len, depth)

class PositionalEmbedding(layers.Layer):
    def __init__(self, vocab_size, d_emb):
        super().__init__()
        self.d_emb = d_emb
        self.embedding = Embedding(vocab_size, d_emb, mask_zero=True)
        self.pos_encoding = positional_encoding(seq_len=2048, depth=d_emb)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_emb, tf.float32))
        return x + self.pos_encoding[tf.newaxis, :seq_len, :]

# -----------------------------
# Attention blocks
# -----------------------------
class BaseAttention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = MultiHeadAttention(**kwargs)
        self.layernorm = LayerNormalization()
        self.add = Add()

class GlobalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(query=x, value=x, key=x)
        return self.layernorm(self.add([x, attn_output]))

class CrossAttention(BaseAttention):
    def call(self, x, context):
        attn_output, attn_scores = self.mha(
            query=x, key=context, value=context, return_attention_scores=True
        )
        self.last_attn_scores = attn_scores
        return self.layernorm(self.add([x, attn_output]))

class CausalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(query=x, value=x, key=x, use_causal_mask=True)
        return self.layernorm(self.add([x, attn_output]))

# -----------------------------
# Feed Forward
# -----------------------------
class FeedForward(keras.layers.Layer):
    def __init__(self, d_emb, d_ff, dropout_rate=0.1):
        super().__init__()
        self.seq = keras.Sequential([
            Dense(d_ff, activation='relu'),
            Dense(d_emb),
            Dropout(dropout_rate),
        ])
        self.add = Add()
        self.layer_norm = LayerNormalization()

    def call(self, x):
        return self.layer_norm(self.add([x, self.seq(x)]))

# -----------------------------
# Encoder
# -----------------------------
class EncoderLayer(layers.Layer):
    def __init__(self, *, d_emb, n_heads, d_ff, dropout_rate=0.1):
        super().__init__()
        self.attention = GlobalSelfAttention(num_heads=n_heads, key_dim=d_emb, dropout=dropout_rate)
        self.ffn = FeedForward(d_emb, d_ff, dropout_rate=dropout_rate)

    def call(self, x):
        x = self.attention(x)
        return self.ffn(x)

class Encoder(layers.Layer):
    def __init__(self, *, n_layers, d_emb, n_heads, d_ff, dropout_rate=0.1):
        super().__init__()
        self.n_layers = n_layers
        self.dropout = Dropout(dropout_rate)
        self.enc_layers = [
            EncoderLayer(d_emb=d_emb, n_heads=n_heads, d_ff=d_ff, dropout_rate=dropout_rate)
            for _ in range(n_layers)
        ]

    def call(self, x):
        x = self.dropout(x)
        for layer in self.enc_layers:
            x = layer(x)
        return x

# -----------------------------
# Decoder
# -----------------------------
class DecoderLayer(layers.Layer):
    def __init__(self, *, d_emb, n_heads, d_ff, dropout_rate=0.1):
        super().__init__()
        self.causal_self_attention = CausalSelfAttention(num_heads=n_heads, key_dim=d_emb, dropout=dropout_rate)
        self.cross_attention = CrossAttention(num_heads=n_heads, key_dim=d_emb, dropout=dropout_rate)
        self.ffn = FeedForward(d_emb, d_ff, dropout_rate=dropout_rate)

    def call(self, x, context):
        x = self.causal_self_attention(x)
        x = self.cross_attention(x, context)
        self.last_attn_scores = self.cross_attention.last_attn_scores
        return self.ffn(x)

class Decoder(layers.Layer):
    def __init__(self, *, n_layers, d_emb, n_heads, d_ff, dropout_rate=0.1):
        super().__init__()
        self.dropout = Dropout(dropout_rate)
        self.dec_layers = [
            DecoderLayer(d_emb=d_emb, n_heads=n_heads, d_ff=d_ff, dropout_rate=dropout_rate)
            for _ in range(n_layers)
        ]
        self.last_attn_scores = None

    def call(self, x, context):
        x = self.dropout(x)
        for layer in self.dec_layers:
            x = layer(x, context)
        self.last_attn_scores = self.dec_layers[-1].last_attn_scores
        return x

# -----------------------------
# Transformer Model
# -----------------------------
class Transformer(keras.Model):
    def __init__(self, *, n_layers, d_emb, n_heads, d_ff, src_vocab_size, tgt_vocab_size, dropout_rate=0.1):
        super().__init__()
        self.src_pos_embedding = PositionalEmbedding(vocab_size=src_vocab_size, d_emb=d_emb)
        self.tgt_pos_embedding = PositionalEmbedding(vocab_size=tgt_vocab_size, d_emb=d_emb)
        self.encoder = Encoder(n_layers=n_layers, d_emb=d_emb, n_heads=n_heads, d_ff=d_ff, dropout_rate=dropout_rate)
        self.decoder = Decoder(n_layers=n_layers, d_emb=d_emb, n_heads=n_heads, d_ff=d_ff, dropout_rate=dropout_rate)
        self.final_layer = Dense(tgt_vocab_size)

    def call(self, inputs):
        src, tgt = inputs
        src_emb = self.src_pos_embedding(src)
        tgt_emb = self.tgt_pos_embedding(tgt)
        context = self.encoder(src_emb)
        x = self.decoder(tgt_emb, context)
        logits = self.final_layer(x)
        try:
            del logits._keras_mask
        except AttributeError:
            pass
        return logits

# -----------------------------
# Optional quick sanity check
# -----------------------------
if __name__ == "__main__":
    vocab_size = 1000
    seq_len = 10
    d_emb = 128

    en = ['good morning', 'how are you']
    sp = ['buen dia', 'como estas']

    src_vectorizer = TextVectorization(max_tokens=vocab_size, output_mode="int", output_sequence_length=seq_len)
    tgt_vectorizer = TextVectorization(max_tokens=vocab_size, output_mode="int", output_sequence_length=seq_len)
    src_vectorizer.adapt(en)
    tgt_vectorizer.adapt(sp)

    seq_en = src_vectorizer(en)
    seq_sp = tgt_vectorizer(sp)

    emb_en = PositionalEmbedding(vocab_size=vocab_size, d_emb=d_emb)(seq_en)
    emb_sp = PositionalEmbedding(vocab_size=vocab_size, d_emb=d_emb)(seq_sp)

    n_heads = 4
    d_ff = 512

    encoder = Encoder(n_layers=2, d_emb=d_emb, n_heads=n_heads, d_ff=d_ff)
    decoder = Decoder(n_layers=2, d_emb=d_emb, n_heads=n_heads, d_ff=d_ff)
    out_enc = encoder(emb_en, training=False)
    out_dec = decoder(emb_sp, out_enc)
    print("Encoder out:", out_enc.shape, "Decoder out:", out_dec.shape)
