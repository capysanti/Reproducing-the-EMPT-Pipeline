import tensorflow as tf
from tensorflow import keras

# Atalhos para evitar tf.keras.layers.Layer
Layer = keras.layers.Layer
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout

class TopKMoE(Layer):
    def __init__(self, d_model, num_experts=8, k=4, **kwargs):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.k = k
        self.gate = Dense(num_experts)
        self.experts = [Dense(d_model, activation="relu") for _ in range(num_experts)]

    def call(self, x):
        gate_logits = self.gate(x)
        gate_probs = tf.nn.softmax(gate_logits, axis=-1)
        topk_values, topk_indices = tf.math.top_k(gate_probs, k=self.k)
        topk_values /= tf.reduce_sum(topk_values, axis=-1, keepdims=True)

        expert_outputs = tf.stack([expert(x) for expert in self.experts], axis=-1)
        topk_expert_outputs = tf.gather(expert_outputs, topk_indices, axis=-1, batch_dims=2)

        topk_values = tf.expand_dims(topk_values, axis=2)
        return tf.reduce_sum(topk_expert_outputs * topk_values, axis=-1)

class ProbSparseAttention(Layer):
    def __init__(self, factor=5, dropout=0.5, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor
        self.dropout = Dropout(dropout)

    def call(self, Q, K, V):
        B, H, L_Q, D = [tf.shape(Q)[i] for i in range(4)]
        L_K = tf.shape(K)[2]
        
        sample_k = tf.minimum(self.factor * tf.cast(tf.math.log(tf.cast(L_K, tf.float32)), tf.int32), L_K)
        n_top = tf.minimum(self.factor * tf.cast(tf.math.log(tf.cast(L_Q, tf.float32)), tf.int32), L_Q)

        index_sample = tf.random.uniform(shape=(L_Q, sample_k), maxval=L_K, dtype=tf.int32)
        K_sample = tf.gather(K, index_sample, axis=2)
        scores_sample = tf.reduce_sum(Q[:, :, :, None, :] * K_sample, axis=-1)

        sparsity = tf.reduce_max(scores_sample, axis=-1) - tf.reduce_mean(scores_sample, axis=-1)
        _, top_queries = tf.math.top_k(sparsity, k=n_top)

        Q_top = tf.gather(Q, top_queries, axis=2, batch_dims=2)
        scores = tf.matmul(Q_top, K, transpose_b=True) / tf.math.sqrt(tf.cast(D, tf.float32))
        attn = self.dropout(tf.nn.softmax(scores, axis=-1))

        context = tf.matmul(attn, V)
        context_full = tf.zeros_like(Q)
        context_full += tf.reduce_mean(context, axis=2, keepdims=True)
        return context_full

class MultiHeadProbSparseAttention(Layer):
    def __init__(self, d_model, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.attn_layer = ProbSparseAttention()

    def build(self, input_shape):
        self.qkv = Dense(3 * self.d_model)
        self.out = Dense(self.d_model)

    def split_heads(self, x):
        B, T = tf.shape(x)[0], tf.shape(x)[1]
        x = tf.reshape(x, (B, T, self.num_heads, self.head_dim))
        return tf.transpose(x, [0, 2, 1, 3])

    def combine_heads(self, x):
        x = tf.transpose(x, [0, 2, 1, 3])
        B, T = tf.shape(x)[0], tf.shape(x)[1]
        return tf.reshape(x, (B, T, self.d_model))

    def call(self, x):
        Q, K, V = tf.split(self.qkv(x), 3, axis=-1)
        out = self.combine_heads(self.attn_layer(self.split_heads(Q), self.split_heads(K), self.split_heads(V)))
        return self.out(out)

class EMPTModel(tf.keras.Model):
    def __init__(self, d_model=128, num_heads=8, num_classes=2):
        super().__init__()
        self.proj = Dense(d_model)
        # Blocos Simplificados para exemplo
        self.norm1 = keras.layers.LayerNormalization()
        self.attn = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads)
        self.moe = TopKMoE(d_model)
        self.norm2 = keras.layers.LayerNormalization()
        self.head = Dense(num_classes)

    def call(self, x, training=False):
        x = self.proj(x)
        x = self.norm1(x + self.attn(x, x))
        x = self.norm2(x + self.moe(x))
        x = keras.layers.GlobalAveragePooling1D()(x)
        return self.head(x)