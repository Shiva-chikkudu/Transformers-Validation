import math
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
# clustering models live in sklearn.cluster, not sklearn.ensemble
from sklearn.cluster import KMeans



class MultiheadAttentionlayer(nn.Module):
    def __init__(self, num_heads, d_model):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(weights, v)
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )

        return self.out_proj(attended), weights

mha = MultiheadAttentionlayer(num_heads=8, d_model=64)
x_with_position = torch.randn(1, 3, 64)
output, weights = mha(x_with_position)

print(output.shape)   # torch.Size([1, 3, 64])
print(weights.shape)  # torch.Size([1, 8, 3, 3])
# sample predictions 
class MultiheadAttentionlayer(nn.Module):
    def __init__(self, num_heads, d_model):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(weights, v)
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )

        return self.out_proj(attended), weights


mha = MultiheadAttentionlayer(num_heads=8, d_model=64)
x_with_position = torch.randn(1, 3, 64)
output, weights = mha(x_with_position)

# Cluster the attention output; KMeans is unsupervised, so it needs no y or z.
x = output.detach().cpu().numpy().reshape(-1, output.size(-1))

x_train, x_test = train_test_split(
    x, test_size=0.33, random_state=56
)

clusterer = KMeans(n_clusters=2, random_state=56, n_init=10)
clusterer.fit(x_train)

predictions = clusterer.predict(x_test)
train_predictions = clusterer.predict(x_train)

print("Output shape:", output.shape)
print("Attention weights shape:", weights.shape)
print("Attention output:\n", output)
print("Test cluster labels:", predictions)
print("Train cluster labels:", train_predictions)
# `v` is local to SelfAttention.forward(), so it is not available here.
# Use the attention output already created in this cell.
y = output.detach().cpu().numpy().reshape(-1, output.size(-1))
z = output.detach().numpy().reshape(-2, output.size(-2))
# Make x, y, and z have the same number of rows
x = output.detach().cpu().numpy().reshape(-1, output.size(-1))
y = output.detach().cpu().numpy().reshape(-1, output.size(-1))
z = output.detach().cpu().numpy().reshape(-1, output.size(-1))  # (3, 64), same as x/y

x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(
    x, y, z, test_size=0.33, random_state=56
)
# KMeans is unsupervised, so we fit only on the feature matrix.
lr = KMeans(n_clusters=2, random_state=56, n_init=10)
lr.fit(x_train)

predictions = lr.predict(x_test)
train_predictions = lr.predict(x_train)

print("Test cluster labels:", predictions)
print("Train cluster labels:", train_predictions)
print("Cluster centers shape:", lr.cluster_centers_.shape)
predictions = lr.predict(x_test)
train_predictions = lr.predict(x_train)
# KMeans is unsupervised; predict() takes only the feature data.
train_predictions = lr.predict(x_train)

print("Output shape:", output.shape)
print("Attention weights shape:", weights.shape)
print("Attention output:\n", output)
print("predcitions\n", predictions)
print("train_predictions\n", train_predictions)


             