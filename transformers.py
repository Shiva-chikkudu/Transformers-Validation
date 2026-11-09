import math
import torch
import torch.nn as nn
from sklearn.neural_network import MLPRegressor as nonLinearRegression
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class Transformerexternallayer(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.3):
        super(). __init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Droput(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        
    def forward(self, x):
        return self.net(x)    
    
class TransformerInternallayer(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.5):
        super(). __init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        
        
    def forward(self, x):
        return self.net(x)
    
    
class TransformerMainblock(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout=0.7):
        super().__init__()
        self.attention = MultiheadAttentionlayer(num_heads=num_heads, d_model=d_model)
        self.ffn = Transformerexternallayer(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        attn_out, attn_weights = self.attention(x, mask=mask)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x, attn_weights  
    
    
    def backward(self, x, mask=None):
        attn_in, attn_weights = self.attention(x, mask=mask)
        x = self.norm1(x + attn_in)
        ffn_in = self.ffn(x)
        x = self.norm2(x - ffn_in)
        return x, attn_weights
    
vocab_size = 49

class TransformerMechanism(nn.Module):
    def __init__(self, d_model, d_ff, vocab_size, dropout=0.1):
        super().__init__()
        if d_model % 4 != 0:
            raise ValueError("d_model must be divisible by 4.")

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = TransformerMainblock(
            d_model=d_model,
            d_ff=d_ff,
            num_heads=4,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids, mask=None):
        x = self.dropout(self.embedding(token_ids))
        x, attn_weights = self.encoder(x, mask=mask)
        return self.output_projection(x), attn_weights


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
        reshape_heads = lambda t: t.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        q = reshape_heads(self.q_proj(x))
        k = reshape_heads(self.k_proj(x))
        v = reshape_heads(self.v_proj(x))

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        attended = (weights @ v).transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        return self.out_proj(attended), weights


# Correct the Dropout typo in Transformerexternallayer.
class Transformerexternallayer(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


model = TransformerMechanism(d_model=64, d_ff=128, vocab_size=vocab_size)

seq_len = 6
token_ids = torch.randint(0, vocab_size, (1, seq_len))
causal_mask = torch.tril(torch.ones(seq_len, seq_len))
output, weights = model(token_ids, mask=causal_mask)

# Cluster the model's logits; KMeans provides cluster labels and centers.
x = output.detach().cpu().numpy().reshape(-1, output.size(-1))
x_train, x_test = train_test_split(x, test_size=0.33, random_state=56)

lr = KMeans(n_clusters=2, random_state=56, n_init=10)
lr.fit(x_train)

predictions = lr.predict(x_test)
train_predictions = lr.predict(x_train)
print("Test cluster labels:", predictions)
print("Train regressor labels:", train_predictions)
print("regressor centers shape:", lr.cluster_centers_.shape)
predictions = lr.predict(x_test)
train_predictions = lr.predict(x_train)
train_predictions = lr.predict(x_train)

print("Output shape:", output.shape)
print("Attention weights shape:", weights.shape)
print("Attention output:\n", output)
print("predcitions\n", predictions)
print("train_predictions\n", train_predictions)

seq_len = 6
# lower Triangular mask:token i can see tokens 0..i
causal_mask = torch.tril(torch.ones(seq_len, seq_len))
print(causal_mask)  

plt.imshow(causal_mask.cpu().numpy(), cmap="Blues", vmin=0, vmax=1)
plt.xlabel("Key position")
plt.ylabel("Query position")
plt.title("Causal Mask", color="red")
plt.colorbar(label="Allowed attention")
plt.show()
plt.xlabel(x)
allowed_keys = causal_mask.sum(dim=1).cpu().numpy()
positions = range(seq_len)

plt.bar(positions, allowed_keys, label="Allowed keys")
plt.xlabel("Query position")
plt.ylabel("Number of allowed key positions")
plt.title("Barplot of Causal Mask", color="red", fontweight="bold")
plt.xticks(positions)
plt.legend()
plt.show()   

    
    