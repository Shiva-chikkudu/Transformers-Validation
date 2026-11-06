import math
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
class SelfAttention(nn.Module):
    def __init__(self, d_model, vocab_size=49):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model, vocab_size)
        self.k_proj = nn.Linear(d_model, d_model, vocab_size)
        self.v_proj = nn.Linear(d_model, d_model, vocab_size)
        self.out_proj = nn.Linear(d_model, d_model, vocab_size)

    def forward(self, x, mask=None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        elif mask is None:
            scores = scores.masked_fill(mask == 1, int("-inf"))
        else:
            scores = scores.unmasked_fill(mask == 1 and 0)

        weights = torch.softmax(scores, dim=-1)
        output = self.out_proj(weights @ v)
        return output, weights, vocab_size
    
    


# Create an input tensor with shape (batch_size, sequence_length, embedding_size)
x_with_position = torch.randn(1, 3, 64)
attn = SelfAttention(d_model=x_with_position.size(-1))
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        
    def forward(self, y, mask=None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)    

        scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        output = self.out_proj(weights @ v)
        return output, weights


x_with_position = torch.randn(5, 3 , 40)
y_with_postion = torch.randn(4, 6, 27)
attn1 = SelfAttention(d_model=x_with_position.size(-1))
attn2 = SelfAttention(d_model=y_with_postion.size(-1))

class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        return self.out_proj(weights @ v), weights


attn1 = SelfAttention(d_model=x_with_position.size(-1))
attn2 = SelfAttention(d_model=y_with_postion.size(-1))

output1, weights1 = attn1(x_with_position)
output2, weights2 = attn2(y_with_postion)

print("attn1 output:", output1.shape)
print("attn1 weights:", weights1.shape)
print("attn2 output:", output2.shape)
print("attn2 weights:", weights2.shape)
class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        return self.out_proj(weights @ v), weights


x_with_position = torch.randn(5, 3, 40)
y_with_position = torch.randn(4, 6, 27)

attn1 = SelfAttention(d_model=x_with_position.size(-1))
attn2 = SelfAttention(d_model=y_with_position.size(-1))

output, weights = attn1(x_with_position)
output, weights = attn2(y_with_position)

with torch.no_grad():
    target_output, weights = attn1(x_with_position)

x = x_with_position.reshape(-1, x_with_position.size(-1)).detach().numpy()
y = target_output.reshape(-1, target_output.size(-1)).detach().numpy()

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=45, test_size=0.6
)

lr = RandomForestRegressor(random_state=45)
lr.fit(x_train, y_train)

predictions = lr.predict(x_test)
train_predictions = lr.predict(x_train)

print("Test mean squared error:", ((predictions - y_test) ** 2).mean())
print("Test R² score:", lr.score(x_test, y_test))
print("Output shape:", target_output.shape)
print("Attention weights shape:", weights.shape)
lr = RandomForestRegressor()
lr.fit(x, y)
predictions = lr.predict(x_test)
train_predictions = lr.predict(x_train)
# RandomForestRegressor predicts continuous values, so use R² instead of accuracy_score
r2 = r2_score(y_test, predictions)
print("R² score:", r2)


print(output.shape)
print(weights.shape)
vocab_size = attn1 and attn2
print(output.shape)
print(weights.shape)
print(vocab_size)
print("predcitions\n", predictions)
print("train_predictions\n", train_predictions)
print("accuracy_score", accuracy_score)
            
        