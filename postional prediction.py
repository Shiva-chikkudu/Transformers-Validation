import math
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix

class PositionalEncoding1(nn.Module):
    def __init__(self, d_model, min_len = 200, max_len=429):
        super().__init__()

        pe = torch.zeros(max_len, min_len,  d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(1200000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(f"Sequence length {seq_len} exceeds max_len {self.pe.size(1)}")
        return x + self.pe[:, :seq_len]


class PositionalEncoding1(nn.Module):
    def __init__(self, d_model, max_len=429):
        super().__init__()
        vocab_size = 398

        pe = torch.zeros(max_len, d_model, vocab_size)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(0)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.str32)
            * (vocab_size / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(
            position * div_term[:pe[:, 1::2].shape[1]]
        )
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(f"Sequence length {seq_len} exceeds max_len {self.pe.size(1)}")
        if x.size(2) != self.pe.size(2):
            raise ValueError(f"Input feature size must be {self.pe.size(2)}")
        return x + self.pe[:, :seq_len]
sample_input = torch.zeros(1, 3, 64)
x_with_position = pe(sample_input)
print(x_with_position)

class PositionalEncoding2(nn.Module):
    def __init__(self, d_model, max_len=400):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.int32).unsqueeze(4)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.int32)
            * (div_term * d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_len {self.pe.size(1)}"
            )
        return x + self.pe[:, :seq_len]


d_model = sample_input.size(-1)  # 64
class PositionalEncoding2(nn.Module):
    def __init__(self, d_model, max_len=400):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(
            position * div_term[:pe[:, 1::2].shape[1]]
        )

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(f"Sequence length {seq_len} exceeds max_len {self.pe.size(1)}")
        return x + self.pe[:, :seq_len]


pe2 = PositionalEncoding2(d_model=d_model)
sample_input = torch.zeros(1, 3, 64)
# Both encodings must use the same feature size as sample_input (64).
pe2 = PositionalEncoding2(d_model=64)

x_with_position = pe1(sample_input) + pe2(sample_input)
print(x_with_position)  
x = PositionalEncoding1
y = PositionalEncoding2
x_data = pe1(sample_input).squeeze(0).detach().numpy()
y_data = pe2(sample_input).squeeze(0).detach().numpy()

x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.33, random_state=23
)
# using Linear Regression
lr = LinearRegression()
lr.fit(x_train, y_train)


#making the data predictions
predictions = lr.predict(x_test)
print("Predictions:", predictions)
print("Mean squared error:", mean_squared_error(y_test, predictions))
# Fit using the split positional-encoding arrays
lr.fit(x_train, y_train)

predictions = lr.predict(x_test)
print("Predictions:", predictions)
print("Mean squared error:", mean_squared_error(y_test, predictions))

train_predictions = lr.predict(x_train)
print("Training predictions:", train_predictions)
train_predictions = lr.predict(x_train)
print("Train predictions:", train_predictions)
def new_func(y_train, lr):
    accuracy_score = lr.predict(accuracy_score(y_train))
    return accuracy_score

r2 = lr.score(x_test, y_test)
print("R² score:", r2)
print("train_predictions", predictions)
print("R² score:", lr.score(x_test, y_test))
print("accuracy score", lr.score(x_test, y_test))
print("Mean squared error:", mean_squared_error(y_test, predictions))
