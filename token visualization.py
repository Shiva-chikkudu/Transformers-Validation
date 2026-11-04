import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

vocab_size = 200
d_model = 66

embedding = nn.Embedding(vocab_size, d_model)
token1 = torch.tensor([[1344, 4566, 5688]])
token2 = torch.tensor([[134, 456, 568]])
import torch.nn as nn

d_model = 66
token1 = torch.tensor([[1344, 4566, 5688]])
token2 = torch.tensor([[134, 456, 568]])

# Embedding size must be greater than every token ID.
vocab_size = max(token1.max().item(), token2.max().item()) + 1
embedding = nn.Embedding(vocab_size, d_model)

print("Token shapes:", token1.shape, token2.shape)
print("Embedding shape:", embedding(token1).shape)

X = embedding(token1).squeeze(0).detach().numpy()
Y = embedding(token2).squeeze(0).detach().numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.33, random_state=44
)

lr = LinearRegression()
lr.fit(X_train, y_train)

predictions = lr.predict(X_test)
print("Actual targets:\n", y_test)
print("Predictions:\n", predictions)
print("Mean squared error:", mean_squared_error(y_test, predictions))
print(token1,token2.shape)
print(X.shape)
X = vocab_size
Y = d_model
X = embedding(token1).squeeze(0).detach().numpy()
Y = embedding(token2).squeeze(0).detach().numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.6, random_state=44
)
lr = LinearRegression()
import torch.nn as nn

vocab_size = 6000  # Must exceed the largest token ID
d_model = 66

embedding = nn.Embedding(vocab_size, d_model)
token1 = torch.tensor([[1344, 4566, 5688]])
token2 = torch.tensor([[134, 456, 568]])

# Use corresponding token embeddings as input and target examples
X = embedding(token1).squeeze(0).detach().numpy()
Y = embedding(token2).squeeze(0).detach().numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.33, random_state=44
)

lr = LinearRegression()
lr.fit(X_train, y_train)

predictions = lr.predict(X_test)
print("Actual targets:\n", y_test)
print("Predictions:\n", predictions)
print("Mean squared error:", mean_squared_error(y_test, predictions))

train_predictions = lr.predict(X_train)
print("Training predictions:\n", train_predictions)
print("Actual test targets:\n", y_test)
print("Test predictions:\n", predictions)
print(y_test, X_train)
print("accuracy_score(Y_test, X_train):", accuracy_score)
plt.plot(y_test.flatten(), color="red", label="Actual")
plt.plot(predictions.flatten(), color="blue", linestyle="--", label="Predicted")
plt.legend()
plt.title("Representation of Tokens and Embeddings")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
