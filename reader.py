# %% [markdown]
# # Neural nets using PyTorch
# 
# A neural network is a stack of layers, each applying a linear transformation followed by a nonlinearity. In this notebook we'll build one of the simplest possible networks — a single small hidden layer — and train it by hand to recognize handwritten digits. The goal here is to understand each step of the process, not to squeeze out the highest possible accuracy.

# %%
# Import torch, torch.nn, numpy, matplotlib.pyplot, train_test_split, and load_digits
import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

# %%
# Load the digits dataset into X (features) and y (labels)

digits = load_digits()
x = digits.data
y = digits.target

# %% [markdown]
# The two functions below are just for visualization — one plots a single digit image, the other plots a model's prediction (a bar chart of predicted probabilities) next to the image it was predicting on. Neither is part of the neural net itself.

# %%
def plot_digit(i):
  """Plot a single digit from the dataset."""
  plt.figure(figsize=(1, 1))
  plt.imshow(np.array(x)[i].reshape(8,8), cmap='grey', interpolation='bicubic')
  plt.title(f'Index = {i}\nLabel = {y[i]}', fontsize=12, loc='center', pad=20)
  plt.axis('off')  # Hide the axes
  plt.show()

# %%
def plot_predictions(i):
  predicted_class = np.argmax(predictions[i]) # class with maximum probability
  confidence = np.max(predictions[i]) * 100

  plt.figure(figsize=(8, 4))

  plt.title(f'#{i}. Label: {y_test[i]}. Predicted: {predicted_class}. Confidence: {confidence:.2f}%.')
  plt.xticks([])
  plt.yticks([])

  plt.subplot(1, 2, 1)
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(x_test[i].reshape(8, 8), cmap='grey', interpolation='bicubic')

  plt.subplot(1, 2, 2)
  plt.bar(range(10), predictions[i], color='lightblue')
  plt.bar(predicted_class, predictions[i][predicted_class], color='red')
  plt.bar(y_test[i], predictions[i][y_test[i]], color='green')
  plt.xticks(range(10))
  plt.yticks([])
  plt.tight_layout()
  plt.show()

# %% [markdown]
# ## Train-test split

# %%
# Print the pixel values of digit 1 reshaped to 8x8, and its label
print(x[1].reshape(8,8))
print(y[1])

# %%
# Plot digit index 1 with plot_digit
plot_digit(1)


# %%
# Normalize X by dividing by 16, then split into train/test sets
# (75/25, random_state=0)
x = x/16

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25, random_state = 0)




# %%
# Print the shapes of y, y_train, and y_test
print(y.shape)
print(y_train.shape)
print(y_test.shape)

# %% [markdown]
# ## Neural Net

# %% [markdown]
# A PyTorch model needs float tensors as inputs, with each value between 0 and 1 (same normalization as before). PyTorch has no built-in training loop — we define the architecture (the layers), and then write the training loop ourselves (the forward pass, the loss, and the backward pass). That's more code than a one-line `.fit()` would be, but it makes every step of training visible.
# 
# **Architecture.** A `nn.Sequential` model is just a stack of layers applied one after another — this kind of stack is called a *multilayer perceptron* (MLP). `nn.Linear(in_features, out_features)` is a *fully-connected* layer: every input is connected to every output by a learned weight, plus a learned bias. Between `nn.Linear` layers we need a nonlinear *activation function* (like `nn.ReLU` or `nn.Tanh`) — without one, stacking linear layers would collapse into a single linear layer, no matter how many we stack.
# 
# We'll start with a **tiny** hidden layer, just a handful of neurons, on purpose: the point of this notebook is to see how the pieces fit together, not to maximize accuracy. See the [`torch.nn` documentation](https://pytorch.org/docs/stable/nn.html) for the full list of layers and activation functions, and the [`torch.optim` documentation](https://pytorch.org/docs/stable/optim.html) for other optimizers — both are worth exploring once you're comfortable with the basics here.

# %%
# Convert X_train, y_train, X_test, y_test to torch tensors
# (float32 for X, long for y)

x_train_t = torch.tensor(x_train, dtype = torch.float32)
y_train_t = torch.tensor(y_train, dtype = torch.long)
x_test_t = torch.tensor(x_test, dtype = torch.float32)
y_test_t = torch.tensor(y_test, dtype = torch.long)



# %%
# Define a sequential model: a tiny Linear hidden layer (64->4) with ReLU,
# then a Linear output layer (4->10) producing raw scores.
# Create a CrossEntropyLoss loss function and an SGD optimizer (lr=0.3)

model = nn.Sequential(
    nn.Linear(64,4),
    nn.ReLU(),
    nn.Linear(4,10)
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.3)


# %%
# Train for 300 epochs: each epoch is one full-batch update (zero gradients,
# forward pass on the whole training set, compute loss, backward pass,
# optimizer step). Print loss/accuracy every 30 epochs

epochs = 300

for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(x_train_t)
    loss = loss_fn(outputs, y_train_t)
    loss.backward()
    optimizer.step()
    accuracy = (outputs.argmax(1) == y_train_t).float().mean().item()

    if(epoch + 1) % 30 == 0:
        print(f" Epoch {epoch + 1}/{epochs} - loss : {loss.item():.4f} - accuracy: {accuracy:.4f}")

# %% [markdown]
# The raw output of the model (before softmax) is a list of 10 numbers, one per digit — the largest one is the model's guess. Applying softmax (below) turns these into probabilities that sum to 1, which is easier to interpret and compare.

# %% [markdown]
# ## Making Predictions

# %%
# Compute softmax probabilities on X_test (use torch.no_grad())
with torch.no_grad():
    predictions = torch.softmax(model(x_test_t), dim = 1).numpy()

# %%
# Print the predicted probabilities for the first test example
print(predictions[0])

# %%
# Plot the predicted probabilities for the second test example as a bar chart
plt.bar(range(10), predictions[1])
plt.show()

# %%
# Pick a random index into X_test, then visualize that prediction with plot_predictions

i = np.random.randint(0, len(x_test))
plot_predictions(i)

# %% [markdown]
# ## Weights and biases
# 
# Everything the model learned during training lives in its weights and biases — one weight matrix and one bias vector per `nn.Linear` layer. `model.state_dict()` returns all of them as a dictionary, in the order the layers were defined.

# %%
# Get the model's learned weights and biases (model.state_dict())
weights_and_biases = list(model.state_dict().items())
weights_and_biases


# %% [markdown]
# ## Visualizing learned weights
# 
# Each of our hidden layer's 4 neurons has 64 weights, one per input pixel — so each neuron's weights can be reshaped back into an 8x8 grid and viewed as an image. Since each weight is tied to one specific pixel position, the picture below is a *global* template: a rough sketch of the whole-image pattern that neuron responds to, rather than a local pattern-detector.

# %%
# Get the first Linear layer's weights (model[0].weight.data), then plot
# each of the 4 neurons' weights, reshaped to 8x8, as a small grayscale image
weights = model[0].weight.data
print(weights.shape)

fig, axes = plt.subplots(1,4,  figsize = (6,2))
for i, ax in enumerate(axes):
    ax.imshow(weights[i].reshape(8,8), cmap = 'gray')
    ax.set_title(f"Neuron {i}")
    ax.axis('off')
plt.show()

# %% [markdown]
# **Try it yourself:** increase the hidden layer's size, swap `nn.ReLU` for `nn.Tanh` (or another activation), or swap `SGD` for `Adam`, and re-run training. Compare the accuracy and how quickly the loss drops. You could also split the training data into shuffled mini-batches (`torch.utils.data.DataLoader`) instead of using the whole training set every epoch — the standard approach once a dataset is too large to fit in memory at once. The [`torch.nn`](https://pytorch.org/docs/stable/nn.html) and [`torch.optim`](https://pytorch.org/docs/stable/optim.html) documentation linked above list everything you can try.


