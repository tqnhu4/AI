
-----

## MNIST Digit Classification: A Complete Project Guide

**Goal:** Build a simple neural network to recognize handwritten digits from 0–9.

**Skills You'll Learn:**

  * **Loading and Preprocessing Data with `torchvision.datasets`**: How to effectively manage and prepare your dataset for training.
  * **Building Models with `nn.Sequential` or `nn.Module`**: Understanding PyTorch's fundamental building blocks for creating neural networks.
  * **Using Loss Functions and Optimizers**: Implementing key components for training your model and minimizing errors.
  * **Plotting Loss/Accuracy**: Visualizing your model's performance during training to identify trends and areas for improvement.

-----

### Step 1: Setting Up Your Environment

First, you need to set up your Python environment and install the necessary libraries.

1.  **Install Python:** If you don't have Python installed, download it from [python.org](https://www.python.org/downloads/). It's recommended to use Python 3.8 or newer.
2.  **Create a Virtual Environment (Recommended):** This isolates your project's dependencies from other Python projects.
    ```bash
    python -m venv mnist_env
    ```
3.  **Activate the Virtual Environment:**
      * **On Windows:**
        ```bash
        mnist_env\Scripts\activate
        ```
      * **On macOS/Linux:**
        ```bash
        source mnist_env/bin/activate
        ```
4.  **Install Required Libraries:**
    ```bash
    pip install torch torchvision matplotlib scikit-learn
    ```
      * **`torch`**: The main PyTorch library.
      * **`torchvision`**: Provides datasets, models, and image transformations specifically for computer vision.
      * **`matplotlib`**: For plotting graphs (loss/accuracy).
      * **`scikit-learn`**: For generating the confusion matrix (useful for testing).

-----

### Step 2: Project Code

Create a Python file (e.g., `mnist_classifier.py`) and add the following code step-by-step.

#### 1\. Imports and Device Setup

Start by importing all necessary libraries and setting up the device (CPU or GPU). If you have a CUDA-enabled GPU, PyTorch can leverage it for faster training.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

# Set device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

#### 2\. Load and Preprocess Data

The MNIST dataset consists of 28x28 grayscale images of handwritten digits. We'll use `torchvision.datasets.MNIST` to download and load it. The `transforms.Compose` pipeline will:

  * Convert images to PyTorch tensors (`transforms.ToTensor()`).
  * Normalize the pixel values (`transforms.Normalize()`) using the mean and standard deviation specific to the MNIST dataset.



```python
# Define a transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # Mean and Std Dev for MNIST
])

# Download and load the training data
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Download and load the test data
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of test samples: {len(test_dataset)}")
```

#### 3\. Define the Neural Network

You have two main options here: a simple **Fully Connected Network (FCN)** using `nn.Sequential` or a more robust **Convolutional Neural Network (CNN)**. For beginners, start with the FCN, then try the CNN as an extension.

**Option A: Fully Connected Network (FCN)**

This network flattens the 28x28 image into a 784-element vector and processes it through linear layers.

```python
class SimpleFCN(nn.Module):
    def __init__(self):
        super(SimpleFCN, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(), # Flattens the 28x28 image into a 784-element vector
            nn.Linear(28 * 28, 128), # Input layer (784 features) to hidden layer (128 neurons)
            nn.ReLU(),           # Activation function
            nn.Linear(128, 64),  # Hidden layer to another hidden layer
            nn.ReLU(),
            nn.Linear(64, 10)    # Output layer (10 classes for digits 0-9)
        )

    def forward(self, x):
        return self.network(x)

# Initialize the model and move it to the configured device (CPU/GPU)
model = SimpleFCN().to(device)
print("\n--- Simple Fully Connected Network (FCN) Architecture ---")
print(model)
```

**Option B: Convolutional Neural Network (CNN) - Extension**

CNNs are generally better for image data. They use convolutional layers to extract features and pooling layers to reduce dimensionality.

```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), # 1 input channel (grayscale), 32 output channels
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Reduces image size by half (28x28 -> 14x14)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # Reduces image size by half (14x14 -> 7x7)
        )
        # Calculate the input size for the fully connected layer
        # After two MaxPool2d layers on a 28x28 image, the size becomes 7x7.
        # So, 64 channels * 7 * 7 = 3136 features
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10) # Output layer for 10 digits
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 64 * 7 * 7) # Flatten the output for the fully connected layer
        x = self.fc_layers(x)
        return x

# UNCOMMENT THE LINE BELOW TO USE THE CNN MODEL INSTEAD OF FCN
# model = CNN().to(device)
# print("\n--- Convolutional Neural Network (CNN) Architecture ---")
# print(model)
```

#### 4\. Define Loss Function and Optimizer

  * **Loss Function (`nn.CrossEntropyLoss`):** Suitable for multi-class classification problems. It combines `LogSoftmax` and `NLLLoss` in one.
  * **Optimizer (`torch.optim.Adam`):** A popular choice for its efficiency and good performance in many scenarios. `lr` (learning rate) controls how much the model's weights are adjusted during training.



```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

#### 5\. Training Loop

This is where the model learns. We'll iterate through the data for a set number of `epochs`.

  * `model.train()`: Sets the model to training mode (important for layers like Dropout or BatchNorm).
  * `optimizer.zero_grad()`: Clears gradients from the previous iteration.
  * `outputs = model(images)`: Performs a forward pass to get predictions.
  * `loss = criterion(outputs, labels)`: Calculates the loss based on predictions and true labels.
  * `loss.backward()`: Computes gradients (backpropagation).
  * `optimizer.step()`: Updates model weights using the gradients.



```python
num_epochs = 10
train_losses = []
train_accuracies = []

print(f"\n--- Starting Training for {num_epochs} Epochs ---")
for epoch in range(num_epochs):
    model.train() # Set the model to training mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device) # Move data to device

        optimizer.zero_grad() # Clear previous gradients
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward() # Backpropagation
        optimizer.step() # Update weights

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

        if (batch_idx + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_accuracy = correct_predictions / total_samples
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

    print(f'Epoch {epoch+1} Complete: Average Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')

print("--- Training Finished ---")
```

#### 6\. Evaluation Loop (Testing the Model)

After training, evaluate the model's performance on the unseen test set.

  * `model.eval()`: Sets the model to evaluation mode (disables dropout, etc.).
  * `with torch.no_grad()`: Disables gradient calculations, saving memory and speeding up computation as we are no longer training.



```python
print("\n--- Evaluating Model on Test Set ---")
model.eval() # Set the model to evaluation mode
test_loss = 0.0
correct_predictions = 0
total_samples = 0
all_labels = []
all_predictions = []

with torch.no_grad(): # Disable gradient calculations during evaluation
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device) # Move data to device
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)

        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

test_loss /= len(test_loader.dataset)
test_accuracy = correct_predictions / total_samples

print(f'\nTest Results: Average Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}')
```

#### 7\. Plotting Loss and Accuracy

Visualizing training progress helps understand if the model is learning effectively.

```python
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy over Epochs')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

-----

### Step 3: Running Your Project

1.  **Save the Code:** Save the entire code block above into a file named `mnist_classifier.py`.
2.  **Open Terminal/Command Prompt:** Navigate to the directory where you saved `mnist_classifier.py`.
3.  **Activate Virtual Environment:**
      * **On Windows:** `mnist_env\Scripts\activate`
      * **On macOS/Linux:** `source mnist_env/bin/activate`
4.  **Run the Script:**
    ```bash
    python mnist_classifier.py
    ```

You will see the training progress printed in the console. After training, the test results will be displayed, and two plots (training loss and accuracy) will pop up.

-----

### Step 4: Testing and Further Analysis

Beyond the basic evaluation, here are ways to test and understand your model more deeply. Add these code snippets to your `mnist_classifier.py` file after the evaluation loop (Step 6).

#### 1\. Confusion Matrix

A confusion matrix visualizes the performance of a classification model, showing where it made correct and incorrect predictions.

```python
print("\n--- Generating Confusion Matrix ---")
# Create confusion matrix
cm = confusion_matrix(all_labels, all_predictions)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
print("Confusion Matrix plot displayed.")
```

#### 2\. Single Image Prediction (Optional)

This allows you to test the model with an individual image. You would need to provide an image file (e.g., a handwritten digit you drew). Make sure the image is preprocessed in the same way as the MNIST dataset.

```python
from PIL import Image

# --- Function to preprocess a single image for prediction ---
def preprocess_custom_image(image_path):
    image = Image.open(image_path).convert('L') # Convert to grayscale
    image = image.resize((28, 28)) # Resize to 28x28
    image_array = np.array(image) # Convert to numpy array
    # Invert colors if necessary (MNIST is white digit on black background)
    # Check your custom image: if it's black digit on white background, invert it.
    # image_array = 255 - image_array # Uncomment this line if your background is white

    image_tensor = transforms.ToTensor()(image_array).unsqueeze(0) # Add batch dimension
    # Apply the same normalization as used for MNIST dataset
    image_tensor = transforms.Normalize((0.1307,), (0.3081,))(image_tensor)
    return image_tensor.to(device)

# --- Example of single image prediction ---
# You need to replace 'path/to/your/digit_image.png' with an actual path to an image file.
# For example, draw a digit '3' in MS Paint, save it as 'my_digit_3.png'.
custom_image_path = 'path/to/your/digit_image.png' # <<< CHANGE THIS PATH

print(f"\n--- Testing Single Image Prediction ---")
try:
    input_image_tensor = preprocess_custom_image(custom_image_path)

    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        output = model(input_image_tensor)
        probabilities = torch.softmax(output, dim=1) # Get probabilities for each class
        predicted_class = torch.argmax(probabilities, dim=1).item() # Get the class with highest probability

    print(f"The model predicts: {predicted_class}")
    # print(f"Probabilities for each digit: {[f'{p:.4f}' for p in probabilities.squeeze().tolist()]}")

except FileNotFoundError:
    print(f"Error: Image file not found at {custom_image_path}. Please create or verify the path.")
except Exception as e:
    print(f"An error occurred during single image prediction: {e}")

# This will keep the plot windows open until you close them manually
plt.show()
```

