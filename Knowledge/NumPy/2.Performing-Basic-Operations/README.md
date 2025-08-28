
-----

## 🟢 Performing Basic Operations

###  Requirement

This section focuses on essential arithmetic operations and common statistical calculations using NumPy arrays.

  * **Objective:** Perform element-wise addition, subtraction, multiplication, and division on arrays.
  * **Challenge:** Calculate the sum, mean, and standard deviation of an array.
  * ✅ Knowledge: `np.sum`, `np.mean`, `np.std`, vectorized operations.

-----

### **Application Guide**

Let's break down how to create the Python application step-by-step.

#### **Step 1: Import NumPy**

First, you'll need to import the NumPy library, which is standard practice for any numerical operations.

```python
import numpy as np
```

#### **Step 2: Create Sample Arrays**

To perform operations, you'll need some sample NumPy arrays. Let's create two 1D arrays for element-wise operations and one for aggregate calculations.

```python
# For element-wise operations
array_a = np.array([10, 20, 30, 40, 50])
array_b = np.array([1, 2, 3, 4, 5])

# For aggregate calculations
data_array = np.array([10, 12, 15, 18, 20, 22, 25])
```

#### **Step 3: Perform Element-wise Arithmetic Operations**

NumPy allows you to perform arithmetic operations directly on arrays. These operations are **vectorized**, meaning they apply to each corresponding element without needing explicit loops, which is much faster and more efficient.

  * **Addition:** `array_a + array_b`
  * **Subtraction:** `array_a - array_b`
  * **Multiplication:** `array_a * array_b`
  * **Division:** `array_a / array_b`



```python
print("--- Element-wise Operations ---")
print(f"Array A: {array_a}")
print(f"Array B: {array_b}")
print(f"Addition (A + B): {array_a + array_b}")
print(f"Subtraction (A - B): {array_a - array_b}")
print(f"Multiplication (A * B): {array_a * array_b}")
print(f"Division (A / B): {array_a / array_b}")
print("-" * 30 + "\n")
```

#### **Step 4: Calculate Aggregate Statistics**

NumPy provides built-in functions to easily compute common statistics on arrays:

  * **Sum:** `np.sum(array)`
  * **Mean (Average):** `np.mean(array)`
  * **Standard Deviation:** `np.std(array)`



```python
print("--- Aggregate Calculations ---")
print(f"Data Array: {data_array}")
print(f"Sum of elements: {np.sum(data_array)}")
print(f"Mean of elements: {np.mean(data_array)}")
print(f"Standard deviation of elements: {np.std(data_array)}")
print("-" * 30 + "\n")
```

#### **Step 5: Assemble the Complete Application**

Combine all the code snippets into a single Python file (e.g., `numpy_operations.py`). You can encapsulate the logic within a function for better organization.

```python
import numpy as np

def perform_basic_numpy_operations():
    """
    Demonstrates basic element-wise arithmetic and aggregate operations
    on NumPy arrays.
    """
    print("### 2. Performing Basic Operations\n")

    print("* **Objective:** Perform element-wise addition, subtraction, multiplication, and division on arrays.")
    print("* **Challenge:** Calculate the sum, mean, and standard deviation of an array.")
    print("✅ Knowledge: `np.sum`, `np.mean`, `np.std`, vectorized operations.\n")

    # For element-wise operations
    array_a = np.array([10, 20, 30, 40, 50])
    array_b = np.array([1, 2, 3, 4, 5])

    # For aggregate calculations
    data_array = np.array([10, 12, 15, 18, 20, 22, 25])

    print("--- Element-wise Operations ---")
    print(f"Array A: {array_a}")
    print(f"Array B: {array_b}")
    print(f"Addition (A + B): {array_a + array_b}")
    print(f"Subtraction (A - B): {array_a - array_b}")
    print(f"Multiplication (A * B): {array_a * array_b}")
    print(f"Division (A / B): {array_a / array_b}")
    print("-" * 30 + "\n")

    print("--- Aggregate Calculations ---")
    print(f"Data Array: {data_array}")
    print(f"Sum of elements: {np.sum(data_array)}")
    print(f"Mean of elements: {np.mean(data_array)}")
    print(f"Standard deviation of elements: {np.std(data_array)}")
    print("-" * 30 + "\n")

if __name__ == "__main__":
    perform_basic_numpy_operations()
```

-----

### Environment Setup

1.  **Clone this repository (if applicable)** or create the folder structure as above.
2.  **Create a virtual environment (recommended)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate  # On Windows

    (venv) $ deactivate #exit
    ```
3.  **Install required libraries**:
    ```bash
    pip install -r requirements.txt
    ```
### **How to Run the Application**

1.  **Save:** Save the code above into a file named `numpy_operations.py`.
2.  **Install NumPy:** If you don't have NumPy installed, open your terminal or command prompt and run:
    ```bash
    pip install numpy
    ```
3.  **Execute:** Navigate to the directory where you saved the file in your terminal and run:
    ```bash
    python numpy_operations.py
    ```

