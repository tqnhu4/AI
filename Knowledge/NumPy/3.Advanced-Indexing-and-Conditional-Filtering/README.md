Here's a guide on how to write a Python application that demonstrates advanced indexing and conditional filtering in NumPy.

-----

## 🟢 Level 1: Basic > Advanced Indexing and Conditional Filtering

### Requirement

This section delves into powerful NumPy features for selecting and filtering data based on conditions.

  * **Objective:** Filter data based on a given condition.
  * **Challenge:** Create a random array and filter out numbers greater than 50.
  *  Knowledge: **Boolean indexing**, **masking**.

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

### **Application Guide**

Let's walk through building this Python application step by step.

#### **Step 1: Import NumPy**

As always, start by importing the NumPy library.

```python
import numpy as np
```

#### **Step 2: Create a Random Array**

For this challenge, you need an array with random numbers. NumPy's `np.random.randint()` function is perfect for this. It generates integers within a specified range.

Let's create an array of 10 random integers between 1 and 100.

```python
# Create a random array of 10 integers between 1 and 100
random_array = np.random.randint(1, 101, size=10)
print(f"Original Random Array: {random_array}")
```

#### **Step 3: Define a Condition (Boolean Mask)**

The core of conditional filtering in NumPy is creating a **boolean mask**. A boolean mask is an array of `True` or `False` values, where `True` indicates that the corresponding element in the original array meets the condition, and `False` indicates it doesn't.

For our challenge, the condition is "numbers greater than 50."

```python
# Create a boolean mask: True where elements are > 50, False otherwise
condition = random_array > 50
print(f"Boolean Mask (random_array > 50): {condition}")
```

When you print `condition`, you'll see an array of `True` and `False` values corresponding to each element in `random_array`.

#### **Step 4: Apply the Boolean Mask (Filtering)**

Once you have the boolean mask, you can use it directly to "index" the original array. This technique is called **Boolean indexing**. When you pass a boolean array of the same shape as your original array, NumPy returns only the elements where the mask is `True`.

```python
# Apply the boolean mask to filter the array
filtered_array = random_array[condition]
print(f"Numbers greater than 50: {filtered_array}")
```

#### **Step 5: Assemble the Complete Application**

Combine all the code snippets into a single Python file (e.g., `numpy_filtering.py`). Encapsulating the logic within a function is a good practice.

```python
import numpy as np

def demonstrate_advanced_indexing():
    """
    Demonstrates advanced indexing and conditional filtering in NumPy.
    """
    print("### 3. Advanced Indexing and Conditional Filtering\n")

    print("* **Objective:** Filter data based on a given condition.")
    print("* **Challenge:** Create a random array and filter out numbers greater than 50.")
    print("✅ Knowledge: **Boolean indexing**, **masking**.\n")

    # Step 1: Create a random array
    # We'll create 15 random integers between 1 and 100
    random_array = np.random.randint(1, 101, size=15)
    print(f"Original Random Array (size={random_array.size}): {random_array}")
    print("-" * 30)

    # Step 2: Define a condition (Boolean Mask)
    # Our condition is to find numbers greater than 50
    condition_gt_50 = random_array > 50
    print(f"Boolean Mask (random_array > 50): {condition_gt_50}")
    print("-" * 30)

    # Step 3: Apply the Boolean Mask (Filtering)
    # This selects only the elements where the mask is True
    filtered_numbers = random_array[condition_gt_50]
    print(f"Filtered Numbers (> 50): {filtered_numbers}")
    print("-" * 30 + "\n")

    # You can also combine steps directly:
    print("--- Direct Filtering Example ---")
    another_random_array = np.random.randint(1, 101, size=10)
    print(f"Another Random Array: {another_random_array}")
    # Filter numbers less than 20 directly
    filtered_less_than_20 = another_random_array[another_random_array < 20]
    print(f"Numbers less than 20: {filtered_less_than_20}")
    print("-" * 30 + "\n")


if __name__ == "__main__":
    demonstrate_advanced_indexing()
```

-----

### **How to Run the Application**

1.  **Save:** Save the code above into a file named `numpy_filtering.py`.
2.  **Install NumPy:** If you haven't already, install NumPy using pip:
    ```bash
    pip install numpy
    ```
3.  **Execute:** Open your terminal or command prompt, navigate to the directory where you saved the file, and run:
    ```bash
    python numpy_filtering.py
    ```

Each time you run the script, you'll get a different set of random numbers, but the filtering logic will consistently extract those greater than 50 (or less than 20 in the second example). This demonstrates how powerful and intuitive boolean indexing is for data selection in NumPy.

What's the next concept you'd like to explore in your NumPy learning journey?