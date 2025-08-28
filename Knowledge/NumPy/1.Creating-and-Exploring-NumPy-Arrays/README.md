### Requirement

* **Goal:** Get familiar with `np.array`, the `.shape`, `.ndim`, `.dtype`, etc. properties.

* **Challenge:** Create a 1D, 2D, 3D array and print out the details.

* Knowledge: Initialization, access, slicing.

### Environment Setup

1.  **Clone this repository (if applicable)** or create the folder structure as above.
2.  **Create a virtual environment (recommended)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate  # On Windows
    ```
3.  **Install required libraries**:
    ```bash
    pip install -r requirements.txt
    ```

### Library Requirements

Make sure you have the following libraries installed:

  * `numpy`

You can install them by running the following command in your terminal or Anaconda Prompt:
`pip install numpy`

```python
import numpy as np

def explore_numpy_arrays():
    """
    Demonstrates creating and exploring 1D, 2D, and 3D NumPy arrays.
    """
    print("## 🟢 Level 1: Basic\n")
    print("### 1. Creating and Exploring NumPy Arrays\n")

    print("* **Objective:** Get familiar with `np.array`, and attributes like `.shape`, `.ndim`, `.dtype`, etc.\n")
    print("* **Challenge:** Create 1D, 2D, and 3D arrays and print their detailed information.\n")
    print("✅ Knowledge: Initialization, access, slicing.\n")

    # --- 1D Array ---
    print("--- Exploring a 1D Array ---")
    array_1d = np.array([1, 2, 3, 4, 5])
    print(f"Array: {array_1d}")
    print(f"Shape: {array_1d.shape}")    # (5,) - 5 elements
    print(f"Dimensions: {array_1d.ndim}") # 1
    print(f"Data type: {array_1d.dtype}")  # int64 (or similar, depends on system)
    print(f"Size: {array_1d.size}")      # 5 - total number of elements
    print(f"Item size: {array_1d.itemsize} bytes") # Size of one element in bytes
    print(f"Total bytes: {array_1d.nbytes} bytes") # Total memory consumed by array data
    print("-" * 30 + "\n")

    # --- 2D Array ---
    print("--- Exploring a 2D Array ---")
    array_2d = np.array([[10, 11, 12], [20, 21, 22]])
    print(f"Array:\n{array_2d}")
    print(f"Shape: {array_2d.shape}")    # (2, 3) - 2 rows, 3 columns
    print(f"Dimensions: {array_2d.ndim}") # 2
    print(f"Data type: {array_2d.dtype}")
    print(f"Size: {array_2d.size}")
    print(f"Item size: {array_2d.itemsize} bytes")
    print(f"Total bytes: {array_2d.nbytes} bytes")
    print("-" * 30 + "\n")

    # --- 3D Array ---
    print("--- Exploring a 3D Array ---")
    array_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    print(f"Array:\n{array_3d}")
    print(f"Shape: {array_3d.shape}")    # (2, 2, 2) - 2 "layers", 2 rows, 2 columns
    print(f"Dimensions: {array_3d.ndim}") # 3
    print(f"Data type: {array_3d.dtype}")
    print(f"Size: {array_3d.size}")
    print(f"Item size: {array_3d.itemsize} bytes")
    print(f"Total bytes: {array_3d.nbytes} bytes")
    print("-" * 30 + "\n")

if __name__ == "__main__":
    explore_numpy_arrays()
```

-----

### **Explanation of the Application**

This Python script uses the **NumPy** library, which is fundamental for numerical computing in Python.

-----

### **Key Concepts Demonstrated:**

  * **`import numpy as np`**: This line imports the NumPy library and assigns it the conventional alias `np`, making it easier to refer to NumPy functions.
  * **`np.array()`**: This is the primary function used to create a NumPy array from a Python list or tuple.
  * **Array Attributes**:
      * **`.shape`**: Returns a tuple indicating the size of the array in each dimension. For example, `(5,)` for a 1D array with 5 elements, or `(2, 3)` for a 2D array with 2 rows and 3 columns.
      * **`.ndim`**: Returns the number of dimensions (axes) of the array.
      * **`.dtype`**: Returns the data type of the elements in the array (e.g., `int64`, `float64`). NumPy tries to infer the most appropriate data type when the array is created.
      * **`.size`**: Returns the total number of elements in the array.
      * **`.itemsize`**: Returns the size in bytes of each element of the array.
      * **`.nbytes`**: Returns the total number of bytes consumed by the array data.

This application provides a clear demonstration of how to instantiate arrays of different dimensions and inspect their fundamental properties, laying a basic foundation for further NumPy operations.