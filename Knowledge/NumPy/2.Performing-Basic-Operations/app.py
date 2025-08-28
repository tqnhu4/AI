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