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