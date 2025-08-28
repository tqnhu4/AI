import pandas as pd

def clean_customer_data(file_path):
    """
    Cleans customer data by removing duplicates and handling missing values.

    Args:
        file_path (str): The path to the customers CSV file.

    Returns:
        pandas.DataFrame: A cleaned DataFrame, or None if an error occurs.
    """
    try:
        # Load the customer data
        df = pd.read_csv(file_path)

        # Display initial info
        print("--- Original Data Info ---")
        print(f"Shape: {df.shape}")
        print("Missing values before cleaning:")
        print(df.isnull().sum())
        print("\nDuplicates before cleaning:")
        print(f"Number of duplicate rows: {df.duplicated().sum()}")
        print("-" * 30)

        # 1. Remove duplicate rows
        # We'll consider a row a duplicate if all its column values are the same.
        initial_rows = df.shape[0]
        df.drop_duplicates(inplace=True)
        print(f"\nRemoved {initial_rows - df.shape[0]} duplicate rows.")
        print(f"New Shape after dropping duplicates: {df.shape}")

        # 2. Handle missing values
        # Option A: Fill missing values (e.g., for 'phone' and 'email')
        # You can choose a strategy:
        # - Fill with a placeholder like 'N/A' or an empty string
        # - Fill with a default value
        print("\n--- Handling Missing Values ---")

        # For 'email', let's fill missing values with an empty string
        if 'email' in df.columns:
            df['email'].fillna('', inplace=True)
            print("Filled missing 'email' values with empty string.")

        # For 'phone', let's fill missing values with a placeholder 'UNKNOWN'
        if 'phone' in df.columns:
            df['phone'].fillna('UNKNOWN', inplace=True)
            print("Filled missing 'phone' values with 'UNKNOWN'.")

        # For 'age', let's fill missing values with the median age.
        # It's good practice to check if the column is numeric before calculating median.
        if 'age' in df.columns and pd.api.types.is_numeric_dtype(df['age']):
            median_age = df['age'].median()
            df['age'].fillna(median_age, inplace=True)
            print(f"Filled missing 'age' values with median ({median_age}).")
        elif 'age' in df.columns:
            print("Warning: 'age' column is not numeric. Skipping median fill.")


        # Option B: Drop rows with missing critical values
        # If 'name' is absolutely essential, we might drop rows where 'name' is missing.
        # This uses dropna() for specific columns.
        if 'name' in df.columns:
            rows_before_name_drop = df.shape[0]
            df.dropna(subset=['name'], inplace=True)
            if df.shape[0] < rows_before_name_drop:
                print(f"Dropped {rows_before_name_drop - df.shape[0]} rows due to missing 'name'.")


        print("\n--- Cleaned Data Info ---")
        print(f"Final Shape: {df.shape}")
        print("Missing values after cleaning:")
        print(df.isnull().sum())
        print("-" * 30)

        return df

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

if __name__ == "__main__":
    # Create a dummy customers.csv file for demonstration
    dummy_data = {
        'name': ['Alice', 'Bob', 'Alice', 'Charlie', 'David', None, 'Eve', 'Charlie'],
        'email': ['alice@example.com', 'bob@example.com', 'alice@example.com', 'charlie@example.com', None, 'frank@example.com', 'eve@example.com', 'charlie@example.com'],
        'phone': ['111-222-3333', '444-555-6666', '111-222-3333', None, '777-888-9999', '101-202-3030', '999-888-7777', None],
        'age': [25, 30, 25, 40, None, 50, 35, 40]
    }
    dummy_df = pd.DataFrame(dummy_data)
    dummy_df.to_csv('customers.csv', index=False)
    print("Created 'customers.csv' with dummy data for testing.")

    # Clean the customer data
    cleaned_data = clean_customer_data('customers.csv')

    if cleaned_data is not None:
        print("\n--- Displaying Cleaned Data (First 10 rows) ---")
        print(cleaned_data.head(10))