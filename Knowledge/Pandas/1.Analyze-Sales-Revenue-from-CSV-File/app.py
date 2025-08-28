import pandas as pd

# Step 1: Load the sales data from the CSV file
# This reads your 'sales.csv' file into a Pandas DataFrame.
try:
    df_sales = pd.read_csv('sales.csv')
    print("Original Sales DataFrame:")
    print(df_sales.head())
    print("\n")
except FileNotFoundError:
    print("Error: sales.csv not found. Please make sure the file is in the same directory.")
    # Create a dummy DataFrame for demonstration if file not found
    data = {
        'product': ['Laptop', 'Mouse', 'Keyboard', 'Laptop', 'Monitor', 'Mouse'],
        'quantity': [2, 5, 3, 1, 2, 3],
        'price': [1200.0, 25.0, 75.0, 1200.0, 300.0, 25.0],
        'date': ['2024-01-15', '2024-01-20', '2024-02-01', '2024-02-10', '2024-03-05', '2024-03-10']
    }
    df_sales = pd.DataFrame(data)
    print("Using dummy data for demonstration:")
    print(df_sales.head())
    print("\n")


# Step 2: Convert the 'date' column to datetime objects
# This is crucial for enabling date-based operations like extracting the month.
df_sales['date'] = pd.to_datetime(df_sales['date'])
print("DataFrame after converting 'date' to datetime:")
print(df_sales.info()) # Check data types to confirm 'date' is now datetime
print("\n")

# Step 3: Calculate the 'revenue' for each sales record
# Revenue is simply quantity multiplied by price for each row.
df_sales['revenue'] = df_sales['quantity'] * df_sales['price']
print("DataFrame after adding 'revenue' column:")
print(df_sales.head())
print("\n")

# Step 4: Extract the month from the 'date' column
# We use .dt accessor to get datetime properties, and .to_period('M') to get the month period (e.g., '2024-01').
df_sales['month'] = df_sales['date'].dt.to_period('M')
print("DataFrame after adding 'month' column:")
print(df_sales.head())
print("\n")

# Step 5: Group the data by 'month' and sum the 'revenue'
# This aggregates all revenues for the same month.
monthly_revenue = df_sales.groupby('month')['revenue'].sum()

# Step 6: Display the results
print("Total Monthly Sales Revenue:")
print(monthly_revenue)