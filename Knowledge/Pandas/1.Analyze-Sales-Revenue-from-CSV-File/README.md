

## 1\. Analyze Sales Revenue from CSV File

### Goal

The main objective is to read sales data from a CSV file, calculate the revenue for each transaction (quantity multiplied by price), and then group this revenue by month to find the total sales for each month.

### Data

We assume you have a CSV file named `sales.csv` with the following columns:

  * `product`: The name of the product.
  * `quantity`: The number of units sold.
  * `price`: The price per unit.
  * `date`: The date of the sale.

**Example `sales.csv` content:**

```csv
product,quantity,price,date
Laptop,2,1200.0,2024-01-15
Mouse,5,25.0,2024-01-20
Keyboard,3,75.0,2024-02-01
Laptop,1,1200.0,2024-02-10
Monitor,2,300.0,2024-03-05
Mouse,3,25.0,2024-03-10
```

### Pandas Skills Used

  * `pd.read_csv()`: To load data from a CSV file into a DataFrame.
  * `pd.to_datetime()`: To convert a column to datetime objects, which is essential for working with dates (like extracting the month).
  * `.dt.to_period('M')`: A datetime accessor method to extract the month part from a datetime series.
  * `.groupby()`: To group data based on one or more columns (in this case, by month).
  * `.sum()`: To calculate the sum of revenues within each group.

### Create virtual environment (venv)
    ```
    python3 -m venv myenv
    source myenv/bin/activate      # Trên Linux/macOS
    # hoặc
    myenv\Scripts\activate.bat     # Trên Windows

    pip install pandas  
    ```

### Program Steps and Code

```python
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

```

### Expected Output

If your `sales.csv` is similar to the example provided:

```
Original Sales DataFrame:
    product  quantity   price        date
0    Laptop         2  1200.0  2024-01-15
1     Mouse         5    25.0  2024-01-20
2  Keyboard         3    75.0  2024-02-01
3    Laptop         1  1200.0  2024-02-10
4   Monitor         2   300.0  2024-03-05

DataFrame after converting 'date' to datetime:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 6 entries, 0 to 5
Data columns (total 4 columns):
 #   Column    Non-Null Count  Dtype         
---  ------    --------------  -----         
 0   product   6 non-null      object        
 1   quantity  6 non-null      int64         
 2   price     6 non-null      float64       
 3   date      6 non-null      datetime64[ns]
dtypes: datetime64[ns](1), float64(1), int64(1), object(1)
memory usage: 320.0+ bytes
None

DataFrame after adding 'revenue' column:
    product  quantity   price       date  revenue
0    Laptop         2  1200.0 2024-01-15   2400.0
1     Mouse         5    25.0 2024-01-20    125.0
2  Keyboard         3    75.0 2024-02-01    225.0
3    Laptop         1  1200.0 2024-02-10   1200.0
4   Monitor         2   300.0 2024-03-05    600.0

DataFrame after adding 'month' column:
    product  quantity   price       date  revenue    month
0    Laptop         2  1200.0 2024-01-15   2400.0  2024-01
1     Mouse         5    25.0 2024-01-20    125.0  2024-01
2  Keyboard         3    75.0 2024-02-01    225.0  2024-02
3    Laptop         1  1200.0 2024-02-10   1200.0  2024-02
4   Monitor         2   300.0 2024-03-05    600.0  2024-03

Total Monthly Sales Revenue:
month
2024-01    2525.0
2024-02    1425.0
2024-03     675.0
Freq: M, Name: revenue, dtype: float64
```