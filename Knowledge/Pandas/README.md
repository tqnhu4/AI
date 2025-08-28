
## Pandas Cheat Sheet

Pandas is a powerful open-source data analysis and manipulation library for Python. It provides data structures like DataFrames and Series that are designed for working with tabular data.

### 1\. Installation

```bash
pip install pandas
```

### 2\. Importing Pandas

```python
import pandas as pd
import numpy as np # Often used with pandas
```

### 3\. Data Structures

#### a. Series

A Series is a one-dimensional labeled array capable of holding any data type (integers, strings, floats, Python objects, etc.).

**Creation:**

```python
s = pd.Series([1, 2, 3, 4, 5])
s_indexed = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
s_dict = pd.Series({'apple': 100, 'banana': 200, 'cherry': 300})
```

**Accessing Elements:**

```python
s[0]          # By position
s_indexed['a'] # By label
s[:2]         # Slicing
s[[0, 2]]     # Multiple elements by position
s_indexed[['a', 'c']] # Multiple elements by label
```

#### b. DataFrame

A DataFrame is a two-dimensional labeled data structure with columns of potentially different types. You can think of it like a spreadsheet or SQL table.

**Creation:**

```python
# From a dictionary of lists/Series
data = {'col1': [1, 2, 3], 'col2': ['A', 'B', 'C']}
df = pd.DataFrame(data)

# From a list of dictionaries
data_list = [{'col1': 1, 'col2': 'A'}, {'col1': 2, 'col2': 'B'}]
df_list = pd.DataFrame(data_list)

# From a NumPy array
arr = np.array([[1, 2], [3, 4]])
df_np = pd.DataFrame(arr, columns=['colA', 'colB'])

# From CSV/Excel (see I/O section)
```

**Basic Info:**

```python
df.head()        # First 5 rows
df.tail()        # Last 5 rows
df.info()        # Summary of DataFrame, including dtypes and non-null values
df.describe()    # Descriptive statistics (numerical columns)
df.shape         # (rows, columns)
df.columns       # Column names
df.index         # Row index
df.dtypes        # Data types of each column
```

### 4\. Selection and Indexing

#### a. Column Selection

```python
df['col1']          # Select a single column (returns a Series)
df[['col1', 'col2']] # Select multiple columns (returns a DataFrame)
```

#### b. Row Selection

**Using `loc` (label-based):**

```python
df.loc[0]           # Select row by index label
df.loc[0:2]         # Select rows by label slice (inclusive)
df.loc[[0, 2]]      # Select multiple rows by label list
df.loc[df['col1'] > 1] # Boolean indexing (filter rows)
df.loc[df['col1'] > 1, ['col2']] # Filter rows and select specific columns
```

**Using `iloc` (integer-location based):**

```python
df.iloc[0]          # Select row by integer position
df.iloc[0:2]        # Select rows by integer slice (exclusive of end)
df.iloc[[0, 2]]     # Select multiple rows by integer list
df.iloc[0, 0]       # Select a single scalar value by row/column position
```

#### c. Combining Row and Column Selection

```python
df.loc[0, 'col1']       # Select specific cell by label
df.iloc[0, 0]       # Select specific cell by integer position
```

### 5\. Data Manipulation

#### a. Adding/Modifying Columns

```python
df['new_col'] = 10       # Add a new column with a single value
df['new_col_calc'] = df['col1'] * 2 # Add a new column based on existing ones
df.loc[:, 'col1'] = df['col1'] + 1 # Modify an existing column
```

#### b. Deleting Columns/Rows

```python
df.drop('col1', axis=1, inplace=True) # Delete column 'col1' (axis=1)
df.drop(['col1', 'col2'], axis=1, inplace=True) # Delete multiple columns
df.drop(0, axis=0, inplace=True)     # Delete row with index 0 (axis=0)
df.drop([0, 2], axis=0, inplace=True) # Delete multiple rows
```

**Note:** `inplace=True` modifies the DataFrame directly. If `inplace=False` (default), it returns a new DataFrame with the changes and leaves the original untouched.

#### c. Handling Missing Data (NaN)

```python
df.isnull()         # Returns boolean DataFrame indicating missing values
df.notnull()        # Opposite of isnull()
df.dropna()         # Drop rows with any NaN values
df.dropna(how='all') # Drop rows where all values are NaN
df.dropna(subset=['col1']) # Drop rows with NaN in specific columns
df.fillna(0)        # Fill NaN values with a specific value (e.g., 0)
df.fillna(df.mean()) # Fill NaN with column mean
```

#### d. Duplicates

```python
df.duplicated()     # Returns boolean Series indicating duplicate rows
df.drop_duplicates() # Drop duplicate rows
df.drop_duplicates(subset=['col1']) # Drop duplicates based on specific column(s)
```

#### e. Applying Functions

```python
df['col1'].apply(lambda x: x * 2) # Apply a function to a Series
df.apply(np.sum, axis=0)       # Apply a function column-wise (axis=0)
df.apply(np.sum, axis=1)       # Apply a function row-wise (axis=1)
```

#### f. Renaming Columns

```python
df.rename(columns={'old_name': 'new_name'}, inplace=True)
```

### 6\. Grouping and Aggregation

```python
# Group by 'category_col' and calculate the mean of 'value_col'
df.groupby('category_col')['value_col'].mean()

# Group by multiple columns and apply multiple aggregations
df.groupby(['col1', 'col2']).agg({'col3': 'sum', 'col4': 'mean'})

# More complex aggregations
df.groupby('category_col').agg(
    total_sales=('sales', 'sum'),
    avg_price=('price', 'mean')
)
```

### 7\. Merging, Joining, Concatenating

#### a. Concatenating (Stacking)

```python
df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})

pd.concat([df1, df2])         # Stack rows (default axis=0)
pd.concat([df1, df2], axis=1) # Stack columns
```

#### b. Merging (Joining like SQL)

```python
# Create sample DataFrames
df_left = pd.DataFrame({'key': ['A', 'B', 'C'], 'value_l': [1, 2, 3]})
df_right = pd.DataFrame({'key': ['A', 'B', 'D'], 'value_r': [4, 5, 6]})

# Inner Join (default): Only rows with matching keys in both DataFrames
pd.merge(df_left, df_right, on='key', how='inner')

# Left Join: All rows from left, matching rows from right. NaN if no match.
pd.merge(df_left, df_right, on='key', how='left')

# Right Join: All rows from right, matching rows from left. NaN if no match.
pd.merge(df_left, df_right, on='key', how='right')

# Outer Join: All rows from both DataFrames. NaN if no match.
pd.merge(df_left, df_right, on='key', how='outer')

# Merge on different column names
pd.merge(df_left, df_right, left_on='key_left', right_on='key_right')
```

### 8\. Input/Output (I/O)

#### a. Reading Data

```python
df = pd.read_csv('data.csv')
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')
df = pd.read_sql('SELECT * FROM my_table', con) # Requires SQLAlchemy
df = pd.read_json('data.json')
```

#### b. Writing Data

```python
df.to_csv('output.csv', index=False) # index=False prevents writing DataFrame index as a column
df.to_excel('output.xlsx', sheet_name='Sheet1', index=False)
df.to_json('output.json', orient='records')
```

### 9\. Time Series Functionality

```python
# Convert to datetime objects
df['date_col'] = pd.to_datetime(df['date_col'])

# Set index to datetime
df.set_index('date_col', inplace=True)

# Resampling (e.g., daily to monthly)
df.resample('M')['value'].mean()

# Datetime properties
df.index.year
df.index.month
df.index.day
df.index.dayofweek
df.index.hour
```

-----