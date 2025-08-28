Let's create an application to analyze and visualize weather data\!

-----

## Weather Data Analyzer & Visualizer

This application processes weather data from a `weather.csv` file to visualize daily temperature trends and calculate monthly average temperatures. It leverages pandas for data manipulation and matplotlib for plotting.

### Requirement

Data: weather.csv contains date, temperature, humidity

Objective: Plot temperature chart by day, calculate monthly average.

Skills: to_datetime(), resample(), plot()

```python
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def analyze_weather_data(file_path):
    """
    Analyzes weather data: plots daily temperature and calculates monthly averages.

    Args:
        file_path (str): The path to the weather CSV file.

    Returns:
        None
    """
    try:
        # Load the weather data
        df = pd.read_csv(file_path)

        # Ensure necessary columns exist
        required_columns = ['date', 'temperature', 'humidity']
        if not all(col in df.columns for col in required_columns):
            print(f"Error: The CSV file must contain the following columns: {', '.join(required_columns)}")
            return

        # Convert 'date' column to datetime objects
        df['date'] = pd.to_datetime(df['date'])

        # Set 'date' as the DataFrame index for time series operations
        df.set_index('date', inplace=True)

        # Ensure 'temperature' is numeric
        if not pd.api.types.is_numeric_dtype(df['temperature']):
            print("Error: The 'temperature' column must contain numeric values.")
            return

        print("--- Analyzing Weather Data ---")

        # 1. Plot Temperature over Time
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['temperature'], marker='o', linestyle='-', markersize=4, color='skyblue')
        plt.title('Daily Temperature Over Time')
        plt.xlabel('Date')
        plt.ylabel('Temperature (°C)') # Assuming Celsius, adjust if Fahrenheit
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout() # Adjust layout to prevent labels from overlapping

        # Format x-axis for better date display
        formatter = mdates.DateFormatter('%Y-%m-%d')
        plt.gca().xaxis.set_major_formatter(formatter)
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gcf().autofmt_xdate() # Auto-formats date labels for readability

        print("\nDisplaying daily temperature plot...")
        plt.show()

        # 2. Calculate Monthly Average Temperature
        # Use resample() to group data by month ('M') and then calculate the mean
        monthly_avg_temp = df['temperature'].resample('M').mean().reset_index()
        monthly_avg_temp.rename(columns={'date': 'Month', 'temperature': 'Average Temperature (°C)'}, inplace=True)

        print("\nMonthly Average Temperature:")
        print(monthly_avg_temp)

        # Optional: Plot monthly average temperature
        plt.figure(figsize=(10, 5))
        plt.bar(monthly_avg_temp['Month'], monthly_avg_temp['Average Temperature (°C)'], color='lightcoral', width=20)
        plt.title('Monthly Average Temperature')
        plt.xlabel('Month')
        plt.ylabel('Average Temperature (°C)')
        plt.grid(axis='y')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        print("\nDisplaying monthly average temperature plot...")
        plt.show()

        print("\nWeather data analysis complete!")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Create a dummy weather.csv file for demonstration
    # Covering a few months for monthly average calculation
    dummy_data = {
        'date': pd.to_datetime([
            '2024-01-01', '2024-01-02', '2024-01-15', '2024-01-28',
            '2024-02-05', '2024-02-15', '2024-02-20',
            '2024-03-10', '2024-03-20', '2024-03-30', '2024-04-05'
        ]),
        'temperature': [15, 16, 14, 17, 18, 19, 20, 22, 21, 23, 25],
        'humidity': [70, 72, 68, 75, 80, 82, 78, 85, 88, 87, 90]
    }
    dummy_df = pd.DataFrame(dummy_data)
    dummy_df.to_csv('weather.csv', index=False)
    print("Created 'weather.csv' with dummy data for testing.\n")

    # Analyze the weather data
    analyze_weather_data('weather.csv')
```

-----

### How the Script Works:

1.  **`analyze_weather_data(file_path)` Function:**

      * **Load Data:** It loads your `weather.csv` file into a pandas DataFrame.
      * **Error Handling & Validation:** Includes `try-except` blocks for `FileNotFoundError` and general exceptions. It also checks for the presence of required columns (`date`, `temperature`, `humidity`) and ensures the `temperature` column is numeric.
      * **Date Conversion & Indexing:**
          * `df['date'] = pd.to_datetime(df['date'])`: This crucial step converts the 'date' column into pandas datetime objects. This is essential for time series operations.
          * `df.set_index('date', inplace=True)`: Setting the 'date' column as the DataFrame's index is best practice for time series analysis, enabling powerful time-based operations like `resample()`.
      * **Plot Daily Temperature (`plot()`):**
          * `plt.figure(figsize=(12, 6))`: Creates a new figure for the plot with a specified size.
          * `plt.plot(df.index, df['temperature'], ...)`: Generates a line plot. `df.index` (which is your 'date' column) is used for the x-axis, and `df['temperature']` for the y-axis.
          * **Matplotlib Customization:** Lines like `plt.title()`, `plt.xlabel()`, `plt.ylabel()`, `plt.grid()`, and `plt.xticks(rotation=45)` enhance the readability and appearance of the plot.
          * `mdates.DateFormatter` and `mdates.AutoDateLocator`: These are used to format the x-axis (dates) nicely, preventing labels from overlapping and ensuring a clean display.
          * `plt.show()`: Displays the generated plot.
      * **Calculate Monthly Average Temperature (`resample()`):**
          * `df['temperature'].resample('M').mean()`: This is the core of the monthly average calculation.
              * `resample('M')`: This downsamples the time series data, grouping it by **month** (the 'M' alias). If you wanted weekly averages, you'd use 'W'; for daily, 'D'.
              * `.mean()`: After resampling, it calculates the mean (average) of the 'temperature' for each monthly group.
          * `.reset_index()` and `.rename()`: These steps format the output DataFrame for better readability, similar to the previous applications.
      * **Optional: Plot Monthly Average Temperature:** A bar chart is generated to visually represent the monthly average temperatures, providing another way to interpret the data.

2.  **`if __name__ == "__main__":` Block:**

      * **Dummy Data Creation:** This part generates a sample `weather.csv` file with dates spanning a few months, which is ideal for testing the monthly averaging and plotting functionalities.
      * **Execute Analysis:** It calls the `analyze_weather_data()` function with the path to the dummy CSV, triggering the analysis and plot generation.

-----

### To Use This Application:

1.  **Save the Code:** Save the script as a Python file (e.g., `weather_analyzer.py`).
2.  **Ensure `weather.csv` Exists:** Place your `weather.csv` file in the same directory as the script, or update the `file_path` in the `analyze_weather_data()` function call with the correct path.
3.  **Install Libraries:** You'll need pandas and matplotlib. If you don't have them, install them via pip:
    ```bash
    pip install pandas matplotlib
    ```
4.  **Run the Script:** Open your terminal or command prompt, navigate to the script's directory, and execute:
    ```bash
    python weather_analyzer.py
    ```
