import pandas as pd

def generate_attendance_report(file_path):
    """
    Generates an attendance report from a CSV file, calculating
    working days and leave days for each employee.

    Args:
        file_path (str): The path to the attendance CSV file.

    Returns:
        pandas.DataFrame: A DataFrame containing the attendance report,
                          or None if an error occurs.
    """
    try:
        # Load the attendance data
        df = pd.read_csv(file_path)

        # Ensure necessary columns exist
        required_columns = ['employee_id', 'name', 'date', 'status']
        if not all(col in df.columns for col in required_columns):
            print(f"Error: The CSV file must contain the following columns: {', '.join(required_columns)}")
            return None

        # Convert 'date' column to datetime objects (optional but good practice)
        df['date'] = pd.to_datetime(df['date'])

        # --- Method 1: Using groupby() and value_counts() ---
        # Group by employee and status, then count occurrences
        attendance_summary = df.groupby(['employee_id', 'name', 'status']).size().unstack(fill_value=0)

        # Rename columns for clarity if 'status' values are 'Work' and 'Leave'
        # You might need to adjust these column names based on your actual 'status' values in the CSV
        attendance_summary.rename(columns={'Work': 'Working Days', 'Leave': 'Leave Days'}, inplace=True)

        # Ensure 'Working Days' and 'Leave Days' columns exist even if no data for them
        for col in ['Working Days', 'Leave Days']:
            if col not in attendance_summary.columns:
                attendance_summary[col] = 0

        # Reset index to make employee_id and name regular columns
        attendance_summary = attendance_summary.reset_index()

        # --- Method 2: Using pivot_table() (alternative for a different structure) ---
        # This method is useful if you want a pivot table with employee_id as index
        # and status as columns. It's often less direct for just counts per employee
        # but demonstrates the skill.
        # attendance_pivot = df.pivot_table(index=['employee_id', 'name'],
        #                                   columns='status',
        #                                   aggfunc='size',
        #                                   fill_value=0)
        # attendance_pivot.rename(columns={'Work': 'Working Days', 'Leave': 'Leave Days'}, inplace=True)
        # attendance_pivot = attendance_pivot.reset_index()


        print("Attendance Report Generated Successfully:")
        return attendance_summary[['employee_id', 'name', 'Working Days', 'Leave Days']]

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

if __name__ == "__main__":
    # Create a dummy attendance.csv file for demonstration
    dummy_data = {
        'employee_id': [1, 1, 1, 2, 2, 3, 3, 3, 1, 2],
        'name': ['Alice', 'Alice', 'Alice', 'Bob', 'Bob', 'Charlie', 'Charlie', 'Charlie', 'Alice', 'Bob'],
        'date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-01', '2024-01-02', '2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-03'],
        'status': ['Work', 'Leave', 'Work', 'Work', 'Work', 'Work', 'Leave', 'Work', 'Work', 'Leave']
    }
    dummy_df = pd.DataFrame(dummy_data)
    dummy_df.to_csv('attendance.csv', index=False)
    print("Created 'attendance.csv' with dummy data for testing.")

    # Generate the report
    report = generate_attendance_report('attendance.csv')

    if report is not None:
        print(report)