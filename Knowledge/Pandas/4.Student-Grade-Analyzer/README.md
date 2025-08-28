
-----

## Student Grade Analyzer

This script processes a `grades.csv` file to calculate the average grade for each student and for each subject. It demonstrates the use of pandas functions like `groupby()`, `mean()`, `unstack()`, and `pivot_table()`.

### Requirement
Data: grades.csv contains student, subject, grade

Objective: Calculate the average score of each student and each subject.

Skills: groupby(), mean(), unstack(), pivot_table()

```python
import pandas as pd

def analyze_grades(file_path):
    """
    Analyzes student grades from a CSV file, calculating average grades
    per student and per subject.

    Args:
        file_path (str): The path to the grades CSV file.

    Returns:
        tuple: A tuple containing two pandas DataFrames:
               - student_avg_grades (DataFrame): Average grade for each student.
               - subject_avg_grades (DataFrame): Average grade for each subject.
               Or (None, None) if an error occurs.
    """
    try:
        # Load the grades data
        df = pd.read_csv(file_path)

        # Ensure necessary columns exist
        required_columns = ['student', 'subject', 'grade']
        if not all(col in df.columns for col in required_columns):
            print(f"Error: The CSV file must contain the following columns: {', '.join(required_columns)}")
            return None, None

        # Ensure 'grade' column is numeric
        if not pd.api.types.is_numeric_dtype(df['grade']):
            print("Error: The 'grade' column must contain numeric values.")
            return None, None

        print("--- Analyzing Grades ---")

        # 1. Calculate average grade for each student
        # Using groupby() and mean()
        student_avg_grades = df.groupby('student')['grade'].mean().reset_index()
        student_avg_grades.rename(columns={'grade': 'Average Grade'}, inplace=True)
        print("\nAverage Grade Per Student:")
        print(student_avg_grades)

        # 2. Calculate average grade for each subject
        # Using groupby() and mean()
        subject_avg_grades = df.groupby('subject')['grade'].mean().reset_index()
        subject_avg_grades.rename(columns={'grade': 'Average Grade'}, inplace=True)
        print("\nAverage Grade Per Subject:")
        print(subject_avg_grades)

        # Optional: Display a pivot table for grades per student per subject
        # Using pivot_table() for a more detailed view
        print("\n--- Detailed Grades (Student vs. Subject) ---")
        # You can use fill_value=0 if you want to show 0 for missing grades,
        # otherwise, NaN indicates a student didn't take that subject.
        detailed_grades_pivot = df.pivot_table(index='student',
                                               columns='subject',
                                               values='grade')
        print(detailed_grades_pivot)

        # Optional: Another way to achieve subject averages using unstack()
        # This is more complex for just subject averages, but shows unstack() usage
        # df_pivot_unstack = df.groupby(['subject', 'student'])['grade'].mean().unstack(fill_value=0)
        # print("\nSubject grades unstacked:")
        # print(df_pivot_unstack)

        print("\nGrade analysis complete!")
        return student_avg_grades, subject_avg_grades

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None

if __name__ == "__main__":
    # Create a dummy grades.csv file for demonstration
    dummy_data = {
        'student': ['Alice', 'Bob', 'Alice', 'Charlie', 'Bob', 'Alice', 'Charlie', 'Bob'],
        'subject': ['Math', 'Math', 'Science', 'Math', 'Science', 'History', 'History', 'History'],
        'grade': [85, 78, 92, 65, 88, 75, 90, 80]
    }
    dummy_df = pd.DataFrame(dummy_data)
    dummy_df.to_csv('grades.csv', index=False)
    print("Created 'grades.csv' with dummy data for testing.\n")

    # Analyze the grades
    student_avgs, subject_avgs = analyze_grades('grades.csv')

    # You can further process student_avgs or subject_avgs DataFrames here if needed
    # For example, save them to new CSVs:
    # if student_avgs is not None:
    #     student_avgs.to_csv('student_average_grades.csv', index=False)
    #     print("\nStudent average grades saved to 'student_average_grades.csv'")
```

-----

### How the Script Works:

1.  **`analyze_grades(file_path)` Function:**

      * **Load Data:** It begins by loading your `grades.csv` file into a pandas DataFrame.
      * **Error Handling & Validation:** It includes robust `try-except` blocks to catch `FileNotFoundError` and general exceptions. Crucially, it verifies that all **required columns** (`student`, `subject`, `grade`) are present and that the `grade` column is **numeric**, preventing errors if your data isn't in the expected format.
      * **Average Grade Per Student:**
          * `df.groupby('student')['grade'].mean()`: This is the core for this calculation. It groups all rows by the `student` column and then calculates the `mean()` (average) of the `grade` for each student group.
          * `.reset_index()`: Converts the 'student' column from being the DataFrame's index back into a regular column.
          * `.rename()`: Renames the 'grade' column (which now holds the average) to 'Average Grade' for clarity.
      * **Average Grade Per Subject:**
          * `df.groupby('subject')['grade'].mean()`: Similar to the student average, this groups by `subject` and calculates the average `grade` for each subject.
      * **Detailed Grades (`pivot_table()`):**
          * `df.pivot_table(index='student', columns='subject', values='grade')`: This powerful function creates a pivot table where:
              * `index='student'` sets students as rows.
              * `columns='subject'` sets subjects as columns.
              * `values='grade'` populates the table with the corresponding grades.
          * This provides a clear, matrix-like view of each student's grade in each subject. If a student didn't take a particular subject, that cell will contain `NaN` (Not a Number) by default.
      * **Return Values:** The function returns two DataFrames: one for student averages and one for subject averages.

2.  **`if __name__ == "__main__":` Block:**

      * **Dummy Data Creation:** This section creates a sample `grades.csv` file, making the script immediately runnable for demonstration purposes.
      * **Execution:** It calls the `analyze_grades()` function and stores the returned DataFrames. It then prints them to the console.
      * **Further Processing (Commented Out):** It includes commented-out lines showing how you might save the resulting average grade DataFrames to new CSV files, which is a common next step in data analysis workflows.

-----

### To Use This Application:

1.  **Save the Code:** Save the script as a Python file (e.g., `grade_analyzer.py`).
2.  **Ensure `grades.csv` Exists:** Place your `grades.csv` file in the same directory as the script, or provide the full path to your CSV file in the `analyze_grades()` function call.
3.  **Install Pandas:** If you haven't already, install the pandas library:
    ```bash
    pip install pandas
    ```
4.  **Run the Script:** Open your terminal or command prompt, navigate to the script's directory, and execute:
    ```bash
    python grade_analyzer.py
    ```
