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