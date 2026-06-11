
import pandas as pd


def data_health_checker(file_path):

    try:
        df = pd.read_csv(file_path)
        print()
        print("DATA HEALTH REPORT")
        print()
        print(" Dataset Shape")
        print(df.shape)
        print()
        print("Missing Values")
        print(df.isnull().sum())
        print()
        print("Duplicate Rows")
        print(df.duplicated().sum())
        print()
        print("Data Types")
        print(df.dtypes)
        print()
        print("Statistical Summary")
        print(df.describe(include="all"))
        print()
        print(" Suggested Cleaning Actions")

        if df.isnull().sum().sum() > 0:
            print("- Fill or remove missing values")

        if df.duplicated().sum() > 0:
            print("- Remove duplicate rows")

        for column in df.columns:
            if df[column].dtype == "object":
                print(f"- Check text consistency in '{column}'")

        print("\nData inspection completed.")

    except Exception as e:
        print("Error:", e)


path = input("Enter CSV file path: ")

data_health_checker(path)