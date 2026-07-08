import pandas as pd


def load_dataset():
    """Load CSV file."""
    while True:
        try:
            path = input("Enter CSV file path: ")
            df = pd.read_csv(path)
            print("\nDataset loaded successfully!\n")
            return df
        except FileNotFoundError:
            print("File not found. Please enter a valid path.")
        except Exception as e:
            print("Error:", e)


def dataset_info(df):
    """Basic dataset information."""
    report = {}

    report["Rows"] = df.shape[0]
    report["Columns"] = df.shape[1]
    report["Column Names"] = list(df.columns)
    report["Data Types"] = df.dtypes.astype(str).to_dict()

    return report


def missing_values(df):
    """Missing value analysis."""
    missing = df.isnull().sum()
    percentage = (missing / len(df) * 100).round(2)

    report = pd.DataFrame({
        "Missing Values": missing,
        "Percentage (%)": percentage
    })

    return report


def duplicate_rows(df):
    """Duplicate row analysis."""
    duplicates = df[df.duplicated()]

    return {
        "Total Duplicates": duplicates.shape[0],
        "Preview": duplicates.head()
    }


def numeric_summary(df):
    """Summary of numeric columns."""
    return df.describe()


def categorical_summary(df):
    """Summary of categorical columns."""
    cat = df.select_dtypes(include="object")

    summary = {}

    for column in cat.columns:
        summary[column] = {
            "Unique Values": cat[column].nunique(),
            "Most Frequent": cat[column].mode()[0] if not cat[column].mode().empty else "None"
        }

    return pd.DataFrame(summary).T


def final_recommendation(df):
    """Generate recommendations."""

    recommendations = []

    if df.isnull().sum().sum() > 0:
        recommendations.append(
            "- Missing values detected. Consider filling or removing them."
        )

    if df.duplicated().sum() > 0:
        recommendations.append(
            "- Duplicate rows found. Remove duplicates before analysis."
        )

    if len(df.select_dtypes(include="number").columns) > 0:
        recommendations.append(
            "- Numeric columns are available for statistical analysis."
        )

    if len(df.select_dtypes(include="object").columns) > 0:
        recommendations.append(
            "- Categorical columns may need encoding or cleaning."
        )

    if len(recommendations) == 0:
        recommendations.append(
            "- Dataset looks clean and ready for analysis."
        )

    return recommendations


def generate_report(df):
    """Create structured report."""

    report = {
        "Dataset Information": dataset_info(df),
        "Missing Values": missing_values(df),
        "Duplicate Rows": duplicate_rows(df),
        "Numeric Summary": numeric_summary(df),
        "Categorical Summary": categorical_summary(df),
        "Recommendations": final_recommendation(df)
    }

    return report


def display_report(report):
    """Print report neatly."""

    print("=" * 60)
    print("DATA HEALTH CHECK REPORT")
    print("=" * 60)

    print("\nDataset Information")
    for key, value in report["Dataset Information"].items():
        print(f"{key}: {value}")

    print("\nMissing Values")
    print(report["Missing Values"])

    print("\nDuplicate Rows")
    print("Total Duplicates:", report["Duplicate Rows"]["Total Duplicates"])

    if report["Duplicate Rows"]["Total Duplicates"] > 0:
        print("\nDuplicate Preview:")
        print(report["Duplicate Rows"]["Preview"])

    print("\nNumeric Column Summary")
    print(report["Numeric Summary"])

    print("\nCategorical Column Summary")
    print(report["Categorical Summary"])

    print("\nFinal Recommendations")
    for rec in report["Recommendations"]:
        print(rec)

    print("=" * 60)


def main():
    df = load_dataset()
    report = generate_report(df)
    display_report(report)


if __name__ == "__main__":
    main()