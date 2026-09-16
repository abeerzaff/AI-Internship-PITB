
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_utils import (
    get_dataset_overview,
    get_missing_values,
    get_basic_statistics,
    get_numeric_columns,
    get_correlation_matrix,
    generate_key_insights,
    describe_distribution,
    describe_relationship,
    describe_correlation_matrix
)

from ai_client import generate_ai_insights



st.set_page_config(
    page_title="AI-Powered Data Insights Dashboard",
    page_icon="📊",
    layout="wide"
)


with st.sidebar:
    st.title("📊 Dashboard")

    st.markdown("""
    ### Sections

    📁 Upload Data

    📊 Dataset Overview

    🔍 Data Quality

    📈 Visualizations

    🔗 Correlation

    💡 Key Insights

    🤖 AI Analysis
    """)

    st.info(
        "Upload a CSV file to automatically analyze "
        "your dataset and generate AI-powered insights."
    )


st.title("📊 AI-Powered Data Insights Dashboard")

st.write(
    "Upload a CSV file to explore your data, "
    "visualize patterns, analyze correlations, "
    "and generate AI-powered insights and recommendations."
)



st.header("📁 Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)


if uploaded_file is None:

    st.info(
        "Please upload a CSV file to begin the analysis."
    )

    st.stop()



try:

    df = pd.read_csv(uploaded_file)

except Exception as e:

    st.error(
        f"Unable to read the uploaded CSV file: {e}"
    )

    st.stop()



if df.empty:

    st.error(
        "The uploaded CSV file is empty."
    )

    st.stop()


st.success(
    "CSV uploaded successfully!"
)


st.header("1. 📊 Dataset Overview")

overview = get_dataset_overview(df)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Rows",
        overview["Rows"]
    )

with col2:
    st.metric(
        "Columns",
        overview["Columns"]
    )

with col3:
    st.metric(
        "Duplicate Rows",
        overview["Duplicate Rows"]
    )

with col4:
    st.metric(
        "Missing Values",
        overview["Missing Values"]
    )


st.subheader("Dataset Preview")

st.dataframe(
    df.head(10),
    use_container_width=True
)


st.subheader("Column Data Types")

dtype_df = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.astype(str).values
})

st.dataframe(
    dtype_df,
    use_container_width=True
)


st.header("2. 🔍 Missing Value Analysis")

missing_df = get_missing_values(df)

st.dataframe(
    missing_df,
    use_container_width=True
)


if missing_df["Missing Values"].sum() == 0:

    st.success(
        "No missing values were found."
    )

else:

    st.warning(
        "Missing values were detected in the dataset."
    )


st.header("3. 📈 Basic Statistics")

statistics = get_basic_statistics(df)

st.dataframe(
    statistics,
    use_container_width=True
)

numeric_columns = get_numeric_columns(df)



st.header("4. 📊 Data Visualization")


if len(numeric_columns) > 0:

    selected_column = st.selectbox(
        "Select a numeric column for distribution analysis",
        numeric_columns
    )

    fig, ax = plt.subplots()

    ax.hist(
        df[selected_column].dropna(),
        bins=20
    )

    ax.set_title(
        f"Distribution of {selected_column}"
    )

    ax.set_xlabel(
        selected_column
    )

    ax.set_ylabel(
        "Frequency"
    )

    st.pyplot(fig)

    st.caption(
        describe_distribution(
            selected_column,
            df[selected_column]
        )
    )


else:

    st.warning(
        "No numeric columns are available for visualization."
    )


if len(numeric_columns) >= 2:

    st.subheader("Relationship Between Variables")

    col1, col2 = st.columns(2)

    with col1:

        x_column = st.selectbox(
            "Select X-axis",
            numeric_columns,
            key="x_axis"
        )

    with col2:

        y_default_index = (
            1 if len(numeric_columns) > 1 else 0
        )

        y_column = st.selectbox(
            "Select Y-axis",
            numeric_columns,
            index=y_default_index,
            key="y_axis"
        )

    fig, ax = plt.subplots()

    ax.scatter(
        df[x_column],
        df[y_column]
    )

    ax.set_xlabel(
        x_column
    )

    ax.set_ylabel(
        y_column
    )

    ax.set_title(
        f"{x_column} vs {y_column}"
    )

    st.pyplot(fig)

    st.caption(
        describe_relationship(
            x_column,
            y_column,
            df
        )
    )



st.header("5. 🔗 Correlation Analysis")

correlation = get_correlation_matrix(df)


if correlation is not None:

    st.subheader("Correlation Matrix")

    st.dataframe(
        correlation,
        use_container_width=True
    )

    st.subheader("Correlation Heatmap")

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    ax.set_title(
        "Correlation Heatmap"
    )

    st.pyplot(fig)

    st.caption(
        describe_correlation_matrix(correlation)
    )

else:

    st.warning(
        "At least two numeric columns are required "
        "for correlation analysis."
    )



st.header("6. 💡 Key Insights")

key_insights = generate_key_insights(df)

if key_insights:

    for insight in key_insights:

        st.write(
            f"• {insight}"
        )

else:

    st.info(
        "No automatic numeric insights could be generated."
    )



st.header("7. 🤖 AI-Generated Insights")

st.write(
    "The AI analyzes the prepared dataset information "
    "and provides a summary, findings, trends, "
    "data-quality observations, and recommendations."
)


if st.button(
    "✨ Generate AI Insights"
):

    with st.spinner(
        "Analyzing dataset with AI..."
    ):

        data_summary = f"""
Dataset Shape:
{df.shape}

Columns:
{list(df.columns)}

Data Types:
{df.dtypes.to_string()}

Missing Values:
{df.isnull().sum().to_string()}

Duplicate Rows:
{df.duplicated().sum()}

Basic Statistics:
{df.describe().to_string()}

Correlation:
{correlation.to_string() if correlation is not None else "Not available"}

Key Insights:
{chr(10).join(key_insights)}
"""

        ai_result = generate_ai_insights(
            data_summary
        )

        st.markdown(
            ai_result
        )



st.divider()

st.caption(
    "AI-Powered Data Insights Dashboard"
)