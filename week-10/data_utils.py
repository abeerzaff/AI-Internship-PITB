import pandas as pd


def get_dataset_overview(df):

    return {
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Duplicate Rows": df.duplicated().sum(),
        "Missing Values": df.isnull().sum().sum()
    }


def get_missing_values(df):

    missing = df.isnull().sum()

    result = pd.DataFrame({
        "Column": missing.index,
        "Missing Values": missing.values
    })

    result["Missing Percentage"] = (
        result["Missing Values"]
        / len(df)
        * 100
    ).round(2)

    return result


def get_basic_statistics(df):

    return df.describe(
        include="all"
    ).transpose()


def get_numeric_columns(df):

    return df.select_dtypes(
        include="number"
    ).columns.tolist()


def get_correlation_matrix(df):

    numeric_df = df.select_dtypes(
        include="number"
    )

    if numeric_df.shape[1] < 2:
        return None

    return numeric_df.corr()


def describe_distribution(column_name, series):
    """
    Returns a simple, human-friendly description of a numeric column's
    distribution — written for someone with no statistics background.
    """

    clean_series = series.dropna()

    if clean_series.empty:
        return f"No data is available yet for **{column_name}**."

    mean_value = clean_series.mean()
    median_value = clean_series.median()
    min_value = clean_series.min()
    max_value = clean_series.max()
    skew_value = clean_series.skew()

    typical_range_low = clean_series.quantile(0.25)
    typical_range_high = clean_series.quantile(0.75)

    sentence = (
        f"Most values for **{column_name}** fall between "
        f"{typical_range_low:,.0f} and {typical_range_high:,.0f}, "
        f"with a typical value around {median_value:,.0f}."
    )

    if skew_value > 0.75:
        sentence += (
            f" A few unusually high values pull the average up to "
            f"{mean_value:,.0f} — most entries are actually lower than that."
        )
    elif skew_value < -0.75:
        sentence += (
            f" A few unusually low values pull the average down to "
            f"{mean_value:,.0f} — most entries are actually higher than that."
        )
    else:
        sentence += (
            f" The values are fairly evenly spread out, ranging overall "
            f"from {min_value:,.0f} to {max_value:,.0f}."
        )

    return sentence


def describe_relationship(x_col, y_col, df):
    """
    Returns a simple, human-friendly description of the relationship
    between two numeric columns — written for someone with no
    statistics background.
    """

    if x_col == y_col:
        return (
            f"Both axes are showing **{x_col}** — pick two different "
            f"columns above to see how they relate to each other."
        )

    valid = df[[x_col, y_col]].dropna()

    if len(valid) < 2:
        return (
            f"There isn't enough shared data between **{x_col}** "
            f"and **{y_col}** yet to say how they're related."
        )

    x_series = valid[x_col]
    y_series = valid[y_col]

    correlation = x_series.corr(y_series)

    if pd.isna(correlation):
        return (
            f"**{x_col}** and **{y_col}** don't vary enough to "
            f"show a relationship."
        )

    abs_corr = abs(correlation)
    trend_word = "rises" if correlation > 0 else "falls"

    if abs_corr >= 0.7:
        return (
            f"There's a clear pattern here: as **{x_col}** goes up, "
            f"**{y_col}** almost always {trend_word} along with it. "
            f"This is one of the strongest relationships in the dataset."
        )
    elif abs_corr >= 0.4:
        return (
            f"There's a noticeable pattern: as **{x_col}** goes up, "
            f"**{y_col}** tends to {trend_word} too, though not "
            f"perfectly every time."
        )
    elif abs_corr >= 0.2:
        return (
            f"There's a slight tendency for **{y_col}** to {trend_word} "
            f"as **{x_col}** goes up, but it's a weak pattern — other "
            f"factors are likely playing a bigger role."
        )
    else:
        return (
            f"**{x_col}** and **{y_col}** don't appear to be related — "
            f"knowing one doesn't really tell you much about the other."
        )


def describe_correlation_matrix(correlation_df):
    """
    Returns a simple, human-friendly summary of the strongest and
    weakest relationships found in a correlation matrix.
    """

    columns = correlation_df.columns
    pairs = []

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            col_a = columns[i]
            col_b = columns[j]
            value = correlation_df.iloc[i, j]

            if pd.notna(value):
                pairs.append((col_a, col_b, value))

    if not pairs:
        return "At least two numeric columns are needed to compare relationships."

    strongest = max(pairs, key=lambda pair: abs(pair[2]))
    weakest = min(pairs, key=lambda pair: abs(pair[2]))

    strongest_trend = "move up together" if strongest[2] > 0 else "move in opposite directions"

    summary = (
        f"**{strongest[0]}** and **{strongest[1]}** show the clearest "
        f"connection in this dataset — they tend to {strongest_trend}."
    )

    if abs(weakest[2]) < 0.2 and (
        weakest[0] != strongest[0] or weakest[1] != strongest[1]
    ):
        summary += (
            f" On the other hand, **{weakest[0]}** and **{weakest[1]}** "
            f"don't seem connected at all."
        )

    return summary


def generate_key_insights(df):

    insights = []

    numeric_columns = get_numeric_columns(df)

    for column in numeric_columns:

        mean_value = df[column].mean()
        max_value = df[column].max()
        min_value = df[column].min()

        insights.append(
            f"{column}: average is "
            f"{mean_value:.2f}, minimum is "
            f"{min_value:.2f}, and maximum is "
            f"{max_value:.2f}."
        )

    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:

        insights.append(
            f"The dataset contains "
            f"{duplicate_count} duplicate rows."
        )

    missing_count = df.isnull().sum().sum()

    if missing_count > 0:

        insights.append(
            f"The dataset contains "
            f"{missing_count} missing values."
        )

    return insights