import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


st.set_page_config(
    page_title="Public Data Insights Dashboard",
    page_icon="🎬",
    layout="wide"
)

sns.set_style("whitegrid")
CLR_TEAL = "#2A9D8F"
CLR_CORAL = "#E76F51"
CLR_AMBER = "#F4A261"
CLR_INDIGO = "#3D5A80"
CLR_PLUM = "#8E5572"
CLR_SAGE = "#6A994E"
PALETTE = [CLR_TEAL, CLR_CORAL, CLR_AMBER, CLR_INDIGO, CLR_PLUM, CLR_SAGE]

st.markdown("""
<style>
    .block-container {padding-top: 2rem;}
    div[data-testid="stMetric"] {
        background-color: #F4FAF9;
        border: 1px solid #D8EFEC;
        border-radius: 10px;
        padding: 10px 15px;
    }
    h2 {border-bottom: 3px solid #2A9D8F; padding-bottom: 6px; margin-top: 2rem; color: #264653;}
    h3 {color: #264653;}
    .section-note {
        color: #6B6560;
        font-size: 0.92rem;
        margin-top: -6px;
        margin-bottom: 14px;
        font-style: italic;
    }
    .filter-panel {
        background-color: #F7F5F2;
        border: 1px solid #EAE5DE;
        border-radius: 12px;
        padding: 16px 20px 4px 20px;
        margin-bottom: 1.2rem;
    }
    .chart-desc {
        color: #6B6560;
        font-size: 0.9rem;
        margin-top: -6px;
        margin-bottom: 10px;
    }
    .insight-box {
        background-color: #EEF6F5;
        border-left: 4px solid #2A9D8F;
        padding: 10px 16px;
        border-radius: 6px;
        margin-top: 8px;
        margin-bottom: 18px;
        font-size: 0.95rem;
        color: #264653;
    }
    .insight-line {
        background-color: #FDF6F3;
        border-left: 3px solid #E76F51;
        padding: 10px 16px;
        border-radius: 5px;
        margin-bottom: 12px;
        font-size: 0.95rem;
    }
    .insight-why {
        color: #6B6560;
        font-size: 0.87rem;
        margin-top: 4px;
    }
    .rec-card {
        background-color: #FFFFFF;
        border: 1px solid #E5E0D8;
        border-left: 5px solid #2A9D8F;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 12px;
    }
    .rec-card.high { border-left-color: #E76F51; }
    .rec-card.medium { border-left-color: #F4A261; }
    .rec-card.low { border-left-color: #6A994E; }
    .rec-priority {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        padding: 2px 9px;
        border-radius: 20px;
        margin-bottom: 6px;
        text-transform: uppercase;
    }
    .rec-priority.high { background-color: #FBE4DD; color: #C0431E; }
    .rec-priority.medium { background-color: #FDEDDB; color: #B96A1B; }
    .rec-priority.low { background-color: #E4EFDA; color: #4A7326; }
    .rec-evidence {
        color: #6B6560;
        font-size: 0.85rem;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_netflixx.csv")
    return df

df = load_data()

st.title("🎬 Public Data Insights Dashboard")


st.markdown(f"""
### Introduction

The dashboard analyzes a cleaned Netflix dataset containing **{df.shape[0]:,} titles**, including both Movies and TV Shows released between **{int(df['release_year'].min())}** and **{int(df['release_year'].max())}**. The dataset covers content from **{df['country'].dropna().str.split(', ').explode().nunique()} countries** and includes important attributes such as content type, release year, rating, genre, duration, country of origin, and the date each title was added to Netflix.

The primary goal of this dashboard is to transform the raw Netflix dataset into meaningful and interactive visualizations that help identify trends, content distribution, and viewing patterns. By exploring the dashboard, users can better understand Netflix's content strategy, discover key insights about its library, and identify potential opportunities for expanding content across different countries, genres, and audience categories. Users can also apply filters based on content type, release year, and country to perform more focused and interactive analysis.

""")

st.markdown('<div class="filter-panel">', unsafe_allow_html=True)
st.markdown("##### 🔍 Filters")

f1, f2, f3 = st.columns(3)

with f1:
    content_type = st.multiselect(
        "Content Type",
        options=sorted(df["type"].dropna().unique()),
        default=sorted(df["type"].dropna().unique())
    )

with f2:
    year_min, year_max = int(df["release_year"].min()), int(df["release_year"].max())
    year_range = st.slider(
        "Release Year Range",
        min_value=year_min,
        max_value=year_max,
        value=(2010, year_max)
    )

with f3:
    top_countries_list = (
        df["country"].dropna().str.split(", ").explode().value_counts().head(15).index.tolist()
    )
    country_filter = st.multiselect(
        "Country (top 15 shown)",
        options=top_countries_list,
        default=[]
    )

st.markdown('</div>', unsafe_allow_html=True)

filtered_df = df[
    (df["type"].isin(content_type)) &
    (df["release_year"].between(year_range[0], year_range[1]))
]

if country_filter:
    filtered_df = filtered_df[
        filtered_df["country"].dropna().apply(
            lambda x: any(c in x for c in country_filter)
        )
    ]

if filtered_df.empty:
    st.warning("No titles match the current filters. Try widening your selection above.")
    st.stop()

st.header("📁 Dataset Overview")


col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Titles", f"{filtered_df.shape[0]:,}")
col2.metric("Movies", f"{(filtered_df['type'] == 'Movie').sum():,}")
col3.metric("TV Shows", f"{(filtered_df['type'] == 'TV Show').sum():,}")
col4.metric("Countries Represented", f"{filtered_df['country'].dropna().str.split(', ').explode().nunique():,}")

with st.expander("🔎 Preview the raw data"):
    st.dataframe(filtered_df.head(20), use_container_width=True)

st.markdown("---")


st.header("📈 Charts")


st.subheader("1. Top 10 Countries by Number of Titles")
st.markdown('<p class="chart-desc">What this shows: how many titles come from each of the top 10 producing countries.</p>', unsafe_allow_html=True)
top_countries = (
    filtered_df["country"].dropna().str.split(", ").explode().value_counts().head(10)
)
c1, c2 = st.columns([2, 1])
with c1:
    fig, ax = plt.subplots(figsize=(9, 4))
    top_countries.plot(kind="bar", ax=ax, color=CLR_TEAL)
    ax.set_xlabel("Country")
    ax.set_ylabel("Number of Titles")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    st.pyplot(fig)
with c2:
    leader = top_countries.index[0]
    leader_share = top_countries.iloc[0] / filtered_df.shape[0] * 100
    st.markdown(f"""
    <div class="insight-box">
    <b>Takeaway:</b> Most of these titles come from just one place — {leader}, at about
    {leader_share:.0f}% of everything shown here. Everything else is split thinly across
    many smaller countries.
    </div>
    """, unsafe_allow_html=True)


st.subheader("2. Titles Released Over Time")
st.markdown('<p class="chart-desc">What this shows: the number of titles released each year, tracking growth or decline over time.</p>', unsafe_allow_html=True)
year_counts = filtered_df["release_year"].value_counts().sort_index()
c1, c2 = st.columns([2, 1])
with c1:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(year_counts.index, year_counts.values, marker="o", color=CLR_INDIGO)
    ax.set_xlabel("Release Year")
    ax.set_ylabel("Number of Titles")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
with c2:
    peak_year = year_counts.idxmax()
    st.markdown(f"""
    <div class="insight-box">
    <b>Takeaway:</b> Netflix released way more titles in recent years than it used to —
    {int(peak_year)} was the busiest year, with {int(year_counts.max())} titles. Something
    clearly picked up speed after 2015.
    </div>
    """, unsafe_allow_html=True)

st.subheader("3. Distribution of Top Content Ratings")
st.markdown('<p class="chart-desc">What this shows: the share of titles that fall under each of the six most common content ratings.</p>', unsafe_allow_html=True)
top_ratings = filtered_df["rating"].value_counts().head(6)
c1, c2 = st.columns([2, 1])
with c1:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.pie(top_ratings, labels=top_ratings.index, autopct="%1.1f%%",
           startangle=90, colors=PALETTE)
    fig.tight_layout()
    st.pyplot(fig)
with c2:
    top_rating = top_ratings.index[0]
    top_rating_share = top_ratings.iloc[0] / filtered_df["rating"].dropna().shape[0] * 100
    st.markdown(f"""
    <div class="insight-box">
    <b>Takeaway:</b> More than a third of titles ({top_rating_share:.0f}%) are rated {top_rating} —
    made for grown-up audiences. There's a lot less made specifically for kids and families.
    </div>
    """, unsafe_allow_html=True)


st.subheader("4. Distribution of Release Years")
st.markdown('<p class="chart-desc">What this shows: how many titles were released in each stretch of years, from the oldest to the newest.</p>', unsafe_allow_html=True)
c1, c2 = st.columns([2, 1])
with c1:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(filtered_df["release_year"], bins=20, color=CLR_AMBER, edgecolor="#8a5a2b")
    ax.set_xlabel("Release Year")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    st.pyplot(fig)
with c2:
    recent_share = (filtered_df["release_year"] >= 2015).sum() / filtered_df.shape[0] * 100
    st.markdown(f"""
    <div class="insight-box">
    <b>Takeaway:</b> Most titles here are pretty new — about {recent_share:.0f}% came out in 2015
    or later. There isn't much older, classic content in the catalog.
    </div>
    """, unsafe_allow_html=True)


st.subheader("5. Release Year vs. Duration (Movies)")
st.markdown('<p class="chart-desc">What this shows: each dot is one movie, placed by the year it came out and how long it runs, so we can see if movies have gotten longer or shorter over time.</p>', unsafe_allow_html=True)
movie_df = filtered_df[filtered_df["duration_unit"] == "min"]
c1, c2 = st.columns([2, 1])
with c1:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(movie_df["release_year"], movie_df["duration_value"], alpha=0.5, color=CLR_CORAL)
    ax.set_xlabel("Release Year")
    ax.set_ylabel("Duration (minutes)")
    fig.tight_layout()
    st.pyplot(fig)
with c2:
    st.markdown(f"""
    <div class="insight-box">
    <b>Takeaway:</b> Movie length hasn't really changed much over the years. An older movie
    and a brand-new one are usually a similar length, around {movie_df['duration_value'].median():.0f}
    minutes.
    </div>
    """, unsafe_allow_html=True)


movie_only_df = filtered_df[filtered_df["duration_unit"] == "min"]
tv_only_df = filtered_df[filtered_df["duration_unit"] != "min"]

st.subheader("6. Movie Runtime Spread (Minutes)")
st.markdown('<p class="chart-desc">What this shows: the usual range of movie runtimes in minutes.</p>', unsafe_allow_html=True)
c1, c2 = st.columns([2, 1])
with c1:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=movie_only_df, y="duration_value", ax=ax, color=CLR_SAGE)
    ax.set_ylabel("Duration (minutes)")
    fig.tight_layout()
    st.pyplot(fig)
with c2:
    st.markdown(f"""
    <div class="insight-box">
    <b>Takeaway:</b> Most movies run about an hour and a half to two hours. A few titles are much
    longer or shorter, but those are rare exceptions, not the norm.
    </div>
    """, unsafe_allow_html=True)

st.subheader("7. TV Show Length Spread (Seasons)")
st.markdown('<p class="chart-desc">What this shows: the usual number of seasons for TV shows, shown separately since it uses a different unit than movie runtime.</p>', unsafe_allow_html=True)
c1, c2 = st.columns([2, 1])
with c1:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=tv_only_df, y="duration_value", ax=ax, color=CLR_INDIGO)
    ax.set_ylabel("Duration (seasons)")
    fig.tight_layout()
    st.pyplot(fig)
with c2:
    st.markdown(f"""
    <div class="insight-box">
    <b>Takeaway:</b> Most TV shows have relatively few seasons. Long-running shows are the
    exception, not the norm.
    </div>
    """, unsafe_allow_html=True)


st.subheader("8. Correlation Heatmap")
st.markdown('<p class="chart-desc">What this shows: whether release year, duration, and add-date tend to move together. A number close to 0 means "no real connection"; closer to 1 or -1 means they move together.</p>', unsafe_allow_html=True)
numeric_cols = ["release_year", "year_added", "month_added", "duration_value"]
corr = filtered_df[numeric_cols].corr()
corr = corr.rename(columns={"release_year":"Release Year","year_added":"Year Added","month_added":"Month Added","duration_value":"Duration"},
                    index={"release_year":"Release Year","year_added":"Year Added","month_added":"Month Added","duration_value":"Duration"})
c1, c2 = st.columns([2, 1])
with c1:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(corr, annot=True, cmap="BuGn", ax=ax, linewidths=0.5, linecolor="white")
    fig.tight_layout()
    st.pyplot(fig)
with c2:
    st.markdown("""
    <div class="insight-box">
    <b>Takeaway:</b> None of these numbers are strongly connected to each other. Knowing a
    title's release year doesn't really tell you much about how long it is or when it was added.
    </div>
    """, unsafe_allow_html=True)


st.subheader("9. Top Genres (Category Comparison)")
st.markdown('<p class="chart-desc">What this shows: the 10 most common genre labels in the catalog, side by side, so we can see which ones Netflix leans on most.</p>', unsafe_allow_html=True)
top_genres = (
    filtered_df["listed_in"].dropna().str.split(", ").explode().value_counts().head(10)
)
c1, c2 = st.columns([2, 1])
with c1:
    fig, ax = plt.subplots(figsize=(9, 4))
    top_genres.plot(kind="barh", ax=ax, color=CLR_PLUM)
    ax.set_xlabel("Number of Titles")
    ax.invert_yaxis()
    fig.tight_layout()
    st.pyplot(fig)
with c2:
    top_genre = top_genres.index[0]
    st.markdown(f"""
    <div class="insight-box">
    <b>Takeaway:</b> "{top_genre}" is the single most common genre. Along with Dramas and
    Comedies, just a few genres make up most of the catalog — everything else is a much
    smaller slice.
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")


st.header("Key Insights")

_type_counts = df["type"].value_counts()
_movie_pct = round(_type_counts["Movie"] / _type_counts.sum() * 100, 0)

insights = [
    ("Movies make up about {:.0f}% of the whole catalog".format(_movie_pct),
     "TV shows keep people coming back season after season. A catalog this movie-heavy gives "
     "viewers fewer reasons to stay subscribed month after month."),
    ("Most titles come from just one or two countries",
     "When most content comes from the same one or two places, it mostly reflects those cultures. "
     "Viewers in other regions may not see much of themselves in the catalog, which can hurt growth there."),
    ("The catalog is mostly recent content, with output rising sharply after 2015",
     "A young, fast-growing catalog is exciting, but it also means there isn't much older or "
     "'classic' content — a gap for viewers who enjoy nostalgia or older favorites."),
    ("The catalog leans heavily toward mature and teen ratings",
     "Families with young kids may struggle to find enough suitable options, which could push "
     "them toward a competitor with a bigger kids' section."),
    ("Movie lengths are consistent and unremarkable (around 90-110 minutes)",
     "This isn't a problem to fix — it's just confirmation that movie lengths already match "
     "what audiences expect."),
    ("A handful of genres dominate the catalog",
     "When a few genres take up most of the space, fans of other genres (documentaries, sci-fi, "
     "family films) have less to choose from — a missed chance to keep a wider range of viewers happy."),
]

for pattern, why in insights:
    st.markdown(f"""
    <div class="insight-line">
    📌 <b>{pattern}.</b>
    <div class="insight-why">Why it matters: {why}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")


st.header("Recommendations")



_movie_tv_ratio = round(_type_counts["Movie"] / _type_counts["TV Show"], 1)

_family_ratings = ["G", "PG", "TV-Y", "TV-Y7", "TV-G", "TV-Y7-FV"]
_family_pct = round(df[df["rating"].isin(_family_ratings)].shape[0] / df.shape[0] * 100, 1)

_country_exploded = df[df["country"] != "Unknown"]["country"].str.split(", ").explode()
_top2_country_share = round(_country_exploded.value_counts().head(2).sum() / _country_exploded.shape[0] * 100, 1)

_genre_exploded = df["listed_in"].dropna().str.split(", ").explode()
_top3_genre_share = round(_genre_exploded.value_counts().head(3).sum() / _genre_exploded.shape[0] * 100, 1)

_pre2000_pct = round((df["release_year"] < 2000).sum() / df.shape[0] * 100, 1)

recommendations = [
    ("high", "Add more TV Shows",
     f"Right now there are about {_movie_tv_ratio} movies for every 1 TV show. Since people tend "
     f"to stick around longer for shows than movies, growing this side of the catalog could help "
     f"keep viewers subscribed."),
    ("high", "Add more content for kids and families",
     f"Only about {_family_pct}% of titles are made for kids or the whole family. Most of what's "
     f"there is meant for teens and adults, leaving a real gap for family viewing."),
    ("medium", "Bring in content from more countries",
     f"About {_top2_country_share}% of titles come from just two countries — the US and India. "
     f"Adding more local shows from other regions could help attract new viewers there."),
    ("medium", "Widen the mix of genres",
     f"Three genres alone make up {_top3_genre_share}% of the catalog. Adding more of the "
     f"less-common genres, like documentaries or sci-fi, could appeal to a broader range of tastes."),
    ("low", "Add a small classics section",
     f"Only about {_pre2000_pct}% of titles are from before 2000. A modest collection of older "
     f"favorites could appeal to nostalgic viewers, without needing a big investment."),
    ("low", "No changes needed for movie length",
     "Movie lengths are already right where audiences expect them to be — about an hour and a "
     "half on average. There's nothing to fix here."),
]

for priority, title, evidence in recommendations:
    label = priority.upper() + " PRIORITY"
    st.markdown(f"""
    <div class="rec-card {priority}">
    <span class="rec-priority {priority}">{label}</span><br>
    <b>{title}</b>
    <div class="rec-evidence">{evidence}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.header("Conclusion")
st.markdown("""

The analysis shows that Netflix's content library is primarily dominated by movies, with most titles originating from a limited number of countries, particularly the United States and India. It also contains a large proportion of content designed for mature audiences, while the platform has experienced significant growth in content additions since the mid-2010s.

From a business perspective, these findings highlight an opportunity for Netflix to broaden its content strategy. Expanding the variety of countries represented, offering a wider range of audience ratings, and increasing diversity across genres could help attract new viewers and better serve audience segments that are currently underrepresented. A more balanced and diverse content catalog would strengthen Netflix's ability to reach a broader global audience and support future growth.



""")