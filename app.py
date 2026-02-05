import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from data_processor import (
    load_and_enrich_data, generate_podcast_data,
    get_timeline_events, get_semester_date_range
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Harmony: Music & Mood Dashboard",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS — DARK THEME
# ============================================================
st.markdown("""
<style>
    /* Force dark theme globally */
    :root { color-scheme: dark !important; }
    .stApp, .main, .block-container {
        background-color: #0E1117 !important;
        color: #FAFAFA !important;
    }
    header[data-testid="stHeader"] {
        background-color: #0E1117 !important;
        border-bottom: 1px solid #262730;
    }
    header[data-testid="stHeader"] * { color: #FAFAFA !important; }
    [data-testid="stToolbar"], [data-testid="stDecoration"] {
        background-color: #0E1117 !important;
    }
    .stDeployButton, [data-testid="stStatusWidget"] {
        background-color: #0E1117 !important;
    }
    h1, h2, h3, h4, h5, h6, p, li, span, label { color: #FAFAFA !important; }

    /* Typography */
    .stApp { font-size: 1.05rem !important; }
    p, li, span, label, div { font-size: 1.05rem; line-height: 1.6; }
    h1 { font-size: 2.4rem !important; }
    h2 { font-size: 1.8rem !important; }
    h3 { font-size: 1.5rem !important; }
    h1, h2, h3 { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; font-weight: 600; }
    .stCaption, figcaption { font-size: 0.95rem !important; }

    /* Metric Cards */
    .metric-card {
        background-color: #262730;
        border: 1px solid #41444C;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        text-align: center;
    }
    .metric-value {
        font-size: 2.2em;
        font-weight: bold;
        color: #FF4B4B !important;
    }
    .metric-label {
        font-size: 1.1em;
        color: #A3A8B8 !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div {
        background-color: #262730 !important;
    }
    [data-testid="stSidebar"] * { color: #FAFAFA !important; }
    [data-testid="stSidebar"] label { font-size: 1rem !important; }
    [data-testid="stSidebar"] .stSelectbox label {
        font-size: 1.05rem !important; font-weight: 500;
    }

    /* Selectbox / Dropdown dark mode */
    [data-baseweb="select"] { background-color: #262730 !important; }
    [data-baseweb="select"] > div {
        background-color: #262730 !important;
        border-color: #41444C !important;
    }
    [data-baseweb="select"] span { color: #FAFAFA !important; }
    [data-baseweb="popover"] { background-color: #262730 !important; }
    [data-baseweb="popover"] li {
        background-color: #262730 !important; color: #FAFAFA !important;
    }
    [data-baseweb="popover"] li:hover { background-color: #3d4051 !important; }
    [role="listbox"] { background-color: #262730 !important; }
    [role="option"] { background-color: #262730 !important; color: #FAFAFA !important; }
    [role="option"]:hover { background-color: #3d4051 !important; }
    .stSelectbox > div > div {
        background-color: #262730 !important; color: #FAFAFA !important;
    }
    .stSelectbox [data-baseweb="input"] { background-color: #262730 !important; }
    .stSelectbox svg { fill: #FAFAFA !important; }

    /* DataFrame/Table — wrapper background only (canvas renderer handles its own theme via config.toml) */
    .stDataFrame, [data-testid="stDataFrame"] { background-color: #262730 !important; }
    .stDataFrame > div, [data-testid="stDataFrame"] > div { background-color: #262730 !important; }

    /* Expander styling */
    [data-testid="stExpander"] {
        background: linear-gradient(135deg, #1E2130 0%, #262730 100%) !important;
        border: 2px solid #FF4B4B !important;
        border-radius: 12px !important;
        margin: 8px 0 !important;
        overflow: hidden;
    }
    [data-testid="stExpander"] summary {
        background-color: #262730 !important;
        color: #FAFAFA !important;
        padding: 12px 16px !important;
        font-weight: 600 !important;
    }
    [data-testid="stExpander"] summary:hover { background-color: #2d303d !important; }
    [data-testid="stExpander"] summary span { color: #FAFAFA !important; }
    [data-testid="stExpander"] > div {
        background-color: #1a1c23 !important; color: #FAFAFA !important;
    }
    [data-testid="stExpander"] p, [data-testid="stExpander"] span,
    [data-testid="stExpander"] div, [data-testid="stExpander"] li {
        color: #FAFAFA !important;
    }

    /* Input fields */
    div[data-baseweb="base-input"] { background-color: #262730 !important; }
    div[data-baseweb="input"] { background-color: #262730 !important; }

    /* Color documentation cards */
    .color-doc {
        background: linear-gradient(135deg, #1a1c23 0%, #262730 100%);
        border: 1px solid #41444C;
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
    }
    .color-swatch {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        margin: 4px 8px 4px 0;
        padding: 4px 10px;
        background: rgba(255,255,255,0.05);
        border-radius: 6px;
    }
    .color-box {
        width: 18px; height: 18px;
        border-radius: 4px;
        border: 1px solid rgba(255,255,255,0.2);
    }
    .color-name { color: #A3A8B8 !important; font-size: 0.85em; }

    /* Section divider */
    .section-header {
        background: linear-gradient(90deg, #FF4B4B 0%, #FF6B35 100%);
        padding: 8px 16px;
        border-radius: 8px 8px 0 0;
        margin-bottom: -10px;
    }
    .section-header span {
        color: #FAFAFA; font-weight: 600; font-size: 0.9em;
    }

    /* Radio buttons & sliders */
    .stRadio > div { flex-direction: row !important; gap: 12px; }
    .stSlider label { color: #FAFAFA !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data
def get_data():
    return load_and_enrich_data('youtube_music_dataset_temp.csv')

@st.cache_data
def get_podcasts():
    return generate_podcast_data()

@st.cache_data
def get_events():
    return get_timeline_events()

df = get_data()
pod_df = get_podcasts()
events_df = get_events()

if df is None:
    st.error("Dataset not found. Please ensure 'youtube_music_dataset_temp.csv' is in the project directory.")
    st.stop()

# ============================================================
# HELPER: Consistent Plotly dark layout
# ============================================================
def apply_dark_layout(fig, height=None):
    layout_args = dict(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#FAFAFA', size=12),
        margin=dict(l=50, r=30, t=40, b=40),
        xaxis=dict(gridcolor='#262730', zerolinecolor='#41444C'),
        yaxis=dict(gridcolor='#262730', zerolinecolor='#41444C'),
    )
    if height:
        layout_args['height'] = height
    fig.update_layout(**layout_args)
    return fig

# ============================================================
# TITLE
# ============================================================
st.title("🎵 Harmony: Personal Music & Mood Analytics")
st.markdown("Understanding your emotional landscape through listening patterns — linking music, mood, and life events.")

# ============================================================
# SIDEBAR FILTERS
# ============================================================
st.sidebar.header("Filter Your Music")

# Semester selector
unique_semesters = df['Semester'].unique().tolist()
sem_order = {
    "Fall 2024-2025 (Brussels)": 1,
    "Spring 2024-2025 (Barcelona)": 2,
    "Summer 2025 (Ankara)": 3,
    "Fall 2025-2026 (Paris)": 4
}
unique_semesters.sort(key=lambda x: sem_order.get(x, 99))
available_options = unique_semesters + ["Entire Academic Program"]
selected_semester = st.sidebar.selectbox("Select Semester / Location", available_options)

# Filter by semester
if selected_semester == "Entire Academic Program":
    semester_df = df
else:
    semester_df = df[df['Semester'] == selected_semester]

# Genre filter (scoped to semester)
genres = ['All'] + sorted(semester_df['Genre'].dropna().unique().tolist())
selected_genre = st.sidebar.selectbox("Select Genre", genres)

# Artist filter (scoped to semester & genre)
if selected_genre != 'All':
    filtered_artists = semester_df[semester_df['Genre'] == selected_genre]['Artist'].unique()
else:
    filtered_artists = semester_df['Artist'].unique()
artists = ['All'] + sorted(filtered_artists.tolist())
selected_artist = st.sidebar.selectbox("Select Artist", artists)

# Apply cascading filters
mask = pd.Series(True, index=semester_df.index)
if selected_genre != 'All':
    mask &= (semester_df['Genre'] == selected_genre)
if selected_artist != 'All':
    mask &= (semester_df['Artist'] == selected_artist)
filtered_df = semester_df[mask]

# Get date range for filtering podcast/events
sem_start, sem_end = get_semester_date_range(selected_semester)

# ============================================================
# KPI CARDS
# ============================================================
col1, col2, col3, col4 = st.columns(4)

total_songs = len(filtered_df)
avg_valence = filtered_df['Valence'].mean() if not filtered_df.empty else 0
top_genre = filtered_df['Genre'].mode()[0] if not filtered_df.empty else "N/A"
dominant_mood = filtered_df['Mood_Valence'].mode()[0] if not filtered_df.empty else "N/A"

# Valence descriptor
if avg_valence >= 0.65:
    valence_desc = "Bright"
elif avg_valence >= 0.45:
    valence_desc = "Balanced"
elif avg_valence >= 0.3:
    valence_desc = "Pensive"
else:
    valence_desc = "Deep"

with col1:
    label = "Total Songs" if selected_semester == "Entire Academic Program" else "Songs This Semester"
    st.markdown(f'<div class="metric-card"><div class="metric-value">{total_songs}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_valence:.2f}</div><div class="metric-label">Avg. Valence ({valence_desc})</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{top_genre}</div><div class="metric-label">Top Genre</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{dominant_mood}</div><div class="metric-label">Dominant Mood</div></div>', unsafe_allow_html=True)

st.markdown("---")

# ============================================================
# MOOD JOURNEY TIMELINE (Replaces Horizon Chart)
# ============================================================
st.subheader("🌊 Your Mood Journey Over Time")
st.caption("Smoothed daily mood valence. Orange = positive/energetic mood, Blue = calm/melancholic mood. Vertical markers show life events and world developments.")

if not filtered_df.empty:
    # Prepare time series
    time_df = filtered_df.sort_values('Date_Added').set_index('Date_Added')
    daily_valence = time_df['Valence'].resample('D').mean().ffill().fillna(0.5)
    smoothed = daily_valence.rolling(window=7, min_periods=1, center=True).mean()

    dates = smoothed.index
    vals = smoothed.values

    fig_mood = go.Figure()

    # Baseline reference (0.5)
    fig_mood.add_trace(go.Scatter(
        x=dates, y=[0.5] * len(dates),
        mode='lines',
        line=dict(color='rgba(128,132,149,0.3)', width=1, dash='dot'),
        showlegend=False, hoverinfo='skip'
    ))

    # Positive fill (above 0.5 — warm orange)
    fig_mood.add_trace(go.Scatter(
        x=dates, y=np.maximum(vals, 0.5),
        fill='tonexty',
        fillcolor='rgba(255, 107, 53, 0.35)',
        line=dict(color='rgba(0,0,0,0)', width=0),
        showlegend=True, name='Positive Mood',
        hoverinfo='skip'
    ))

    # Second baseline for negative fill reference
    fig_mood.add_trace(go.Scatter(
        x=dates, y=[0.5] * len(dates),
        mode='lines',
        line=dict(color='rgba(0,0,0,0)', width=0),
        showlegend=False, hoverinfo='skip'
    ))

    # Negative fill (below 0.5 — cool blue)
    fig_mood.add_trace(go.Scatter(
        x=dates, y=np.minimum(vals, 0.5),
        fill='tonexty',
        fillcolor='rgba(2, 136, 209, 0.35)',
        line=dict(color='rgba(0,0,0,0)', width=0),
        showlegend=True, name='Negative Mood',
        hoverinfo='skip'
    ))

    # Main valence line on top
    fig_mood.add_trace(go.Scatter(
        x=dates, y=vals,
        mode='lines',
        line=dict(color='#FAFAFA', width=2),
        name='Mood Valence',
        hovertemplate='<b>%{x|%b %d, %Y}</b><br>Valence: %{y:.2f}<extra></extra>'
    ))

    # Add event annotations
    relevant_events = events_df[
        (events_df['date'] >= sem_start) & (events_df['date'] <= sem_end)
    ]

    for idx, event in relevant_events.iterrows():
        is_world = event['type'] == 'world'
        color = '#FF4B4B' if is_world else '#A78BFA'
        dash = 'dash' if is_world else 'dot'

        fig_mood.add_vline(
            x=event['date'], line_dash=dash,
            line_color=color, opacity=0.7, line_width=1
        )
        fig_mood.add_annotation(
            x=event['date'], y=1.02, yref='paper',
            text=event['label'],
            showarrow=False,
            font=dict(size=9, color=color),
            textangle=-35,
            xanchor='left'
        )

    fig_mood.update_layout(
        yaxis=dict(
            range=[0, 1], dtick=0.25,
            title='Valence',
            tickvals=[0, 0.25, 0.5, 0.75, 1.0],
            ticktext=['Low', '', 'Neutral', '', 'High'],
        ),
        xaxis=dict(title='Date'),
        legend=dict(
            orientation='h', yanchor='top', y=-0.08,
            xanchor='center', x=0.5,
            font=dict(color='#FAFAFA')
        ),
    )
    apply_dark_layout(fig_mood, height=450)
    # Extra top margin for annotations, bottom margin for legend
    fig_mood.update_layout(margin=dict(t=100, b=80))

    st.plotly_chart(fig_mood, use_container_width=True)

    # Legend explanation
    st.markdown("""
    <div style="display: flex; justify-content: center; gap: 25px; font-size: 0.85em; color: #A3A8B8; margin-top: -10px;">
        <div style="display: flex; align-items: center; gap: 6px;">
            <span style="display: inline-block; width: 20px; height: 2px; border-top: 2px dashed #A78BFA;"></span> City Move
        </div>
        <div style="display: flex; align-items: center; gap: 6px;">
            <span style="display: inline-block; width: 20px; height: 2px; border-top: 2px dashed #FF4B4B;"></span> World Event
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("No data available for the selected filters.")

st.markdown("---")

# ============================================================
# ROW 2: RADAR CHART & SUNBURST
# ============================================================
row2_col1, row2_col2 = st.columns([1, 1])

with row2_col1:
    st.subheader("🎧 Audio Features (Radar)")

    if not filtered_df.empty:
        radar_features = ['Energy', 'Danceability', 'Valence', 'Acousticness', 'Instrumentalness', 'Liveness']
        avg_features = filtered_df[radar_features].mean().values.tolist()

        r_vals = avg_features + [avg_features[0]]
        theta_vals = radar_features + [radar_features[0]]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=r_vals, theta=theta_vals,
            fill='toself', name='Current Selection',
            line_color='#FF4B4B', fillcolor='rgba(255,75,75,0.2)'
        ))

        global_avg = df[radar_features].mean().values.tolist()
        global_r = global_avg + [global_avg[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=global_r, theta=theta_vals,
            name='Global Average',
            line_color='#808495', line_dash='dot'
        ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], gridcolor='#41444C', linecolor='#41444C'),
                bgcolor='rgba(0,0,0,0)'
            ),
            showlegend=True,
            legend=dict(font=dict(color='white'), orientation='h',
                        yanchor='bottom', y=-0.2, xanchor='center', x=0.5)
        )
        apply_dark_layout(fig_radar)
        st.plotly_chart(fig_radar, use_container_width=True)
    else:
        st.info("No data available for Radar Chart.")

with row2_col2:
    st.subheader("📊 Mood Sentiment Breakdown")

    if not filtered_df.empty:
        mood_counts = filtered_df['Mood_Valence'].value_counts().reset_index()
        mood_counts.columns = ['Mood', 'Count']

        def categorize_mood(mood):
            positive = ['Energetic', 'Happy', 'Uplifting', 'Euphoric', 'Joyful',
                        'Romantic', 'Passionate', 'Sensual', 'Seductive',
                        'Hopeful', 'Optimistic', 'Confident', 'Empowering',
                        'Carefree', 'Groovy', 'Adrenaline', 'Exciting', 'Heroic',
                        'Peaceful', 'Relaxed', 'Dreamy', 'Ethereal', 'Adventurous',
                        'Epic', 'Triumphant', 'Transcendent', 'Quirky', 'Spiritual']
            negative = ['Melancholic', 'Sad', 'Somber', 'Depressive', 'Sorrowful',
                        'Anxious', 'Conflicted', 'Desperate', 'Heartbroken',
                        'Dark', 'Ominous', 'Haunting', 'Unsettling', 'Tense',
                        'Resigned', 'Fatalistic', 'Lonely', 'Alienated',
                        'Disorienting', 'Vulnerable', 'Rainy', 'Aggressive',
                        'Nocturnal', 'Dystopian']
            if mood in positive:
                return 'Positive'
            elif mood in negative:
                return 'Negative'
            return 'Neutral'

        mood_counts['Sentiment'] = mood_counts['Mood'].apply(categorize_mood)
        sentiment_colors = {
            'Positive': '#FF6B35',
            'Negative': '#0288D1',
            'Neutral': '#808495'
        }

        top_moods = mood_counts.groupby('Sentiment').apply(
            lambda x: x.nlargest(5, 'Count')
        ).reset_index(drop=True)

        sunburst_fig = go.Figure(go.Sunburst(
            ids=[f"{row['Sentiment']}-{row['Mood']}" for _, row in top_moods.iterrows()] +
                list(top_moods['Sentiment'].unique()) + ['All Moods'],
            labels=[row['Mood'] for _, row in top_moods.iterrows()] +
                   list(top_moods['Sentiment'].unique()) + ['All Moods'],
            parents=[row['Sentiment'] for _, row in top_moods.iterrows()] +
                    ['All Moods'] * len(top_moods['Sentiment'].unique()) + [''],
            values=[row['Count'] for _, row in top_moods.iterrows()] +
                   [top_moods[top_moods['Sentiment'] == s]['Count'].sum()
                    for s in top_moods['Sentiment'].unique()] + [top_moods['Count'].sum()],
            branchvalues='total',
            marker=dict(
                colors=[sentiment_colors.get(row['Sentiment'], '#808495')
                       for _, row in top_moods.iterrows()] +
                       [sentiment_colors.get(s, '#808495')
                        for s in top_moods['Sentiment'].unique()] + ['#262730'],
                line=dict(color='#0E1117', width=2)
            ),
            hovertemplate='<b>%{label}</b><br>Songs: %{value}<br>Percentage: %{percentRoot:.1%}<extra></extra>',
            textinfo='label+percent entry',
            insidetextorientation='auto',
            textfont=dict(size=11, color='#FAFAFA'),
            rotation=90
        ))

        sunburst_fig.update_layout(
            font=dict(color='#FAFAFA', size=11),
            margin=dict(l=5, r=5, t=5, b=5),
            height=450,
            uniformtext=dict(minsize=8, mode='hide')
        )
        apply_dark_layout(sunburst_fig)
        sunburst_fig.update_layout(margin=dict(l=5, r=5, t=5, b=5))
        st.plotly_chart(sunburst_fig, use_container_width=True)
        st.caption("Click segments to drill down. Inner = sentiment, outer = top 5 moods per sentiment.")
    else:
        st.info("No data available.")

st.markdown("---")

# ============================================================
# MOOD DISTRIBUTION BY SEMESTER (Violin Plots)
# ============================================================
st.subheader("📈 Mood Distribution Across Your Journey")
st.caption("Compare emotional spread across semesters. Wider sections = more songs at that valence level. All semesters shown for context; selected semester is highlighted.")

semester_order = [
    "Fall 2024-2025 (Brussels)",
    "Spring 2024-2025 (Barcelona)",
    "Summer 2025 (Ankara)",
    "Fall 2025-2026 (Paris)"
]
semester_colors = {
    "Fall 2024-2025 (Brussels)": '#A78BFA',
    "Spring 2024-2025 (Barcelona)": '#34D399',
    "Summer 2025 (Ankara)": '#FBBF24',
    "Fall 2025-2026 (Paris)": '#F87171'
}

fig_violin = go.Figure()
city_names = []
for sem in semester_order:
    sem_data = df[df['Semester'] == sem]['Valence']
    is_selected = (sem == selected_semester) or (selected_semester == "Entire Academic Program")
    city = sem.split('(')[1].rstrip(')')
    city_names.append(city)

    mean_val = sem_data.mean()
    median_val = sem_data.median()

    fig_violin.add_trace(go.Violin(
        x0=city,
        y=sem_data,
        name=city,
        box_visible=True,
        meanline_visible=True,
        fillcolor=semester_colors[sem],
        opacity=1.0 if is_selected else 0.3,
        line_color='#FAFAFA' if is_selected else '#41444C',
        line_width=1.5 if is_selected else 0.5,
        points=False,
        scalemode='width',
        width=0.7,
        hoveron='violins',
        hovertemplate='%{x}<br>Valence: %{y:.2f}<extra></extra>',
    ))

fig_violin.update_layout(
    xaxis=dict(title=''),
    yaxis=dict(title='Valence', range=[0, 1]),
    showlegend=False,
)
apply_dark_layout(fig_violin, height=350)
st.plotly_chart(fig_violin, use_container_width=True)

st.markdown("---")

# ============================================================
# PODCAST SECTION: Stacked Area + Treemap
# ============================================================
st.subheader("🎙️ Podcast Listening Patterns")
st.markdown("Tracking how listening interests have evolved over time. The shift toward political content reflects growing engagement with the situation in Iran.")

# Filter podcast data by semester date range
pod_filtered = pod_df[(pod_df['Date'] >= sem_start) & (pod_df['Date'] <= sem_end)]

pod_category_colors = {
    'Politics': '#FF4B4B',
    'Science': '#4FC3F7',
    'Religion': '#FFB74D'
}

if not pod_filtered.empty:
    pod_col1, pod_col2 = st.columns([3, 2])

    with pod_col1:
        st.markdown("**Focus Shift Over Time**")

        # Aggregate by week and category
        pod_weekly = (pod_filtered
                      .set_index('Date')
                      .groupby([pd.Grouper(freq='W'), 'Category'])['Listens']
                      .sum()
                      .reset_index())

        fig_pod = px.area(
            pod_weekly, x='Date', y='Listens', color='Category',
            color_discrete_map=pod_category_colors,
            category_orders={'Category': ['Science', 'Religion', 'Politics']},
            line_shape='spline'
        )

        fig_pod.update_traces(
            hovertemplate='<b>Week of %{x|%b %d, %Y}</b><br>%{data.name}: %{y} episodes<extra></extra>'
        )

        fig_pod.update_layout(
            xaxis=dict(title=''),
            yaxis=dict(title='Weekly Episodes'),
            legend=dict(
                title=dict(text=''),
                orientation='h', yanchor='top', y=-0.08,
                xanchor='center', x=0.5,
                font=dict(color='#FAFAFA', size=13),
            ),
        )
        apply_dark_layout(fig_pod, height=520)
        fig_pod.update_layout(margin=dict(b=80))
        st.plotly_chart(fig_pod, use_container_width=True)

    with pod_col2:
        st.markdown("**Listening Breakdown**")

        # Treemap by category and show
        treemap_data = (pod_filtered
                        .groupby(['Category', 'Show'])['Listens']
                        .sum()
                        .reset_index())

        fig_tree = px.treemap(
            treemap_data,
            path=['Category', 'Show'],
            values='Listens',
            color='Category',
            color_discrete_map=pod_category_colors,
        )

        fig_tree.update_traces(
            hovertemplate='<b>%{label}</b><br>Episodes: %{value}<br>%{percentParent:.1%} of category<extra></extra>',
            textinfo='label+percent parent',
            textfont=dict(color='#FAFAFA', family='Arial Black, Helvetica Neue, sans-serif'),
            marker=dict(line=dict(color='#0E1117', width=2)),
        )

        apply_dark_layout(fig_tree, height=500)
        fig_tree.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            treemapcolorway=['#FF4B4B', '#4FC3F7', '#FFB74D'],
        )
        st.plotly_chart(fig_tree, use_container_width=True)

    st.caption("The stacked area shows weekly listening volume per category. The treemap shows total proportion by show. Both filtered by selected semester.")
else:
    st.info("No podcast data available for the selected time period.")

st.markdown("---")

# ============================================================
# DATASET EXPLORER (Enhanced with Filters)
# ============================================================
st.markdown("""
<div class="section-header">
    <span>📝 CLICK TO EXPAND: Dataset Explorer</span>
</div>
""", unsafe_allow_html=True)

with st.expander("Browse, filter, and sort your song data", expanded=False):
    if not filtered_df.empty:
        # Filter controls
        filter_cols = st.columns([2, 2, 1, 2, 1])

        with filter_cols[0]:
            filter_feature = st.selectbox(
                "Filter by Feature",
                ["None", "Energy", "Danceability", "Valence", "Acousticness", "Instrumentalness"]
            )

        with filter_cols[1]:
            if filter_feature != "None":
                threshold = st.slider(f"{filter_feature} Threshold", 0.0, 1.0, 0.5, 0.05)
            else:
                threshold = 0.5

        with filter_cols[2]:
            if filter_feature != "None":
                comparison = st.radio("Show", ["Above", "Below"], horizontal=True)
            else:
                comparison = "Above"

        with filter_cols[3]:
            sort_by = st.selectbox(
                "Sort by",
                ["Date", "Title", "Artist", "Genre", "Energy", "Danceability", "Valence"]
            )

        with filter_cols[4]:
            sort_order = st.radio("Order", ["Asc", "Desc"], horizontal=True)

        # Prepare display dataframe
        display_df = filtered_df[['Title', 'Artist', 'Album', 'Genre', 'Mood_Valence',
                                   'Energy', 'Danceability', 'Valence', 'Acousticness',
                                   'Instrumentalness', 'Date_Added']].copy()
        display_df['Date'] = pd.to_datetime(display_df['Date_Added']).dt.strftime('%Y-%m-%d')

        # Apply feature filter
        if filter_feature != "None":
            if comparison == "Above":
                display_df = display_df[display_df[filter_feature] >= threshold]
            else:
                display_df = display_df[display_df[filter_feature] <= threshold]

        # Apply sorting
        sort_col_map = {
            "Date": "Date_Added", "Title": "Title", "Artist": "Artist",
            "Genre": "Genre", "Energy": "Energy", "Danceability": "Danceability",
            "Valence": "Valence"
        }
        ascending = sort_order == "Asc"
        display_df = display_df.sort_values(sort_col_map[sort_by], ascending=ascending)

        # Format for display
        show_df = display_df[['Title', 'Artist', 'Genre', 'Mood_Valence', 'Energy',
                               'Danceability', 'Valence', 'Date']].copy()
        show_df.columns = ['Title', 'Artist', 'Genre', 'Mood', 'Energy', 'Danceability', 'Valence', 'Date']

        # Format percentages
        for col in ['Energy', 'Danceability']:
            show_df[col] = show_df[col].apply(lambda x: f"{x:.0%}")
        show_df['Valence'] = show_df['Valence'].apply(lambda x: f"{x:.2f}")

        st.markdown(f"Showing **{len(show_df)}** songs")
        st.dataframe(
            show_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Title": st.column_config.TextColumn("Title", width="medium"),
                "Artist": st.column_config.TextColumn("Artist", width="small"),
                "Mood": st.column_config.TextColumn("Mood", width="small"),
            }
        )
    else:
        st.info("No data to display.")

st.markdown("---")

# ============================================================
# METHODOLOGY & DESIGN DOCUMENTATION
# ============================================================
st.markdown("""
<div style="background: linear-gradient(90deg, #0288D1 0%, #4FC3F7 100%); padding: 8px 16px; border-radius: 8px 8px 0 0; margin-bottom: -10px;">
    <span style="color: #FAFAFA; font-weight: 600; font-size: 0.9em;">📖 CLICK TO EXPAND: Methodology & Design Documentation</span>
</div>
""", unsafe_allow_html=True)

with st.expander("How mood was extracted, color psychology, and chart rationale", expanded=False):
    meth_col1, meth_col2 = st.columns(2)

    with meth_col1:
        st.markdown("""
        ### 🔬 Mood Extraction Methodology

        **How were mood labels and valence scores derived?**

        The mood classification for this dataset was performed using a local LLM
        (Large Language Model) through **Ollama**. The process:

        1. **Initial Approach**: The raw YouTube Music listening history was fed to a
           locally-hosted LLM via Ollama, which classified each track's mood based on
           song title, artist, genre, and available metadata.

        2. **Valence Mapping**: Each mood label (e.g., "Melancholic", "Energetic") was
           then mapped to a numerical valence score on a 0–1 scale, informed by music
           psychology research (Russell's circumplex model of affect).

        3. **Privacy Decision**: During the process, it became apparent that too much
           personal information was leaking through the raw listening data — specific
           timestamps, listening patterns, and song choices could reveal sensitive details
           about daily routines and emotional states. To protect privacy while maintaining
           analytical value, **the majority of the dataset was regenerated synthetically
           using the LLM**, preserving the statistical properties and mood patterns of
           the original data without exposing private listening habits.

        4. **Synthetic Expansion**: The mood drift algorithm uses weighted sampling to
           simulate realistic listening patterns across semesters, with event-based
           modifiers that reflect real-world emotional impacts (city moves, global events).

        ---

        ### 🎙️ Podcast Data

        Podcast listening data was generated to reflect real listening patterns:
        - **Primary shows**: Kyle Kulinski Show (politics), Professor Dave Explains
          (science), Alex O'Connor (religion/philosophy)
        - **Pattern**: Interest shifted toward political content correlating with
          developments in the Iran situation, with notable spikes in July-August 2025
          and January 2026.
        """)

    with meth_col2:
        st.markdown("""
        ### 🎨 Color Psychology

        The color palette is designed based on **color psychology research**
        to make emotional data intuitive:

        ---

        **Mood Valence Colors** — Used in the Mood Timeline and Sunburst
        <div class="color-doc">
        <div class="color-swatch"><div class="color-box" style="background:#FF6B35;"></div><span class="color-name">Orange fill — Positive mood (above baseline)</span></div>
        <div class="color-swatch"><div class="color-box" style="background:#0288D1;"></div><span class="color-name">Blue fill — Negative mood (below baseline)</span></div>
        </div>

        Orange = **energy, optimism, warmth** (uplifting emotions).
        Blue = **calm, depth, contemplation** (introspective emotions).
        This warm-vs-cool split follows established affective color research,
        where warm hues map to high-arousal positive states and cool hues
        map to low-arousal or negative states.

        ---

        **Podcast Category Colors**
        <div class="color-doc">
        <div class="color-swatch"><div class="color-box" style="background:#FF4B4B;"></div><span class="color-name">Red — Politics (urgency, intensity)</span></div>
        <div class="color-swatch"><div class="color-box" style="background:#4FC3F7;"></div><span class="color-name">Sky Blue — Science (analytical, rational)</span></div>
        <div class="color-swatch"><div class="color-box" style="background:#FFB74D;"></div><span class="color-name">Warm Orange — Religion (spiritual, contemplative)</span></div>
        </div>

        Red draws attention to the growing political content — its urgency
        mirrors the real-world intensity of the situation being followed.
        Blue for science echoes analytical thinking. Orange for religion/philosophy
        evokes warmth and spiritual contemplation.

        ---

        **Semester / City Colors** — Used in Violin Plots
        <div class="color-doc">
        <div class="color-swatch"><div class="color-box" style="background:#A78BFA;"></div><span class="color-name">Lavender — Brussels (autumn, beginnings)</span></div>
        <div class="color-swatch"><div class="color-box" style="background:#34D399;"></div><span class="color-name">Mint — Barcelona (spring, freshness)</span></div>
        <div class="color-swatch"><div class="color-box" style="background:#FBBF24;"></div><span class="color-name">Gold — Ankara (summer, warmth)</span></div>
        <div class="color-swatch"><div class="color-box" style="background:#F87171;"></div><span class="color-name">Coral — Paris (autumn, intensity)</span></div>
        </div>

        Each city gets a seasonally-inspired hue that avoids conflict with
        the valence warm/cool system. They serve as neutral identifiers.

        ---

        **Accent & UI Colors**
        <div class="color-doc">
        <div class="color-swatch"><div class="color-box" style="background:#FF4B4B;"></div><span class="color-name">Red — Primary accent, KPI values, world events</span></div>
        <div class="color-swatch"><div class="color-box" style="background:#A78BFA;"></div><span class="color-name">Lavender — City move markers on timeline</span></div>
        <div class="color-swatch"><div class="color-box" style="background:#808495;"></div><span class="color-name">Gray — Baseline, neutral reference, deselected items</span></div>
        <div class="color-swatch"><div class="color-box" style="background:#0E1117;"></div><span class="color-name">Near-black — Background (reduces eye strain, emphasizes data)</span></div>
        </div>

        ---

        ### 📊 Chart Selection Rationale

        **1. Mood Journey Timeline** — *"Your Mood Journey Over Time"*
        - **Type**: Diverging area chart with event annotations
        - **Purpose**: Shows the temporal evolution of mood valence across
          your entire listening history
        - **How to read**: White line = smoothed daily valence. Orange fill above
          the 0.5 baseline = positive mood periods. Blue fill below = negative.
          Dashed vertical lines mark city moves (lavender) and world events (red)
        - **Why chosen**: Replaces the horizon chart — uses the full display space,
          has a clear x-axis with dates, and the event annotations let you connect
          mood dips/peaks to real life events

        **2. Radar Chart** — *"Audio Features"*
        - **Type**: Polar/spider chart
        - **Purpose**: Compare 6 audio features (Energy, Danceability, Valence,
          Acousticness, Instrumentalness, Liveness) between your current selection
          and the global average
        - **How to read**: Larger area = more of that feature. Dotted gray line =
          global average baseline for comparison
        - **Why chosen**: Multi-dimensional comparison at a glance — the filled
          polygon shape makes differences immediately visible

        **3. Sunburst Chart** — *"Mood Sentiment Breakdown"*
        - **Type**: Hierarchical pie/sunburst
        - **Purpose**: Shows the breakdown of moods by sentiment category
          (Positive / Negative / Neutral) with drill-down to individual moods
        - **How to read**: Inner ring = sentiment categories, outer ring = top 5
          specific moods per sentiment. Click to drill down. Segment size = proportion
        - **Why chosen**: Reveals both macro-level emotional balance and the specific
          moods driving it, with interactive exploration

        **4. Violin Plots** — *"Mood Distribution Across Your Journey"*
        - **Type**: Violin + box plot hybrid
        - **Purpose**: Compare the full shape of mood distributions across semesters
        - **How to read**: Width = density of songs at that valence. Box inside shows
          quartiles and median. Wider sections = more songs at that valence level
        - **Why chosen**: Shows mood *distribution*, not just averages — reveals whether
          a semester was consistently stable or had emotional swings

        **5. Podcast Stacked Area** — *"Focus Shift Over Time"*
        - **Type**: Stacked area chart (weekly aggregation)
        - **Purpose**: Shows how podcast listening volume and category proportions
          changed over time, highlighting the shift toward political content
        - **How to read**: Total height = weekly listening volume. Color bands show
          the proportion per category. Growing red = more politics
        - **Why chosen**: Shows both absolute volume increase and proportional shifts,
          making the Iran-related interest spike clearly visible

        **6. Podcast Treemap** — *"Listening Breakdown"*
        - **Type**: Interactive treemap
        - **Purpose**: Part-to-whole overview of total podcast consumption by
          category and individual show
        - **How to read**: Box size = total episodes. Nested boxes = shows within
          categories. Colors match the stacked area chart
        - **Why chosen**: Gives an at-a-glance summary of which shows dominate
          your listening, with consistent color coding for category recognition
        """, unsafe_allow_html=True)

st.markdown("---")
st.caption("Harmony — Personal Music & Mood Analytics Dashboard. Color palette based on established color psychology principles.")
