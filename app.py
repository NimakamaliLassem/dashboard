import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import altair as alt
from data_processor import load_and_enrich_data, generate_podcast_data


# Page Configuration
st.set_page_config(
    page_title="Harmony: Music & Mood Dashboard",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Premium" Look
st.markdown("""
<style>
    /* FORCE Dark Theme on EVERYTHING */
    .stApp, .main, .block-container {
        background-color: #0E1117 !important;
        color: #FAFAFA !important;
    }

    /* Hide/Fix top header bar */
    header[data-testid="stHeader"] {
        background-color: #0E1117 !important;
        border-bottom: 1px solid #262730;
    }
    header[data-testid="stHeader"] * {
        color: #FAFAFA !important;
    }

    /* Fix the toolbar/decoration bar at top */
    [data-testid="stToolbar"], [data-testid="stDecoration"] {
        background-color: #0E1117 !important;
    }
    .stDeployButton, [data-testid="stStatusWidget"] {
        background-color: #0E1117 !important;
    }

    h1, h2, h3, h4, h5, h6, p, li, span, label {
        color: #FAFAFA !important;
    }

    /* Global font size increase */
    .stApp {
        font-size: 1.05rem !important;
    }
    p, li, span, label, div {
        font-size: 1.05rem;
        line-height: 1.6;
    }

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

    /* Headers */
    h1 {
        font-size: 2.4rem !important;
    }
    h2 {
        font-size: 1.8rem !important;
    }
    h3 {
        font-size: 1.5rem !important;
    }
    h1, h2, h3 {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 600;
    }

    /* Sidebar text */
    [data-testid="stSidebar"] label {
        font-size: 1rem !important;
    }
    [data-testid="stSidebar"] .stSelectbox label {
        font-size: 1.05rem !important;
        font-weight: 500;
    }

    /* Captions and small text */
    .stCaption, figcaption {
        font-size: 0.95rem !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div {
        background-color: #262730 !important;
    }
    [data-testid="stSidebar"] * {
        color: #FAFAFA !important;
    }

    /* SELECTBOX / DROPDOWN - Dark Mode Fix */
    [data-baseweb="select"] {
        background-color: #262730 !important;
    }
    [data-baseweb="select"] > div {
        background-color: #262730 !important;
        border-color: #41444C !important;
    }
    [data-baseweb="select"] span {
        color: #FAFAFA !important;
    }
    [data-baseweb="popover"] {
        background-color: #262730 !important;
    }
    [data-baseweb="popover"] li {
        background-color: #262730 !important;
        color: #FAFAFA !important;
    }
    [data-baseweb="popover"] li:hover {
        background-color: #3d4051 !important;
    }
    /* Dropdown menu */
    [role="listbox"] {
        background-color: #262730 !important;
    }
    [role="option"] {
        background-color: #262730 !important;
        color: #FAFAFA !important;
    }
    [role="option"]:hover {
        background-color: #3d4051 !important;
    }
    /* Select input field */
    .stSelectbox > div > div {
        background-color: #262730 !important;
        color: #FAFAFA !important;
    }
    .stSelectbox [data-baseweb="input"] {
        background-color: #262730 !important;
    }
    .stSelectbox svg {
        fill: #FAFAFA !important;
    }

    /* Fix DataFrame/Table dark mode */
    .stDataFrame, [data-testid="stDataFrame"] {
        background-color: #262730 !important;
    }
    .stDataFrame > div, [data-testid="stDataFrame"] > div {
        background-color: #262730 !important;
    }
    [data-testid="stDataFrame"] iframe {
        background-color: #262730 !important;
    }

    /* PROMINENT Expander styling */
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
    [data-testid="stExpander"] summary:hover {
        background-color: #2d303d !important;
    }
    [data-testid="stExpander"] summary span {
        color: #FAFAFA !important;
    }
    [data-testid="stExpander"] > div {
        background-color: #1a1c23 !important;
        color: #FAFAFA !important;
    }
    [data-testid="stExpander"] p, [data-testid="stExpander"] span,
    [data-testid="stExpander"] div, [data-testid="stExpander"] li {
        color: #FAFAFA !important;
    }

    /* Fix any white backgrounds */
    div[data-baseweb="base-input"] {
        background-color: #262730 !important;
    }
    div[data-baseweb="input"] {
        background-color: #262730 !important;
    }

    /* Color Psychology Card */
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
        width: 18px;
        height: 18px;
        border-radius: 4px;
        border: 1px solid rgba(255,255,255,0.2);
    }
    .color-name {
        color: #A3A8B8 !important;
        font-size: 0.85em;
    }

    /* Expander hint styling */
    .expander-hint {
        display: inline-block;
        background: #FF4B4B;
        color: #FAFAFA !important;
        font-size: 0.7em;
        padding: 2px 8px;
        border-radius: 4px;
        margin-left: 8px;
        font-weight: normal;
    }
</style>
""", unsafe_allow_html=True)

# Application Title
st.title("🎵 Harmony: Personal Music Analytics")
st.markdown("Explore your listening habits, mood patterns, and audio features.")

# Load Data
@st.cache_data
def get_data():
    return load_and_enrich_data('youtube_music_dataset_temp.csv')

df = get_data()

if df is None:
    st.error("Dataset not found. Please ensure 'youtube_music_dataset_temp.csv' is in the project directory.")
    st.stop()

# --- SIDEBAR FILTERS ---
st.sidebar.header("Filter Your Music")

# 1. Semester Selector (Primary Filter)
# Get unique semesters from the data
unique_semesters = df['Semester'].unique().tolist()

# Custom sort order
sem_order = {
    "Fall 2024-2025 (Brussels)": 1,
    "Spring 2024-2025 (Barcelona)": 2,
    "Summer 2025 (Ankara)": 3,
    "Fall 2025-2026 (Paris)": 4
}
unique_semesters.sort(key=lambda x: sem_order.get(x, 99))

# Add "Entire Program" option at the END
available_options = unique_semesters + ["Entire Academic Program"]

selected_semester = st.sidebar.selectbox("Select Semester / Location", available_options)

# Filter data by Semester first
if selected_semester == "Entire Academic Program":
    semester_df = df
else:
    semester_df = df[df['Semester'] == selected_semester]

# 2. Genre Filter (Scoped to Semester)
genres = ['All'] + sorted(semester_df['Genre'].dropna().unique().tolist())
selected_genre = st.sidebar.selectbox("Select Genre", genres)

# 3. Artist Filter (Scoped to Semester & Genre)
if selected_genre != 'All':
    filtered_artists = semester_df[semester_df['Genre'] == selected_genre]['Artist'].unique()
else:
    filtered_artists = semester_df['Artist'].unique()
    
artists = ['All'] + sorted(filtered_artists.tolist())
selected_artist = st.sidebar.selectbox("Select Artist", artists)

# Apply Filters
mask = pd.Series(True, index=semester_df.index)
if selected_genre != 'All':
    mask &= (semester_df['Genre'] == selected_genre)
if selected_artist != 'All':
    mask &= (semester_df['Artist'] == selected_artist)

filtered_df = semester_df[mask]

# --- KPI METRICS ---
col1, col2, col3, col4 = st.columns(4)

total_songs = len(filtered_df)
top_genre = filtered_df['Genre'].mode()[0] if not filtered_df.empty else "N/A"
avg_energy = filtered_df['Energy'].mean() if not filtered_df.empty else 0
avg_mood = filtered_df['Mood_Valence'].mode()[0] if not filtered_df.empty else "N/A"

with col1:
    label_text = "Total Songs" if selected_semester == "Entire Academic Program" else "Songs This Semester"
    st.markdown(f'<div class="metric-card"><div class="metric-value">{total_songs}</div><div class="metric-label">{label_text}</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{top_genre}</div><div class="metric-label">Top Genre</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_energy:.0%}</div><div class="metric-label">Avg. Energy</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_mood}</div><div class="metric-label">Dominant Mood</div></div>', unsafe_allow_html=True)

st.markdown("---")

# --- VISUALIZATIONS ---

# Row 1: Radar Chart & Genre/Mood Distribution
row1_col1, row1_col2 = st.columns([1, 1])

with row1_col1:
    st.subheader("🎧 Audio Features Analysis (Radar Chart)")
    
    if not filtered_df.empty:
        # Calculate averages for radar chart
        features = ['Energy', 'Danceability', 'Valence', 'Acousticness', 'Intense'] # 'Intense' is mapped from Instrumentalness/Liveness just for label variety or use raw
        # Let's use the raw columns we generated
        radar_features = ['Energy', 'Danceability', 'Valence', 'Acousticness', 'Instrumentalness', 'Liveness']
        avg_features = filtered_df[radar_features].mean().values.tolist()
        
        # Close the loop for radar chart
        r_vals = avg_features + [avg_features[0]]
        theta_vals = radar_features + [radar_features[0]]
        
        fig_radar = go.Figure()
        
        # Add current selection
        fig_radar.add_trace(go.Scatterpolar(
            r=r_vals,
            theta=theta_vals,
            fill='toself',
            name='Current Selection',
            line_color='#FF4B4B'
        ))
        
        # Add baseline (global average) for comparison
        global_avg = df[radar_features].mean().values.tolist()
        global_r = global_avg + [global_avg[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=global_r,
            theta=theta_vals,
            name='Global Average',
            line_color='#808495',
            line_dash='dot'
        ))
        
        fig_radar.update_layout(
            template="plotly_dark",
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    gridcolor='#41444C',
                    linecolor='#41444C'
                ),
                bgcolor='rgba(0,0,0,0)'
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#FAFAFA'),
            margin=dict(l=40, r=40, t=20, b=20),
            showlegend=True,
            legend=dict(
                font=dict(color="white"),
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            )
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    else:
        st.info("No data available for Radar Chart.")

with row1_col2:
    st.subheader("📊 Mood Sentiment Breakdown")
    if not filtered_df.empty:
        mood_counts = filtered_df['Mood_Valence'].value_counts().reset_index()
        mood_counts.columns = ['Mood', 'Count']

        # Categorize moods by sentiment for hierarchical sunburst
        def categorize_mood(mood):
            positive_moods = ['Energetic', 'Happy', 'Uplifting', 'Euphoric', 'Joyful',
                           'Romantic', 'Passionate', 'Sensual', 'Seductive',
                           'Hopeful', 'Optimistic', 'Confident', 'Empowering',
                           'Carefree', 'Groovy', 'Adrenaline', 'Exciting', 'Heroic',
                           'Peaceful', 'Relaxed', 'Dreamy', 'Ethereal', 'Adventurous',
                           'Epic', 'Triumphant', 'Transcendent', 'Quirky', 'Spiritual']
            negative_moods = ['Melancholic', 'Sad', 'Somber', 'Depressive', 'Sorrowful',
                            'Anxious', 'Conflicted', 'Desperate', 'Heartbroken',
                            'Dark', 'Ominous', 'Haunting', 'Unsettling', 'Tense',
                            'Resigned', 'Fatalistic', 'Lonely', 'Alienated',
                            'Disorienting', 'Vulnerable', 'Rainy', 'Aggressive',
                            'Nocturnal', 'Dystopian']
            if mood in positive_moods:
                return 'Positive'
            elif mood in negative_moods:
                return 'Negative'
            else:
                return 'Neutral'

        mood_counts['Sentiment'] = mood_counts['Mood'].apply(categorize_mood)

        # Custom color mapping matching dark theme
        sentiment_colors = {
            'Positive': '#FF6B35',   # Warm orange (matches horizon positive)
            'Negative': '#0288D1',   # Cool blue (matches horizon negative)
            'Neutral': '#808495'     # Gray (matches global average line)
        }

        # Only show top moods per sentiment to reduce clutter
        top_moods = mood_counts.groupby('Sentiment').apply(
            lambda x: x.nlargest(5, 'Count')
        ).reset_index(drop=True)

        # Build sunburst data with cleaner structure
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
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#FAFAFA', size=11),
            margin=dict(l=5, r=5, t=5, b=5),
            height=450,
            uniformtext=dict(minsize=8, mode='hide')
        )

        st.plotly_chart(sunburst_fig, use_container_width=True)
        st.caption("Click segments to drill down. Inner = sentiment, outer = top 5 moods. Hover for details.")
    else:
        st.info("No data available.")

st.markdown("---")

# --- CUSTOM HORIZON CHART LOGIC (STUDIO QUALITY) ---
from scipy.interpolate import make_interp_spline

def horizon_chart(ax, data, x, bands=4):
    """
    Creates a HIGH-FIDELITY Diverging Horizon Chart.
    """
    # 1. Center data
    y_centered = data - 0.5
    
    # 2. Config for Visuals
    # Lower deviation = taller peaks = more drama
    max_deviation = 0.35 
    band_height = max_deviation / bands
    
    # PREMIUM PALETTES (Designer Picked)
    # Positive: Warm, energetic, glowing
    # Steps: Cream -> Peach -> Vivid Orange -> Deep Red-Orange
    pos_colors = ['#FFF3E0', '#FFB74D', '#ff6f00', '#BF360C']
    
    # Negative: Cool, calm, deep
    # Steps: Icy Blue -> Sky Blue -> Electric Blue -> Midnight Blue
    neg_colors = ['#E1F5FE', '#4FC3F7', '#0288D1', '#01579B']
    
    # 3. High-Res Interpolation for "Curved" look
    # We create a new density X axis (e.g., 500 points)
    x_new = np.linspace(x.min(), x.max(), 500)
    
    # Cubic Spline for smoothness
    try:
        spl = make_interp_spline(x, y_centered, k=3)
        y_smooth = spl(x_new)
    except:
        # Fallback if too few points
        x_new = x
        y_smooth = y_centered

    # 4. Plot Bands with new smooth data
    for i in range(bands):
        lower = band_height * i
        upper = band_height * (i + 1)
        
        # --- POSITIVE BANDS ---
        pos_signal = np.clip(y_smooth, lower, upper) - lower
        # Plot generally everywhere, masking happens visually by layers
        # But for fill_between 'where' argument is cleaner
        ax.fill_between(x_new, 0, pos_signal, where=(y_smooth > 0), 
                        color=pos_colors[i % len(pos_colors)], alpha=1.0, linewidth=0)
        
        # --- NEGATIVE BANDS ---
        neg_signal = np.clip(np.abs(y_smooth), lower, upper) - lower
        ax.fill_between(x_new, 0, neg_signal, where=(y_smooth < 0), 
                        color=neg_colors[i % len(neg_colors)], alpha=1.0, linewidth=0)

    ax.set_ylim(0, band_height)
    ax.set_xlim(x_new[0], x_new[-1])
    ax.axis('off')

# Row 2: Horizon Chart
st.subheader("🌊 Mood Variance (Horizon Chart)")
st.caption("A visualization of emotional intensity over time. Blue = Calm/Melancholy | Orange = Energetic/Happy.")

if not filtered_df.empty:
    # Prepare data
    time_df = filtered_df.sort_values('Date_Added').set_index('Date_Added')
    
    # Resample to Daily
    resampled = time_df['Valence'].resample('D').mean().fillna(0.5)
    
    # Rolling window for "macro" trend smoothness (removes daily jitter)
    smoothed = resampled.rolling(window=7, min_periods=1, center=True).mean()
    
    # Figure setup for High DPI
    fig_horizon, ax = plt.subplots(figsize=(14, 3), dpi=200) # High Resolution
    fig_horizon.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    
    x_nums = np.arange(len(smoothed))
    
    if len(x_nums) > 3:
        horizon_chart(ax, smoothed.values, x_nums, bands=4)
        
        st.pyplot(fig_horizon, use_container_width=True)
        
        # Custom Legend for Horizon Chart
        st.markdown("""
        <div style="display: flex; justify-content: center; align-items: center; gap: 20px; font-size: 0.9em; color: #A3A8B8; margin-top: -10px;">
            <div style="display: flex; align-items: center; gap: 5px;">
                <span style="display: inline-block; width: 12px; height: 12px; background-color: #01579B;"></span> Considerably Negative
            </div>
            <div style="display: flex; align-items: center; gap: 5px;">
                <span style="display: inline-block; width: 12px; height: 12px; background-color: #4FC3F7;"></span> Moderately Negative
            </div>
            <div style="display: flex; align-items: center; gap: 5px;">
                <span style="display: inline-block; width: 1px; height: 15px; background-color: #555;"></span>
            </div>
            <div style="display: flex; align-items: center; gap: 5px;">
                <span style="display: inline-block; width: 12px; height: 12px; background-color: #FFB74D;"></span> Moderately Positive
            </div>
            <div style="display: flex; align-items: center; gap: 5px;">
                <span style="display: inline-block; width: 12px; height: 12px; background-color: #BF360C;"></span> Considerably Positive
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        st.info("Not enough data points for the Horizon Chart.")

# Row 3: Data Table
st.markdown("""
<div style="background: linear-gradient(90deg, #FF4B4B 0%, #FF6B35 100%); padding: 8px 16px; border-radius: 8px 8px 0 0; margin-bottom: -10px;">
    <span style="color: #FAFAFA; font-weight: 600; font-size: 0.9em;">📝 CLICK TO EXPAND: Detailed Dataset</span>
</div>
""", unsafe_allow_html=True)
with st.expander("View all song data with mood, energy, and danceability scores", expanded=False):
    display_df = filtered_df[['Title', 'Artist', 'Album', 'Genre', 'Mood_Valence', 'Energy', 'Danceability', 'Date_Added']].copy()
    display_df['Energy'] = display_df['Energy'].apply(lambda x: f"{x:.0%}")
    display_df['Danceability'] = display_df['Danceability'].apply(lambda x: f"{x:.0%}")
    display_df['Date_Added'] = pd.to_datetime(display_df['Date_Added']).dt.strftime('%Y-%m-%d')
    display_df.columns = ['Title', 'Artist', 'Album', 'Genre', 'Mood', 'Energy', 'Danceability', 'Date']

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Title": st.column_config.TextColumn("Title", width="medium"),
            "Artist": st.column_config.TextColumn("Artist", width="small"),
            "Mood": st.column_config.TextColumn("Mood", width="small"),
        }
    )

st.markdown("---")

# --- PODCAST SECTION ---
st.subheader("🎙️ Podcast Focus Shift")
st.markdown("Analyzing how listening habits have evolved over time, highlighting the shift in interest.")

pod_df = generate_podcast_data()

if not pod_df.empty:
    # Altair Ridge Plot (Joyplot)
    # We want to show density of listens per category over time.
    
    # Selection for interactive filtering (optional but nice)
    step = 20
    overlap = 1
    
    # We need to aggregate counts by day or week to make density meaningful if data is sparse? 
    # Or just use the dense date data directly.
    
    # Create the Ridge Plot
    # X: Date
    # Y: Category
    # Color: Category
    # Height: Density/Count
    
    ridge = alt.Chart(pod_df).transform_density(
        'Date',
        as_=['Date', 'density'],
        groupby=['Category']
    ).mark_area(
        orient='vertical',
        opacity=0.8,
        interpolate='monotone',
        strokeWidth=1
    ).encode(
        x=alt.X('Date:T', title='Time', axis=alt.Axis(titleColor='#FAFAFA', labelColor='#FAFAFA')),
        y=alt.Y('density:Q', stack=None, axis=None), # Vertical overlap
        row=alt.Row('Category:N', header=alt.Header(title=None, labelAngle=0, labelAlign='left', labelColor='#FAFAFA')),
        color=alt.Color('Category:N', legend=None, scale=alt.Scale(
            domain=['Science', 'Religion', 'Politics'],
            range=['#4FC3F7', '#FFB74D', '#FF4B4B']  # Consistent with app theme
        )),
        tooltip=['Category', 'Date']
    ).properties(
        width='container',
        height=50,
        background='rgba(0,0,0,0)'
    ).configure_axis(
         gridColor='#41444C',
         labelColor='#FAFAFA',
         titleColor='#FAFAFA'
    ).configure_view(
        stroke=None,
        step=40 
    ).configure_facet(
        spacing=0
    ).configure_title(
        anchor='start',
        color='#FAFAFA'
    )
    
    st.altair_chart(ridge, use_container_width=True)
    
    st.caption("Shift in attention observed around Dec 2025, moving significantly towards Political content.")


else:
    st.write("No podcast data avaliable")

st.markdown("---")

# Color Psychology & Design Documentation (Main Content)
st.markdown("""
<div style="background: linear-gradient(90deg, #0288D1 0%, #4FC3F7 100%); padding: 8px 16px; border-radius: 8px 8px 0 0; margin-bottom: -10px;">
    <span style="color: #FAFAFA; font-weight: 600; font-size: 0.9em;">📖 CLICK TO EXPAND: Design Documentation &amp; Color Psychology</span>
</div>
""", unsafe_allow_html=True)
with st.expander("Learn why these colors and chart types were chosen", expanded=False):
    doc_col1, doc_col2 = st.columns(2)

    with doc_col1:
        st.markdown("""
        ### 🎨 Color Psychology

        The color palette in this dashboard is intentionally designed based on **color psychology research**
        to make emotional data instantly intuitive:

        ---

        **Warm Colors (Orange Spectrum)**
        <div class="color-doc">
        <div class="color-swatch"><div class="color-box" style="background:#FFF3E0;"></div><span class="color-name">Cream - Subtle positivity</span></div>
        <div class="color-swatch"><div class="color-box" style="background:#FFB74D;"></div><span class="color-name">Peach - Moderate energy</span></div>
        <div class="color-swatch"><div class="color-box" style="background:#FF6B35;"></div><span class="color-name">Orange - High energy/happiness</span></div>
        <div class="color-swatch"><div class="color-box" style="background:#BF360C;"></div><span class="color-name">Deep Orange - Intense positivity</span></div>
        </div>

        Orange and warm tones are universally associated with **energy, optimism, and warmth**.
        In music context, they represent uplifting, energetic, and happy emotional states.
        The gradient intensity reflects the strength of positive emotion.

        ---

        **Cool Colors (Blue Spectrum)**
        <div class="color-doc">
        <div class="color-swatch"><div class="color-box" style="background:#E1F5FE;"></div><span class="color-name">Ice - Light calm</span></div>
        <div class="color-swatch"><div class="color-box" style="background:#4FC3F7;"></div><span class="color-name">Sky - Moderate calm</span></div>
        <div class="color-swatch"><div class="color-box" style="background:#0288D1;"></div><span class="color-name">Blue - Introspection</span></div>
        <div class="color-swatch"><div class="color-box" style="background:#01579B;"></div><span class="color-name">Midnight - Deep melancholy</span></div>
        </div>

        Blue tones evoke **calm, depth, and contemplation**. In this dashboard, they
        represent lower-valence emotions like melancholy, sadness, and introspection.
        Darker blues indicate stronger negative emotional intensity.

        ---

        **Accent & Neutral Colors**
        <div class="color-doc">
        <div class="color-swatch"><div class="color-box" style="background:#FF4B4B;"></div><span class="color-name">Red - Primary accent/attention</span></div>
        <div class="color-swatch"><div class="color-box" style="background:#808495;"></div><span class="color-name">Gray - Baseline/neutral reference</span></div>
        </div>
        """, unsafe_allow_html=True)

    with doc_col2:
        st.markdown("""
        ### 📊 Chart Selection Rationale

        Each visualization type was chosen for specific analytical purposes:

        ---

        **Radar Chart** (Audio Features)
        - **Why**: Multi-dimensional data comparison at a glance
        - **Strength**: The filled polygon shape makes it easy to compare your selection
          against the global average baseline
        - **Reading**: Larger area = more of that characteristic; overlapping regions
          show similarity to average

        ---

        **Sunburst Chart** (Mood Sentiment)
        - **Why**: Hierarchical part-to-whole relationships
        - **Strength**: Shows both macro (sentiment categories) and micro (individual moods)
          simultaneously with drill-down capability
        - **Reading**: Inner ring = sentiment categories, outer ring = specific moods.
          Segment size = proportion of listening time

        ---

        **Horizon Chart** (Mood Variance)
        - **Why**: Space-efficient time-series for detecting subtle patterns
        - **Strength**: Band layering amplifies small variations that traditional line
          charts would flatten. Shows both direction AND intensity of mood shifts
        - **Reading**: Orange bands above baseline = positive mood deviation;
          Blue bands = negative deviation. Darker = stronger intensity

        ---

        **Ridge Plot** (Podcast Focus)
        - **Why**: Comparing distributions across categories over time
        - **Strength**: Overlapping density curves reveal when and how listening
          patterns shifted between topics
        - **Reading**: Peak height = listening density; horizontal position = time.
          Compare peak positions across categories to see behavioral shifts
        """, unsafe_allow_html=True)

st.markdown("---")
st.caption("Developed as an MVP for Visual Analytics. Color palette based on established color psychology principles.")
