import pandas as pd
import numpy as np


def load_and_enrich_data(filepath):
    """
    Loads the music dataset and generates synthetic listening history.
    Mood drift is influenced by life events (city moves, Iran situation).
    Uses weighted song sampling based on daily mood targets.
    """
    try:
        source_df = pd.read_csv(filepath)
    except FileNotFoundError:
        return None

    # --- 1. Mood Valence Mapping ---
    mood_map = {
        # High Valence (Positive / Warm)
        'Energetic': 0.85, 'Happy': 0.9, 'Uplifting': 0.85, 'Euphoric': 0.95, 'Joyful': 0.9,
        'Romantic': 0.7, 'Passionate': 0.75, 'Sensual': 0.7, 'Seductive': 0.65,
        'Hopeful': 0.75, 'Optimistic': 0.8, 'Confident': 0.8, 'Empowering': 0.85,
        'Carefree': 0.8, 'Groovy': 0.75, 'Adrenaline': 0.9, 'Exciting': 0.85, 'Heroic': 0.8,
        'Peaceful': 0.65, 'Relaxed': 0.6, 'Dreamy': 0.6, 'Ethereal': 0.6, 'Adventurous': 0.75,
        'Cinema': 0.6, 'Cinematic': 0.55, 'Epic': 0.8, 'Triumphant': 0.9, 'Transcendent': 0.7,
        'Hypnotic': 0.55, 'Mystical': 0.55, 'Exploratory': 0.65,
        # Low Valence (Negative / Cool)
        'Melancholic': 0.2, 'Sad': 0.1, 'Somber': 0.15, 'Depressive': 0.1, 'Sorrowful': 0.1,
        'Anxious': 0.25, 'Conflicted': 0.3, 'Desperate': 0.2, 'Heartbroken': 0.15,
        'Dark': 0.15, 'Ominous': 0.1, 'Haunting': 0.2, 'Unsettling': 0.2, 'Drama': 0.35,
        'Dramatic': 0.35, 'Tense': 0.25, 'Resigned': 0.3, 'Fatalistic': 0.2, 'Wistful': 0.35,
        'Nostalgic': 0.4, 'Bittersweet': 0.4, 'Lonely': 0.15, 'Alienated': 0.2,
        'Disorienting': 0.3, 'Vulnerable': 0.35, 'Rainy': 0.3, 'Sarcastic': 0.4,
        'Rebellious': 0.45, 'Aggressive': 0.3, 'Introspective': 0.4, 'Contemplative': 0.4,
        'Reflective': 0.4, 'Meditative': 0.45, 'Abstract': 0.5, 'Experimental': 0.5,
        'Mysterious': 0.4, 'Nocturnal': 0.3, 'Dystopian': 0.2, 'Rootsy': 0.55,
        'Cathartic': 0.45, 'Quirky': 0.65, 'Spiritual': 0.6,
        # Additional moods from dataset
        'Powerful': 0.75, 'Defiant': 0.5, 'Intense': 0.45, 'Emotional': 0.4,
        'Storytelling': 0.55, 'Yearning': 0.3, 'Questioning': 0.4, 'Longing': 0.35
    }

    if 'Mood_Valence' in source_df.columns:
        source_df['Valence'] = source_df['Mood_Valence'].map(mood_map)
        missing_mask = source_df['Valence'].isna()
        source_df.loc[missing_mask, 'Valence'] = np.random.uniform(0.4, 0.6, size=missing_mask.sum())
    else:
        source_df['Valence'] = np.random.uniform(0.0, 1.0, size=len(source_df))

    # --- 2. Semester Definitions ---
    semesters = [
        ("Fall 2024-2025 (Brussels)",    pd.Timestamp("2024-09-01"), pd.Timestamp("2025-01-31")),
        ("Spring 2024-2025 (Barcelona)", pd.Timestamp("2025-02-01"), pd.Timestamp("2025-06-30")),
        ("Summer 2025 (Ankara)",         pd.Timestamp("2025-07-01"), pd.Timestamp("2025-08-31")),
        ("Fall 2025-2026 (Paris)",       pd.Timestamp("2025-09-01"), pd.Timestamp("2026-02-05")),
    ]

    # --- 3. Event-Based Mood Modifiers ---
    # Negative values = mood drops, positive = mood lifts
    mood_events = {
        pd.Timestamp("2025-07-10"): -0.15,   # Iran situation begins
        pd.Timestamp("2025-07-25"): -0.12,   # Continued tension
        pd.Timestamp("2025-08-10"): -0.08,   # Ongoing concern
        pd.Timestamp("2025-09-05"):  0.06,   # Settling into Paris, slight recovery
        pd.Timestamp("2025-10-15"):  0.03,   # Routine establishes
        pd.Timestamp("2026-01-05"): -0.22,   # Iran: major escalation
        pd.Timestamp("2026-01-15"): -0.18,   # Crisis deepens
        pd.Timestamp("2026-01-25"): -0.14,   # Continued stress
    }

    # --- 4. Generate Synthetic Listening History ---
    np.random.seed(42)
    expanded_rows = []
    current_mood_target = 0.5

    for sem_name, start_date, end_date in semesters:
        days_range = pd.date_range(start_date, end_date, freq='D')

        # Semester-specific mood baseline
        if "Summer" in sem_name:
            current_mood_target = 0.65   # Summer starts optimistic but drops with Iran news
        elif "Fall 2025" in sem_name:
            current_mood_target = 0.45   # Paris: Iran weighing on mood
        elif "Fall 2024" in sem_name:
            current_mood_target = 0.55   # Brussels: fresh start
        else:
            current_mood_target = 0.58   # Barcelona: positive overall

        semester_data = []
        t = 0

        for day in days_range:
            t += 1
            # Monthly oscillation (simulates natural mood cycles)
            seasonality = 0.15 * np.sin(2 * np.pi * t / 30)

            # Random walk drift
            drift = np.random.normal(0, 0.04)

            # Apply event effects (Gaussian decay over ~30 days)
            event_effect = 0
            for event_date, impact in mood_events.items():
                days_since = (day - event_date).days
                if 0 <= days_since <= 30:
                    event_effect += impact * np.exp(-0.5 * (days_since / 10) ** 2)

            current_mood_target += drift
            current_mood_target = np.clip(current_mood_target, 0.15, 0.85)

            daily_target = current_mood_target + seasonality + event_effect
            daily_target = np.clip(daily_target, 0.05, 0.95)

            # Weighted song selection: prefer songs close to daily mood target
            weights = 1 / (np.abs(source_df['Valence'] - daily_target) + 0.05)
            weights = weights / weights.sum()

            daily_songs_count = np.random.randint(8, 25)
            daily_songs = source_df.sample(n=daily_songs_count, replace=True, weights=weights).copy()

            # Assign timestamps throughout the day
            hours = np.random.randint(0, 24, size=daily_songs_count)
            minutes = np.random.randint(0, 60, size=daily_songs_count)
            daily_timestamps = [day + pd.Timedelta(hours=int(h), minutes=int(m))
                                for h, m in zip(hours, minutes)]

            daily_songs['Date_Added'] = daily_timestamps
            daily_songs['Semester'] = sem_name
            semester_data.append(daily_songs)

        expanded_rows.extend(semester_data)

    # Combine all semesters
    df = pd.concat(expanded_rows).reset_index(drop=True)
    df['Date_Added'] = pd.to_datetime(df['Date_Added'])
    df = df.sort_values('Date_Added')

    # --- 5. Synthetic Audio Features ---
    n = len(df)
    df['Energy'] = np.random.uniform(0.1, 0.95, size=n)
    df['Danceability'] = np.random.uniform(0.1, 0.95, size=n)
    df['Acousticness'] = np.random.uniform(0.0, 0.8, size=n)
    df['Instrumentalness'] = np.random.exponential(scale=0.2, size=n).clip(0, 1)
    df['Liveness'] = np.random.uniform(0.05, 0.4, size=n)
    df['Speechiness'] = np.random.beta(a=1, b=5, size=n)

    # Tempo-based adjustments for realism
    if 'Tempo_Category' in df.columns:
        mask_fast = df['Tempo_Category'] == 'Fast'
        df.loc[mask_fast, 'Energy'] = df.loc[mask_fast, 'Energy'].clip(lower=0.6)
        df.loc[mask_fast, 'Danceability'] = df.loc[mask_fast, 'Danceability'].clip(lower=0.5)
        mask_slow = df['Tempo_Category'] == 'Slow'
        df.loc[mask_slow, 'Energy'] = df.loc[mask_slow, 'Energy'].clip(upper=0.6)

    return df


def generate_podcast_data():
    """
    Generates synthetic podcast listening data with specific show names.
    Reflects shifting interest toward politics due to Iran developments:
    - Before July 2025: balanced across categories
    - July-Aug 2025: first politics spike (Iran tensions)
    - Sept-Dec 2025: settles but politics remains elevated
    - Jan 2026+: major politics spike (Iran escalation)
    """
    np.random.seed(99)
    start_date = pd.Timestamp("2024-09-01")
    end_date = pd.Timestamp("2026-02-05")
    days = pd.date_range(start_date, end_date, freq='D')

    # Podcast catalog by category
    podcasts = {
        'Politics': {
            'shows': ['Kyle Kulinski Show', 'Breaking Points', 'Pod Save America', 'The Daily'],
            'weights': [0.45, 0.25, 0.15, 0.15]
        },
        'Science': {
            'shows': ['Professor Dave Explains', 'Lex Fridman Podcast', 'StarTalk', 'Radiolab'],
            'weights': [0.40, 0.30, 0.15, 0.15]
        },
        'Religion': {
            'shows': ["Alex O'Connor", 'Philosophize This!', 'The Bible Project', 'Unbelievable?'],
            'weights': [0.50, 0.25, 0.15, 0.10]
        }
    }

    categories = ['Science', 'Religion', 'Politics']

    # Phase boundaries
    phase_1_end = pd.Timestamp("2025-07-01")   # Balanced
    phase_2_end = pd.Timestamp("2025-09-01")   # First politics spike
    phase_3_end = pd.Timestamp("2026-01-01")   # Settles somewhat
    # phase 4: Jan 2026+ big spike

    data = []

    for day in days:
        if day < phase_1_end:
            # Balanced listening
            cat_probs = [0.35, 0.35, 0.30]
            n_listens = np.random.randint(0, 4)
        elif day < phase_2_end:
            # First spike: Iran tensions rise (July-Aug 2025)
            cat_probs = [0.18, 0.17, 0.65]
            n_listens = np.random.randint(1, 6)
        elif day < phase_3_end:
            # Settles but politics stays elevated (Sept-Dec 2025)
            cat_probs = [0.25, 0.25, 0.50]
            n_listens = np.random.randint(0, 4)
        else:
            # Big spike: Iran situation escalates (Jan 2026+)
            cat_probs = [0.08, 0.07, 0.85]
            n_listens = np.random.randint(3, 8)

        if n_listens > 0:
            todays_cats = np.random.choice(categories, size=n_listens, p=cat_probs)
            for cat in todays_cats:
                info = podcasts[cat]
                show = np.random.choice(info['shows'], p=info['weights'])
                data.append({
                    'Date': day,
                    'Category': cat,
                    'Show': show,
                    'Listens': 1
                })

    return pd.DataFrame(data)


def get_timeline_events():
    """
    Returns life events (city moves) and world events (Iran) for timeline annotations.
    """
    events = [
        # City moves between semesters
        {'date': pd.Timestamp("2024-09-01"), 'label': 'Moved to Brussels',
         'type': 'move', 'detail': 'Fall semester begins'},
        {'date': pd.Timestamp("2025-02-01"), 'label': 'Moved to Barcelona',
         'type': 'move', 'detail': 'Spring semester begins'},
        {'date': pd.Timestamp("2025-07-01"), 'label': 'Moved to Ankara',
         'type': 'move', 'detail': 'Summer break begins'},
        {'date': pd.Timestamp("2025-09-01"), 'label': 'Moved to Paris',
         'type': 'move', 'detail': 'Fall semester begins'},
        # Iran-related events
        {'date': pd.Timestamp("2025-07-15"), 'label': 'Iran: Tensions rise',
         'type': 'world', 'detail': 'Political situation in Iran intensifies'},
        {'date': pd.Timestamp("2025-08-05"), 'label': 'Iran: Developments continue',
         'type': 'world', 'detail': 'Ongoing political developments'},
        {'date': pd.Timestamp("2026-01-05"), 'label': 'Iran: Situation escalates',
         'type': 'world', 'detail': 'Major escalation in political crisis'},
        {'date': pd.Timestamp("2026-01-20"), 'label': 'Iran: Crisis deepens',
         'type': 'world', 'detail': 'Situation continues to develop'},
    ]
    return pd.DataFrame(events)


def get_semester_date_range(semester_name):
    """Returns (start_date, end_date) tuple for a given semester."""
    ranges = {
        "Fall 2024-2025 (Brussels)":    (pd.Timestamp("2024-09-01"), pd.Timestamp("2025-01-31")),
        "Spring 2024-2025 (Barcelona)": (pd.Timestamp("2025-02-01"), pd.Timestamp("2025-06-30")),
        "Summer 2025 (Ankara)":         (pd.Timestamp("2025-07-01"), pd.Timestamp("2025-08-31")),
        "Fall 2025-2026 (Paris)":       (pd.Timestamp("2025-09-01"), pd.Timestamp("2026-02-05")),
        "Entire Academic Program":      (pd.Timestamp("2024-09-01"), pd.Timestamp("2026-02-05")),
    }
    return ranges.get(semester_name, (pd.Timestamp("2024-09-01"), pd.Timestamp("2026-02-05")))


if __name__ == "__main__":
    df = load_and_enrich_data('youtube_music_dataset_temp.csv')
    if df is not None:
        print(f"Music data: {df.shape[0]} rows, {df.shape[1]} cols")
        print(f"Date range: {df['Date_Added'].min()} to {df['Date_Added'].max()}")
        print(f"Semesters: {df['Semester'].unique()}")

    pod_df = generate_podcast_data()
    print(f"\nPodcast data: {len(pod_df)} rows")
    if not pod_df.empty:
        print(f"Shows: {pod_df['Show'].unique()}")
        print(f"Post Jan 2026 Politics: {len(pod_df[(pod_df['Date'] >= '2026-01-01') & (pod_df['Category'] == 'Politics')])}")

    events_df = get_timeline_events()
    print(f"\nTimeline events: {len(events_df)}")
    print(events_df[['date', 'label', 'type']])
