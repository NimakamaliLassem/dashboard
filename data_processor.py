import pandas as pd
import numpy as np
import datetime

def load_and_enrich_data(filepath):
    """
    Loads the music dataset and adds synthetic features for the dashboard.
    Expands the dataset to cover 4 semesters with realistic volume.
    """
    try:
        source_df = pd.read_csv(filepath)
    except FileNotFoundError:
        return None

    # --- 1. Pre-calculate Mood Valence (CRITICAL STEP) ---
    # We do this FIRST so we can select songs based on their actual numerical score.
    # This ensures the "Mood Walk" works correctly.
    
    mood_map = {
        # High Valence (Positive / Orange)
        'Energetic': 0.85, 'Happy': 0.9, 'Uplifting': 0.85, 'Euphoric': 0.95, 'Joyful': 0.9,
        'Romantic': 0.7, 'Passionate': 0.75, 'Sensual': 0.7, 'Seductive': 0.65,
        'Hopeful': 0.75, 'Optimistic': 0.8, 'Confident': 0.8, 'Empowering': 0.85,
        'Carefree': 0.8, 'Groovy': 0.75, 'Adrenaline': 0.9, 'Exciting': 0.85, 'Heroic': 0.8,
        'Peaceful': 0.65, 'Relaxed': 0.6, 'Dreamy': 0.6, 'Ethereal': 0.6, 'Adventurous': 0.75,
        'Cinema': 0.6, 'Cinematic': 0.55, 'Epic': 0.8, 'Triumphant': 0.9, 'Transcendent': 0.7,
        'Hypnotic': 0.55, 'Mystical': 0.55, 'Exploratory': 0.65,
        
        # Low Valence (Negative / Blue)
        'Melancholic': 0.2, 'Sad': 0.1, 'Somber': 0.15, 'Depressive': 0.1, 'Sorrowful': 0.1,
        'Anxious': 0.25, 'Conflicted': 0.3, 'Desperate': 0.2, 'Heartbroken': 0.15,
        'Dark': 0.15, 'Ominous': 0.1, 'Haunting': 0.2, 'Unsettling': 0.2, 'Drama': 0.35, 'Dramatic': 0.35,
        'Tense': 0.25, 'Resigned': 0.3, 'Fatalistic': 0.2, 'Wistful': 0.35, 'Nostalgic': 0.4,
        'Bittersweet': 0.4, 'Lonely': 0.15, 'Alienated': 0.2, 'Disorienting': 0.3, 
        'Vulnerable': 0.35, 'Rainy': 0.3, 'Sarcastic': 0.4, 'Rebellious': 0.45, 'Aggressive': 0.3,
        'Introspective': 0.4, 'Contemplative': 0.4, 'Reflective': 0.4, 'Meditative': 0.45,
        'Abstract': 0.5, 'Experimental': 0.5, 'Mysterious': 0.4, 'Nocturnal': 0.3, 'Dystopian': 0.2,
        'Rootsy': 0.55, 'Cathartic': 0.45, 'Quirky': 0.65, 'Spiritual': 0.6
    }
    
    # Map and fill defaults
    if 'Mood_Valence' in source_df.columns:
        source_df['Valence'] = source_df['Mood_Valence'].map(mood_map)
        # Fill missing with random, but try to infer from 'Genre' if possible? No, random is fine for fallback.
        missing_mask = source_df['Valence'].isna()
        source_df.loc[missing_mask, 'Valence'] = np.random.uniform(0.4, 0.6, size=missing_mask.sum())
    else:
        source_df['Valence'] = np.random.uniform(0.0, 1.0, size=len(source_df))

    # --- 2. Define Semester Ranges ---
    semesters = [
        ("Fall 2024-2025 (Brussels)",  pd.Timestamp("2024-09-01"), pd.Timestamp("2025-01-31")),
        ("Spring 2024-2025 (Barcelona)", pd.Timestamp("2025-02-01"), pd.Timestamp("2025-06-30")),
        ("Summer 2025 (Ankara)",       pd.Timestamp("2025-07-01"), pd.Timestamp("2025-08-31")),
        ("Fall 2025-2026 (Paris)",     pd.Timestamp("2025-09-01"), pd.Timestamp("2026-01-31")),
    ]

    # --- 3. Generate Expanded Synthetic History (Mood Drift) ---
    np.random.seed(42)
    expanded_rows = []
    
    # Global mood target starts neutral
    current_mood_target = 0.5
    
    for sem_name, start_date, end_date in semesters:
        days_range = pd.date_range(start_date, end_date, freq='D')
        
        # Reset mood target per semester to avoid stuck state? Or let it flow?
        # Let's force a "Season Start" mood.
        # Fall/Winter = slightly lower start, Summer = higher start
        if "Summer" in sem_name:
            current_mood_target = 0.7
        elif "Fall" in sem_name:
            current_mood_target = 0.4
        else:
            current_mood_target = 0.5

        semester_data = []
        
        # We want the walk to be "swingy".
        # We'll use a sine wave component + random walk to force oscillation.
        t = 0
        
        for day in days_range:
            t += 1
            # Sine wave oscillation (period ~30 days for a "monthly mood phase")
            seasonality = 0.2 * np.sin(2 * np.pi * t / 30)
            
            # Random Walk Drift
            drift = np.random.normal(0, 0.05)
            
            # Update target
            # Combine drift and seasonality
            # We treat 'current_mood_target' as the center point which drifts slowly
            current_mood_target += drift
            current_mood_target = np.clip(current_mood_target, 0.2, 0.8) # keep center sane
            
            # Daily Target is Center + Seasonality
            daily_target = current_mood_target + seasonality
            daily_target = np.clip(daily_target, 0.05, 0.95)
            
            # Selection Logic
            # Pick songs close to daily_target
            # We calculate distance = abs(Valence - target)
            # We prefer smaller distance.
            # Convert distance to probability weight: weight = 1 / (distance + 0.1)
            
            weights = 1 / (np.abs(source_df['Valence'] - daily_target) + 0.05)
            weights = weights / weights.sum()
            
            # Sampling
            daily_songs_count = np.random.randint(8, 25)
            daily_songs = source_df.sample(n=daily_songs_count, replace=True, weights=weights).copy()
            
            # Add timestamps
            hours = np.random.randint(0, 24, size=daily_songs_count)
            minutes = np.random.randint(0, 60, size=daily_songs_count)
            daily_timestamps = [day + pd.Timedelta(hours=h, minutes=m) for h, m in zip(hours, minutes)]
            
            daily_songs['Date_Added'] = daily_timestamps
            daily_songs['Semester'] = sem_name
            
            semester_data.append(daily_songs)
            
        expanded_rows.extend(semester_data)

    # Combine all
    df = pd.concat(expanded_rows).reset_index(drop=True)
    df['Date_Added'] = pd.to_datetime(df['Date_Added'])
    
    # Sort for safety
    df = df.sort_values('Date_Added')

    # --- 4. Other Synthetic Features ---
    # Noise for already calculated Valence?
    # Maybe a tiny bit to avoid exact duplicates visually, but mapping is better kept pure for logic.
    # df['Valence'] += np.random.normal(0, 0.01, size=len(df))
    # df['Valence'] = df['Valence'].clip(0, 1)

    df['Energy'] = np.random.uniform(0.1, 0.95, size=len(df))
    df['Danceability'] = np.random.uniform(0.1, 0.95, size=len(df))
    df['Acousticness'] = np.random.uniform(0.0, 0.8, size=len(df))
    df['Instrumentalness'] = np.random.exponential(scale=0.2, size=len(df)).clip(0, 1)
    df['Liveness'] = np.random.uniform(0.05, 0.4, size=len(df))
    df['Speechiness'] = np.random.beta(a=1, b=5, size=len(df))
    
    # Tempo refinements
    if 'Tempo_Category' in df.columns:
        mask_fast = df['Tempo_Category'] == 'Fast'
        df.loc[mask_fast, 'Energy'] = df.loc[mask_fast, 'Energy'].clip(lower=0.6)
        df.loc[mask_fast, 'Danceability'] = df.loc[mask_fast, 'Danceability'].clip(lower=0.5)
        mask_slow = df['Tempo_Category'] == 'Slow'
        df.loc[mask_slow, 'Energy'] = df.loc[mask_slow, 'Energy'].clip(upper=0.6)

    return df


def generate_podcast_data():
    """
    Generates synthetic podcast listening history.
    Scenario: Random topics -> Shift to Politics after Dec 25, 2025.
    """
    np.random.seed(99)
    
    # Define range covers all semesters
    start_date = pd.Timestamp("2024-09-01")
    end_date = pd.Timestamp("2026-01-22") # Today
    days = pd.date_range(start_date, end_date, freq='D')
    
    shift_date = pd.Timestamp("2025-12-25")
    
    data = []
    
    categories = ['Science', 'Religion', 'Politics']
    
    for day in days:
        # Determine probabilities based on date
        if day < shift_date:
            # Random / Balanced distribution
            probs = [0.4, 0.4, 0.2] # Science, Religion, Politics (low)
            n_listens = np.random.randint(0, 3) # Occasional listening
        else:
            # Shift to Politics
            # Ramp up politics probability slightly over time or jump? User said "Dec 25 ... A LOT of political"
            probs = [0.1, 0.1, 0.8] 
            n_listens = np.random.randint(2, 6) # Increased frequency
            
        if n_listens > 0:
            todays_cats = np.random.choice(categories, size=n_listens, p=probs)
            for cat in todays_cats:
                data.append({
                    'Date': day,
                    'Category': cat,
                    'Listens': 1 # Count of 1 for this listen event
                })
                
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Test run
    df = load_and_enrich_data('youtube_music_dataset_temp.csv')
    if df is not None:
        print("Data loaded successfully!")
        print(f"Shape: {df.shape}")
        
    pod_df = generate_podcast_data()
    print("Podcast Data generated!")
    print(f"Podcast Rows: {len(pod_df)}")
    if not pod_df.empty:
        print(pod_df.head())
        print(f"Post-Shift Politics Count: {len(pod_df[(pod_df['Date'] > '2025-12-25') & (pod_df['Category'] == 'Politics')])}")

