# Harmony: Personal Music & Mood Analytics

An interactive dashboard for exploring music listening habits, mood patterns, podcast consumption, and audio features across academic semesters in four European cities.

**[Live Demo](https://harmonyam.streamlit.app/)**

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)
![License](https://img.shields.io/badge/License-MIT-green)

## Features

- **Mood Journey Timeline** — Diverging area chart with life event annotations (city moves, world events)
- **Radar Chart** — Compare audio features (energy, danceability, valence, acousticness, instrumentalness, liveness) against global averages
- **Sunburst Chart** — Hierarchical mood sentiment breakdown with drill-down capability
- **Violin Plots** — Full mood distribution comparison across semesters/cities
- **Podcast Stacked Area** — Weekly listening volume and category proportion shifts over time
- **Podcast Treemap** — Part-to-whole breakdown of podcast consumption by category and show
- **Dataset Explorer** — Filterable, sortable data table with genre, artist, and valence threshold filters
- **Semester Filter** — All charts respond to a global semester selector (Brussels, Barcelona, Ankara, Paris, or full program)

---

## Mood Extraction Methodology

**How were mood labels and valence scores derived?**

The mood classification for this dataset was performed using a local LLM (Large Language Model) through **Ollama**. The process:

1. **Initial Approach**: The raw YouTube Music listening history was fed to a locally-hosted LLM via Ollama, which classified each track's mood based on song title, artist, genre, and available metadata.

2. **Valence Mapping**: Each mood label (e.g., "Melancholic", "Energetic") was then mapped to a numerical valence score on a 0–1 scale, informed by music psychology research (Russell's circumplex model of affect).

3. **Privacy Decision**: During the process, it became apparent that too much personal information was leaking through the raw listening data — specific timestamps, listening patterns, and song choices could reveal sensitive details about daily routines and emotional states. To protect privacy while maintaining analytical value, **the majority of the dataset was regenerated synthetically using the LLM**, preserving the statistical properties and mood patterns of the original data without exposing private listening habits.

4. **Synthetic Expansion**: The mood drift algorithm uses weighted sampling to simulate realistic listening patterns across semesters, with event-based modifiers that reflect real-world emotional impacts (city moves, global events).

## Podcast Data

Podcast listening data was generated to reflect real listening patterns:
- **Primary shows**: Kyle Kulinski Show (politics), Professor Dave Explains (science), Alex O'Connor (religion/philosophy)
- **Pattern**: Interest shifted toward political content correlating with developments in the Iran situation, with notable spikes in July–August 2025 and January 2026.

---

## Color Psychology

The color palette is designed based on **color psychology research** to make emotional data intuitive.

### Mood Valence Colors
*Used in the Mood Timeline and Sunburst*

| Color | Hex | Meaning |
|-------|-----|---------|
| Orange | `#FF6B35` | Positive mood — energy, optimism, warmth |
| Blue | `#0288D1` | Negative mood — calm, depth, contemplation |

This warm-vs-cool split follows established affective color research, where warm hues map to high-arousal positive states and cool hues map to low-arousal or negative states.

### Podcast Category Colors

| Color | Hex | Category |
|-------|-----|----------|
| Red | `#FF4B4B` | Politics — urgency, intensity |
| Sky Blue | `#4FC3F7` | Science — analytical, rational |
| Warm Orange | `#FFB74D` | Religion — spiritual, contemplative |

Red draws attention to the growing political content — its urgency mirrors the real-world intensity of the situation being followed. Blue for science echoes analytical thinking. Orange for religion/philosophy evokes warmth and spiritual contemplation.

### Semester / City Colors
*Used in Violin Plots*

| Color | Hex | City |
|-------|-----|------|
| Lavender | `#A78BFA` | Brussels — autumn, beginnings |
| Mint | `#34D399` | Barcelona — spring, freshness |
| Gold | `#FBBF24` | Ankara — summer, warmth |
| Coral | `#F87171` | Paris — autumn, intensity |

Each city gets a seasonally-inspired hue that avoids conflict with the valence warm/cool system.

### Accent & UI Colors

| Color | Hex | Usage |
|-------|-----|-------|
| Red | `#FF4B4B` | Primary accent, KPI values, world events |
| Lavender | `#A78BFA` | City move markers on timeline |
| Gray | `#808495` | Baseline, neutral reference, deselected items |
| Near-black | `#0E1117` | Background (reduces eye strain, emphasizes data) |

---

## Chart Selection Rationale

### 1. Mood Journey Timeline — *"Your Mood Journey Over Time"*
- **Type**: Diverging area chart with event annotations
- **Purpose**: Shows the temporal evolution of mood valence across your entire listening history
- **How to read**: White line = smoothed daily valence. Orange fill above the 0.5 baseline = positive mood periods. Blue fill below = negative. Dashed vertical lines mark city moves (lavender) and world events (red)
- **Why chosen**: Uses the full display space, has a clear x-axis with dates, and the event annotations let you connect mood dips/peaks to real life events

### 2. Radar Chart — *"Audio Features"*
- **Type**: Polar/spider chart
- **Purpose**: Compare 6 audio features (Energy, Danceability, Valence, Acousticness, Instrumentalness, Liveness) between your current selection and the global average
- **How to read**: Larger area = more of that feature. Dotted gray line = global average baseline for comparison
- **Why chosen**: Multi-dimensional comparison at a glance — the filled polygon shape makes differences immediately visible

### 3. Sunburst Chart — *"Mood Sentiment Breakdown"*
- **Type**: Hierarchical pie/sunburst
- **Purpose**: Shows the breakdown of moods by sentiment category (Positive / Negative / Neutral) with drill-down to individual moods
- **How to read**: Inner ring = sentiment categories, outer ring = top 5 specific moods per sentiment. Click to drill down. Segment size = proportion
- **Why chosen**: Reveals both macro-level emotional balance and the specific moods driving it, with interactive exploration

### 4. Violin Plots — *"Mood Distribution Across Your Journey"*
- **Type**: Violin + box plot hybrid
- **Purpose**: Compare the full shape of mood distributions across semesters
- **How to read**: Width = density of songs at that valence. Box inside shows quartiles and median. Wider sections = more songs at that valence level
- **Why chosen**: Shows mood *distribution*, not just averages — reveals whether a semester was consistently stable or had emotional swings

### 5. Podcast Stacked Area — *"Focus Shift Over Time"*
- **Type**: Stacked area chart (weekly aggregation)
- **Purpose**: Shows how podcast listening volume and category proportions changed over time, highlighting the shift toward political content
- **How to read**: Total height = weekly listening volume. Color bands show the proportion per category. Growing red = more politics
- **Why chosen**: Shows both absolute volume increase and proportional shifts, making the Iran-related interest spike clearly visible

### 6. Podcast Treemap — *"Listening Breakdown"*
- **Type**: Interactive treemap
- **Purpose**: Part-to-whole overview of total podcast consumption by category and individual show
- **How to read**: Box size = total episodes. Nested boxes = shows within categories. Colors match the stacked area chart
- **Why chosen**: Gives an at-a-glance summary of which shows dominate your listening, with consistent color coding for category recognition

---

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tech Stack

Streamlit | Plotly | Pandas | NumPy

---

*Developed for Visual Analytics — Harmony Personal Music & Mood Analytics Dashboard*
