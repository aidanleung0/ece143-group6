import pandas as pd
import numpy as np

def build_from_stress_df(df):
    """
    Returns a processed and cleaned DataFrame from the stress level detection dataset.
    """
    assert isinstance(df, pd.DataFrame)

    tmp = pd.DataFrame()
    tmp["age"] = df["Age"]
    tmp["gender"] = df["Gender"]

    tmp["screen_time_hours"] = pd.to_numeric(df["Screen time"], errors="coerce")
    tmp["sleep_hours"] = pd.to_numeric(df["Sleep time"], errors="coerce")
    tmp["exercise_freq"] = pd.to_numeric(df["Exercise frequency"], errors="coerce")

    mood_raw = df["Mood Stability"]

    if pd.api.types.is_numeric_dtype(mood_raw):
        tmp["stress_level"] = 10 - pd.to_numeric(mood_raw, errors="coerce")
    else:
        mood_map = {"Low": 8,"Medium": 5,"High": 2,"Stable": 3,"Unstable": 8,"Very High": 1,"Very Low": 9}
        tmp["stress_level"] = mood_raw.map(mood_map)

    tmp["happiness_index"] = np.nan
    return tmp


def build_from_mh_df(df):
    """
    Returns a processed and cleaned DataFrame from the mental health dataset.
    """
    assert isinstance(df, pd.DataFrame)
    
    tmp = pd.DataFrame()
    tmp["age"] = df["age"]
    tmp["gender"] = df["gender"]
    tmp["sleep_hours"] = df["sleep_hours"]
    tmp["exercise_freq"] = df["physical_activity_days"]
    tmp["screen_time_hours"] = np.nan
    tmp["stress_level"] = df["stress_level"]

    if "depression_score" in df.columns and "anxiety_score" in df.columns:
        tmp["happiness_index"] = 10 - (df["depression_score"] + df["anxiety_score"]) / 2
    else:
        tmp["happiness_index"] = np.nan
    return tmp


def build_from_sm_balance_df(df):
    """
    Returns a processed and cleaned DataFrame from the mental health and social media balance dataset.
    """
    assert isinstance(df, pd.DataFrame)

    tmp = pd.DataFrame()
    tmp["age"] = df["Age"]
    tmp["gender"] = df["Gender"]
    tmp["screen_time_hours"] = df["Daily_Screen_Time(hrs)"]
    tmp["sleep_hours"] = df["Sleep_Quality(1-10)"]
    tmp["exercise_freq"] = df["Exercise_Frequency(week)"]
    tmp["stress_level"] = df["Stress_Level(1-10)"]
    tmp["happiness_index"] = df["Happiness_Index(1-10)"]
    tmp["platform"] = df["Social_Media_Platform"]
    return tmp


def build_from_sm_emo_df(df):
    """
    Returns a processed and cleaned DataFrame from the social media and emotions dataset.
    """
    assert isinstance(df, pd.DataFrame)

    tmp = pd.DataFrame()
    tmp["age"] = df["Age"]
    tmp["gender"] = df["Gender"]
    tmp["screen_time_hours"] = df["Daily_Usage_Time (minutes)"] / 60
    tmp["sleep_hours"] = np.nan
    tmp["exercise_freq"] = np.nan

    emotion_map = {"Happy": 4,"Excited": 4,"Calm": 3,"Neutral": 2,
                   "Sad": 1,"Anxious": 1,"Stressed": 1,"Depressed": 0,"Angry": 0}
    tmp["happiness_index"] = df["Dominant_Emotion"].map(emotion_map)
    tmp["stress_level"] = np.nan
    tmp["platform"] = df["Platform"]
    return tmp

def convert_age(x):
    """
    Helper function to process age ranges.
    """
    try:
        return float(x)
    except:
        pass
    if isinstance(x, str) and "-" in x:
        parts = x.replace(" ", "").split("-")
        if len(parts) == 2:
            try:
                return (float(parts[0]) + float(parts[1])) / 2
            except:
                return np.nan
    return np.nan

