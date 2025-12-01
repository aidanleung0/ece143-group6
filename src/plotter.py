import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_age_v_stress(df):
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 17, 24, 34, 44, 60, 120],
        labels=["<=17", "18-24", "25-34", "35-44", "45-60", "60+"]
    )

    age_valid = df.dropna(subset=["age_group", "stress_level"])

    age_groups = age_valid["age_group"].unique().tolist()

    avg_stress = [
        age_valid[age_valid["age_group"] == g]["stress_level"].mean()
        for g in age_groups
    ]

    plt.figure(figsize=(8, 4))
    plt.bar(age_groups, avg_stress)
    plt.xlabel("Age Group")
    plt.ylabel("Average Stress Level")
    plt.title("Average Stress Level by Age Group")
    plt.xticks(rotation=45)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_screen_time_v_stress(df):
    clean_df = df.dropna(subset=["screen_time_hours", "stress_level"])

    plt.figure(figsize=(8, 4))
    plt.scatter(clean_df["screen_time_hours"], clean_df["stress_level"], alpha=0.4)
    plt.xlabel("Daily Screen Time (hours)")
    plt.ylabel("Stress Level")
    plt.title("Screen Time vs Stress Level")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_screen_time_v_happiness(df):
    clean_df = df.dropna(subset=["screen_time_hours", "happiness_index"])

    plt.figure(figsize=(8, 4))
    plt.scatter(clean_df["screen_time_hours"], clean_df["happiness_index"], alpha=0.4)
    plt.xlabel("Daily Screen Time (hours)")
    plt.ylabel("Happiness Index")
    plt.title("Screen Time vs Happiness Index")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_sleep_v_stress(df):
    clean_df = df.dropna(subset=["sleep_hours", "stress_level"])

    plt.figure(figsize=(8, 4))
    plt.scatter(clean_df["sleep_hours"], clean_df["stress_level"], alpha=0.4)
    plt.xlabel("Sleep Hours")
    plt.ylabel("Stress Level")
    plt.title("Sleep Hours vs Stress Level")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_exercise_v_stress(df):
    clean_df = df.dropna(subset=["exercise_freq", "stress_level"])

    plt.figure(figsize=(8, 4))
    plt.scatter(clean_df["exercise_freq"], clean_df["stress_level"], alpha=0.4)
    plt.xlabel("Exercise Frequency (per week)")
    plt.ylabel("Stress Level")
    plt.title("Exercise Frequency vs Stress Level")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_platform_analysis(df):
    platform_df = df.dropna(subset=["platform"])
    platform_df = platform_df.groupby("platform")[["stress_level", "happiness_index"]].mean()

    # Plot bar chart
    plt.figure(figsize=(12, 5))
    platform_df.plot(kind="bar", figsize=(12, 5))
    plt.title("Average Stress Level and Happiness Index by Platform")
    plt.xlabel("Platform")
    plt.ylabel("Score")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_true_v_pred(y_true, y_pred, target):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(f"True vs Predicted: {target}")
    plt.grid(alpha=0.3)
    plt.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()],
             linestyle='--', color='red')
    plt.tight_layout()
    plt.show()

def plot_residual_distribution(y_true, y_pred, target):
    residuals = y_true - y_pred

    plt.figure(figsize=(7, 5))
    plt.hist(residuals, bins=30, edgecolor='k', alpha=0.7)
    plt.title(f"Residual Distribution: {target}")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_feature_importances(feature_names, importances, title="Feature Importances"):
    sorted_idx = np.argsort(importances)

    plt.figure(figsize=(8, 6))
    plt.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx])
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()
