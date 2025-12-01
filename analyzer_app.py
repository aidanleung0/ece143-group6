import pandas as pd
import numpy as np
import os
from src.utils import build_from_stress_df, build_from_mh_df, build_from_sm_balance_df, build_from_sm_emo_df, convert_age
from src.dataset import Dataset
from src.model import PredictionModel

def load_and_preprocess_data():
    stress_df = pd.read_csv("data/Stress Level Detection Based on Daily Activities.csv")
    mh_df = pd.read_csv("data/mental_health_dataset.csv")
    sm_balance_df = pd.read_csv("data/Mental_Health_and_Social_Media_Balance_Dataset.csv")
    sm_emo_df = pd.concat([pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")], ignore_index=True)

    df_list = [
        build_from_stress_df(stress_df),
        build_from_mh_df(mh_df),
        build_from_sm_balance_df(sm_balance_df),
        build_from_sm_emo_df(sm_emo_df)
    ]

    full_df = pd.concat(df_list, ignore_index=True)
    full_df = full_df.dropna(subset=["stress_level", "happiness_index"], how="all")

    full_df["gender"] = full_df["gender"].astype(str).str.lower().str.strip()
    full_df["gender"] = full_df["gender"].replace({
        "m": "male", "male": "male", "f": "female", "female": "female",
        "non-binary": "other", "other": "other"
    })

    mask_quality = full_df["sleep_hours"] <= 10
    full_df.loc[mask_quality, "sleep_hours"] = full_df.loc[mask_quality, "sleep_hours"] / 10 * 8

    df = full_df.copy()
    df["age"] = df["age"].apply(convert_age)
    return df

def get_user_input():
    print("Please enter the following information:\n")

    while True:
        try:
            age = float(input("Age (expected: 18-100): "))
            if 18 <= age <= 100:
                break
            print("Invalid age. Please enter a value between 18 and 100.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    while True:
        gender = input("Gender (expected: male, female, or other): ").lower().strip()
        if gender in ["male", "female", "other"]:
            break
        print("Invalid gender. Please enter 'male', 'female', or 'other'.")

    while True:
        try:
            screen_time = float(input("Screen time hours per day (expected: 0-24): "))
            if 0 <= screen_time <= 24:
                break
            print("Invalid screen time. Please enter a value between 0 and 24.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    while True:
        try:
            sleep_hours = float(input("Sleep hours per day (expected: 0-24): "))
            if 0 <= sleep_hours <= 24:
                break
            print("Invalid sleep hours. Please enter a value between 0 and 24.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    while True:
        try:
            exercise_freq = float(input("Exercise frequency per week (expected: 0-7): "))
            if 0 <= exercise_freq <= 7:
                break
            print("Invalid exercise frequency. Please enter a value between 0 and 7.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    return pd.DataFrame({
        "age": [age],
        "gender": [gender],
        "screen_time_hours": [screen_time],
        "sleep_hours": [sleep_hours],
        "exercise_freq": [exercise_freq]
    })

def main():
    model_path = "model.pkl"

    if os.path.exists(model_path):
        print("Loading saved model...")
        model = PredictionModel.load_model(model_path)
    else:
        print("Training model (this may take a moment)...")
        df = load_and_preprocess_data()
        dataset = Dataset(df)
        model = PredictionModel(candidates=["LinearRegression", "RandomForest", "GradientBoosting", "XGBoost"])
        model.fit(dataset.X_train, dataset.y_train, dataset.X_test, dataset.y_test)
        model.save_model(model_path)
        print("Model trained and saved.\n")

    user_data = get_user_input()
    predictions = model.predict(user_data)

    stress_level = predictions[0][0]
    happiness_index = predictions[0][1]

    print("\n" + "*"*50)
    print("PREDICTION RESULTS")
    print("*"*50)

    stress_label = "high" if stress_level > 7 else "low" if stress_level < 4 else "moderate"
    print(f"\nStress Level: {min(max(stress_level, 1), 10):.2f} ({stress_label})")
    print("  Range: 1-10 scale, where 10 is highest stress")

    happiness_label = "high" if happiness_index > 7 else "low" if happiness_index < 4 else "moderate"
    print(f"\nHappiness Index: {min(max(happiness_index, 0), 10):.2f} ({happiness_label})")
    print("  Range: 0-10 scale, where 10 is highest happiness")
    print('\n'+"*"*50)

if __name__ == "__main__":
    main()

