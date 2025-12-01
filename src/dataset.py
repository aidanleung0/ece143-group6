from sklearn.model_selection import train_test_split

class Dataset:
    def __init__(self, df, mode="train", test_size=0.2, random_state=42):
        feature_cols = ["age", "gender", "screen_time_hours", "sleep_hours", "exercise_freq"]
        target_cols = ["stress_level", "happiness_index"]
        model_df = df[feature_cols + target_cols].dropna()
        X = model_df[feature_cols]
        y = model_df[target_cols]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        self.model_df = model_df
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.set_mode(mode)

    def __len__(self):
        if self.mode == "train":
            return self.X_train.shape[0]
        elif self.mode == "test":
            return self.X_test.shape[0]
        else:
            raise NotImplementedError

    def __getitem__(self, idx):
        if self.mode == "train":
            return self.X_train[idx], self.y_train[idx]
        elif self.mode == "test":
            return self.X_test[idx], self.y_test[idx]
        else:
            raise NotImplementedError

    def set_mode(self, mode):
        assert mode in ["train", "test"]
        self.mode = mode

