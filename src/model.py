import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb

models = {
    "LinearRegression": MultiOutputRegressor(LinearRegression()),
    "RandomForest": MultiOutputRegressor(RandomForestRegressor(n_estimators=200)),
    "GradientBoosting": MultiOutputRegressor(GradientBoostingRegressor()),
    "XGBoost": MultiOutputRegressor(xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.9, colsample_bytree=0.9, objective='reg:squarederror'))
}

def get_feature_names_from_pipeline(model, numeric_features, categorical_features):
    preprocessor = model.named_steps["preprocessor"]

    # numeric names are just the original columns
    num_features = list(numeric_features)

    # categorical names from OHE
    ohe = preprocessor.named_transformers_["cat"]["onehot"]
    ohe_features = ohe.get_feature_names_out(categorical_features)

    feature_names = num_features + list(ohe_features)
    return feature_names


def get_linear_importances(model):
    """
    :param model: model to plot importance
    :return: 1D np.array of importance (per transformed feature)
    """
    reg = model.named_steps["regressor"]
    estimators = reg.estimators_

    # shape: (n_targets, n_features)
    coefs = np.vstack([est.coef_.ravel() for est in estimators])

    # aggregate over targets: mean absolute coefficient
    importances = np.mean(np.abs(coefs), axis=0)
    return importances, coefs

def get_tree_importances(model):
    """
    :param model: model to plot importance
    :return: 1D np.array of importance (per transformed feature)
    """
    reg = model.named_steps["regressor"]
    estimators = reg.estimators_

    # shape: (n_targets, n_features)
    imps = np.vstack([est.feature_importances_ for est in estimators])

    # aggregate over targets: mean importance
    importances = np.mean(imps, axis=0)
    return importances, imps

importance_function = {
    "LinearRegression": get_linear_importances,
    "RandomForest": get_tree_importances,
    "GradientBoosting": get_tree_importances,
    "XGBoost": get_tree_importances,
}

class PredictionModel:
    def __init__(self, candidates: list[str]):
        for candidate in candidates:
            assert candidate in models, f"{candidate} is not supported"
        self.candidates = candidates
        self.model = None
        self.model_name = None

        self.numeric_features = ["age", "screen_time_hours", "sleep_hours", "exercise_freq"]
        self.categorical_features = ["gender"]
        numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
        categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])
        self.preprocessor = ColumnTransformer([
            ("num", numeric_transformer, self.numeric_features),
            ("cat", categorical_transformer, self.categorical_features),
        ])

    def fit(self, X_train, y_train, X_test, y_test):
        """
        This function will fit the model to the training data, and select the best one from candidates
        :param X_train: features
        :param y_train: label
        :param X_test: features
        :param y_test: label
        :return: final model
        """
        results = {}
        for name in self.candidates:
            reg = models[name]
            print(f"Training {name}")
            model = Pipeline([("preprocessor", self.preprocessor), ("regressor", reg)])
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred, multioutput="raw_values")
            results[name] = {"model": model, "r2": r2, "rmse": rmse}

        best_model_name = min(results, key=lambda x: results[x]["rmse"])
        best_model = results[best_model_name]["model"]
        self.model = best_model
        self.model_name = best_model_name
        return best_model

    def predict(self, X_test):
        assert self.model is not None, "Model not fit"
        return self.model.predict(X_test)

    def get_importance(self):
        assert self.model_name is not None, "Model not fit"
        assert self.model is not None, "Model not fit"

        importance, _ = importance_function[self.model_name](self.model)
        features = get_feature_names_from_pipeline(
            model=self.model, numeric_features=self.numeric_features, categorical_features=self.categorical_features
        )
        return features, importance

    def save_model(self, path):
        assert self.model is not None, "Model not fit"
        assert self.model_name is not None, "Model not fit"
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'model_name': self.model_name,
                'numeric_features': self.numeric_features,
                'categorical_features': self.categorical_features
            }, f)

    @classmethod
    def load_model(cls, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        instance = cls(candidates=[data['model_name']])
        instance.model = data['model']
        instance.model_name = data['model_name']
        instance.numeric_features = data['numeric_features']
        instance.categorical_features = data['categorical_features']
        return instance