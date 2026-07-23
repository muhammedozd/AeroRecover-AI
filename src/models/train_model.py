from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data.preprocess_data import preprocess_data


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "logistic_regression.pkl"


def train_model() -> Pipeline:
    """Pipeline modelini eğit ve diske kaydet."""

    X_train, X_test, y_train, y_test = preprocess_data()

    categorical_columns = [
        "OP_UNIQUE_CARRIER",
        "ORIGIN",
        "DEST",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns,
            )
        ],
        remainder="passthrough",
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"Model kaydedildi: {MODEL_PATH}")

    y_pred = model.predict(X_test)

    print("İlk 10 tahmin:")
    print(y_pred[:10])

    return model


if __name__ == "__main__":
    train_model()