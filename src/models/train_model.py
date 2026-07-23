from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from src.data.preprocess_data import preprocess_data


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_model.pkl"


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
            XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                scale_pos_weight=3.52,
               random_state=42,
               eval_metric="logloss",
                
            ),
        ),
    ]
)

    model.fit(X_train, y_train)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"Model kaydedildi: {MODEL_PATH}")

    y_pred = model.predict(X_test)

    print("\nTahmin edilen gecikmeyen ucus:", (y_pred == 0).sum())
    print("Tahmin edilen geciken ucus:", (y_pred == 1).sum())

    print("\nGercekte gecikmeyen ucus:", (y_test == 0).sum())
    print("Gercekte geciken ucus:", (y_test == 1).sum())

    print("İlk 10 tahmin:")
    print(y_pred[:10])
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return model


if __name__ == "__main__":
    train_model()