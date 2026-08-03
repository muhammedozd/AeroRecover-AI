from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

import joblib
from pathlib import Path
import pandas as pd


from src.models.train_rotation_model import (
    MODEL_COLUMNS,
    create_time_masks,
    prepare_features)



PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rotation_dataset_2023.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "xgboost_propagation_2023_time_split.pkl"
)



def load_validation_data():
    data = pd.read_csv(
        DATA_PATH,
        usecols=MODEL_COLUMNS,
    )

    X, y, _, _ = prepare_features(data)

    _, validation_mask, _ = create_time_masks(data)

    X_validation = X.loc[validation_mask]
    y_validation = y.loc[validation_mask]

    print(f"Validation samples: {len(X_validation):,}")
    print(
        "Propagation rate: "
        f"{y_validation.mean():.4f}"
    )

    return X_validation, y_validation


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model bulunamadı: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    print(f"Model yüklendi: {MODEL_PATH.name}")

    return model

def predict_probabilities(model, X_validation):
    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    print(
        "Üretilen olasılık sayısı: "
        f"{len(probabilities):,}"
    )

    return probabilities


def analyze_thresholds(
    y_validation,
    probabilities,
):
    thresholds = [
    value / 100
    for value in range(40, 51)

]

    results = []

    for threshold in thresholds:
        y_pred = (
            probabilities >= threshold
        ).astype(int)
# .ravel() iki boyutlu tabloyu tek sıra haline getirir.
        tn, fp, fn, tp = confusion_matrix(
            y_validation,
            y_pred,
            labels=[0, 1],
        ).ravel()

        results.append(
            {
                "threshold": threshold,
                "precision": precision_score(
                    y_validation,
                    y_pred,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y_validation,
                    y_pred,
                    zero_division=0,
                ),
                "f1_score": f1_score(
                    y_validation,
                    y_pred,
                    zero_division=0,
                ),
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "alert_count": int(y_pred.sum()),
            }
        )

    return pd.DataFrame(results)



def main():
    X_validation, y_validation = load_validation_data()

    model = load_model()

    probabilities = predict_probabilities(
        model,
        X_validation,
    )

    threshold_results = analyze_thresholds(
        y_validation,
        probabilities,
    )

    print("\nThreshold analysis")
    print("-" * 100)

    print(
        threshold_results.to_string(
            index=False,
        )
    )


if __name__ == "__main__":
    main()