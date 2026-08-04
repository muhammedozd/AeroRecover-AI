from pathlib import Path

import joblib
import pandas as pd

from src.models.train_rotation_model import (
    MODEL_COLUMNS,
    create_time_masks,
    prepare_features,
)

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


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


def load_dataset():
    """
    Load the required columns from the 2023 rotation dataset.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset bulunamadı: {DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH,
        usecols=MODEL_COLUMNS,
        low_memory=False,
    )

    print("Dataset yüklendi.")
    print(f"Shape: {df.shape}")

    return df

#40 sütun yerine 12 sütun okur 
df = pd.read_csv(
    DATA_PATH,
    usecols=MODEL_COLUMNS,
    low_memory=False,
)

# _ ifadesi bu değer döndü ama kullanmayacağım anlamına gelir.

def prepare_validation_data(df):
    """
    Select the September-October validation dataset.
    """

    (
        X,
        y,
        _,
        _,
    ) = prepare_features(df)

    _, validation_mask, _ = create_time_masks(df)

    X_validation = X.loc[validation_mask]
    y_validation = y.loc[validation_mask]

    return X_validation, y_validation

def load_model():
    model = joblib.load(MODEL_PATH)

    print("Model başarıyla yüklendi.")

    return model


def evaluate_model(
    model,
    X_validation,
    y_validation,
):
    validation_predictions = model.predict(
        X_validation
    )

    validation_probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    accuracy = accuracy_score(
        y_validation,
        validation_predictions,
    )

    precision = precision_score(
        y_validation,
        validation_predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_validation,
        validation_predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_validation,
        validation_predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_validation,
        validation_probabilities,
    )

    pr_auc = average_precision_score(
        y_validation,
        validation_probabilities,
    )

    brier_score = brier_score_loss(
        y_validation,
        validation_probabilities,
    )

    matrix = confusion_matrix(
        y_validation,
        validation_predictions,
    )

    print("\n===== VALIDATION RESULTS =====")

    print(f"Accuracy   : {accuracy:.4f}")
    print(f"Precision  : {precision:.4f}")
    print(f"Recall     : {recall:.4f}")
    print(f"F1 Score   : {f1:.4f}")
    print(f"ROC-AUC    : {roc_auc:.4f}")
    print(f"PR-AUC     : {pr_auc:.4f}")
    print(f"Brier Score: {brier_score:.4f}")

    print("\nConfusion Matrix:")
    print(matrix)

    print("\nClassification Report:")
    print(
        classification_report(
            y_validation,
            validation_predictions,
            zero_division=0,
        )
    )

if __name__ == "__main__":
    df = load_dataset()

    X_validation, y_validation = (
    prepare_validation_data(df)
)

    model = load_model()

    evaluate_model(
    model,
    X_validation,
    y_validation,
)