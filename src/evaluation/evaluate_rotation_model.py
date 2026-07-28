from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rotation_dataset.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "xgboost_propagation_classifier.pkl"
)

def load_dataset():
    df = pd.read_csv(DATA_PATH)

    print("Dataset yüklendi.")
    print(f"Shape: {df.shape}")

    return df


def prepare_test_data(df):
    categorical_features = [
        "PREV_DEST",
        "PREV_DELAY_LEVEL"
    ]

    numerical_features = [
        "ROTATION_POSITION",
        "PREV_ARR_DELAY",
        "PREV_ARR_MIN",
        "PREV_CRS_ARR_MIN",
        "PLANNED_TURNAROUND",
        "TURN_BUFFER",
        "PREV_DELAY_RATIO",
        "HAS_BUFFER",
        "IS_SHORT_TURN",
        "PREV_DELAYED"
    ]

    feature_columns = (
        categorical_features
        + numerical_features
    )

    target_column = "IS_DELAY_PROPAGATED"

    X = df[feature_columns]
    y = df[target_column]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_test, y_test


def load_model():
    model = joblib.load(MODEL_PATH)

    print("Model başarıyla yüklendi.")

    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_test,
        y_pred
    )

    print("\n===== MODEL DEĞERLENDİRME SONUÇLARI =====")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(matrix)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )


if __name__ == "__main__":
    df = load_dataset()

    X_test, y_test = prepare_test_data(df)

    model = load_model()

    evaluate_model(
        model,
        X_test,
        y_test
    )