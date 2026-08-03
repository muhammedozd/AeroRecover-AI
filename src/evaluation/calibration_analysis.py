from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from src.models.train_rotation_model import (
    MODEL_COLUMNS,
    create_time_masks,
    prepare_features,
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

FIGURE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "calibration_curve_2023_validation.png"
)
#Klasör zaten varsa hiçbir şey yapmaz
FIGURE_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

def load_validation_data():
    """
    Load and prepare the September-October validation data.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset bulunamadı: {DATA_PATH}"
        )

    data = pd.read_csv(
        DATA_PATH,
        usecols=MODEL_COLUMNS,
        low_memory=False,
    )

    (
        X,
        y,
        _,
        _,
    ) = prepare_features(data)

    _, validation_mask, _ = create_time_masks(data)

    X_validation = X.loc[validation_mask]
    y_validation = y.loc[validation_mask]

    print(
        f"Validation samples: {len(X_validation):,}"
    )

    print(
        f"Propagation rate: {y_validation.mean():.4f}"
    )

    return X_validation, y_validation

def load_model():
    """
    Load the trained temporal XGBoost pipeline.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model bulunamadı: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    print(f"Model yüklendi: {MODEL_PATH.name}")

    return model


def predict_validation_probabilities(
    model,
    X_validation,
):
    """
    Predict propagation probabilities for validation flights.
    """

    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    print(
        f"Üretilen olasılık sayısı: {len(probabilities):,}"
    )

    print(
        f"Ortalama tahmin olasılığı: {probabilities.mean():.4f}"
    )

    return probabilities


def calculate_calibration(
    y_validation,
    probabilities,
):
    """
    Calculate validation calibration values and Brier score.
    """
#strategy uniform ile her aralığın eşit sayıda örnek içermesi sağlanır.
    actual_rates, predicted_rates = calibration_curve(
        y_validation,
        probabilities,
        n_bins=10,
        strategy="uniform",
    )
#brier_score Her uçuş için model olasılığı ile gerçek 0/1 cevabının farkını ölçer.
    brier_score = brier_score_loss(
        y_validation,
        probabilities,
    )
#predicted_rates Her dolu aralıktaki ortalama model tahminidir.
#actual_rates Her dolu aralıktaki gerçek propagation oranıdır.

    calibration_table = pd.DataFrame({
        "predicted_probability": predicted_rates,
        "actual_propagation_rate": actual_rates,
    })

    print("\nCalibration table")
    print("-" * 60)
    print(calibration_table.to_string(index=False))

    print(f"\nBrier Score: {brier_score:.4f}")

    return (
        calibration_table,
        brier_score,
    )

def plot_calibration_curve(calibration_table):
    FIGURE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(8, 6))

    # Kusursuz kalibrasyon çizgisi
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        label="Perfect Calibration",
    )

    # Modelimizin kalibrasyon eğrisi
    plt.plot(
        calibration_table["predicted_probability"],
        calibration_table["actual_propagation_rate"],
        marker="o",
        label="Temporal XGBoost",
    )

    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Propagation Rate")
    plt.title("Calibration Curve - Validation (Sep-Oct 2023)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


def main():
    X_validation, y_validation = load_validation_data()

    model = load_model()

    probabilities = predict_validation_probabilities(
        model,
        X_validation,
    )

    calibration_table, brier_score = calculate_calibration(
        y_validation,
        probabilities,
    )
    print(f"Brier Score: {brier_score:.4f}")

    plot_calibration_curve(calibration_table)


if __name__ == "__main__":
     main()
