"""Compare baseline models on the validation period."""

from pathlib import Path
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.models.train_rotation_model import (
    MODEL_PATH,
    build_preprocessor,
    create_time_masks,
    load_dataset,
    prepare_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "baseline_model_comparison.csv"
)
def load_train_validation_data():
    data = load_dataset()

    X, y, categorical_features, numerical_features = (
        prepare_features(data)
    )

    train_mask, validation_mask, _ = create_time_masks(
        data
    )

    return (
        X.loc[train_mask],
        y.loc[train_mask],
        X.loc[validation_mask],
        y.loc[validation_mask],
        categorical_features,
        numerical_features,
    )

def build_baseline_models(
    categorical_features,
    numerical_features,
):
    dummy_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(
                    categorical_features,
                    numerical_features,
                ),
            ),
            (
                "classifier",
                DummyClassifier(
                    strategy="prior",
                ),
            ),
        ]
    )

    logistic_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(
                    categorical_features,
                    numerical_features,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    solver="saga",
                    max_iter=200,
                    random_state=42,
                ),
            ),
        ]
    )

    return {
        "Dummy": dummy_pipeline,
        "Logistic Regression": logistic_pipeline,
    }
def evaluate_model(
    model_name,
    model,
    X_validation,
    y_validation,
    threshold=0.50,
):
    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {
        "MODEL": model_name,
        "THRESHOLD": threshold,
        "ACCURACY": accuracy_score(
            y_validation,
            predictions,
        ),
        "PRECISION": precision_score(
            y_validation,
            predictions,
            zero_division=0,
        ),
        "RECALL": recall_score(
            y_validation,
            predictions,
            zero_division=0,
        ),
        "F1": f1_score(
            y_validation,
            predictions,
            zero_division=0,
        ),
        "ROC_AUC": roc_auc_score(
            y_validation,
            probabilities,
        ),
        "PR_AUC": average_precision_score(
            y_validation,
            probabilities,
        ),
    }

def main():
    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        categorical_features,
        numerical_features,
    ) = load_train_validation_data()

    baseline_models = build_baseline_models(
        categorical_features,
        numerical_features,
    )

    results = []

    for model_name, model in baseline_models.items():
        print(f"Eğitiliyor: {model_name}")

        model.fit(
            X_train,
            y_train,
        )

        results.append(
            evaluate_model(
                model_name=model_name,
                model=model,
                X_validation=X_validation,
                y_validation=y_validation,
                threshold=0.50,
            )
        )

    print("Yükleniyor: XGBoost")

    xgboost_model = joblib.load(
        MODEL_PATH
    )

    results.append(
        evaluate_model(
            model_name="XGBoost",
            model=xgboost_model,
            X_validation=X_validation,
            y_validation=y_validation,
            threshold=0.50,
        )
    )

    results.append(
        evaluate_model(
            model_name="XGBoost Operational",
            model=xgboost_model,
            X_validation=X_validation,
            y_validation=y_validation,
            threshold=0.46,
        )
    )

    results_df = pd.DataFrame(
        results
    )

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        RESULTS_PATH,
        index=False,
    )

    print("\nValidation baseline comparison")
    print("-" * 90)
    print(
        results_df.to_string(
            index=False
        )
    )
    print(
        f"\nSaved to: {RESULTS_PATH}"
    )


if __name__ == "__main__":
    main()