from pathlib import Path
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier


# PROJECT PATHS


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
    """
    Load the processed aircraft rotation dataset.
    """

    df = pd.read_csv(DATA_PATH)

    print("=" * 50)
    print("Dataset loaded successfully.")
    print(f"Shape: {df.shape}")
    print("=" * 50)

    return df

def prepare_features(df):
    """
    Select features and target for delay propagation prediction.
    """

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

    target_column = "IS_DELAY_PROPAGATED"

    feature_columns = (
        categorical_features
        + numerical_features
    )

    X = df[feature_columns]
    y = df[target_column]

    return (
        X,
        y,
        categorical_features,
        numerical_features
    )


    print("\nFeature kontrolü")
    print("-" * 50)

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    print("\nKullanılan feature'lar:")
    print(X.columns.tolist())

    print("\nTarget dağılımı:")
    print(y.value_counts())

    print("\nTarget oranı:")
    print(y.value_counts(normalize=True))


def build_preprocessor(categorical_features, numerical_features):
    """
    Build the preprocessing pipeline.
    """

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            ),
            (
                "numerical",
                "passthrough",
                numerical_features
            )
        ]
    )

    return preprocessor

def build_pipeline(preprocessor):
    """
    Build the machine learning pipeline.
    """

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                XGBClassifier(
                    random_state=42,
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    eval_metric="logloss"
                )
            )
        ]
    )

    return pipeline
    
if __name__ == "__main__":
    df = load_dataset()


    (
    X,
    y,
    categorical_features,
    numerical_features
    )   = prepare_features(df)



    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    preprocessor = build_preprocessor(
        categorical_features,
        numerical_features
    )
    pipeline = build_pipeline(preprocessor)
    #model eğitme başlıyor

    pipeline.fit(X_train, y_train)
    print("Model eğitimi başarıyla tamamlandı.")

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        pipeline,
        MODEL_PATH
    )

    print(f"Model kaydedildi: {MODEL_PATH}")

