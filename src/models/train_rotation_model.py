from pathlib import Path
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier


# PROJECT PATHS


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    /  "rotation_dataset_2023.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    /  "xgboost_propagation_2023_time_split.pkl"
)


CATEGORICAL_FEATURES = [
        "PREV_DEST",
        "PREV_DELAY_LEVEL"
    ]

NUMERICAL_FEATURES = [
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

TARGET_COLUMN = "IS_DELAY_PROPAGATED"

MODEL_COLUMNS = (
    ["FL_DATE"]
    + CATEGORICAL_FEATURES
    + NUMERICAL_FEATURES
    + [TARGET_COLUMN]
)

def load_dataset():
    """
    Load the processed aircraft rotation dataset.
    """

    df = pd.read_csv(
    DATA_PATH,
    usecols=MODEL_COLUMNS,
    low_memory=False,
)

    print("=" * 50)
    print("Dataset loaded successfully.")
    print(f"Shape: {df.shape}")
    print("=" * 50)

    return df


def create_time_masks(df):
    """
    Create chronological train, validation and test masks.
    """

    flight_dates = pd.to_datetime(
        df["FL_DATE"],
        format="%Y-%m-%d",
    )

    train_mask = flight_dates < "2023-09-01"

    validation_mask = (
        (flight_dates >= "2023-09-01")
        & (flight_dates < "2023-11-01")
    )

    test_mask = flight_dates >= "2023-11-01"

    return (
        train_mask,
        validation_mask,
        test_mask,
    )


def prepare_features(df):
    """
    Select features and target for delay propagation prediction.
    """

    feature_columns = (
        CATEGORICAL_FEATURES
        + NUMERICAL_FEATURES
    )

    X = df[feature_columns]
    y = df[TARGET_COLUMN]

    return (
        X,
        y,
        CATEGORICAL_FEATURES,
        NUMERICAL_FEATURES,
    )




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
                #tree_method="hist" → Büyük veriyi daha verimli eğitir.
                #n_jobs=-1 → Tüm CPU çekirdeklerini kullanır.
                XGBClassifier(
        random_state=42,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
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


    train_mask, validation_mask, test_mask = (
    create_time_masks(df)
)


    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]


    X_validation = X.loc[validation_mask]
    y_validation = y.loc[validation_mask]

    X_test = X.loc[test_mask]
    y_test = y.loc[test_mask]

    print("\nTime-based dataset split")
    print("-" * 50)

    print(f"Train samples: {len(X_train):,}")
    print(f"Validation samples: {len(X_validation):,}")
    print(f"Test samples: {len(X_test):,}") 

    total_split_samples = (
    len(X_train)
    + len(X_validation)
    + len(X_test)
)

    print(f"Total split samples: {total_split_samples:,}")
    print(f"Original samples: {len(X):,}")


  
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

