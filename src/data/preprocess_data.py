import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.load_flights import load_flights


def preprocess_data():
    """Uçuş verisini temizle, kodla ve eğitim/test kümelerine ayır."""
    df = load_flights()

    # Hedef değişkeni eksik olan satırları sil.
    df = df.dropna(subset=["ARR_DEL15"])

    # Modelde kullanılmayacak sütunları kaldır.
    columns_to_drop = [
        "ARR_DELAY",
        "ARR_DELAY_NEW",
        "ARR_TIME",
        "ACTUAL_ELAPSED_TIME",
        "AIR_TIME",
        "CARRIER_DELAY",
        "WEATHER_DELAY",
        "NAS_DELAY",
        "SECURITY_DELAY",
        "LATE_AIRCRAFT_DELAY",
        "CANCELLATION_CODE",
        "DEP_TIME",
        "DEP_DELAY",
        "DEP_DELAY_NEW",
        "DEP_DEL15",
        "FL_DATE",
        "TAIL_NUM",
        "ORIGIN_CITY_NAME",
        "DEST_CITY_NAME",
    ]
    df = df.drop(columns=columns_to_drop)

    # Özellikleri ve hedef değişkeni ayır.
    X = df.drop(columns=["ARR_DEL15"])
    y = df["ARR_DEL15"]

    categorical_columns = X.select_dtypes(
        include=["object", "string"]
    ).columns

    print("Kategorik sütunlar:")
    print(categorical_columns.tolist())

    X_encoded = pd.get_dummies(
        X,
        columns=categorical_columns,
        dtype=int,
    )

    print("Encoding öncesi X boyutu:", X.shape)
    print("Encoding sonrası X boyutu:", X_encoded.shape)
    print(X_encoded.head())
    print(X_encoded.dtypes)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("X_train boyutu:", X_train.shape)
    print("X_test boyutu:", X_test.shape)
    print("y_train boyutu:", y_train.shape)
    print("y_test boyutu:", y_test.shape)

    return X_train, X_test, y_train, y_test
