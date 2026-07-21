from src.data.load_flights import load_flights
import pandas as pd
# Veriyi yükle
df = load_flights()

# Hedefi eksik olanları sil
df = df.dropna(subset=["ARR_DEL15"])

# Kullanılmayacak sütunları kaldır
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

# X ve y ayır
X = df.drop(columns=["ARR_DEL15"])
y = df["ARR_DEL15"]
# print(X.dtypes)

categorical_columns = X.select_dtypes(
include=["object", "string"]
).columns

print("Kategorik sütunlar:")
print(categorical_columns.tolist())

X_encoded = pd.get_dummies(
    X,
    columns=categorical_columns,
    dtype=int
)

print("Encoding öncesi X boyutu:", X.shape)

print("Encoding sonrası X boyutu:", X_encoded.shape)

print(X_encoded.head())
print(X_encoded.dtypes)
