from src.data.load_flights import load_flights


df = load_flights()


print("Temizleme öncesi satır sayısı:")
print(df.shape)

# Hedef değişkeni eksik olan satırları sil
df = df.dropna(subset=["ARR_DEL15"])

print("\nTemizleme sonrası satır sayısı:")
print(df.shape)

# Modelde kullanılmayacak sütunlar
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
    "CANCELLATION_CODE"
]

df = df.drop(columns=columns_to_drop)

print("\nKalan sütunlar:")
print(df.columns.tolist())

print("\nToplam sütun sayısı:")
print(df.shape[1])


# Özellikler (Features)
X = df.drop(columns=["ARR_DEL15"])

# Hedef değişken (Target)
y = df["ARR_DEL15"]

print("\nX boyutu:")
print(X.shape)

print("\ny boyutu:")
print(y.shape)

print("\nX sütunları:")
print(X.columns.tolist())