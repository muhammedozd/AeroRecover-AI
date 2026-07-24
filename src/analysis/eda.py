from operator import lt

from numpy import rint

from src.data.load_flights import load_flights
import matplotlib.pyplot as plt

import pandas as pd

def main():
    df = load_flights()

    print("Veri seti boyutu:", df.shape)

    # Hedef değişkeni eksik olan satırları kaldır
    df = df.dropna(subset=["ARR_DEL15"])

    # Gecikme sayıları
    delay_counts = df["ARR_DEL15"].value_counts()

    print("\nGecikme Sayıları:")
    print(delay_counts)

    # Gecikme yüzdeleri
    delay_percentages = (
        df["ARR_DEL15"]
        .value_counts(normalize=True)
        * 100
    )

    daily_delay = (
    df.groupby("DAY_OF_WEEK")["ARR_DEL15"]
    .mean()
    * 100
)

    print("\nHaftanın Günlerine Göre Gecikme Oranları (%):")
    print(daily_delay)
    plt.figure(figsize=(9, 5))

    plt.bar(
    daily_delay.index,
    daily_delay.values
)

    plt.title("Haftanın Günlerine Göre Gecikme Oranı")
    plt.xlabel("Haftanın Günü")
    plt.ylabel("Gecikme Oranı (%)")

    plt.xticks(
    range(1, 8),
    [
        "Pazartesi",
        "Salı",
        "Çarşamba",
        "Perşembe",
        "Cuma",
        "Cumartesi",
        "Pazar",
    ],
    rotation=30
)

    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    print("\nGecikme Yüzdeleri:")
    print(delay_percentages)
    print(df["MONTH"].unique())
    print(df["MONTH"].value_counts().sort_index())  

    carrier_delay = (
    df.groupby("OP_UNIQUE_CARRIER")["ARR_DEL15"]
    .mean()
    * 100
)

    carrier_delay = carrier_delay.sort_values(ascending=False)

    print("\nHavayolu Şirketlerine Göre Gecikme Oranı (%):")
    print(carrier_delay)
    plt.figure(figsize=(12, 6))

    plt.bar(
    carrier_delay.index,
    carrier_delay.values
)

    plt.title("Havayolu Şirketlerine Göre Gecikme Oranı")
    plt.xlabel("Havayolu")
    plt.ylabel("Gecikme Oranı (%)")

    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()

    plt.show()


    df["DEP_HOUR"] = df["CRS_DEP_TIME"] // 100

    hourly_delay = (df.groupby("DEP_HOUR")["ARR_DEL15"].mean() * 100)
    print("\nPlanlanan Kalkış Saatine Göre Gecikme Oranları (%):")
    print(hourly_delay)

# Grafiği oluştur
    plt.figure(figsize=(12, 6))

    plt.plot(
    hourly_delay.index,
    hourly_delay.values,
    marker="o"
)

    plt.title("Planlanan Kalkış Saatine Göre Gecikme Oranı")
    plt.xlabel("Kalkış Saati")
    plt.ylabel("Gecikme Oranı (%)")

    plt.xticks(range(0, 24))
    plt.grid(axis="both", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()
    
    hourly_count = df.groupby("DEP_HOUR").size()

    print(hourly_count)

    # Her havaalanındaki uçuş sayısını hesapla
    origin_count = df.groupby("ORIGIN").size()

# En az 1000 uçuşu olan havaalanlarını seç
    valid_airports = origin_count[origin_count >= 1000].index

# Veri setini filtrele
    filtered_df = df[df["ORIGIN"].isin(valid_airports)]

    origin_delay = (
    filtered_df.groupby("ORIGIN")["ARR_DEL15"]
    .mean()
    * 100
)

    origin_delay = origin_delay.sort_values(
    ascending=False
)

    top10_origin = origin_delay.head(10)

    plt.figure(figsize=(10, 6))

    plt.bar(
    top10_origin.index,
    top10_origin.values
)

    plt.title("En Yüksek Gecikme Oranına Sahip İlk 10 Çıkış Havaalanı (1000+ Uçuş)")
    plt.xlabel("Çıkış Havaalanı")
    plt.ylabel("Gecikme Oranı (%)")

    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()



    dest_count = df.groupby("DEST").size()
    valid_destinations = dest_count[dest_count >= 1000].index

    filtered_df = df[df["DEST"].isin(valid_destinations)]
    dest_delay = (
    filtered_df.groupby("DEST")["ARR_DEL15"]
    .mean()
    * 100
)
    dest_delay = dest_delay.sort_values(ascending=False)
    top10_dest = dest_delay.head(10)

    plt.figure(figsize=(10, 6))

    top10_dest.plot(kind="bar")

    plt.title("Top 10 Destination Airports by Delay Rate (Min. 1000 Flights)")
    plt.xlabel("Destination Airport")
    plt.ylabel("Delay Rate (%)")

    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

# ==============================
# Distance Analysis
# ==============================

    distance_bins = [0, 500, 1000, 1500, 2000, 3000, float("inf")]

    distance_labels = [
    "0-500",
    "500-1000",
    "1000-1500",
    "1500-2000",
    "2000-3000",
    "3000+"
]

    df["DISTANCE_GROUP"] = pd.cut(
    df["DISTANCE"],
    bins=distance_bins,
    labels=distance_labels
)

    distance_delay = (
    df.groupby("DISTANCE_GROUP")["ARR_DEL15"]
    .mean()
    * 100
)

    plt.figure(figsize=(8, 5))

    distance_delay.plot(kind="bar")

    plt.title("Delay Rate by Flight Distance")
    plt.xlabel("Distance Group (Miles)")
    plt.ylabel("Delay Rate (%)")

    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()