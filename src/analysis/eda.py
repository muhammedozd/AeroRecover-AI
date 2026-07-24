from operator import lt

from numpy import rint

from src.data.load_flights import load_flights
import matplotlib.pyplot as plt

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

if __name__ == "__main__":
    main()