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

    # Aylara göre gecikme oranı
    monthly_delay = (
    df.groupby("MONTH")["ARR_DEL15"]
    .mean()
    * 100
)

    plt.figure(figsize=(10, 5))

    plt.bar(
    monthly_delay.index,
    monthly_delay.values
)

    plt.title("Aylara Göre Gecikme Oranı")
    plt.xlabel("Ay")
    plt.ylabel("Gecikme Oranı (%)")

    plt.xticks(range(1, 13))

    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.show()

    print("\nAylara Göre Gecikme Oranları (%):")
    print(monthly_delay)

    print("\nGecikme Yüzdeleri:")
    print(delay_percentages)


if __name__ == "__main__":
    main()