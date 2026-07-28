from src.data.load_flights import load_flights


def preprocess_data():
    df = load_flights()

    print("Ham veri boyutu:", df.shape)

    return df


if __name__ == "__main__":
    preprocess_data()