from pathlib import Path
from xml.parsers.expat import model
from src.data.preprocess_data import preprocess_data
import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Projenin ana klasörünü bulur
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Kaydedilmiş modelin yolu
MODEL_PATH = PROJECT_ROOT / "models" / "logistic_regression.pkl"


def load_model():
    """Kaydedilmiş modeli diskten yükler."""

    model = joblib.load(MODEL_PATH)

    year = int(input("Yil: "))
    month = int(input("Ay: "))
    day_of_month = int(input("Ayin gunu: "))
    day_of_week = int(input("Haftanin gunu (1-7): "))

    carrier = input("Havayolu kodu (Ornek: TK): ").upper()
    flight_number = int(input("Ucus numarasi: "))

    origin = input("Kalkis havalimani kodu (Ornek: IST): ").upper()
    dest = input("Varis havalimani kodu (Ornek: ESB): ").upper()

    crs_dep_time = int(input("Planlanan kalkis saati (Ornek: 1430): "))
    crs_arr_time = int(input("Planlanan varis saati (Ornek: 1615): "))

    cancelled = int(input("Ucus iptal edildi mi? (0=Hayir, 1=Evet): "))
    diverted = int(input("Ucus baska havalimanina yonlendirildi mi? (0=Hayir, 1=Evet): "))

    crs_elapsed_time = float(input("Planlanan ucus suresi (dakika): "))
    distance = float(input("Ucus mesafesi (mil): "))

    yeni_ucus = pd.DataFrame(
    [
        {
            "YEAR": year,
            "MONTH": month,
            "DAY_OF_MONTH": day_of_month,
            "DAY_OF_WEEK": day_of_week,
            "OP_UNIQUE_CARRIER": carrier,
            "OP_CARRIER_FL_NUM": flight_number,
            "ORIGIN": origin,
            "DEST": dest,
            "CRS_DEP_TIME": crs_dep_time,
            "CRS_ARR_TIME": crs_arr_time,
            "CANCELLED": cancelled,
            "DIVERTED": diverted,
            "CRS_ELAPSED_TIME": crs_elapsed_time,
            "DISTANCE": distance,
        }
    ]
)

    tahmin = model.predict(yeni_ucus)

    print("\n===== TAHMIN SONUCU =====")

    if tahmin[0] == 1:
        print("Ucusun gecikmesi bekleniyor.")
    else:
        print("Ucusun zamaninda varmasi bekleniyor.")



    X_train, X_test, y_train, y_test = preprocess_data()
    tek_ucus = X_test.iloc[[0]]
    tek_ucus_tahmini = model.predict(tek_ucus)
    gercek_sonuc = y_test.iloc[0]

    print("\nTek uçuş tahmini:")

    if tek_ucus_tahmini[0] == 1:
        print("Sonuç: Uçuşun gecikmesi bekleniyor.")
    else:
        print("Sonuç: Uçuşun gecikmesi beklenmiyor.")

    if gercek_sonuc == 1:
        print("Gerçek sonuç: Uçuş gecikmiş.")
    else:
         print("Gerçek sonuç: Uçuş gecikmemiş.")

    #accuracy = accuracy_score(y_test, tahminler)
    #print("\nİlk 5 tahmin:", tahminler[:5])
    #print(f"\nAccuracy: {accuracy:.2f}")

    #matrix = confusion_matrix(y_test, tahminler)

    #print("\nConfusion Matrix:")
    #print(matrix)

    #report = classification_report(y_test, tahminler)
    #print("\nClassification Report:")
    #print(report)

    
    print(f"Model başarıyla yüklendi: {MODEL_PATH}")
    

    return model

if __name__ == "__main__":
    load_model()