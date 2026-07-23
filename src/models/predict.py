from pathlib import Path

import joblib
import pandas as pd
# Projenin ana klasörünü bulur
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Kaydedilmiş modelin yolu
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_model.pkl"


def load_model():
    """Kaydedilmiş modeli diskten yükler."""

    model = joblib.load(MODEL_PATH)

    
    print(f"Model başarıyla yüklendi: {MODEL_PATH}")
    

    return model

def get_flight_information():
    """Kullanicidan ucus bilgilerini alir."""

    year = int(input("Yil: "))
    month = int(input("Ay: "))
    day_of_month = int(input("Ayin gunu: "))
    day_of_week = int(input("Haftanin gunu (1-7): "))

    carrier = input("Havayolu kodu: ").strip().upper()
    flight_number = int(input("Ucus numarasi: "))

    origin = input("Kalkis havalimani kodu: ").strip().upper()
    dest = input("Varis havalimani kodu: ").strip().upper()

    crs_dep_time = int(input("Planlanan kalkis saati (Ornek: 1430): "))
    crs_arr_time = int(input("Planlanan varis saati (Ornek: 1615): "))

    cancelled = int(input("Ucus iptal edildi mi? (0=Hayir, 1=Evet): "))
    diverted = int(
        input("Ucus baska havalimanina yonlendirildi mi? (0=Hayir, 1=Evet): ")
    )

    crs_elapsed_time = float(input("Planlanan ucus suresi (dakika): "))
    distance = float(input("Ucus mesafesi (mil): "))

    flight = pd.DataFrame(
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

    return flight

def predict_delay(model, flight):
    """Ucusun gecikme tahminini ve olasiligini hesaplar."""

    prediction = model.predict(flight)
    probabilities = model.predict_proba(flight)

    delay_probability = probabilities[0][1]

    return prediction[0], delay_probability

def show_result(prediction, delay_probability):
    """Tahmin sonucunu kullaniciya gosterir."""

    print("\n===== AERORECOVER AI TAHMIN SONUCU =====")
    print(f"Gecikme olasiligi: %{delay_probability * 100:.2f}")

    if prediction == 1:
        print("Tahmin: Ucusun gecikmesi bekleniyor.")
    else:
        print("Tahmin: Ucusun zamaninda varmasi bekleniyor.")

    print("========================================")



if __name__ == "__main__":
    model = load_model()
    flight = get_flight_information()
    prediction, delay_probability = predict_delay(model, flight)
    show_result(prediction, delay_probability)