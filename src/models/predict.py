from pathlib import Path
from xml.parsers.expat import model
from src.data.preprocess_data import preprocess_data
import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

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
    X_train, X_test, y_train, y_test = preprocess_data()
    tahminler = model.predict(X_test)

    accuracy = accuracy_score(y_test, tahminler)
    print("\nİlk 5 tahmin:", tahminler[:5])
    print(f"\nAccuracy: {accuracy:.2f}")

    matrix = confusion_matrix(y_test, tahminler)

    print("\nConfusion Matrix:")
    print(matrix)

    report = classification_report(y_test, tahminler)
    print("\nClassification Report:")
    print(report)

    
    print(f"Model başarıyla yüklendi: {MODEL_PATH}")
    

    return model


if __name__ == "__main__":
    load_model()