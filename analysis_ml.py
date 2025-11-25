import joblib
from pathlib import Path
from typing import List, Dict


MODEL_PATH = Path("models/mbti_model.joblib")


def predict_mbti_ml(texts: List[str]) -> Dict:
    """
    texts: 상대방 대화 문장 리스트
    반환: {"mbti": "INTJ", "confidence": 0.73}
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError("ML 모델이 존재하지 않습니다. train_mbti_model.py를 먼저 실행하세요.")

    model_bundle = joblib.load(MODEL_PATH)

    vectorizer = model_bundle["vectorizer"]
    model = model_bundle["model"]

    combined_text = " ".join(texts)

    X = vectorizer.transform([combined_text])
    proba = model.predict_proba(X)[0]
    pred = model.predict(X)[0]

    confidence = float(max(proba))

    return {
        "mbti": pred,
        "confidence": round(confidence, 3)
    }
