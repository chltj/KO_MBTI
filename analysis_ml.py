# mbti_project/analysis_ml.py

import joblib
from pathlib import Path
from typing import List, Dict

# 학습된 MBTI 모델 경로
MODEL_PATH = Path("models/mbti_model.joblib")

def predict_mbti_ml(texts: List[str]) -> Dict:
    """
    texts: 대화 문장 리스트 (상대방이든 나든 아무나)
    반환: {"mbti": "INTJ", "confidence": 0.73} 형태
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "ML 모델이 존재하지 않습니다. "
            "mbti_project 폴더에서 train_mbti_model.py를 먼저 실행해서 "
            "models/mbti_model.joblib 파일을 만들어 주세요."
        )

    # 모델 번들 로드 (vectorizer + model)
    model_bundle = joblib.load(MODEL_PATH)
    vectorizer = model_bundle["vectorizer"]
    model = model_bundle["model"]

    # 여러 문장을 하나의 문자열로 합치기
    combined_text = " ".join(texts)

    # 벡터화
    X = vectorizer.transform([combined_text])

    # 예측
    proba = model.predict_proba(X)[0]
    pred = model.predict(X)[0]

    confidence = float(max(proba))

    return {
        "mbti": pred,
        "confidence": round(confidence, 3),
    }
