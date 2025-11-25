import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path


DATA_PATH = Path("data/kakao_mbti_dataset.csv")
MODEL_PATH = Path("models/mbti_model.joblib")


def load_dataset():
    df = pd.read_csv(DATA_PATH)

    # 필수 컬럼: text, mbti
    if not {"text", "mbti"}.issubset(df.columns):
        raise ValueError("CSV에 'text'와 'mbti' 컬럼이 필요합니다.")

    return df


def train_model():
    df = load_dataset()

    X = df["text"].astype(str)
    y = df["mbti"].astype(str)

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2
    )

    X_vec = vectorizer.fit_transform(X)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

    model.fit(X_vec, y)

    bundle = {
        "vectorizer": vectorizer,
        "model": model
    }

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)

    print("✅ MBTI 모델 학습 완료!")
    print(f"저장 경로: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
