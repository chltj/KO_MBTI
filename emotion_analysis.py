from typing import List, Dict
from collections import Counter
import re


# 간단 감정 키워드 사전 (추후 고도화 가능)
EMOTION_LEXICON = {
    "기쁨": ["좋아", "행복", "재밌", "웃기", "최고", "ㅋㅋ", "ㅎㅎ", "개꿀", "득템"],
    "슬픔": ["우울", "슬프", "힘들", "눈물", "외롭", "ㅠ", "ㅜ", "허무", "현타"],
    "분노": ["짜증", "화나", "빡치", "열받", "극혐", "개빡", "환장"],
    "불안": ["걱정", "걱정되", "불안", "초조", "긴장", "떨리", "조마조마"],
    "중립": []
}

def detect_emotion(sentence: str) -> str:
    for emotion, keywords in EMOTION_LEXICON.items():
        for kw in keywords:
            if kw in sentence:
                return emotion
    return "중립"


def analyze_emotions(texts: List[str]) -> Dict:
    if not texts:
        return {}

    emotion_results = []
    example_sentences = {}

    for text in texts:
        emo = detect_emotion(text)
        emotion_results.append(emo)
        if emo not in example_sentences:
            example_sentences[emo] = text

    counter = Counter(emotion_results)
    total = sum(counter.values())

    distribution = {emo: round(cnt / total, 3) for emo, cnt in counter.items()}

    top_sorted = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
    top_emotions = dict(top_sorted[:3])

    # 분위기 요약 생성
    main_emotion = top_sorted[0][0] if top_sorted else "중립"

    summary = f"대화의 전반적인 분위기는 '{main_emotion}'에 가깝습니다. "
    if main_emotion == "기쁨":
        summary += "밝고 긍정적인 말투가 자주 나타납니다."
    elif main_emotion == "슬픔":
        summary += "감정적으로 다소 가라앉은 상태가 보입니다."
    elif main_emotion == "분노":
        summary += "공격적이거나 예민한 표현이 일부 관찰됩니다."
    elif main_emotion == "불안":
        summary += "불확실한 상황에 대한 걱정이 느껴집니다."
    else:
        summary += "감정의 큰 기복 없이 비교적 안정적입니다."

    return {
        "distribution": distribution,      # 감정 분포
        "top_emotions": top_emotions,      # 상위 감정
        "examples": example_sentences,     # 대표 문장
        "summary": summary                 # 요약 설명
    }
