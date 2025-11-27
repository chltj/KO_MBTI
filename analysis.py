import re
import pandas as pd
from collections import Counter
from typing import Dict


# -----------------------------
# 1. 카카오톡 파싱 
# -----------------------------
def parse_kakao_chat(text: str, my_name: str) -> pd.DataFrame:
    """
    카카오톡 txt 파일 문자열을 받아 DataFrame으로 변환
    반환 컬럼: [datetime, speaker, message]
    """
    lines = text.splitlines()
    records = []

    # 기존 카톡 공식 내보내기 포맷
    # 예: 2023. 5. 12. 오후 3:21, 고주성 : 안녕
    pattern_export = re.compile(
        r"(\d{4}\. \d{1,2}\. \d{1,2}\. (?:오전|오후) \d{1,2}:\d{2}), (.*) : (.*)"
    )

    # 새로운 "복붙형" 카톡 포맷
    # 예: [유채린] [오후 8:48] 사진
    pattern_bracket = re.compile(
        r"^\[(?P<speaker>.*?)\]\s*\[(?P<time>.*?)\]\s*(?P<message>.*)$"
    )

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 1) 공식 내보내기 형식 매칭
        m1 = pattern_export.match(line)
        if m1:
            dt, speaker, message = m1.groups()
            records.append([dt, speaker.strip(), message.strip()])
            continue

        # 2) 복붙형 카톡 형식 매칭
        m2 = pattern_bracket.match(line)
        if m2:
            speaker = m2.group("speaker").strip()
            t = m2.group("time").strip()
            message = m2.group("message").strip()
            # datetime 을 일단 time 문자열만 저장 (필요시 나중에 처리 가능)
            records.append([t, speaker, message])
            continue

    df = pd.DataFrame(records, columns=["datetime", "speaker", "message"])
    return df

# -----------------------------
# 2. 말투 스타일 분석
# -----------------------------
def analyze_style(df: pd.DataFrame) -> Dict:
    messages = df["message"].astype(str)

    total = len(messages)
    if total == 0:
        return {}

    avg_length = messages.apply(len).mean()
    emoji_count = messages.str.count(r"[ㅋㅎㅠㅜ]").sum()
    question_ratio = messages.str.contains(r"\?").sum() / total
    exclam_ratio = messages.str.contains(r"!").sum() / total

    return {
        "평균 문장 길이": round(avg_length, 2),
        "이모티콘/감정표현 수": int(emoji_count),
        "질문 비율": round(question_ratio, 3),
        "감탄 비율": round(exclam_ratio, 3),
    }


# -----------------------------
# 3. 규칙 기반 MBTI 추정
# -----------------------------
def estimate_mbti(df: pd.DataFrame) -> Dict:
    messages = df["message"].astype(str)

    text = " ".join(messages)

    score = {
        "I": 0, "E": 0,
        "N": 0, "S": 0,
        "T": 0, "F": 0,
        "J": 0, "P": 0,
    }

    # 간단한 예시 룰 (추후 고도화 가능)
    if len(text) / max(len(messages), 1) > 15:
        score["E"] += 1
    else:
        score["I"] += 1

    if re.search(r"상상|미래|가능성|느낌", text):
        score["N"] += 1
    else:
        score["S"] += 1

    if re.search(r"논리|근거|이성적", text):
        score["T"] += 1
    else:
        score["F"] += 1

    if re.search(r"계획|정리|일정", text):
        score["J"] += 1
    else:
        score["P"] += 1

    mbti = (
        ("E" if score["E"] >= score["I"] else "I") +
        ("N" if score["N"] >= score["S"] else "S") +
        ("T" if score["T"] >= score["F"] else "F") +
        ("J" if score["J"] >= score["P"] else "P")
    )

    return {
        "mbti": mbti,
        "detail_score": score
    }
