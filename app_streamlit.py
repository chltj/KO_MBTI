import streamlit as st
import pandas as pd
from pathlib import Path

from analysis import parse_kakao_chat, analyze_style, estimate_mbti
from analysis_ml import predict_mbti_ml
from emotion_analysis import analyze_emotions

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(
    page_title="카카오톡 말투 기반 MBTI + 감정 분석기",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# 커스텀 CSS 로드
# -----------------------------
def load_css():
    css_path = Path("assets/style.css")
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

load_css()

# -----------------------------
# 유틸 함수
# -----------------------------
def show_header():
    st.markdown(
        """
        <div class="main-header">
            <h1>🧠 카카오톡 말투 기반 MBTI & 감정 분석기</h1>
            <p>대화 내용을 업로드하면, 상대방의 말투를 기반으로 MBTI와 감정 패턴을 분석합니다.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

def show_how_to_use():
    with st.expander("❓ 사용 방법", expanded=False):
        st.markdown(
            """
            1. **카카오톡 대화 txt 파일**을 업로드합니다. (내보내기한 원본 txt)
            2. **내 이름**을 정확히 입력합니다. (카톡에 표시된 이름과 동일하게)
            3. 분석 옵션에서 **규칙 기반 / ML 기반 / 둘 다** 중 선택합니다.
            4. [분석 시작] 버튼을 누르면  
               - 상대방 대화만 추출  
               - MBTI 추정 (규칙 + ML)  
               - 감정 분포 & 키워드  
               - 시각화 차트  
               가 순서대로 출력됩니다.
            """
        )

def split_layout():
    col_left, col_right = st.columns([1.2, 1])
    return col_left, col_right

# -----------------------------
# 메인 앱
# -----------------------------
def main():
    show_header()
    show_how_to_use()

    # 사이드바 설정
    st.sidebar.subheader("⚙️ 분석 설정")

    my_name = st.sidebar.text_input("내 이름 (카톡에 표시된 이름 그대로)", value="")
    analysis_mode = st.sidebar.radio(
        "MBTI 분석 모드 선택",
        options=["규칙 기반", "ML 기반", "둘 다 비교"],
        index=2,
    )

    show_raw_chat = st.sidebar.checkbox("파싱된 대화 DataFrame 보기", value=False)

    uploaded_file = st.file_uploader("📁 카카오톡 대화 txt 업로드", type=["txt"])

    if uploaded_file is None:
        st.info("왼쪽에서 txt 파일을 업로드하고, 이름을 입력하면 분석을 시작할 수 있습니다.")
        return

    if not my_name.strip():
        st.warning("먼저 사이드바에 **내 이름**을 입력해 주세요.")
        return

    # 분석 버튼
    if st.button("🚀 분석 시작", use_container_width=True):
        with st.spinner("카카오톡 대화 파싱 및 분석 중입니다..."):
            try:
                # txt → 문자열
                raw_text = uploaded_file.read().decode("utf-8", errors="ignore")

                # 1) 카톡 파싱 & 상대방 대화만 추출
                df_chat = parse_kakao_chat(raw_text, my_name=my_name)

                if df_chat.empty:
                    st.error("파싱 결과가 비어 있습니다. 이름이 카톡과 동일한지, txt 형식이 맞는지 확인해 주세요.")
                    return

                # 필요시 미리보기
                if show_raw_chat:
                    st.subheader("📄 파싱된 대화 (상대방 포함 전체 or 상대방만)")
                    st.dataframe(df_chat.head(50), use_container_width=True)

                # 상대방 발화만 모으기 (speaker 컬럼 기준 가정)
                if "speaker" in df_chat.columns and "message" in df_chat.columns:
                    other_df = df_chat[df_chat["speaker"] != my_name].copy()
                    other_texts = other_df["message"].astype(str).tolist()
                    full_other_text = "\n".join(other_texts)
                else:
                    # 만약 컬럼명이 다르다면 여기만 수정
                    st.error("parse_kakao_chat 결과에 'speaker', 'message' 컬럼이 필요합니다.")
                    return

                # 레이아웃 분할
                col_left, col_right = split_layout()

                # 2) MBTI 분석
                with col_left:
                    st.subheader("🧬 MBTI 분석 결과")

                    rule_mbti = None
                    ml_mbti = None

                    if analysis_mode in ["규칙 기반", "둘 다 비교"]:
                        rule_result = estimate_mbti(other_df)
                        rule_mbti = rule_result.get("mbti") if isinstance(rule_result, dict) else rule_result
                        st.markdown(f"**규칙 기반 추정 MBTI:** `{rule_mbti}`")

                    if analysis_mode in ["ML 기반", "둘 다 비교"]:
                        ml_result = predict_mbti_ml(other_texts)
                        ml_mbti = ml_result.get("mbti") if isinstance(ml_result, dict) else ml_result
                        st.markdown(f"**ML 기반 추정 MBTI:** `{ml_mbti}`")

                    if rule_mbti and ml_mbti and rule_mbti != ml_mbti:
                        st.info(f"⚖️ 두 방식 결과가 다릅니다. 규칙: **{rule_mbti}**, ML: **{ml_mbti}**")

                # 3) 말투 스타일 분석
                with col_right:
                    st.subheader("✏️ 말투 스타일 분석")
                    style_result = analyze_style(other_df)

                    # 예시: style_result에 이런 값들이 들어있다고 가정
                    # {"avg_length": 23.1, "emoji_count": 120, "question_ratio": 0.32, ...}
                    if isinstance(style_result, dict):
                        for k, v in style_result.items():
                            st.metric(label=k, value=round(v, 3) if isinstance(v, (int, float)) else v)
                    else:
                        st.write(style_result)

                # 4) 감정 분석
                st.markdown("---")
                st.subheader("💬 감정 분석")

                emotion_result = analyze_emotions(other_texts)

                col1, col2 = st.columns([1.2, 1])
                with col1:
                    if isinstance(emotion_result, dict) and "summary" in emotion_result:
                        st.write("**감정 요약**")
                        st.write(emotion_result["summary"])

                    if isinstance(emotion_result, dict) and "examples" in emotion_result:
                        st.write("**대표 문장 예시**")
                        for emo, example in emotion_result["examples"].items():
                            st.markdown(f"- **{emo}**: {example}")

                with col2:
                    if isinstance(emotion_result, dict) and "distribution" in emotion_result:
                        import matplotlib.pyplot as plt

                        emo_labels = list(emotion_result["distribution"].keys())
                        emo_values = list(emotion_result["distribution"].values())

                        fig, ax = plt.subplots()
                        ax.bar(emo_labels, emo_values)
                        ax.set_title("감정 분포")
                        ax.set_ylabel("비율")
                        plt.xticks(rotation=30)

                        st.pyplot(fig)

                # 5) 전체 요약 카드
                st.markdown("---")
                st.subheader("📌 요약")

                summary_cols = st.columns(3)
                with summary_cols[0]:
                    st.markdown("#### 🧬 MBTI 요약")
                    st.write(f"- 규칙 기반: **{rule_mbti or '-'}**")
                    st.write(f"- ML 기반: **{ml_mbti or '-'}**")

                with summary_cols[1]:
                    st.markdown("#### ✏️ 말투 특징")
                    if isinstance(style_result, dict):
                        keys = list(style_result.keys())[:4]
                        for k in keys:
                            v = style_result[k]
                            st.write(f"- {k}: {round(v, 2) if isinstance(v, (int, float)) else v}")
                    else:
                        st.write(style_result)

                with summary_cols[2]:
                    st.markdown("#### 💬 감정 분위기")
                    if isinstance(emotion_result, dict) and "top_emotions" in emotion_result:
                        for emo, score in emotion_result["top_emotions"].items():
                            st.write(f"- {emo}: {round(score, 2)}")
                    elif isinstance(emotion_result, dict) and "distribution" in emotion_result:
                        for emo, score in sorted(
                            emotion_result["distribution"].items(),
                            key=lambda x: x[1],
                            reverse=True
                        )[:4]:
                            st.write(f"- {emo}: {round(score, 2)}")

            except Exception as e:
                st.error(f"알 수 없는 에러가 발생했습니다: {e}")

if __name__ == "__main__":
    main()
