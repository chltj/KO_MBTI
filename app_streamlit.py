import streamlit as st
import pandas as pd
from pathlib import Path
import platform
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

from analysis import parse_kakao_chat, analyze_style, estimate_mbti
from analysis_ml import predict_mbti_ml
from emotion_analysis import analyze_emotions

# -----------------------------
# matplotlib 한글 폰트 설정
# -----------------------------
def set_matplotlib_korean_font():
    system = platform.system()

    try:
        if system == "Windows":
            rc("font", family="Malgun Gothic")
        elif system == "Darwin":
            rc("font", family="AppleGothic")
        else:
            font_path = Path("assets/fonts/NanumGothic.ttf")
            if font_path.exists():
                font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
                rc("font", family=font_name)
        plt.rcParams["axes.unicode_minus"] = False
    except Exception as e:
        print(f"폰트 설정 에러: {e}")


set_matplotlib_korean_font()

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
# 세션 상태 초기값
# -----------------------------
if "run_analysis" not in st.session_state:
    st.session_state["run_analysis"] = False

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
            <p>대화 내용을 업로드하면, 참가자 각각의 말투를 기반으로 MBTI와 감정 패턴을 분석합니다.</p>
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
               - 참가자별 대화를 분리  
               - 각자 MBTI 추정 (규칙 + ML)  
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
        # 파일이 없어졌으면 분석 플래그도 꺼주기
        st.session_state["run_analysis"] = False
        return

    if not my_name.strip():
        st.warning("먼저 사이드바에 **내 이름**을 입력해 주세요.")
        st.session_state["run_analysis"] = False
        return

    # -------------------------
    # 분석 시작 버튼 (플래그만 세팅)
    # -------------------------
    if st.button("🚀 분석 시작", use_container_width=True):
        st.session_state["run_analysis"] = True

    # -------------------------
    # 플래그가 켜져 있을 때만 분석 수행
    # -------------------------
    if not st.session_state["run_analysis"]:
        return

    with st.spinner("카카오톡 대화 파싱 및 분석 중입니다..."):
        try:
            # txt → 문자열 (getvalue()는 매 실행마다 다시 읽을 수 있음)
            raw_bytes = uploaded_file.getvalue()
            if not raw_bytes:
                st.error("업로드된 파일 내용을 읽을 수 없습니다.")
                return

            raw_text = raw_bytes.decode("utf-8", errors="ignore")

            # 1) 카톡 파싱
            df_chat = parse_kakao_chat(raw_text, my_name=my_name)

            if df_chat.empty:
                st.error("파싱 결과가 비어 있습니다. 이름이 카톡과 동일한지, txt 형식이 맞는지 확인해 주세요.")
                return

            # 필요시 미리보기
            if show_raw_chat:
                st.subheader("📄 파싱된 대화 (전체)")
                st.dataframe(df_chat.head(80), use_container_width=True)

            if "speaker" not in df_chat.columns or "message" not in df_chat.columns:
                st.error("parse_kakao_chat 결과에 'speaker', 'message' 컬럼이 필요합니다.")
                return

            # -------------------------
            # 데이터 분리
            # -------------------------
            participants = sorted(df_chat["speaker"].dropna().unique().tolist())

            if not participants:
                st.error("speaker 정보가 비어 있습니다.")
                return

            if my_name not in participants:
                st.error(
                    "입력한 이름이 카카오톡 대화 목록에 없습니다.\n"
                    "카톡에 표시된 이름을 공백/띄어쓰기까지 정확히 입력해 주세요."
                )
                st.session_state["run_analysis"] = False
                return

            # speaker -> df / text 맵
            speaker_dfs = {}
            speaker_texts = {}

            for name in participants:
                sub_df = df_chat[df_chat["speaker"] == name].copy()
                speaker_dfs[name] = sub_df
                speaker_texts[name] = sub_df["message"].astype(str).tolist()

            # 이름 표시용 (나 표시)
            def display_name(name: str) -> str:
                return f"{name} (나)" if name == my_name else name

            # -------------------------
            # 2) MBTI / 스타일 / 감정 분석 계산
            # -------------------------
            mbti_rule = {}
            mbti_ml = {}
            style_results = {}
            emotion_results = {}

            for name in participants:
                df_person = speaker_dfs[name]
                texts_person = speaker_texts[name]

                # MBTI - 규칙 기반
                if analysis_mode in ["규칙 기반", "둘 다 비교"]:
                    rule_result = estimate_mbti(df_person)
                    mbti_rule[name] = (
                        rule_result.get("mbti") if isinstance(rule_result, dict) else rule_result
                    )
                else:
                    mbti_rule[name] = None

                # MBTI - ML 기반
                if analysis_mode in ["ML 기반", "둘 다 비교"] and texts_person:
                    ml_result = predict_mbti_ml(texts_person)
                    mbti_ml[name] = (
                        ml_result.get("mbti") if isinstance(ml_result, dict) else ml_result
                    )
                else:
                    mbti_ml[name] = None

                # 말투 스타일
                style_results[name] = analyze_style(df_person)

                # 감정 분석
                emotion_results[name] = analyze_emotions(texts_person) if texts_person else {}

            # -------------------------
            # 레이아웃 분할
            # -------------------------
            col_left, col_right = split_layout()

            # -------------------------
            # 3) MBTI 분석
            # -------------------------
            with col_left:
                st.subheader("🧬 MBTI 분석 결과")

                for name in participants:
                    st.markdown(f"### {display_name(name)}")
                    rule_val = mbti_rule.get(name)
                    ml_val = mbti_ml.get(name)

                    if analysis_mode in ["규칙 기반", "둘 다 비교"]:
                        st.write(f"- 규칙 기반: `{rule_val or '-'}`")

                    if analysis_mode in ["ML 기반", "둘 다 비교"]:
                        st.write(f"- ML 기반: `{ml_val or '-'}`")

                    if (
                        analysis_mode == "둘 다 비교"
                        and rule_val
                        and ml_val
                        and rule_val != ml_val
                    ):
                        st.info(f"⚖️ 규칙 기반과 ML 기반 결과가 다릅니다. ({rule_val} vs {ml_val})")

                    st.markdown("---")

            # -------------------------
            # 4) 말투 스타일 분석 
            # -------------------------
            with col_right:
                st.subheader("✏️ 말투 스타일 분석")

                tabs = st.tabs([display_name(n) for n in participants])

                for tab, name in zip(tabs, participants):
                    with tab:
                        df_person = speaker_dfs[name]
                        if df_person.empty:
                            st.info("대화가 부족하여 스타일 분석이 어렵습니다.")
                            continue

                        style = style_results.get(name, {})
                        if isinstance(style, dict) and style:
                            for k, v in style.items():
                                st.metric(
                                    label=k,
                                    value=round(v, 3) if isinstance(v, (int, float)) else v,
                                )
                        else:
                            st.write(style)

            # -------------------------
            # 5) 감정 분석 
            # -------------------------
            st.markdown("---")
            st.subheader("💬 감정 분석")

            selected_name = st.selectbox(
                "감정을 자세히 보고 싶은 사람을 선택하세요",
                participants,
                format_func=display_name,
            )

            emo_info = emotion_results.get(selected_name, {})

            col1, col2 = st.columns([1.2, 1])

            with col1:
                st.markdown(f"### {display_name(selected_name)} - 감정 요약")

                if isinstance(emo_info, dict) and emo_info:
                    if "summary" in emo_info:
                        st.write("**감정 요약**")
                        st.write(emo_info["summary"])

                if "examples" in emo_info:
                    st.write("**대표 문장 예시**")
                    dist = emo_info.get("distribution", {})
                    for emo, example in emo_info["examples"].items():
                        percent = round(dist.get(emo, 0) * 100, 1)
                        st.markdown(f"- **{emo}**: {example} ({percent}%)")

                else:
                    st.info("감정 분석 결과가 없습니다.")

            with col2:
                if isinstance(emo_info, dict) and "distribution" in emo_info:
                    emo_labels = list(emo_info["distribution"].keys())
                    emo_values = list(emo_info["distribution"].values())

                    # 감정별 색상
                    color_map = {
                        "기쁨": "#FFB400",   # 주황/노랑
                        "슬픔": "#4A90E2",   # 파랑
                        "분노": "#D0021B",   # 빨강
                        "불안": "#9013FE",   # 보라
                        "중립": "#9B9B9B",   # 회색
                    }
                    bar_colors = [color_map.get(label, "#CCCCCC") for label in emo_labels]

                    # 비율을 % 기준으로 표시
                    emo_values_percent = [v * 100 for v in emo_values]

                    fig, ax = plt.subplots()
                    ax.bar(emo_labels, emo_values_percent, color=bar_colors)
                    ax.set_title(f"감정 분포 - {display_name(selected_name)}")
                    ax.set_ylabel("비율(%)")
                    plt.xticks(rotation=0)

                    st.pyplot(fig)

                    # 텍스트로도 백분율 표시
                    st.write("**감정 분포 (백분율)**")
                    for label, value in zip(emo_labels, emo_values_percent):
                        st.write(f"- {label}: {round(value, 1)}%")

            # -------------------------
            # 6) 요약 
            # -------------------------
            st.markdown("---")
            st.subheader("📌요약")

            # 보조 함수들
            def get_main_emotion(emotion_dict):
                if (
                    isinstance(emotion_dict, dict)
                    and "distribution" in emotion_dict
                    and emotion_dict["distribution"]
                ):
                    return max(
                        emotion_dict["distribution"].items(),
                        key=lambda x: x[1],
                    )[0]
                return "-"

            def style_pick(style_dict, key):
                if isinstance(style_dict, dict) and key in style_dict:
                    v = style_dict[key]
                    return round(v, 2) if isinstance(v, (int, float)) else v
                return "-"

            # 참가자 수에 따라 row/column 배치
            per_row = 3
            for i in range(0, len(participants), per_row):
                row_names = participants[i : i + per_row]
                cols = st.columns(len(row_names))

                for col, name in zip(cols, row_names):
                    with col:
                        style = style_results.get(name, {})
                        emo = emotion_results.get(name, {})
                        main_emo = get_main_emotion(emo)

                        col_mbti_rule = mbti_rule.get(name)
                        col_mbti_ml = mbti_ml.get(name)

                        st.markdown(
                            f"""
                            <div style="
                                border-radius: 16px;
                                padding: 16px 20px;
                                border: 1px solid #eeeeee;
                                background-color: #fafafa;
                                ">
                                <h4>{display_name(name)}</h4>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        st.markdown("**MBTI**")
                        if analysis_mode in ["규칙 기반", "둘 다 비교"]:
                            st.write(f"- 규칙 기반: `{col_mbti_rule or '-'}`")
                        if analysis_mode in ["ML 기반", "둘 다 비교"]:
                            st.write(f"- ML 기반: `{col_mbti_ml or '-'}`")

                        st.markdown("---")
                        st.markdown("**말투 특징**")
                        st.write(
                            f"- 평균 문장 길이: {style_pick(style, '평균 문장 길이')}"
                        )
                        st.write(
                            f"- 이모티콘/감정표현 수: {style_pick(style, '이모티콘/감정표현 수')}"
                        )
                        st.write(f"- 질문 비율: {style_pick(style, '질문 비율')}")
                        st.write(f"- 감탄 비율: {style_pick(style, '감탄 비율')}")

                        st.markdown("---")
                        st.markdown("**감정 분위기**")
                        st.write(f"- 주 감정: **{main_emo}**")
                        if (
                            isinstance(emo, dict)
                            and "distribution" in emo
                            and emo["distribution"]
                        ):
                            for emo_label, score in sorted(
                                emo["distribution"].items(),
                                key=lambda x: x[1],
                                reverse=True,
                            )[:3]:
                                st.write(f"- {emo_label}: {round(score * 100, 1)}%")

        except Exception as e:
            st.error(f"알 수 없는 에러가 발생했습니다: {e}")


if __name__ == "__main__":
    main()
