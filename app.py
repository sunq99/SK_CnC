import streamlit as st
import re
import time
import json

# import openai
from openai import OpenAI
from config import OPENAI_API_KEY
from intent_analysis import intent_analysis
from generate_multiquery_and_retrieve import generate_multiquery_and_retrieve
from generate_answer_and_evaluate import generate_answer_and_evaluate
from all_step import all_step
from utils import linkify_articles
from google_sheets import append_feedback_to_sheet
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chat_models import ChatOpenAI
from sentence_transformers import CrossEncoder

#openai.api_key = OPENAI_API_KEY
client = OpenAI(api_key=OPENAI_API_KEY)

st.set_page_config(page_title="ASAC 법률자문 AI", layout="wide", page_icon="📚")
st.title("ASAC 저작권법 법률 자문에 오신 것을 환영합니다.")

st.markdown("""
<div style='font-size:18px; line-height:1.6'>
저작권법 전문 생성형 AI가 법령, 판례, 해석례를 기반으로 신속하고 신뢰성 있는 자문을 제공합니다.<br><br>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style='font-size:15px; color:#888; border:1px solid #ddd; padding:10px; border-radius:5px; margin-bottom:10px;'>
    <div style='font-size:20px; line-height:1.6; color:#888;'><b>📌 질문 예시</b></div>
    <div>　　Q. 유튜브 영상에 다른 사람의 음악을 배경으로 쓰면 저작권 침해인가요?</div>
    <div>　　Q. 허락 없이 써도 되는 저작물의 조건에 뭐가 있나요?</div>
    <div>　　Q. 유튜브에 올리는 것과 개인 블로그에 쓰는 것 중 뭐가 더 문제인가요?</div>
</div>
""", unsafe_allow_html=True)

# 세션 상태 초기화
for key in ["messages", "chat_sessions", "active_chat", "related_questions", "prompt_input"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key in ["messages", "related_questions"] else {} if key == "chat_sessions" else None

# 사이드바
with st.sidebar:
    if st.button("➕ 새 대화"):
        st.session_state.messages = []
        st.session_state.active_chat = "대화 준비 중..."
        st.session_state.related_questions = []
        st.session_state.prompt_input = None

    st.subheader("📁 이전 대화")
    for title in reversed(list(st.session_state.chat_sessions.keys())):
        if st.button(title):
            st.session_state.messages = st.session_state.chat_sessions[title]["messages"]
            st.session_state.active_chat = title
            st.session_state.related_questions = st.session_state.chat_sessions[title].get("related", [])
            st.session_state.prompt_input = None

# 사용자 입력
user_input = st.chat_input("저작권법에 관한 궁금한 점을 입력하세요.")
if user_input:
    st.session_state["prompt_input"] = user_input

# 질문이 들어온 경우 처리
if st.session_state["prompt_input"]:
    prompt = st.session_state["prompt_input"]
    spinner = st.empty()
    spinner.info("🧠 AI가 신중히 답변을 구성하고 있습니다...")

    try:
        final_result_with_note, final_answer, source_docs, evaluation_result, related_questions, category, title = all_step(prompt)
        st.session_state.related_questions = related_questions

        st.session_state.messages.append({"role": "user", "content": prompt, "source_docs": []})
        st.session_state.messages.append({
            "role": "assistant",
            "content": final_result_with_note,
            "source_docs": source_docs
        })

        st.session_state.active_chat = title
        st.session_state.chat_sessions[title] = {
            "messages": st.session_state.messages,
            "category": category,
            "related": related_questions
        }

        spinner.empty()

    except Exception as e:
        spinner.empty()
        st.error(f"❌ 오류 발생: {e}")
        st.stop()

    finally:
        st.session_state["prompt_input"] = None

# 채팅 메시지 출력
if st.session_state.active_chat and st.session_state.active_chat != "대화 준비 중...":
    category = st.session_state.chat_sessions[st.session_state.active_chat].get("category", "기타")
    if not category or not category.strip():
        category = "기타"
    st.markdown(f"📂 **카테고리:** `{category}`")

    with st.container():
        for idx, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                st.markdown(linkify_articles(msg["content"]), unsafe_allow_html=True)

                if msg["role"] == "assistant" and "source_docs" in msg:
                    st.markdown("📎 **참조 문서 목록**")
                    for i, doc in enumerate(msg["source_docs"]):
                        doc_type = doc.metadata.get("문서유형", "문서")
                        type_icon_map = {
                          "판례": "📄",
                          "해석례": "📘",
                          "법령": "📜",
                          "시행령": "📑",
                          "부칙": "📂",
                        }
                        icon = type_icon_map.get(doc_type, "📎")
                        label = f"{icon} {doc_type or '문서'} {i+1}"
                        st.write(f"**{label}**")
                        st.write(doc.page_content[:300] + "...")
                        visible_keys = ['사건명', '사건번호', '선고일자', '법원명']
                        meta = {k: v for k, v in doc.metadata.items() if k in visible_keys}

                        if meta:
                            court = meta.get('법원명', '')
                            case_title = meta.get('사건명', '')
                            case_number = meta.get('사건번호', '')
                            date = meta.get('선고일자', '')

                            summary = f"""
                            <div style='background-color:#f9f9f9; color:green; border:1px solid #ddd; padding:10px; border-radius:5px; margin-bottom:10px;'>
                            📄 {date}에 {court}에서 '{case_title}' 사건({case_number})에 대한 판결입니다.
                            </div>
                            """
                            st.markdown(summary, unsafe_allow_html=True)

                if msg["role"] == "assistant":
                    feedback_key = f"feedback_{idx}"
                    if not st.session_state.get(feedback_key):
                        with st.expander("이 답변이 도움이 되었나요?"):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                if st.button("👍 도움이 됐어요", key=f"helpful_{idx}"):
                                    append_feedback_to_sheet("도움이 됐어요", msg["content"], st.session_state.active_chat)
                                    st.success("감사합니다.")
                                    st.session_state[feedback_key] = True
                            with col2:
                                if st.button("👎 더 궁금해요", key=f"more_info_{idx}"):
                                    append_feedback_to_sheet("더 궁금해요", msg["content"], st.session_state.active_chat)
                                    st.success("추가 개선하게사합니다.")
                                    st.session_state[feedback_key] = True
                            with col3:
                                if st.button("🤔 이해가 어렵습니다", key=f"difficult_{idx}"):
                                    append_feedback_to_sheet("이해 어렵습니다", msg["content"], st.session_state.active_chat)
                                    st.success("도움 주셔서 감사합니다.")
                                    st.session_state[feedback_key] = True

    valid_related = [q for q in st.session_state.related_questions if q.strip()]
    if valid_related:
        st.markdown("📚 **추가로 궁금할 수 있는 질문**")
        for idx, q in enumerate(valid_related[:3]):
            if st.button(q, key=f"related_q_{idx}"):
                st.session_state["prompt_input"] = q
else:
    st.info("왼쪽에서 대화를 선택하거나 새 대화를 시작하세요.")
