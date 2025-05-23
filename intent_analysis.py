import re
import json
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chat_models import ChatOpenAI
from sentence_transformers import CrossEncoder

def intent_analysis(user_input):
    # 사용자 질문
    # query = user_input
    query = "다른사람의 저작물을 내가 거짓으로 등록하면 어떻게 되나요?"
    # query = "갑은 고등학교 음악선생님인데, 학교 수업시간에 음악이론을 설명하기 위해서 공표되어 있는 노래를 저작자의 허락없이 학생들에게 들려주는 것이 저작권법위반인지?"

    # LLM 초기화
    llm = ChatOpenAI(
          temperature=0,
          model_name = 'gpt-4-turbo',
          openai_api_key = OPENAI_API_KEY
    )

    # 사용자 질문 의도 파악 ########################## 수정 (5,6번 제거, 시행령 추가)
    intent_analysis_prompt = f"""
    분석 대상 질문: {query}

    이 질문에 대해 다음을 분석해주세요:
    1. 이 질문이 저작권법에 관한 것인지? (예/아니오)
    2. 질문이 묻는 주요 문서 유형은 무엇인지? (법령/판례/해석례/복합적)
    3. 특정 조문이나 호를 언급하는지? 언급한다면 어떤 조와 호인지?
    4. 맥락적 설명이 필요한 질문인지? (개념 설명, 차이점 설명 등을 요구하는지)

    각 항목에 대해 간결하게 답변하고, JSON 형식으로 응답해주세요:

    ```json
    {{
    "is_copyright_related": true/false,
    "document_types": ["법령", "판례", "해석례", "시행령"],
    "specific_article": {{"article": "조번호", "ho": "호번호"}},
    "needs_context": true/false
    }}
    """

    intent_analysis_result = llm.predict(intent_analysis_prompt)




    # JSON 문자열을 파이썬 딕셔너리로 변환
    try:
        # JSON 블록 추출을 위한 정규식 (불필요한 마크다운 문법 제거)
        json_str = re.sub(r'```.*?\n|```', '', intent_analysis_result.strip())
        intent_data = json.loads(json_str)

    except json.JSONDecodeError:
        # JSON 파싱 실패 시 기본값 설정
        intent_data = {
            "is_copyright_related": True,
            "document_types": ["법령", "판례", "시행령"], ################# 수정 (시행령 추가)
            "specific_article": {"article": "2", "ho": "8"},
            "needs_context": True
        }

    # 분석 결과 추출
    is_copyright_related = intent_data.get("is_copyright_related", True)
    document_types = intent_data.get("document_types", ["법령", "판례", "시행령"])
    specific_article = intent_data.get("specific_article", {"article": "2", "ho": "8"})
    needs_context = intent_data.get("needs_context", True)

    return {
        "is_copyright_related": is_copyright_related,
        "document_types": document_types,
        "specific_article": specific_article,
        "needs_context": needs_context
    }
