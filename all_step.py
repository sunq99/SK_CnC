import json
import os

from config import OPENAI_API_KEY, FAISS_INDEX_PATH
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.retrievers.document_compressors import CrossEncoderReranker
from transformers import AutoModel
from langchain.retrievers import ContextualCompressionRetriever
from langchain.chat_models import ChatOpenAI
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, ResponseRelevancy
from intent_analysis import intent_analysis
from generate_multiquery_and_retrieve import generate_multiquery_and_retrieve
from generate_answer_and_evaluate import generate_answer_and_evaluate
from langchain.vectorstores import FAISS
from langchain.retrievers.multi_query import MultiQueryRetriever

def all_step(query):
    MAX_RETRIES = 3
    retry_count = 0
    final_result = None
    fallback_results = []
    FAIL_MESSAGE = "죄송합니다. 질문에 대한 정보를 찾을 수 없습니다. 조금 더 구체적인 상황을 추가해서 질문해주세요."
    success_attempt = None

    intent_result = intent_analysis(query)
    if not intent_result['is_copyright_related']:
        return "저작권법과 관련된 질문을 해주세요. 다른 법령에 관한 질문은 답변하기 어렵습니다.", [], None, [], "기타", "기타"

    embeddings_model = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=OPENAI_API_KEY
    )
    FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./faiss_index")
    vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings_model, allow_dangerous_deserialization=True)

    llm = ChatOpenAI(
        temperature=0,
        model_name='gpt-4-turbo',
        openai_api_key=OPENAI_API_KEY
    )

    retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(),
        llm=llm
    )

    # ✅ 대화 제목, 카테고리, 관련 질문 한 번에 생성
    metadata_prompt = f"""
    다음은 사용자의 질문입니다:

    "{query}"

    이 질문에 대해 다음 항목을 순서대로 출력해주세요:

    1. 대화 제목 (15자 이내, 따옴표 없이)
    2. 카테고리 (한 단어나 짧은 문장)
    3. 관련된 추가 질문 3가지 (줄바꿈으로 구분)

    출력 예시:

    제목: 유튜브 음악 저작권
    카테고리: 음악저작권
    관련 질문:
    - 유튜브 영상에 배경음악으로 상업곡을 사용하면 문제가 되나요?
    - 강의자료에 배경음악을 삽입해도 되나요?
    - 다른 사람의 음악을 편집해서 써도 되나요?
    """
    metadata_result = llm.predict(metadata_prompt).strip()
    lines = metadata_result.splitlines()
    title = lines[0].replace("제목:", "").strip()
    category = lines[1].replace("카테고리:", "").strip()
    related_questions = [line.replace("- ", "").strip() for line in lines[3:] if line.strip()]

    model = AutoModel.from_pretrained(
        'jinaai/jina-reranker-m0',
        torch_dtype="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )
    model.to("cuda")

    # ✅ 문서 필터 설정
    document_type_filters = []
    for doc_type in intent_result['document_types']:
        if doc_type == "법령":
            law_filter = {"문서유형": "법령"}
            article_num = intent_result['specific_article'].get("article")
            ho_num = intent_result['specific_article'].get("ho")
            if article_num:
                law_filter["조문번호"] = f"{article_num}조"
                if ho_num:
                    law_filter["호번호"] = ho_num
            document_type_filters.append(law_filter)

        elif doc_type == "부칙":
            sub_filter = {"문서유형": "부칙"}
            article_num = intent_result['specific_article'].get("article")
            if article_num:
                sub_filter["조문번호"] = f"{article_num}조"
            document_type_filters.append(sub_filter)

        elif doc_type == "시행령":
            ord_filter = {"문서유형": "시행령"}
            article_num = intent_result['specific_article'].get("article")
            ho_num = intent_result['specific_article'].get("ho")
            if article_num:
                ord_filter["조문번호"] = f"{article_num}조"
                if ho_num:
                    ord_filter["호번호"] = ho_num
            document_type_filters.append(ord_filter)

        elif doc_type == "판례":
            case_filter = {"문서유형": "판례"}
            article_num = intent_result['specific_article'].get("article")
            ho_num = intent_result['specific_article'].get("ho")
            related_article = []
            if article_num:
                if ho_num:
                    related_article.append(f"저작권법 제{article_num}조 제{ho_num}호")
                else:
                    related_article.append(f"저작권법 제{article_num}조")
            if related_article:
                case_filter["참조조문"] = {"$in": related_article}
            document_type_filters.append(case_filter)

        elif doc_type == "해석례":
            exp_filter = {"문서유형": "해석례"}
            document_type_filters.append(exp_filter)

    final_filter = {"$or": document_type_filters} if document_type_filters else {}
    if final_filter:
        vectorstore.as_retriever().search_kwargs["filter"] = final_filter

    while retry_count < MAX_RETRIES:
        filtered_docs = generate_multiquery_and_retrieve(query, retry_count, retriever, llm, model)
        final_answer, final_ref, evaluation_result = generate_answer_and_evaluate(query, filtered_docs, llm)

        faithfulness = evaluation_result["faithfulness"][0]
        relevancy = evaluation_result["answer_relevancy"][0]

        if faithfulness >= 0.5 and relevancy >= 0.7:
            final_result = final_answer
            success_attempt = retry_count + 1
            break
        elif 0.1 < faithfulness < 0.5 and relevancy >= 0.8:
            fallback_results.append({
                "answer": final_answer,
                "docs": filtered_docs,
                "final_ref": final_ref,
                "faithfulness": faithfulness,
                "relevancy": relevancy,
                "retry": retry_count + 1
            })

        retry_count += 1

    if final_result:
        success_message = f"✅ {success_attempt}회차 시도에 정식 기준으로 답변이 생성되었습니다."
        final_result_with_note = f"{success_message}\n\n{final_result}"
        return final_result_with_note, final_result, filtered_docs, evaluation_result, related_questions, category, title

    elif fallback_results:
        best_fallback = max(fallback_results, key=lambda x: (x["relevancy"] + x["faithfulness"]))
        fallback_msg = f"""⚠️ 정식 기준은 충족하지 못했지만 {best_fallback['retry']}회차 시도에 유사한 답변이 생성되었습니다:\n\n{best_fallback['answer']}\n\n※ 더 정확한 답변을 원하시면 질문을 조금 더 구체적으로 작성해 주세요."""
        return fallback_msg, best_fallback["answer"], best_fallback["docs"], evaluation_result, related_questions, category, title

    else:
        return FAIL_MESSAGE, FAIL_MESSAGE, [], None, related_questions, category, title
