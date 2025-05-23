from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever  # (필요 시)
from langchain.chat_models import ChatOpenAI
from transformers import AutoModel
from intent_analysis import intent_analysis
import re

def generate_multiquery_and_retrieve(query, retry_num, retriever, llm, model):
    intent_result = intent_analysis(query)
    # retry_num에 따른 다양한 프롬프트 구성##################################### (수정: needs_context 추가 및 프롬프트 수정)

    if intent_result['needs_context'] and retry_num == 0:
        prompt_template = f"""
        사용자 질문: {query}

        이 질문은 법률적 개념에 대한 맥락적 설명이 필요합니다.
        따라서 이 질문과 의미는 같지만 다른 표현으로 검색할 수 있는 3가지 다른 질문을 생성해주세요.
        다음과 같은 다양한 관점에서 질문을 작성해주세요:
        1. 법령 검색에 최적화된 질문
        2. 판례 검색에 최적화된 질문
        3. 법률 해석 검색에 최적화된 질문

        질문 목록만 줄바꿈으로 구분하여 작성해주세요.
        """
    elif intent_result['needs_context'] and retry_num == 1:
        prompt_template = f"""
        아래 질문에 대한 맥락적 설명을 위해 더 다양한 키워드와 시각으로 재해석하여 검색 쿼리를 생성해주세요:
        {query}

        - 법률적 키워드 및 동의어를 포함해주세요.
        - 질문의 맥락을 확장하거나 변형하여 검색에 도움이 되도록 바꿔주세요.

        3개의 질문을 줄바꿈으로 구분해주세요.
        """
    elif retry_num == 2:
        prompt_template = f"""
        아래 질문에 대해 다양한 시나리오를 상상하며 검색에 도움이 되는 쿼리를 생성해주세요:
        {query}

        - 판례, 해석례, 법령 등에서 유사 상황을 떠올리며 질문을 구성해주세요.
        - 의미는 유지하되 표현은 새롭게 구성해주세요.

        줄바꿈으로 구분된 3개의 질문만 작성해주세요.
        """
    else:
        prompt_template = f"{query}"

    # 멀티쿼리 생성
    new_queries = llm.predict(prompt_template).strip().split('\n')
    new_queries = [q.strip() for q in new_queries if q.strip()]
    # print(f"\n[멀티쿼리 {retry_num+1}회차 생성 결과] {new_queries}")

    # 멀티쿼리 리트리버 구성
    prompt_obj = PromptTemplate.from_template(prompt_template)
    multiquery_retriever = MultiQueryRetriever.from_llm(
        retriever=retriever,
        llm=llm,
        prompt=prompt_obj
    )

    # 리랭킹
    docs = multiquery_retriever.invoke(query)
    text_pairs = [[query, doc.page_content] for doc in docs]
    scores = model.compute_score(text_pairs, doc_type="text")

    # 점수와 docs 묶기
    scored_docs = list(zip(scores, docs))

    # 점수 기준 내림차순 정렬 후 상위 3개 설정
    top_k = sorted(scored_docs, key=lambda x: x[0], reverse=True)[:3]

    # 중복 문서 필터링
    filtered_docs = []

    # print("\n=== 간단한 중복 문서 필터링 ===")
    # print(f"필터링 전 문서 수: {len(top_k)}")

    for i, (score, doc) in enumerate(top_k):
        content = doc.page_content
        is_duplicate = False

        for j, existing_doc in enumerate(filtered_docs):
            existing_content = existing_doc.page_content

            # 핵심 키워드들을 추출하여 비교 (3글자 이상 한글 단어)
            current_keywords = set(re.findall(r'[가-힣]{3,}', content))
            existing_keywords = set(re.findall(r'[가-힣]{3,}', existing_content))

            # Jaccard 유사도 계산 (교집합/합집합)
            if current_keywords and existing_keywords:
                intersection = current_keywords.intersection(existing_keywords)
                union = current_keywords.union(existing_keywords)
                jaccard_similarity = len(intersection) / len(union)

                print(f"문서 {i+1} vs 기존 문서 {j+1}: Jaccard 유사도 = {jaccard_similarity:.3f}")

                # 70% 이상 유사하면 중복으로 판단
                if jaccard_similarity > 0.7:
                    is_duplicate = True
                    print(f"  → 중복으로 판단됨!")
                    break

        if not is_duplicate:
            filtered_docs.append(doc)
            # print(f"문서 {len(filtered_docs)} 추가: {content[:50]}...")
        # else:
            # print(f"중복 문서 제외: {content[:50]}...")
    return filtered_docs
