import os
import re
import json
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, ResponseRelevancy

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

ragas_llm = ChatOpenAI(model="gpt-4o")
ragas_embeddings = OpenAIEmbeddings()

def generate_answer_and_evaluate(query, filtered_docs, llm): ################################ 수정 (함수 전체적으로 수정- 문서 처리)
    # 프롬프트 구성
    final_ref = []
    response_prompt = f"다음은 '{query}'에 대한 관련 정보입니다:\n\n"

    # 중복이 제거된 문서들을 프롬프트에 추가
    for i, doc in enumerate(filtered_docs):
        doc_type = doc.metadata.get('문서유형', '')

        # 문서 내용 가져오기
        content = doc.page_content

        # 미완성 문장 처리 (여러 패턴 대응)

        # 괄호 밸런스 확인 및 수정
        def fix_unbalanced_parentheses(text):
            # 열린 괄호와 닫힌 괄호 개수 확인
            open_count = text.count('(')
            close_count = text.count(')')

            # 괄호 불균형 확인
            if open_count > close_count:
                # 열린 괄호가 더 많은 경우, 마지막 완전한 괄호 구문 이후 내용 제거
                # 가장 마지막 닫힌 괄호 위치 찾기
                last_close_idx = text.rfind(')')
                if last_close_idx > 0:
                    # 해당 닫힌 괄호와 짝을 이루는 열린 괄호 찾기 시도
                    stack = []
                    for idx, char in enumerate(text[:last_close_idx+1]):
                        if char == '(':
                            stack.append(idx)
                        elif char == ')':
                            if stack:  # 짝이 맞는 열린 괄호가 있으면 pop
                                stack.pop()
                            # 스택이 비어있으면 짝이 안 맞는 닫힌 괄호

                    # 아직 짝이 없는 열린 괄호가 있는 경우
                    if stack:
                        # 가장 마지막 짝이 맞는 괄호 구조 이후 위치 찾기
                        pos = last_close_idx + 1
                        # 불완전한 구문을 제거하고 중략 표시 추가
                        return text[:pos] + "...(후략)"

            # 닫힌 괄호가 더 많은 경우, 첫 번째 닫힌 괄호 위치 전까지만 사용
            elif close_count > open_count:
                first_unmatched_close = -1
                stack = []
                for idx, char in enumerate(text):
                    if char == '(':
                        stack.append(idx)
                    elif char == ')':
                        if stack:  # 짝이 맞는 열린 괄호가 있으면 pop
                            stack.pop()
                        else:  # 짝이 없는 첫 번째 닫힌 괄호 찾기
                            first_unmatched_close = idx
                            break

                if first_unmatched_close > 0:
                    return "(전략)... " + text[first_unmatched_close+1:]

            # 괄호가 균형을 이루거나 수정이 필요없는 경우
            return text

        # 판례 인용 패턴 처리
        def fix_case_citation(text):
            # "선고 OO다OO 판결 참조), " 패턴 찾기
            pattern = r'선고\s+\d+\s*다\s*\d+\s*판결\s*참조\)\s*,'
            match = re.search(pattern, text)
            if match:
                # 매치된 부분의 끝 위치
                end_pos = match.end()
                # 앞부분은 유지하고 패턴 이후의 내용은 중략 표시로 대체
                return text[:end_pos] + " ...(중략)"
            return text

        #### 미완성 문장 처리 (여러 패턴 대응) ###

        # 판례 인용 패턴 처리
        content = fix_case_citation(content)

        # 괄호 밸런스 확인 및 수정
        content = fix_unbalanced_parentheses(content)

        # 내용이 쉼표(,)로 시작하는 경우 처리
        if content.strip().startswith(','):
            content = "...(중략) " + content.strip()[1:].strip()

        # 문장이 특정 패턴으로 시작하는 경우
        if content.startswith('.') or content.startswith(')') or re.match(r'^\s*[0-9]+\.', content):
            content = re.sub(r'^\.\s*', '', content)

        # 미완성 문장이 끝나는 경우
        if content.endswith('(') or content.endswith(',') or re.search(r'[a-zA-Z가-힣]\s*$', content):
            content = content.strip() + "..."

        # 기존 코드의 미완성 문장 처리 (점으로 시작하는 경우)
        if content.startswith('.'):
            content = re.sub(r'^\.\s*[^.]*\)\s*', '', content)

        # 콜론(:) 다음에 콤마(,)가 오는 경우 처리
        content = re.sub(r':\s*,\s*', '', content)

        # 문장이 "이" 또는 "화"와 같은 한글 한 글자로 끝나는 경우 (잘린 문장)
        if re.search(r'[가-힣]\s*$', content) and len(content.strip()) > 0:
            last_char = content.strip()[-1]
            # 문장 종결 조사가 아닌 경우에만 처리
            if last_char not in ['다', '까', '요', '죠', '잖', '죠', '네', '요', '임']:
                # 마지막 완전한 문장 찾기
                last_sentence_end = max(content.rfind('. '), content.rfind('.\n'), content.rfind('? '), content.rfind('! '))
                if last_sentence_end > 0:
                    # 마지막 완전한 문장까지만 유지하고 나머지는 제거
                    content = content[:last_sentence_end+1] + " ...(후략)"
                else:
                    # 완전한 문장 구분이 없으면 그대로 ... 추가
                    content = content.strip() + "..."

        # 특정 판례 인용 패턴 처리 (예: "7. 12. 선고 77다90 판결 참조),")
        pattern = r'\d+\.\s*\d+\.\s*선고\s+\d+다\d+\s*판결\s*참조\)\s*,\s*'
        if re.search(pattern, content):
            match = re.search(pattern, content)
            if match:
                end_pos = match.end()
                if end_pos < len(content):
                    # 패턴 이후 내용이 한 문장 이상인 경우에만 중략 처리
                    sentences_after = re.split(r'[.!?]\s+', content[end_pos:])
                    if len(sentences_after) > 1 and len(sentences_after[0]) > 20:
                        content = content[:end_pos] + "...(이하 생략)"

        # 맨 앞에 숫자 하나만 있는 경우 처리
        content = re.sub(r'^\s*(\d+)\s+', '', content)

        # 빈 문장이나 특수문자만 있는 문장 필터링
        content = content.strip()
        if not content or content in ['.', ',', ')', '(']:
            continue

        # 문서 유형에 따른 참조 형식 생성 ################################################# 수정(부칙, 시행령, 해석례 처리 안됐던 부분 수정)
        if doc_type == '법령':
            article_num = doc.metadata.get('조문번호', '')
            article_title = doc.metadata.get('조문제목', '')
            ho_num = doc.metadata.get('호번호', '')

            if ho_num:
                reference = f"[법령 {i+1}] 저작권법 제{article_num} {article_title} 제{ho_num}호: {content}"
            else:
                reference = f"[법령 {i+1}] 저작권법 제{article_num} {article_title}: {content}"

        elif doc_type == '시행령':
            ord_num = doc.metadata.get('시행령_조문번호', '')
            ord_title = doc.metadata.get('시행령_조문제목', '')
            ord_ho_num = doc.metadata.get('호번호', '')

            if ord_ho_num:  # 'ho_num'이 아닌 'ord_ho_num' 사용 (변수명 수정)
                reference = f"[법령 {i+1}] 저작권법 시행령 제{ord_num} {ord_title} 제{ord_ho_num}호: {content}"
            else:
                reference = f"[법령 {i+1}] 저작권법 시행령 제{ord_num} {ord_title}: {content}"

        elif doc_type == '부칙':
            sub_num = doc.metadata.get('부칙_조문번호', '')
            sub_title = doc.metadata.get('부칙_조문제목', '')
            hang_num = doc.metadata.get('항번호', '')

            if hang_num:  # 'ho_num'이 아닌 'hang_num' 사용 (변수명 수정)
                reference = f"[법령 {i+1}] 저작권법 부칙 제{sub_num} {sub_title} 제{hang_num}호: {content}"
            else:
                reference = f"[법령 {i+1}] 저작권법 부칙 제{sub_num} {sub_title}: {content}"

        elif doc_type == '판례':
            case_num = doc.metadata.get('사건번호', doc.metadata.get('판례번호', ''))
            case_date = doc.metadata.get('선고일자', doc.metadata.get('판결일자', ''))
            court = doc.metadata.get('법원명', '')

            reference = f"[판례 {i+1}] {court} {case_num} ({case_date}): {content}"

        elif doc_type == '해석례':
            exp_name = doc.metadata.get('안건명', '')
            exp_date = doc.metadata.get('회신일자', '')

            reference = f"[해석례 {i+1}] {exp_name} ({exp_date}): {content}"

        else:
            reference = f"[참조 {i+1}] {content}"

        response_prompt += reference + "\n\n"
        final_ref.append(reference)

    response_prompt += """
    위 정보를 바탕으로 사용자의 질문에 답변해주세요. 다음 가이드라인을 따라주세요:
    1. 문서들을 명확히 인용해주세요 (예: [법령 1]에 따르면..., [판례 1]에 따르면..., [해석례 1]에 따르면... 등).
    2. 법률 용어는 일반인이 이해할 수 있도록 풀어서 설명해주세요.
    3. 답변은 논리적이고 단락이 나눠져야 합니다.
    4. 결론을 명확히 제시해주세요.
    """

    # LLM으로 최종 답변 생성
    final_answer = llm.predict(response_prompt)
    print("\n=== 최종 답변 ===\n")
    print(final_answer)
    print("\n\n=== 참조 문서 목록 ===\n")
    for i, reference in enumerate(final_ref):
        print(f"{reference}\n")

    # 평가 수행
    ragas_llm = ChatOpenAI(model="gpt-4o")
    evaluation_dataset = EvaluationDataset.from_list([{
        "user_input": query,
        "retrieved_contexts": [doc.page_content for doc in filtered_docs],
        "response": final_answer,
    }])
    evaluator_llm = LangchainLLMWrapper(ragas_llm)
    result = evaluate(dataset=evaluation_dataset,
                      metrics=[Faithfulness(), ResponseRelevancy()],
                      llm=evaluator_llm)

    return final_answer, final_ref, result
