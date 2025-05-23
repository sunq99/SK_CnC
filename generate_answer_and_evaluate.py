import os
import re
import json
from config import OPENAI_API_KEY
from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, ResponseRelevancy

ragas_llm = ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY)
ragas_embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

def clean_incomplete_sentences(content):
    """
    미완성 문장을 처리하는 개선된 함수
    """

    # 1. 문장 시작 부분의 쉼표 제거 (앞에 아무것도 없이 쉼표로 시작하는 경우)
    if content.strip().startswith(','):
        content = content.strip()[1:].strip()
        # 앞에 내용이 잘린 것이므로 중략 표시 추가
        content = "...(중략) " + content

    # 2. 문장 시작 부분의 닫힌 괄호 제거 (개선된 버전)
    if content.strip().startswith(')'):
        # 닫힌 괄호와 그 다음의 연결어들을 모두 제거
        content = re.sub(r'^\s*\)\s*(및|그리고|또는|혹은)?\s*', '', content).strip()
        if content:
            content = "...(중략) " + content

    # 3. 법률 개정 날짜 패턴 처리 (예: "1. 26. 법률 제16600호로 개정되기 전의 것)")
    # 연도가 없는 날짜 패턴을 찾아서 처리
    pattern = r'^\s*\d{1,2}\.\s*\d{1,2}\.\s*법률\s*제\d+호로\s*개정되기\s*전의\s*것\)\s*'
    if re.match(pattern, content):
        content = re.sub(pattern, '', content)
        if content.strip():
            content = "...(중략) " + content

    # 4. 불완전한 법률 조항 참조 처리 (예: "전의 것) 제35조의3")
    pattern = r'^\s*전의\s*것\)\s*제\d+조(?:의\d+)?\s*'
    if re.match(pattern, content):
        content = re.sub(pattern, '', content)
        if content.strip():
            content = "...(중략) " + content

    # 5. 문장 중간에 나타나는 불완전한 괄호 내용 제거
    # (2019. 1. 26. 법률 제16600호로 개정되기 전의 것) 같은 패턴
    pattern = r'\(\d{4}\.\s*\d{1,2}\.\s*\d{1,2}\.\s*법률\s*제\d+호로\s*개정되기\s*전의\s*것\)'
    content = re.sub(pattern, '', content)

    # 6. 숫자와 점으로만 시작하는 경우 (예: "1. 26. ")
    pattern = r'^\s*\d{1,2}\.\s*\d{1,2}\.\s+'
    if re.match(pattern, content):
        content = re.sub(pattern, '', content)
        if content.strip():
            content = "...(중략) " + content

    # 7. 점으로 시작하는 경우
    if content.strip().startswith('.'):
        content = re.sub(r'^\.\s*', '', content)
        if content.strip():
            content = "...(중략) " + content

    # 8. 숫자만으로 시작하는 경우
    content = re.sub(r'^\s*\d+\s+', '', content)

    # 9. 판례 인용 패턴 처리
    # "선고 OO다OO 판결 참조), " 패턴 찾기
    pattern = r'선고\s+\d+\s*다\s*\d+\s*판결\s*참조\)\s*,'
    match = re.search(pattern, content)
    if match:
        end_pos = match.end()
        content = content[:end_pos] + " ...(중략)"

    # 10. 괄호 밸런스 확인 및 수정 (개선된 버전)
    def fix_parentheses(text):
        open_count = text.count('(')
        close_count = text.count(')')

        # 닫힌 괄호가 더 많은 경우 (앞부분이 잘린 경우)
        if close_count > open_count:
            # 첫 번째 균형이 맞지 않는 닫힌 괄호 찾기
            balance = 0
            for i, char in enumerate(text):
                if char == '(':
                    balance += 1
                elif char == ')':
                    balance -= 1
                    if balance < 0:  # 여기서 불균형 발생
                        # 이 위치부터 시작
                        text = text[i+1:].strip()
                        if text:
                            text = "...(전략) " + text
                        break

        # 열린 괄호가 더 많은 경우 (뒷부분이 잘린 경우)
        elif open_count > close_count:
            # 마지막 균형이 맞는 지점 찾기
            balance = 0
            last_balanced = -1
            for i, char in enumerate(text):
                if char == '(':
                    balance += 1
                elif char == ')':
                    balance -= 1
                if balance == 0:
                    last_balanced = i

            if last_balanced > 0 and last_balanced < len(text) - 1:
                text = text[:last_balanced+1] + " ...(후략)"

        return text

    content = fix_parentheses(content)

    # 11. 콜론(:) 다음에 콤마(,)가 오는 경우 처리
    content = re.sub(r':\s*,\s*', '', content)

    # 12. 문장 끝 처리
    # 조사나 어미 없이 끝나는 경우
    if re.search(r'[가-힣]\s*$', content):
        last_word = content.strip().split()[-1] if content.strip().split() else ''
        # 완전한 문장 종결어미가 아닌 경우
        if not re.search(r'(다|요|죠|음|임|함|까|네)$', last_word):
            # 마지막 완전한 문장 찾기
            last_sentence_end = max(content.rfind('. '), content.rfind('.\n'),
                                   content.rfind('? '), content.rfind('! '))
            if last_sentence_end > 0:
                content = content[:last_sentence_end+1] + " ...(후략)"
            else:
                content = content.strip() + "...(후략)"

    # 13. 쉼표로 끝나는 경우
    if content.strip().endswith(','):
        content = content.strip()[:-1] + "...(후략)"

    # 14. 열린 괄호로 끝나는 경우
    if content.strip().endswith('('):
        content = content.strip()[:-1].strip() + "...(후략)"

    # 15. 빈 문장이나 너무 짧은 문장 처리
    content = content.strip()
    if not content or len(content) < 5 or content in ['.', ',', ')', '(']:
        return None

    return content

def generate_answer_and_evaluate(query, filtered_docs, llm):
    # 프롬프트 구성
    final_ref = []
    response_prompt = f"다음은 '{query}'에 대한 관련 정보입니다:\n\n"

    # 중복이 제거된 문서들을 프롬프트에 추가
    doc_count = 0  # 실제 문서 번호를 추적
    for i, doc in enumerate(filtered_docs):
        doc_type = doc.metadata.get('문서유형', '')
        content = doc.page_content

        # ===== 새로운 미완성 문장 처리 함수 사용 =====
        cleaned_content = clean_incomplete_sentences(content)

        # 처리 결과가 None이면 이 문서는 건너뛰기
        if cleaned_content is None:
            continue

        content = cleaned_content
        doc_count += 1  # 유효한 문서만 카운트
        # =========================================

        # 문서 유형에 따른 참조 형식 생성
        if doc_type == '법령':
            article_num = doc.metadata.get('조문번호', '')
            article_title = doc.metadata.get('조문제목', '')
            ho_num = doc.metadata.get('호번호', '')

            if ho_num:
                reference = f"[법령 {doc_count}] 저작권법 제{article_num} {article_title} 제{ho_num}호: {content}"
            else:
                reference = f"[법령 {doc_count}] 저작권법 제{article_num} {article_title}: {content}"

        elif doc_type == '시행령':
            ord_num = doc.metadata.get('시행령_조문번호', '')
            ord_title = doc.metadata.get('시행령_조문제목', '')
            ord_ho_num = doc.metadata.get('호번호', '')

            if ord_ho_num:
                reference = f"[법령 {doc_count}] 저작권법 시행령 제{ord_num} {ord_title} 제{ord_ho_num}호: {content}"
            else:
                reference = f"[법령 {doc_count}] 저작권법 시행령 제{ord_num} {ord_title}: {content}"

        elif doc_type == '부칙':
            sub_num = doc.metadata.get('부칙_조문번호', '')
            sub_title = doc.metadata.get('부칙_조문제목', '')
            hang_num = doc.metadata.get('항번호', '')

            if hang_num:
                reference = f"[법령 {doc_count}] 저작권법 부칙 제{sub_num} {sub_title} 제{hang_num}호: {content}"
            else:
                reference = f"[법령 {doc_count}] 저작권법 부칙 제{sub_num} {sub_title}: {content}"

        elif doc_type == '판례':
            case_num = doc.metadata.get('사건번호', doc.metadata.get('판례번호', ''))
            case_date = doc.metadata.get('선고일자', doc.metadata.get('판결일자', ''))
            court = doc.metadata.get('법원명', '')

            reference = f"[판례 {doc_count}] {court} {case_num} ({case_date}): {content}"

        elif doc_type == '해석례':
            exp_name = doc.metadata.get('안건명', '')
            exp_date = doc.metadata.get('회신일자', '')

            reference = f"[해석례 {doc_count}] {exp_name} ({exp_date}): {content}"

        else:
            reference = f"[참조 {doc_count}] {content}"

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
