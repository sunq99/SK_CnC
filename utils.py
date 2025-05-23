import re

def linkify_articles(text):
    def replacer(match):
        full = match.group(0)

        # 조문 번호 추출 (ex: 제101조의 3 → 101, 3)
        article_match = re.search(r'제(\d+)(?:조(?:의\s?(\d+))?)?', full)
        if not article_match:
            return full

        base_num = article_match.group(1)  # ex: '101'
        sub_num = article_match.group(2)   # ex: '3' if '의 3'이 있는 경우

        # 조문 표시: 제101조 or 제101조의 3
        if sub_num:
            article_name = f"제{base_num}조의{sub_num}"
        else:
            article_name = f"제{base_num}조"

        # 최종 URL 생성 (항은 URL에 반영하지 않음)
        url = f"https://www.law.go.kr/법령/저작권법/{article_name}"

        return f"[{full}]({url})"

    # 대상 문장: 저작권법 제xx조(의 x)? (제n항)?
    pattern = r"(저작권법 제\d+조(?:의\s?\d+)?(?: 제\d+항)?)"
    return re.sub(pattern, replacer, text)
