import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

def append_feedback_to_sheet(feedback_type, content, chat_title):
    # ✅ 구글 API 접근 범위 설정
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']

    # ✅ credentials.json은 동일 디렉토리에 있어야 함
    creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
    client = gspread.authorize(creds)

    # ✅ 실제 시트 ID로 시트 열기
    sheet_id = "1KSPtZZagGFh-voSnLKImkpNYoJeNc8CBGhj_MrXt4QM"
    sheet = client.open_by_key(sheet_id).sheet1  # 첫 번째 시트 사용

    # ✅ 현재 시간과 함께 한 행 삽입
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sheet.append_row([now, chat_title, feedback_type, content])
