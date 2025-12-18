(프로젝트 루트)
├─ kiosk_test.py          # ✅ 키오스크 메인 실행(UI/주문/결제/흐름 총괄)
├─ voice_ai_kiosk.py      # ✅ 음성 AI 모듈(STT/LLM/TTS)
├─ orders_db.py           # ✅ SQLite DB 연결/초기화/주문저장/관리자검증
├─ admin_login.py         # ✅ 관리자 로그인 다이얼로그(UI 로드 + DB 검증)
├─ admin_window.py        # ✅ 관리자 창(주문내역 조회, 품절 관리)
│
├─ DATA/                  # ✅ 데이터 저장소
│   ├─ orders.db          # SQLite DB (주문/관리자 계정)
│   ├─ data.csv           # 메뉴 원본(이름/가격/분류 등)
│   ├─ drink_price.csv    # 옵션/추가금(샷추가 등) 기준표
│   └─ menu_images.csv    # 메뉴명-이미지 매핑
│
├─ UI/ (또는 ui/)          # ✅ Qt Designer .ui 화면 파일
│   ├─ first_page.ui      # 매장/포장 선택 등 첫 화면
│   ├─ main_page.ui       # 메뉴 선택 메인 화면
│   ├─ mega_detail_page.ui# 상세/옵션 선택 화면
│   ├─ order_payment_page.ui # 주문확인/결제 진입 화면
│   ├─ admin_login.ui
│   └─ admin_main.ui
│
├─ img/                   # ✅ 이미지 리소스
│   ├─ ad/*.jpg,png
│   ├─ drinks/*.jpg,png
│   ├─ food/*.jpg,png
│   ├─ option_img/*.jpg,png
│   ├─ pay/*.jpg,png
│   ├─ qt/*.jpg,png
│   └─ mega_logo.jpg
│
├─ wav/                   # ✅ 음성 안내 wav 리소스(캐시 포함)
│   ├─ ask_place.wav / retry_place.wav
│   ├─ ask_menu.wav / retry_menu.wav
│   ├─ ask_pay.wav / retry_pay.wav
│   ├─ ask_pay_method.wav / retry_pay_method.wav
│   └─ voice_menu_ok.wav / voice_menu_partial.wav / voice_menu_not_found.wav 등
│
├─ MeloTTS/               # ✅ Melo TTS 엔진(프로젝트에 포함된 폴더)
├─ model/                 # (선택) 모델/캐시 폴더(환경에 따라 사용)
├─ __pycache__/           # 파이썬 캐시


## 링크 / 첨부

- GitHub: (링크) : https://github.com/myshell-ai/MeloTTS.git
- https://huggingface.co/andreass123/EEVE-Korean-2.8B-v1.0-Q8_0-GGUF
- https://github.com/guaba98/mega_kiosk (개발툴,ui,이미지 참고)
