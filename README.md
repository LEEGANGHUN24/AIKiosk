## 📁 프로젝트 구조 (Project Structure)

```bash
.
├── kiosk_test.py          # ✅ 키오스크 메인 실행 파일 (UI / 주문 / 결제 흐름 총괄)
├── voice_ai_kiosk.py      # ✅ 음성 AI 모듈 (녹음 → STT → LLM → TTS 파이프라인)
├── orders_db.py           # ✅ SQLite DB 처리 (초기화 / 주문 저장 / 관리자 인증)
├── admin_login.py         # ✅ 관리자 로그인 다이얼로그 (UI 로드 + DB 검증)
├── admin_window.py        # ✅ 관리자 창 (주문 내역 조회 / 메뉴 일시품절 관리)
│
├── DATA/                  # ✅ 데이터 저장소
│   ├── orders.db          # SQLite DB (관리자 계정 + 주문 내역)
│   ├── data.csv           # 메뉴 원본 데이터 (메뉴명/가격/카테고리 등)
│   ├── drink_price.csv    # 옵션/추가금 기준표 (예: 샷추가, 사이즈업 등)
│   └── menu_images.csv    # 메뉴명 ↔ 이미지 파일 매핑
│
├── UI/                    # ✅ Qt Designer .ui 화면 파일 (환경에 따라 ui/로 존재할 수도 있음)
│   ├── first_page.ui      # 매장/포장 선택 등 첫 화면
│   ├── main_page.ui       # 메뉴 선택 메인 화면
│   ├── mega_detail_page.ui# 메뉴 상세/옵션 선택 화면
│   ├── order_payment_page.ui # 주문 확인/결제 진입 화면
│   ├── admin_login.ui     # 관리자 로그인 UI
│   └── admin_main.ui      # 관리자 메인 UI
│
├── img/                   # ✅ 이미지 리소스
│   ├── ad/                # 광고 이미지 폴더 (*.jpg, *.png)
│   ├── drinks/            # 음료 이미지 폴더
│   ├── food/              # 푸드 이미지 폴더
│   ├── option_img/        # 옵션 아이콘/이미지 폴더
│   ├── pay/               # 결제수단 이미지 폴더
│   ├── qt/                # Qt UI용 오버레이/아이콘 (예: 품절 표시 등)
│   └── mega_logo.jpg      # 로고 이미지
│
├── wav/                   # ✅ 음성 안내 wav 파일 (TTS 캐시 포함)
│   ├── ask_place.wav              # 매장/포장 선택 안내
│   ├── retry_place.wav            # 매장/포장 재요청 안내
│   ├── ask_menu.wav               # 메뉴 주문 안내
│   ├── retry_menu.wav             # 메뉴 재요청 안내
│   ├── ask_pay.wav                # 결제 여부 안내
│   ├── retry_pay.wav              # 결제 여부 재요청 안내
│   ├── ask_pay_method.wav         # 결제수단 선택 안내
│   ├── retry_pay_method.wav       # 결제수단 재요청 안내
│   ├── voice_menu_ok.wav          # 주문 성공 안내
│   ├── voice_menu_partial.wav     # 부분 성공(일부만 인식) 안내
│   └── voice_menu_not_found.wav   # 메뉴 인식 실패 안내
│
├── MeloTTS/               # ✅ Melo TTS 엔진 (프로젝트에 포함된 폴더)
├── model/                 # (선택) 모델/캐시 폴더 (환경에 따라 사용)
└── __pycache__/           # 파이썬 캐시 파일

## 링크 / 첨부

- GitHub: (링크) : https://github.com/myshell-ai/MeloTTS.git
- https://huggingface.co/andreass123/EEVE-Korean-2.8B-v1.0-Q8_0-GGUF
- https://github.com/guaba98/mega_kiosk (개발툴,ui,이미지 참고)
