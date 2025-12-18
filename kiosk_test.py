# kiosk.py  (PySide6)

import os
import sys
import csv
from typing import List, Dict, Optional, Set, Tuple
import winsound
import threading
import time
import re

from PySide6.QtCore import Qt, QTimer, QFile, QSize, QEvent, Signal
from PySide6.QtGui import QPixmap, QAction, QCursor, QPainter
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QToolButton, QMessageBox, QDialog, QMenu, QStackedWidget,
    QTabWidget, QFrame, QTextBrowser,
    QTableWidget, QTableWidgetItem
)
from PySide6.QtUiTools import QUiLoader

from orders_db import init_db, save_order
from admin_login import AdminLoginDialog
from admin_window import AdminWindow
from voice_ai_kiosk import VoiceAIKiosk, WAV_DIR,BASE_DIR


def resource_path(rel: str) -> str:
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, rel)


class KioskMain(QMainWindow):
    requestGoMain = Signal(str)          # "for_here" / "to_go"
    requestStartVoiceMenu = Signal()
    # ✅ 추가: 주문확인에서 결제 여부 결과를 UI 스레드로 전달
    requestPayDecision = Signal(bool)    # True=결제 진행, False=메뉴로 돌아감
    requestApplyLLMResult = Signal(dict)
     # ✅ 추가: 결제수단(STT) 결과를 UI 스레드로 전달
    requestPayMethodDecision = Signal(str)   # "카드" / "앱카드" / "네이버페이" / "카카오페이" / "KB Pay"
    def __init__(self):
        super().__init__()

        self.voice_flow_busy = False
        # EEVE 추론 백그라운드
        self.llm_busy = False
        self.llm_last_result = None

        # ✅ 추가: 결제 확인 음성 플로우 중복 방지
        self.pay_voice_busy = False


        # ✅ "자동으로 order_check_page로 넘어온 경우"에만 결제 질문 실행하기 위한 플래그
        self.auto_enter_order_check = False
        # 중복 방지
        self.pay_method_voice_busy = False

        # 시그널 → 슬롯 연결 (UI 스레드에서 실행됨)
        self.requestGoMain.connect(self._go_main)
        self.requestStartVoiceMenu.connect(self._handle_voice_menu_and_confirm)

        # ✅ 추가: 결제 여부 결정 시 UI 이동은 여기서!
        self.requestPayDecision.connect(self._on_pay_decision_from_voice)
        self.requestApplyLLMResult.connect(self.apply_llm_result_to_order)
        self.requestPayMethodDecision.connect(self._on_pay_method_from_voice)


        self.setFixedSize(768, 864)

        self.stack = QStackedWidget(self)
        self.setCentralWidget(self.stack)

        self.loader = QUiLoader()

        # === (Whisper + EEVE + Melo TTS) 로드 ===
        self.voice_ai = VoiceAIKiosk()   # 여기서 모델들이 한 번 로드됨
        self.order_mode = None           # "for_here" / "to_go" 저장용

        # === UI 로드 ===
        self.page_opening = self._load_ui("ui/first_page.ui")
        self.page_main: QWidget = self._load_ui("ui/main_page.ui")
        self.page_detail: QWidget = self._load_ui("ui/mega_detail_page.ui")
        self.page_order: QWidget = self._load_ui("ui/order_payment_page.ui")

        if not self.page_opening or not self.page_main:
            raise RuntimeError("first_page.ui 또는 main_page.ui 로드 실패")

        self.stack.addWidget(self.page_opening)  # 0
        self.stack.addWidget(self.page_main)     # 1
        if self.page_detail:
            self.stack.addWidget(self.page_detail)  # 2
        if self.page_order:
            self.stack.addWidget(self.page_order)   # 3

        self.stack.setCurrentIndex(0)

        # ---------- Opening ----------
        self.logo_label: Optional[QLabel] = self.page_opening.findChild(QLabel, "logo_label")
        self.ad_label: Optional[QLabel] = self.page_opening.findChild(QLabel, "ad_label")
        self.btn_eat_here: Optional[QPushButton] = self.page_opening.findChild(QPushButton, "eat_here_btn")
        self.btn_to_go: Optional[QPushButton] = self.page_opening.findChild(QPushButton, "to_go_btn")
        self.btn_settings: Optional[QToolButton] = self.page_opening.findChild(QToolButton, "settings_btn")
        self.btn_voice_ai: Optional[QToolButton] = self.page_opening.findChild(QToolButton, "voice_ai_btn")  # AI

        # ---------- Main ----------
        self.main_mode_badge: Optional[QLabel] = self.page_main.findChild(QLabel, "mode_badge")
        self.btn_back: Optional[QToolButton] = self.page_main.findChild(QToolButton, "back_btn")
        self.cart_total_label: Optional[QLabel] = self.page_main.findChild(QLabel, "cart_total")
        self.order_check_btn: Optional[QPushButton] = self.page_main.findChild(QPushButton, "to_order_check_btn")
        self.btn_voice_ai_menu: Optional[QToolButton] = self.page_main.findChild(QToolButton, "voice_ai_menu_btn") #menu AI
        self.btn_voice_ai_oc: Optional[QToolButton] = self.page_order.findChild(QToolButton, "voice_ai_oc_btn") #pay AI
        self.voice_ai_pc_btn: Optional[QToolButton] = self.page_order.findChild(QToolButton, "voice_ai_pc_btn")


        if self.btn_settings:
        
            menu = QMenu(self)
            act_admin = QAction("관리자 모드", self)
            menu.addAction(act_admin)
            self.btn_settings.setMenu(menu)
            self.btn_settings.setPopupMode(QToolButton.InstantPopup)
            act_admin.triggered.connect(self._open_manager)

        self.ad_images = self._collect_ad_images()
        self.ad_index = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._next_ad)
        self.timer.start(3000)

        if self.btn_eat_here:
            self.btn_eat_here.clicked.connect(lambda: self._go_main("for_here"))
        if self.btn_to_go:
            self.btn_to_go.clicked.connect(lambda: self._go_main("to_go"))
        if self.btn_voice_ai:  # AI 버튼
            self.btn_voice_ai.clicked.connect(self._handle_voice_place_async)
        # 메인 페이지에서 "메뉴 음성 주문 다시 시작" 버튼
        if self.btn_voice_ai_menu:
            # -> ask_menu.wav 재생 + 녹음 + LLM 추론 (장바구니까지)
            self.btn_voice_ai_menu.clicked.connect(self._handle_voice_menu_and_confirm)
        # 주문확인/결제 페이지의 AI 버튼
        if self.btn_voice_ai_oc:
            self.btn_voice_ai_oc.clicked.connect(self._on_voice_ai_oc_clicked)
        if self.voice_ai_pc_btn:
             self.voice_ai_pc_btn.clicked.connect(self._start_voice_pay_method)

        if self.btn_back:
            self.btn_back.clicked.connect(self._go_opening)

        QTimer.singleShot(0, self._show_logo)
        QTimer.singleShot(0, lambda: self._next_ad(initial=True))

        # 주문 모드 / 결제수단
        self.order_mode: Optional[str] = None          # "for_here" / "to_go"
        self.selected_pay_method: Optional[str] = None  # "카드", "네이버페이" 등

        # ====== 품절 상태 저장 (프로그램 실행 동안만 유지) ======
        self.sold_out_menus: Set[str] = set()

        # 관리자 창 (한 번 만든 뒤 재사용)
        self.admin_window: Optional[AdminWindow] = None

        # 메뉴 이미지 매핑
        self.menu_img_map: Dict[str, str] = {}
        self._load_menu_images()

        # 옵션 정보
        self.drink_option_by_eng: Dict[str, Dict] = {}
        self._load_drink_options()
        self.category_option_eng_map: Dict[str, List[str]] = self._build_category_option_map()

        self.option_image_map: Dict[str, str] = {
            "아몬드밀크변경": "almond_milk_2.jpg",
            "시나몬(O)": "cinnamon.jpg",
            "시나몬(X)": "cinnamon.jpg",
            "헤이즐넛시럽추가": "hazelnet_syrup.jpg",
            "오트밀크변경": "oat-milkjpg.jpg",
            "샷 추가": "one_shot.jpg",
            "2샷 추가": "two_shot.jpg",
            "바닐라시럽추가": "vanilla-syrup-img.jpg",
            "카라멜시럽추가": "simple-syrup-2.jpg",
            "휘핑(O)": "Whipped-Cream.jpg",
            "휘핑(X)": "Whipped-Cream.jpg",
            "제로사이다": "zero_cider_2.jpg",
            "스태비아 추가": "stevia_2.png",
        }

        # 메인 페이지(메뉴)
        self._init_menu_logic()

        # 상세 페이지 / 장바구니
        self.current_detail_data: Optional[Dict] = None
        self.detail_base_price: int = 0
        self.option_frame_index: Dict[QFrame, int] = {}
        self.option_frame_base_styles: Dict[QFrame, str] = {}
        self.option_click_counts: Dict[int, int] = {}
        self.cart_items: List[Dict] = []

        if self.page_detail:
            self._init_detail_page()

        # 결제 이미지 라벨들
        self.pay_img_labels: List[QLabel] = []
        self.pay_img_method_map: Dict[QLabel, str] = {}

        # 주문/결제 페이지
        if self.page_order:
            self._init_order_page()

        # ✅ DB 초기화
        init_db()

        self._recalc_cart_summary()

    # ------------------------------------------------------------------
    # AI wav / 음성 흐름
    # ------------------------------------------------------------------
    def _handle_voice_place_async(self):
        """
        AI 대화형 버튼을 눌렀을 때 실제 음성 흐름은
        별도 스레드에서 돌리고, UI는 계속 반응 가능하게 만든다.
        """
        if self.voice_flow_busy:
            print("⚠ 이미 음성 플로우가 동작 중입니다.")
            return

        self.voice_flow_busy = True

        def worker():
            try:
                # 여기서 실제 매장/포장 + 메뉴/STT/LLM 흐름을 돌린다.
                self._handle_voice_place_flow()
            finally:
                # 끝나면 플래그 해제
                self.voice_flow_busy = False

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def _play_wav_blocking(self, path: str):
        """단순 WAV 재생 (winsound, 메인스레드에서 동기 실행)"""
        if not path or not os.path.exists(path):
            print("⚠ 음성 파일을 찾을 수 없습니다:", path)
            return
        try:
            winsound.PlaySound(path, winsound.SND_FILENAME)
        except Exception as e:
            print("⚠ 음성 재생 중 오류:", e)

    def _parse_place_from_text(self, text: str) -> Optional[str]:
        """
        STT 결과에서 '매장 / 포장' 의도를 구분해서
        - 매장 → "for_here"
        - 포장 → "to_go"
        리턴. 못 알아들으면 None.
        """
        if not text:
            return None

        t = text.replace(" ", "")  # 공백 제거

        # 매장 관련 표현
        if ("매장" in t) or ("먹고" in t) or ("여기서" in t):
            return "for_here"

        # 포장 관련 표현
        if ("포장" in t) or ("싸가" in t) or ("싸갈" in t) or ("테이크아웃" in t):
            return "to_go"

        return None

    def _speak_or_make_tts(self, text: str, filename: str):
        """
        filename.wav 파일이 이미 있으면 바로 재생하고,
        없으면 TTS로 생성 후 재생한다.

        단, assistant_response처럼 내용이 계속 바뀌는 동적 멘트에 사용하는
        "voice_menu_ok.wav" 는 매번 새로 생성하도록 한다.
        """
        wav_path = os.path.join(WAV_DIR, filename)

        # 🔥 동적 멘트용 파일은 캐시를 사용하지 않고 항상 새로 생성
        dynamic_files = {"voice_menu_ok.wav","voice_menu_not_found.wav","voice_menu_partial.wav"}
        if filename in dynamic_files and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
                print(f"🗑 기존 동적 TTS 파일 삭제: {filename}")
            except OSError as e:
                print(f"⚠ 동적 TTS 파일 삭제 실패({filename}):", e)

        # 파일이 있으면 바로 재생 (정적 멘트용)
        if os.path.exists(wav_path):
            print(f"▶ 캐시된 음성 재생: {filename}")
            self._play_wav_blocking(wav_path)
            return wav_path

        # 파일이 없으면 TTS 생성
        print(f"🎤 TTS 생성: {filename}")
        wav_path = self.voice_ai.make_tts(text, out_filename=filename)

        # 생성한 TTS 재생
        self._play_wav_blocking(wav_path)
        return wav_path

    def _handle_voice_place_flow(self):
        """
        백그라운드 스레드에서 도는 실제 음성 플로우 로직.
        '매장/포장' 인식될 때까지 반복.
        ask_place.wav, retry_place.wav 파일이 있으면 즉시 재생하고,
        없으면 TTS로 생성 후 재생함.
        """

        # 1) 첫 질문 (캐시 재생)
        question = "주문을 도와드리겠습니다. 우선 매장에서 드시겠습니까, 포장하시겠습니까?"
        self._speak_or_make_tts(question, "ask_place.wav")

        mode = None

        # 2) 인식될 때까지 무한 반복
        while True:
            # 녹음 + STT
            answer_text = self.voice_ai.record_and_stt(
                sec=5.0,
                in_filename="answer_place.wav"
            )
            print("📝 place STT:", answer_text)

            mode = self._parse_place_from_text(answer_text)

            # 성공 → 종료
            if mode is not None:
                break

            # 실패 → retry 음성 (캐시 재생)
            retry_text = "죄송합니다. 다시 한 번 말씀해 주세요."
            self._speak_or_make_tts(retry_text, "retry_place.wav")

        # 3) 여기까지 오면 mode는 "for_here" / "to_go"
        # 👉 UI 변경은 메인 스레드에게 맡긴다 (Signal 사용)
        print(f"✅ place 결정: {mode}")
        self.requestGoMain.emit(mode)

        # 살짝 기다렸다가 (UI 전환될 시간)
        time.sleep(0.3)

        # 4) 메인화면으로 전환된 뒤, 메뉴 음성 흐름 시작도 메인 스레드에 요청
        self.requestStartVoiceMenu.emit()

    def _is_yes(self, text: str) -> bool:
        if not text:
            return False
        t = text.replace(" ", "")
        yes_keywords = ["네", "예", "맞아", "맞습니다", "응", "그래", "좋아요", "좋습니다", "맞아요"]
        return any(k in t for k in yes_keywords)

    def _is_no(self, text: str) -> bool:
        if not text:
            return False
        t = text.replace(" ", "")
        no_keywords = ["아니", "아니요", "싫", "취소", "다시", "변경"]
        return any(k in t for k in no_keywords)

    def _handle_voice_menu_and_confirm(self):
        """
        메뉴/옵션을 음성으로 듣고
        STT + EEVE 추론을 백그라운드에서 돌린 뒤
        LLM 결과를 cart_items / TTS에 반영한다.

        ✅ 1차 방어:
        - STT 문장 안에 실제 매장 메뉴명이 하나도 없으면
        LLM 호출 자체를 하지 않고 바로 안내 후 종료
        """

        # 1) 질문 TTS (캐시 사용)
        question = "메뉴 주문을 도와드리겠습니다. 원하시는 메뉴와 옵션을 말씀해 주세요."
        self._speak_or_make_tts(question, "ask_menu.wav")

        # 2) STT
        stt_text = self.voice_ai.record_and_stt(
            sec=5.0,
            in_filename="voice_menu.wav"
        )
        print("📝 menu STT:", stt_text)

        # ✅ 마지막 STT 저장 (2차 방어에서 사용)
        self.last_menu_stt_text = stt_text or ""

        # 2-1) STT 자체가 비어 있으면 재질문
        if not stt_text or not stt_text.strip():
            retry = "죄송합니다. 메뉴와 옵션을 다시 한 번 말씀해 주세요."
            self._speak_or_make_tts(retry, "retry_menu.wav")
            return  # 버튼으로 다시 시작

        # ✅✅ 1차 방어: STT에 실제 매장 메뉴명이 없으면 LLM 호출 차단
        if not self._stt_mentions_any_real_menu(stt_text):
            print(f"[GUARD-1] STT에 매장 메뉴명이 없음 -> LLM 호출 차단 (STT='{stt_text}')")
            self._speak_or_make_tts(
                "죄송합니다. 저희 매장에 없는 메뉴이거나 인식이 어렵습니다. 메뉴 이름을 다시 말씀해 주세요.",
                "voice_menu_not_found.wav"
            )
            return  # ❗ 여기서 끝, LLM 실행 안 함

        # 3) 여기까지 통과한 경우에만 EEVE LLM 실행
        self._run_eeve_in_background(stt_text)
    def _on_voice_ai_oc_clicked(self):
        """
        order_payment_page(주문/결제 페이지)에서 voice_ai_oc_btn 눌렀을 때:
        결제하시겠습니까? → 네/아니요 → YES면 결제수단 선택, NO면 메인으로
        """
        print("[PAY] voice_ai_oc_btn 클릭됨 → 결제 여부 음성 확인 시작")

        # (선택) 주문/결제 페이지가 아닌데 눌리는 상황 방어
        if self.stack.currentWidget() != self.page_order:
            self.stack.setCurrentWidget(self.page_order)

        # 주문확인 화면으로 보고 있는 게 안전하면(표 보이게) 그쪽으로 강제
        if self.order_stack and self.order_check_page:
            self.order_stack.setCurrentWidget(self.order_check_page)

        # 표/합계 최신화(선택이지만 추천)
        self._populate_order_table()
        self._recalc_cart_summary()

        # 자동진입 플래그와 무관하게 그냥 실행
        self._start_voice_pay_confirm()


    def _run_eeve_in_background(self, stt_text: str):
        """
        EEVE LLM 추론을 백그라운드 스레드에서 수행.
        - UI(키오스크 화면)는 계속 반응 가능
        - 결과는 apply_llm_result_to_order 로 넘김
        """
        if self.llm_busy:
            print("⚠ LLM 이미 동작 중입니다. 잠시 후 다시 시도하세요.")
            return

        self.llm_busy = True
        self.llm_last_result = None

        def worker():
            try:
                print("\n[LLM] ===== EEVE 추론 시작 =====")
                print("[LLM] STT 텍스트:", stt_text)

                result = self.voice_ai.parse_menu_json(stt_text)
                self.llm_last_result = result

                print("[LLM] ===== EEVE 추론 완료 =====")
                print("[LLM] 결과 JSON:", result)

                items = result.get("items") or []
                assistant_resp = result.get("assistant_response")

                if items and isinstance(items, list):
                    print(f"[LLM] items 개수: {len(items)}")
                    first = items[0]
                    print("  - menu_name      =", first.get("menu_name"))
                    print("  - menu_quantity  =", first.get("menu_quantity"))
                    print("  - menu_option    =", first.get("menu_option"))
                else:
                    print("  - items 없음 또는 형식 오류")

                print("  - assistant_resp =", assistant_resp)

                # 🔥 여기서 바로 주문 로직 호출 (이전처럼)
                print("[LLM] 이제 LLM 결과를 주문에 반영합니다 (apply_llm_result_to_order 호출)")
                try:
                    self.requestApplyLLMResult.emit(result)
                    # self.apply_llm_result_to_order(result)
                except Exception as e:
                    import traceback
                    print("❌ apply_llm_result_to_order 실행 중 예외:")
                    traceback.print_exc()

            except Exception as e:
                print("❌ LLM 추론 중 예외 발생:", e)

            finally:
                self.llm_busy = False

        t = threading.Thread(target=worker, daemon=True)
        t.start()


    def _safe_apply_llm(self, llm_result: dict):
            """
            apply_llm_result_to_order를 예외 안전하게 호출하는 래퍼.
            여기서 예외가 나면 traceback까지 콘솔에 다 찍어줌.
            """
            print("[DEBUG] _safe_apply_llm 호출됨")
            try:
                self.apply_llm_result_to_order(llm_result)
            except Exception as e:
                import traceback
                print("❌ apply_llm_result_to_order 실행 중 예외 발생:")
                traceback.print_exc()

    def _start_voice_pay_method(self):
        """payment_choose_page에서 결제수단을 음성으로 선택"""
        if self.pay_method_voice_busy:
            return
        self.pay_method_voice_busy = True
        print("[PAY] 결제수단 음성 선택 시작")

        t = threading.Thread(target=self._voice_pay_method_flow, daemon=True)
        t.start()


    def _voice_pay_method_flow(self):
        try:
            # ✅ 질문 TTS
            question = "결제수단을 선택해주세요."
            self._speak_or_make_tts(question, "ask_pay_method.wav")

            # 최대 3회 시도
            for attempt in range(3):
                answer = self.voice_ai.record_and_stt(sec=3.0, in_filename="answer_pay_method.wav")
                print("📝 pay_method STT:", answer)

                method = self._infer_pay_method(answer)
                if method:
                    print(f"✅ 결제수단 인식: {method}")
                    self.requestPayMethodDecision.emit(method)
                    return

                retry = "죄송합니다. 다시한번 말씀해주세요."
                self._speak_or_make_tts(retry, "retry_pay_method.wav")

            print("⚠ 결제수단 인식 실패: 결제수단 음성 플로우 종료")

        finally:
            self.pay_method_voice_busy = False


    def _infer_pay_method(self, text: str) -> Optional[str]:
        """STT 결과에서 결제수단 표준값으로 매핑"""
        if not text:
            return None
        t = text.replace(" ", "").lower()

        # 1) 카드(신용/체크/카드)
        if ("카드" in t) or ("신용" in t) or ("체크" in t):
            # "앱카드"와 구분: 앱카드가 더 구체적이니 먼저 처리하고 싶으면 아래 앱카드 조건을 위로 올려도 됨
            # 여기서는 앱카드 키워드를 별도로 더 강하게 잡아준다
            if "앱카드" in t or ("앱" in t and "카드" in t):
                return "앱카드"
            return "카드"

        # 2) 앱카드
        if "앱카드" in t or ("앱" in t and "카드" in t):
            return "앱카드"

        # 3) 네이버페이
        if "네이버페이" in t or ("네이버" in t and "페이" in t) or "npay" in t or ("네이버" in t):
            return "네이버페이"

        # 4) 카카오페이
        if "카카오페이" in t or ("카카오" in t and "페이" in t) or "kakaopay" in t or ("카카오" in t):
            return "카카오페이"

        # 5) KB Pay (케이비 페이)
        if "kbpay" in t or "kb페이" in t or ("케이비" in t and "페이" in t) or "kb" in t  or ("케이비" in t):
            return "KB Pay"

        return None


    def _on_pay_method_from_voice(self, method: str):
        """결제수단 음성 선택 결과 -> charge_page로 이동"""
        if not method:
            return

        # 안전: payment_choose_page로 먼저 맞춰두기(선택사항)
        if self.order_stack and self.payment_choose_page:
            self.order_stack.setCurrentWidget(self.payment_choose_page)

        self._go_charge_page(method)


    # ------------------------------------------------------------------
    # LLM → CSV 매칭용 헬퍼
    # ------------------------------------------------------------------
    def _normalize_menu_name(self, name: str) -> str:
        """
        메뉴 이름 비교용 정규화:
        - 공백 제거
        - (HOT)/(ICE) 제거
        - 프러페 → 프라페 (오타 보정)
        """
        if not name:
            return ""
        s = name.strip()
        s = s.replace(" ", "")
        s = s.replace("(HOT)", "").replace("(ICE)", "")
        s = s.replace("프러페", "프라페")
        return s

    def _find_menu_row_for_llm_name(self, llm_name: str) -> Optional[Dict]:
        """
        LLM이 준 menu_name 을 CSV 메뉴에서 찾아서 row 반환.
        못 찾으면 None.
        """
        if not llm_name:
            return None

        name = llm_name.strip()

        # 1) 완전 일치 우선
        row = self.menu_by_name.get(name)
        if row:
            return row

        # 2) 정규화해서 비교
        target_key = self._normalize_menu_name(name)
        for row in self.menu_all_rows:
            row_name = (row.get("이름") or "").strip()
            if not row_name:
                continue
            if self._normalize_menu_name(row_name) == target_key:
                return row

        return None

    def _add_cart_item_from_llm(self, menu_row: Dict, quantity: int,
                            option_row: Optional[Dict], option_name: Optional[str]):
        """
        LLM 결과를 실제 cart_items 에 반영하는 공통 함수.
        - menu_row : data.csv 한 줄
        - quantity : 잔 수
        - option_row : drink_price.csv 한 줄 (없으면 None)
        - option_name : 옵션 한글 이름 (표시용)
        """
        menu_name = (menu_row.get("이름") or "").strip()
        try:
            menu_id = int(menu_row.get("카테고리번호") or 0)
        except ValueError:
            menu_id = 0

        try:
            base_price = int(menu_row.get("가격") or 0)
        except ValueError:
            base_price = 0

        opt_price = 0
        if option_row is not None:
            try:
                opt_price = int(option_row.get("noraml_drink") or 0)
            except ValueError:
                opt_price = 0

        qty = quantity if isinstance(quantity, int) and quantity > 0 else 1

        print(f"[CART] _add_cart_item_from_llm 호출: menu_name={menu_name}, qty={qty}, base_price={base_price}, opt_price={opt_price}, option_name={option_name}")

        for _ in range(qty):
            option_list: List[Dict] = []
            if option_name and opt_price > 0:
                option_list.append({
                    "kor_name": option_name,
                    "count": 1,
                    "unit_price": opt_price,
                    "total_price": opt_price,
                })

            total_price = base_price + opt_price
            cart_item = {
                "menu_name": menu_name,
                "menu_id": menu_id,
                "base_price": base_price,
                "options": option_list,
                "total_price": total_price,
            }
            self.cart_items.append(cart_item)

            print("[CART]   → cart_item 추가:", cart_item)

        print(f"[CART] 현재 cart_items 개수: {len(self.cart_items)}")
        self._recalc_cart_summary()

    def apply_llm_result_to_order(self, llm_result: dict):
        """
        EEVE LLM 결과(JSON)를 받아서
        - data.csv 메뉴 매칭
        - drink_price.csv 옵션 매칭
        - cart_items / cart_total / 주문내역 테이블로 반영

        ✅ 동작 규칙
        1) 전부 성공: assistant_response 그대로 읽기
        2) 일부 실패(여러 메뉴 중 일부만 성공):
        - "(없는메뉴들)은 없는 메뉴입니다."
        - "(성공한 메뉴들) 주문 도와드리겠습니다."   <-- '인식된 메뉴는' 문구 제거 버전
        3) 전부 실패: "없는 메뉴/옵션" 안내 후 종료(버튼으로 재시도)

        1) 전부 성공: assistant_response 읽고 -> 2초 뒤 order_check_page 자동 이동
        (한 잔 성공도 여기 포함)
        2) 일부 실패(성공+실패 섞임): 안내 TTS만 하고 -> 자동 이동 금지(버튼으로 처리)
        3) 전부 실패: 안내 TTS만 하고 -> 자동 이동 금지(버튼으로 재시도)
         """
        print("\n[LLM-ORDER] ==== LLM 결과 적용 시작 ====")
        print("[LLM-ORDER] raw llm_result:", llm_result)

         # ✅✅ 2차 방어: STT에 '실제 메뉴명'이 없으면 결과 반영 금지

        stt_text = getattr(self, "last_menu_stt_text", "") or ""

        if not self._stt_mentions_any_real_menu(stt_text):
            print(f"[GUARD-2] STT에 매장 메뉴명이 없음 -> 적용 차단 (STT='{stt_text}')")
            self._speak_or_make_tts(
                "죄송합니다. 저희 매장 메뉴가 아닌 것 같습니다. 메뉴 이름을 다시 말씀해 주세요.",
                "voice_menu_not_found.wav"
            )
            return
        
        items = llm_result.get("items") or []
        assistant_resp = (llm_result.get("assistant_response") or "").strip()

        # items 자체가 없으면: 안내 후 바로 재질문 (기존 유지)
        if not items:
            msg = "죄송합니다. 주문을 인식하지 못했습니다. 다시 한 번 말씀해 주세요."
            print("[LLM-ORDER] items 비어 있음 ->", msg)
            self._speak_or_make_tts(msg, "voice_menu_retry2.wav")

            print("[LLM-ORDER] items 없음 -> 메뉴를 다시 음성으로 물어봅니다.")
            self.llm_busy = False
            self._handle_voice_menu_and_confirm()
            return

        any_added = False        # 실제로 장바구니에 들어간 항목이 있는지
        menu_fail = False        # 메뉴 매칭 실패 여부
        option_fail = False      # 옵션 매칭 실패 여부
        unknown_menu_names: List[str] = []
        unknown_option_names: List[str] = []

        # ✅ 추가: 품절 메뉴 목록
        sold_out_names: List[str] = []

        # ✅ 이번 호출에서 성공한 "메뉴 이름"만 모아서 부분 실패 TTS에 사용
        success_menu_names: List[str] = []
        failed_item_count = 0              # 실패한 item 수(메뉴 or 옵션 실패면 +1)

        for idx, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                print(f"[LLM-ORDER] item #{idx} 형식이 dict 아님, 건너뜀:", item)
                continue

            menu_name = (item.get("menu_name") or "").strip()
            menu_qty = item.get("menu_quantity") or 1
            menu_opt = (item.get("menu_option") or "").strip()

            print(f"\n[LLM-ORDER] --- item #{idx} ---")
            print(f"[LLM-ORDER] menu_name={menu_name}, menu_quantity={menu_qty}, menu_option={menu_opt}")

            # 1) LLM이 준 수량 기본 변환
            try:
                menu_qty = int(menu_qty)
            except (TypeError, ValueError):
                menu_qty = 1
            if menu_qty <= 0:
                menu_qty = 1

            # 2) items가 1개뿐이고, 수량이 1 이하이면 assistant_response에서 잔수 보정
            if len(items) == 1 and menu_qty <= 1:
                inferred = self._infer_quantity_from_text(assistant_resp, default=menu_qty)
                if inferred != menu_qty:
                    print(f"[LLM-ORDER] assistant_response 기반으로 잔수 보정: {menu_qty} → {inferred}")
                    menu_qty = inferred

            # 3) 메뉴 찾기 (data.csv 기준)
            #menu_row = self._find_menu_row_for_llm_name(menu_name)
            stt_text = getattr(self, "last_menu_stt_text", "") or ""
            temp_hint = self._infer_temp_from_stt(stt_text)
            if temp_hint:
                print(f"[LLM-ORDER] STT 온도 힌트 감지: {temp_hint} (STT='{stt_text}')")

            menu_row = self._find_menu_row_for_llm_name_with_temp(menu_name, temp_hint)
            if not menu_row:
                menu_fail = True
                failed_item_count += 1     
                unknown_menu_names.append(menu_name)
                print(f"[LLM-ORDER] 메뉴 매칭 실패: '{menu_name}' -> data.csv에서 찾지 못함")
                continue


            # ✅✅ (핵심) 품절 체크: "매칭 성공"이어도 품절이면 장바구니 추가 금지
            real_menu_name = (menu_row.get("이름") or "").strip()
            if real_menu_name and real_menu_name in self.sold_out_menus:
                sold_out_names.append(real_menu_name)
                failed_item_count += 1
                print(f"[LLM-ORDER] 품절 메뉴 감지 -> 장바구니 추가 금지: {real_menu_name}")
                continue


            print("[LLM-ORDER] 메뉴 매칭 성공:")
            print("    - 이름 =", menu_row.get("이름"))
            print("    - 분류 =", menu_row.get("분류"))
            print("    - 카테고리번호 =", menu_row.get("카테고리번호"))
            print("    - HOT/ICE =", menu_row.get("HOT/ICE"))
            print("    - 가격 =", menu_row.get("가격"))

            # 4) 옵션 찾기 (drink_price.csv 기준)
            option_row = None
            disp_opt_name = menu_opt

            if menu_opt:
                option_row = self._find_option_row_by_llm_name(menu_opt)
                if option_row:
                    disp_kor_name = (option_row.get("kor_name") or "").strip()
                    disp_opt_name = disp_kor_name or menu_opt
                    print("[LLM-ORDER] 옵션 매칭 성공:")
                    print("    - kor_name =", option_row.get("kor_name"))
                    print("    - noraml_drink =", option_row.get("noraml_drink"))
                else:
                    option_fail = True
                    failed_item_count += 1
                    unknown_option_names.append(menu_opt)
                    print(f"[LLM-ORDER] 옵션 매칭 실패: '{menu_opt}' -> drink_price.csv에서 찾지 못함")
                    # 옵션이 없는 경우 이 item 전체를 장바구니에 넣지 않음
                    continue
            else:
                print("[LLM-ORDER] 메뉴 옵션 없음(빈 문자열)")

            # 5) 가격 계산 (data.csv 가격 + drink_price noraml_drink)
            try:
                base_price = int(menu_row.get("가격") or 0)
            except (TypeError, ValueError):
                base_price = 0

            opt_price = 0
            if option_row is not None:
                try:
                    opt_price = int(option_row.get("noraml_drink") or 0)
                except (TypeError, ValueError):
                    opt_price = 0

            one_total = base_price + opt_price
            total_for_all = one_total * menu_qty

            print(f"[LLM-ORDER] 가격 계산:")
            print(f"    - base_price(메뉴) = {base_price}")
            print(f"    - opt_price(옵션)  = {opt_price}")
            print(f"    - 1잔 total        = {one_total}")
            print(f"    - 수량             = {menu_qty}")
            print(f"    -> 이 item 전체 금액 = {total_for_all}원")

            # 6) 장바구니 반영
            self._add_cart_item_from_llm(menu_row, menu_qty, option_row, disp_opt_name)
            any_added = True
            print("[LLM-ORDER] 장바구니 추가 완료.")

            # ✅ 성공한 메뉴명 누적 (부분실패 안내문에 사용)
            say_name = ((menu_row.get("이름") or menu_name) or "").strip()
            if say_name:
                # 같은 메뉴를 여러 잔(혹은 여러 row) 추가해도 문장은 깔끔하게 1번만 나오게
                success_menu_names.append(say_name)

        # 7) 결과에 따른 TTS 처리

        # (A) 전부 실패: 기존 로직 유지 + 재시도는 버튼으로
        if not any_added:
            msg_parts = []
            if sold_out_names:
            # "유자차(HOT)은 일시품절입니다." 같이 말하고 싶으면 join 처리
                msg_parts.append(", ".join(sold_out_names) + "은 일시품절입니다.")
            if menu_fail:
                msg_parts.append("저희 매장엔 없는 메뉴입니다.")
            if option_fail:
                msg_parts.append("저희 매장엔 없는 메뉴 옵션입니다.")
            if not msg_parts:
                msg_parts.append("죄송합니다. 주문을 인식하지 못했습니다. 다시 한 번 말씀해 주세요.")

            final_msg = " ".join(msg_parts)
            print("[LLM-ORDER] 장바구니에 추가된 항목이 없음 ->", final_msg)
            self._speak_or_make_tts(final_msg, "voice_menu_not_found.wav")

            print("[LLM-ORDER] 음성 주문 실패 - 사용자가 다시 음성 주문 버튼을 눌러 재시도해야 합니다.")
            return

        # (B) 일부 실패: "(없는메뉴) 없는 메뉴" + "(성공한메뉴) 주문 도와드리겠습니다"
        all_success = (failed_item_count == 0)

        if not all_success:

            if sold_out_names:
                msg_parts.append(", ".join(sold_out_names) + "은 일시품절입니다.")
            if menu_fail:
                print("[LLM-ORDER] 일부 메뉴 매칭 실패 목록:", unknown_menu_names)
            if option_fail:
                print("[LLM-ORDER] 일부 옵션 매칭 실패 목록:", unknown_option_names)
        # if menu_fail or option_fail:
        #     if menu_fail:
        #         print("[LLM-ORDER] 일부 메뉴 매칭 실패 목록:", unknown_menu_names)
        #     if option_fail:
        #         print("[LLM-ORDER] 일부 옵션 매칭 실패 목록:", unknown_option_names)

            msg_parts = []

            # ✅ 없는 메뉴 안내
            if menu_fail and unknown_menu_names:
                msg_parts.append(", ".join(unknown_menu_names) + "은 없는 메뉴입니다.")

            # ✅ (옵션까지 말하고 싶으면 아래 주석 해제)
            # if option_fail and unknown_option_names:
            #     msg_parts.append(", ".join(unknown_option_names) + "은 없는 메뉴 옵션입니다.")

            # ✅ 성공한 메뉴만 주문 안내 ('인식된 메뉴는' 제거)
            if success_menu_names:
                # 중복 제거(순서 유지)
                seen = set()
                uniq_success = []
                for n in success_menu_names:
                    if n not in seen:
                        seen.add(n)
                        uniq_success.append(n)

                msg_parts.append("와 ".join(uniq_success) + " 주문 도와드리겠습니다.")
            else:
                msg_parts.append("주문 가능한 메뉴만 주문 도와드리겠습니다.")

            final_tts = " ".join(msg_parts)
            print("[LLM-ORDER] 최종 응답 TTS(부분실패):", final_tts)
            self._speak_or_make_tts(final_tts, "voice_menu_partial.wav")
            return
            
        # (C) 전부 성공: 기존처럼 assistant_resp 사용
        if not assistant_resp:
            assistant_resp = "주문해 주셔서 감사합니다. 그대로 주문 도와드리겠습니다."
        print("[LLM-ORDER] 최종 응답 TTS:", assistant_resp)
        self._speak_or_make_tts(assistant_resp, "voice_menu_ok.wav")
        # ✅ 자동 진입 표시 (이 진입에서만 결제 질문 자동 실행)
        self.auto_enter_order_check = True

        # ✅ 2초 뒤 주문내역 확인창으로 자동 이동
        QTimer.singleShot(2000, self._open_order_check_page)


    def _infer_temp_from_stt(self, stt_text: str) -> str:
        """
        STT 문장에서 HOT/ICE 힌트를 뽑음.
        return: "HOT" | "ICE" | ""
        """
        t = (stt_text or "").strip()
        if not t:
            return ""

        hot_keywords = ["뜨거운", "따뜻한", "뜨겁게", "따뜻하게", "핫"]
        ice_keywords = ["아이스", "차가운", "시원한", "차갑게", "시원하게", "아이스로"]

        if any(k in t for k in hot_keywords):
            return "HOT"
        if any(k in t for k in ice_keywords):
            return "ICE"
        return ""
    def _find_menu_row_for_llm_name_with_temp(self, menu_name: str, temp_hint: str):
        """
        너 매장 규칙:
        - ICE는 대부분 기본 이름(예: 유자차)
        - HOT은 (HOT) 붙은 이름(예: 유자차(HOT))
        - 예외: '아이스초코' 같은 고유명사는 그대로
        """
        raw = (menu_name or "").strip()
        if not raw:
            return None

        # 0) 혹시 LLM이 '아이스 유자차'처럼 넣으면 공백 제거 정도만 정리
        name = raw.replace(" ", "")

        # 1) LLM이 이미 (HOT)를 붙여준 경우: 무조건 HOT로 찾기
        if "(HOT)" in name:
            return self._find_menu_row_for_llm_name(name)

        # 2) STT가 HOT 힌트면: 무조건 (HOT)로 찾아야 함
        if temp_hint == "HOT":
            hot_name = f"{name}(HOT)"
            row = self._find_menu_row_for_llm_name(hot_name)
            return row  # 없으면 None (핫 메뉴 자체가 없는 것)

        # 3) STT가 ICE 힌트면:
        #    - 아이스초코는 고유명사 그대로
        #    - 나머지는 기본 이름(ICE) 그대로
        if temp_hint == "ICE":
            return self._find_menu_row_for_llm_name(name)

        # 4) 힌트가 없으면: 기본(ICE) 먼저 찾고, 없으면 HOT도 시도 (선택)
        row = self._find_menu_row_for_llm_name(name)
        if row:
            return row
        return self._find_menu_row_for_llm_name(f"{name}(HOT)")
    

    # ------------------------------------------------------------------
    # 공통 UI 로더
    # ------------------------------------------------------------------
    def _load_ui(self, path_rel: str) -> Optional[QWidget]:
        path = resource_path(path_rel)
        if not os.path.exists(path):
            QMessageBox.warning(self, "오류", f"UI 파일이 없습니다: {path_rel}")
            return None
        f = QFile(path)
        if not f.open(QFile.ReadOnly):
            QMessageBox.warning(self, "오류", f"UI 파일을 열 수 없습니다: {path_rel}")
            return None
        w = self.loader.load(f, self)
        f.close()
        if not isinstance(w, QWidget):
            QMessageBox.warning(self, "오류", f"{path_rel} 루트가 QWidget이 아닙니다.")
            return None
        return w
    def _infer_quantity_from_text(self, text: str, default: int = 1) -> int:
        """
        assistant_response 같은 문장에서 '2잔', '두 잔' 등을 찾아
        잔 수를 추론한다. 못 찾으면 default 반환.
        """
        if not text:
            return default

        # 1) 숫자 + '잔' 패턴 (예: 2잔, 3 잔)
        m = re.search(r'(\d+)\s*잔', text)
        if m:
            try:
                n = int(m.group(1))
                if n > 0:
                    print(f"[LLM-ORDER] 텍스트에서 숫자 잔수 추출: {n}잔")
                    return n
            except ValueError:
                pass

        # 2) 한글 숫자 + '잔' 패턴 (예: 두 잔, 세 잔)
        mapping = {
            "한": 1,
            "두": 2,
            "세": 3,
            "네": 4,
            "다섯": 5,
            "여섯": 6,
            "일곱": 7,
            "여덟": 8,
            "아홉": 9,
            "열": 10,
        }
        for word, n in mapping.items():
            if word + " 잔" in text or word + "잔" in text:
                print(f"[LLM-ORDER] 텍스트에서 한글 잔수 추출: {n}잔 ({word})")
                return n

        # 3) 못 찾으면 기본값
        return default


    # ------------------------------------------------------------------
    # Opening
    # ------------------------------------------------------------------
    def _show_logo(self):
        path = resource_path("img/mega_logo.jpg")
        if not (self.logo_label and os.path.exists(path)):
            return
        pix = QPixmap(path)
        if pix.isNull():
            return
        size = self.logo_label.size() if self.logo_label.width() > 0 else QSize(600, 140)
        self.logo_label.setPixmap(pix.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.logo_label.setScaledContents(False)

    def _collect_ad_images(self) -> List[str]:
        base = resource_path("img/ad")
        out: List[str] = []
        for i in range(1, 5):
            for n in (f"ad_img_{i}.jpg", f"ad_img{i}.jpg"):
                p = os.path.join(base, n)
                if os.path.exists(p):
                    out.append(p)
                    break
        return out

    def _next_ad(self, initial: bool = False):
        if self.stack.currentWidget() is not self.page_opening:
            return
        if not (self.ad_images and self.ad_label):
            return
        pix = QPixmap(self.ad_images[self.ad_index % len(self.ad_images)])
        if not pix.isNull():
            size = self.ad_label.size() if self.ad_label.width() > 0 else QSize(700, 420)
            self.ad_label.setPixmap(pix.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.ad_label.setScaledContents(False)
        if not initial:
            self.ad_index = (self.ad_index + 1) % len(self.ad_images)

    def _go_main(self, mode: str):
        self.order_mode = mode
        if self.timer.isActive():
            self.timer.stop()
        if self.main_mode_badge:
            self.main_mode_badge.setText("매장" if mode == "for_here" else "포장")
        self.stack.setCurrentWidget(self.page_main)

    def _reset_cursor(self):
        QApplication.restoreOverrideCursor()

    def _go_opening(self):
        self._reset_cursor()
        if not self.timer.isActive():
            self.timer.start(3000)
        self.stack.setCurrentWidget(self.page_opening)
        QTimer.singleShot(0, self._show_logo)
        QTimer.singleShot(0, lambda: self._next_ad(initial=True))

    def _open_manager(self):
        """톱니바퀴 → 관리자 로그인 → 성공 시 AdminWindow 열기"""
        dlg = AdminLoginDialog(self)
        if dlg.exec() == QDialog.Accepted:
            # 이미 열려있으면 그 창 재사용
            if self.admin_window is None:
                self.admin_window = AdminWindow(self)

            # 키오스크 중심 기준으로 위치 잡기
            my_geo = self.geometry()
            aw_geo = self.admin_window.frameGeometry()
            aw_geo.moveCenter(my_geo.center())
            self.admin_window.move(aw_geo.topLeft())

            self.admin_window.show()
            self.admin_window.raise_()
            self.admin_window.activateWindow()

    # ------------------------------------------------------------------
    # 메뉴 이미지 매핑 CSV
    # ------------------------------------------------------------------
    def _load_menu_images(self):
        csv_path = resource_path("DATA/menu_images.csv")
        if not os.path.exists(csv_path):
            QMessageBox.warning(self, "이미지 로드", "DATA/menu_images.csv 를 찾을 수 없습니다.")
            return

        encodings = ["utf-8-sig", "cp949"]
        for enc in encodings:
            try:
                with open(csv_path, "r", encoding=enc, newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        name = (row.get("menu_name") or "").strip()
                        path_rel = (row.get("img_path") or "").strip()
                        if name and path_rel:
                            self.menu_img_map[name] = path_rel
                break
            except UnicodeDecodeError:
                continue

    # ------------------------------------------------------------------
    # 옵션 CSV
    # ------------------------------------------------------------------
    def _load_drink_options(self):
        csv_path = resource_path("DATA/drink_price.csv")
        if not os.path.exists(csv_path):
            QMessageBox.information(self, "옵션 로드", "DATA/drink_price.csv 를 찾을 수 없습니다.")
            return

        encodings = ["utf-8-sig", "cp949"]
        delimiters = [",", "\t"]

        rows: List[Dict] = []
        success = False

        for enc in encodings:
            for delim in delimiters:
                try:
                    with open(csv_path, "r", encoding=enc, newline="") as f:
                        reader = csv.DictReader(f, delimiter=delim)
                        tmp = list(reader)
                    if tmp and "eng_name" in tmp[0] and "kor_name" in tmp[0] and "noraml_drink" in tmp[0]:
                        rows = tmp
                        success = True
                        break
                except UnicodeDecodeError:
                    continue
            if success:
                break

        if not success:
            QMessageBox.warning(self, "옵션 로드 실패", "drink_price.csv 형식/인코딩을 인식할 수 없습니다.")
            return

        # ✅ 세 개의 인덱스 준비
        self.drink_option_by_eng: Dict[str, Dict] = {}
        self.drink_option_by_kor: Dict[str, Dict] = {}
        self.drink_option_by_kor_norm: Dict[str, Dict] = {}

        for row in rows:
            eng = (row.get("eng_name") or "").strip()
            kor = (row.get("kor_name") or "").strip()
            if not eng or not kor:
                continue
            try:
                price = int(row.get("noraml_drink") or 0)
            except ValueError:
                price = 0

            row["eng_name"] = eng
            row["kor_name"] = kor
            row["noraml_drink"] = price

            # 영문 기준 (기존 상세 옵션 UI에서 사용)
            self.drink_option_by_eng[eng] = row

            # 한글 이름 그대로
            self.drink_option_by_kor[kor] = row

            # 한글 이름 공백 제거 (샷 추가 vs 샷추가 등)
            kor_norm = kor.replace(" ", "")
            if kor_norm and kor_norm not in self.drink_option_by_kor_norm:
                self.drink_option_by_kor_norm[kor_norm] = row

    def _find_option_row_by_llm_name(self, llm_option: str) -> Optional[Dict]:
        """
        LLM이 준 menu_option(예: '샷추가')을 drink_price.csv 옵션과 매칭.
        - kor_name 그대로 비교
        - 공백 제거해서 비교 (샷 추가 vs 샷추가)
        """
        if not llm_option:
            return None

        name = llm_option.strip()
        row = self.drink_option_by_kor.get(name)
        if row:
            return row

        norm = name.replace(" ", "")
        row = self.drink_option_by_kor_norm.get(norm)
        if row:
            return row

        return None

    # ------------------------------------------------------------------
    def _build_category_option_map(self) -> Dict[str, List[str]]:
        coffee_common = [
            "coffee_add_one",
            "coffee_add_two",
            "decaffein_shot",
            "vanilia_syrup_add",
            "light_vanilia_changed",
            "light_vanilia_add",
            "syrup_add_h",
            "syrup_add_c",
            "stevia_changed",
            "stevia_add",
            "choose_milk_a",
            "choose_milk_o",
            "whip_n",
            "whip_y",
            "cinnamon_y",
            "cinnamon_n",
        ]

        decaf_extra = ["coffee_weak", "coffee_weak_2"]

        smoothie_base = [
            "whip_n",
            "whip_y",
            "vanilia_syrup_add",
            "light_vanilia_changed",
            "light_vanilia_add",
            "syrup_add_h",
            "syrup_add_c",
            "cinnamon_y",
            "cinnamon_n",
        ]

        beverage_opts = [
            "coffee_add_one",
            "coffee_add_two",
            "vanilia_syrup_add",
            "light_vanilia_changed",
            "light_vanilia_add",
            "syrup_add_h",
            "syrup_add_c",
            "stevia_changed",
            "stevia_add",
            "choose_milk_a",
            "choose_milk_o",
            "whip_n",
            "whip_y",
            "cinnamon_y",
            "cinnamon_n",
            "honey_add",
        ]

        tea_opts = [
            "vanilia_syrup_add",
            "light_vanilia_changed",
            "light_vanilia_add",
            "syrup_add_h",
            "syrup_add_c",
            "stevia_changed",
            "stevia_add",
            "cinnamon_y",
            "cinnamon_n",
        ]

        return {
            "디카페인": coffee_common + decaf_extra,
            "커피(HOT)": coffee_common,
            "커피(ICE)": coffee_common,
            "커피(콜드브루)": coffee_common + decaf_extra,
            "스무디&프라페": smoothie_base,
            "에이드": ["zero_cider_changed"],
            "음료": beverage_opts,
            "티": tea_opts,
        }

    # ------------------------------------------------------------------
    # 메인 페이지 (카테고리/메뉴)
    # ------------------------------------------------------------------
    def _init_menu_logic(self):
        self.category_tab: QTabWidget = self.page_main.findChild(QTabWidget, "category_tab")
        self.cat_left_btn: QToolButton = self.page_main.findChild(QToolButton, "cat_left_btn")
        self.cat_right_btn: QToolButton = self.page_main.findChild(QToolButton, "cat_right_btn")

        self.menu_car: QStackedWidget = self.page_main.findChild(QStackedWidget, "menu_car")
        self.menu_left_btn: QToolButton = self.page_main.findChild(QToolButton, "menu_left_btn")
        self.menu_right_btn: QToolButton = self.page_main.findChild(QToolButton, "menu_right_btn")

        self.menu_slots: Dict[int, Dict[str, QLabel]] = {}
        for i in range(1, 19):
            img = self.page_main.findChild(QLabel, f"menu_img_{i}")
            name = self.page_main.findChild(QLabel, f"menu_name_label_{i}")
            price = self.page_main.findChild(QLabel, f"menu_price_label_{i}")
            frame = self.page_main.findChild(QFrame, f"menu_frame_{i}")
            if img and name and price and frame:
                self.menu_slots[i] = {
                    "frame": frame,
                    "img": img,
                    "name": name,
                    "price": price,
                }

        self.menu_frames: List[QFrame] = []
        self.menu_frame_sizes: Dict[QFrame, QSize] = {}
        self.menu_frame_index: Dict[QFrame, int] = {}

        for idx, slot in self.menu_slots.items():
            frame: QFrame = slot["frame"]
            self.menu_frames.append(frame)

            s = frame.size()
            if s.width() == 0 or s.height() == 0:
                s = frame.minimumSize()
            self.menu_frame_sizes[frame] = s
            self.menu_frame_index[frame] = idx

            frame.setCursor(Qt.PointingHandCursor)
            frame.installEventFilter(self)

        self.menu_by_tab: Dict[str, List[Dict]] = {}
        self._load_menu_csv()

        if self.category_tab and self.cat_left_btn and self.cat_right_btn:
            self.cat_left_btn.clicked.connect(self._prev_category)
            self.cat_right_btn.clicked.connect(self._next_category)
            self.category_tab.currentChanged.connect(self._on_category_changed)

        if self.menu_car and self.menu_left_btn and self.menu_right_btn:
            self.menu_left_btn.clicked.connect(self._prev_menu_page)
            self.menu_right_btn.clicked.connect(self._next_menu_page)

        if self.order_check_btn:
            self.order_check_btn.clicked.connect(self._open_order_check_page)

        self._render_current_tab()

    def _load_menu_csv(self):
        csv_path = resource_path("DATA/data.csv")
        if not os.path.exists(csv_path):
            QMessageBox.warning(self, "메뉴 로드", "DATA/data.csv 를 찾을 수 없습니다.")
            return

        # 탭 → CSV의 분류값 매핑
        self.tab_category_map: Dict[str, List[str]] = {
            "디카페인": ["디카페인"],
            "커피(ICE)": ["커피(ICE)"],
            "커피(HOT)": ["커피(HOT)"],
            "스무디": ["스무디&프라페"],
            "에이드": ["에이드"],
            "티": ["티"],
            "음료": ["음료"],
            "디저트": ["디저트"],
            "콜드브루": ["커피(콜드브루)"],
        }

        # 탭별 메뉴 초기화
        self.menu_by_tab = {tab: [] for tab in self.tab_category_map.keys()}

        encodings = ["utf-8-sig", "cp949"]
        delimiters = [",", "\t"]

        rows: List[Dict] = []
        success = False

        # CSV 열기 (인코딩/구분자 자동 판별)
        for enc in encodings:
            for delim in delimiters:
                try:
                    with open(csv_path, "r", encoding=enc, newline="") as f:
                        reader = csv.DictReader(f, delimiter=delim)
                        tmp_rows = list(reader)

                    if tmp_rows and \
                       "분류" in tmp_rows[0] and \
                       "카테고리번호" in tmp_rows[0] and \
                       "이름" in tmp_rows[0] and \
                       "가격" in tmp_rows[0]:
                        rows = tmp_rows
                        success = True
                        break
                except UnicodeDecodeError:
                    continue
            if success:
                break

        if not success:
            QMessageBox.warning(self, "메뉴 로드 실패", "CSV 인코딩/형식을 인식할 수 없습니다.")
            return

        # 🔥 전체 메뉴 검색용 인덱스
        self.menu_all_rows: List[Dict] = []
        self.menu_by_name: Dict[str, Dict] = {}       # 정확히 같은 이름
        self.menu_by_name_norm: Dict[str, Dict] = {}  # 공백 제거한 이름

        for row in rows:
            cat = (row.get("분류") or "").strip()

            try:
                row["카테고리번호"] = int(row.get("카테고리번호") or 0)
            except ValueError:
                row["카테고리번호"] = 0

            try:
                row["가격"] = int(row.get("가격") or 0)
            except ValueError:
                row["가격"] = 0

            # 탭별로 분류
            for tab_name, src_list in self.tab_category_map.items():
                if cat in src_list:
                    self.menu_by_tab[tab_name].append(row)
                    break

            # 전체 메뉴 인덱스에 추가
            name = (row.get("이름") or "").strip()
            if name:
                self.menu_all_rows.append(row)
                self.menu_by_name[name] = row

                norm = name.replace(" ", "")
                if norm and norm not in self.menu_by_name_norm:
                    self.menu_by_name_norm[norm] = row

        # ✅ 탭별 정렬
        for tab, items in self.menu_by_tab.items():
            items.sort(key=lambda r: r.get("카테고리번호", 0))


    def _normalize_text(self, s: str) -> str:
        """공백/개행 제거 후 비교용 문자열로 정규화"""
        return (s or "").strip().replace(" ", "").replace("\n", "")


    def _stt_mentions_any_real_menu(self, stt_text: str) -> bool:
        """
        STT 문장 안에 data.csv의 메뉴 '이름'이 하나라도 포함되면 True.
        포함된 메뉴명이 하나도 없으면 LLM이 추측한 메뉴일 가능성이 크므로 False.
        """
        s = self._normalize_text(stt_text)
        if not s:
            return False

        # 메뉴 로드가 안된 경우엔 안전하게 통과시키지 말고 False(=차단) 추천
        rows = getattr(self, "menu_all_rows", None) or []
        if not rows:
            print("[GUARD] menu_all_rows 비어있음(메뉴 로드 전?) -> 안전 차단")
            return False

        for row in rows:
            name = self._normalize_text(row.get("이름") or "")
            if not name:
                continue

            # 1) 그대로 포함
            if name in s:
                return True

            # 2) (HOT)/(ICE) 같은 괄호형이 STT에는 안 잡힐 수 있으니 괄호 제거 형태도 비교
            name2 = name.replace("(HOT)", "").replace("(ICE)", "")
            if name2 and name2 in s:
                return True

        return False

    def _prev_category(self):
        if not self.category_tab:
            return
        idx = (self.category_tab.currentIndex() - 1) % self.category_tab.count()
        self.category_tab.setCurrentIndex(idx)

    def _next_category(self):
        if not self.category_tab:
            return
        idx = (self.category_tab.currentIndex() + 1) % self.category_tab.count()
        self.category_tab.setCurrentIndex(idx)

    def _on_category_changed(self, _index: int):
        if self.menu_car:
            self.menu_car.setCurrentIndex(0)
        self._render_current_tab()

    def _page_count_for_current_tab(self) -> int:
        if not self.category_tab:
            return 1
        tab_text = self.category_tab.tabText(self.category_tab.currentIndex())
        items = self.menu_by_tab.get(tab_text, [])
        total = len(items)
        return max(1, min(2, (total + 8) // 9))

    def _prev_menu_page(self):
        if not self.menu_car:
            return
        cnt = self._page_count_for_current_tab()
        if cnt <= 1:
            return
        idx = (self.menu_car.currentIndex() - 1) % cnt
        self.menu_car.setCurrentIndex(idx)

    def _next_menu_page(self):
        if not self.menu_car:
            return
        cnt = self._page_count_for_current_tab()
        if cnt <= 1:
            return
        idx = (self.menu_car.currentIndex() + 1) % cnt
        self.menu_car.setCurrentIndex(idx)

    # ====== 품절 오버레이용 헬퍼 ======
    def _apply_sold_out_overlay(self, base_pix: QPixmap) -> QPixmap:
        """img/qt/sold_out.png 를 base_pix 위에 덧씌운 새 QPixmap 반환"""
        overlay_path = resource_path("img/qt/sold_out.png")
        if not os.path.exists(overlay_path):
            return base_pix
        ov = QPixmap(overlay_path)
        if ov.isNull():
            return base_pix

        result = QPixmap(base_pix.size())
        result.fill(Qt.transparent)

        painter = QPainter(result)
        painter.drawPixmap(0, 0, base_pix)
        ov_scaled = ov.scaled(
            base_pix.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        painter.drawPixmap(0, 0, ov_scaled)
        painter.end()
        return result

    def _render_current_tab(self):
        if not self.category_tab:
            return

        tab_text = self.category_tab.tabText(self.category_tab.currentIndex())
        items = self.menu_by_tab.get(tab_text, [])

        for i in range(1, 19):
            slot = self.menu_slots.get(i)
            if not slot:
                continue

            if i - 1 < len(items):
                data = items[i - 1]
                self._fill_slot(i, data)
            else:
                self._clear_slot(i)

            slot["frame"].setVisible(True)

    def _fill_slot(self, idx: int, data: Dict):
        slot = self.menu_slots.get(idx)
        if not slot:
            return

        name = data.get("이름", "")
        price = data.get("가격", 0)

        slot["name"].setText(name or "메뉴명")
        if isinstance(price, int):
            slot["price"].setText(f"{price:,}원")
        else:
            slot["price"].setText(str(price))

        label: QLabel = slot["img"]
        img_rel = self.menu_img_map.get(name, "")

        if not img_rel:
            label.setPixmap(QPixmap())
            label.setText("")
            return

        img_path = img_rel
        if not os.path.isabs(img_rel):
            img_path = resource_path(img_rel)

        if os.path.exists(img_path):
            pix = QPixmap(img_path)
            if not pix.isNull():
                # 품절 상태면 sold_out 덧씌우기
                if name in self.sold_out_menus:
                    pix = self._apply_sold_out_overlay(pix)

                size = label.size() if label.width() > 0 else QSize(160, 110)
                label.setPixmap(pix.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                label.setScaledContents(False)
            else:
                label.setPixmap(QPixmap())
                label.setText("")
        else:
            label.setPixmap(QPixmap())
            label.setText("")

    def _clear_slot(self, idx: int):
        slot = self.menu_slots.get(idx)
        if not slot:
            return
        slot["name"].setText("")
        slot["price"].setText("")
        slot["img"].setPixmap(QPixmap())
        slot["img"].setText("")

    # ------------------------------------------------------------------
    # 상세 페이지
    # ------------------------------------------------------------------
    def _init_detail_page(self):
        self.detail_back_btn: Optional[QToolButton] = self.page_detail.findChild(QToolButton, "back_btn")
        self.detail_close_btn: Optional[QPushButton] = self.page_detail.findChild(QPushButton, "close_btn")
        self.detail_add_cart_btn: Optional[QPushButton] = self.page_detail.findChild(QPushButton, "add_cart_btn")
        self.detail_reset_btn: Optional[QPushButton] = self.page_detail.findChild(QPushButton, "reset_opt_btn")

        self.detail_img_label: Optional[QLabel] = self.page_detail.findChild(QLabel, "menu_img")
        self.detail_name_label: Optional[QLabel] = self.page_detail.findChild(QLabel, "menu_name")
        self.detail_price_label: Optional[QLabel] = self.page_detail.findChild(QLabel, "menu_price")

        desc_widget = self.page_detail.findChild(QTextBrowser, "menu_desc")
        if not desc_widget:
            desc_widget = self.page_detail.findChild(QLabel, "menu_desc")
        self.detail_desc_widget: Optional[QWidget] = desc_widget

        self.detail_selected_opt_label: Optional[QLabel] = self.page_detail.findChild(QLabel, "selected_opt_label")

        self.option_slots: Dict[int, Dict[str, QWidget]] = {}
        self.option_frame_index.clear()
        self.option_frame_base_styles.clear()
        self.option_click_counts.clear()

        for i in range(1, 13):
            frame = self.page_detail.findChild(QFrame, f"opt_frame_{i}")
            img = self.page_detail.findChild(QLabel, f"opt_img_{i}")
            name = self.page_detail.findChild(QLabel, f"opt_name_{i}")
            price = self.page_detail.findChild(QLabel, f"opt_price_{i}")
            if frame and img and name and price:
                self.option_slots[i] = {
                    "frame": frame,
                    "img": img,
                    "name": name,
                    "price": price,
                    "row": None,
                }
                self.option_frame_index[frame] = i
                self.option_frame_base_styles[frame] = frame.styleSheet()
                frame.setCursor(Qt.PointingHandCursor)
                frame.installEventFilter(self)

        if self.detail_back_btn:
            self.detail_back_btn.clicked.connect(self._back_from_detail)
        if self.detail_close_btn:
            self.detail_close_btn.clicked.connect(self._back_from_detail)
        if self.detail_add_cart_btn:
            self.detail_add_cart_btn.clicked.connect(self._detail_add_cart)
        if self.detail_reset_btn:
            self.detail_reset_btn.clicked.connect(self._reset_detail_options)

    def _back_from_detail(self):
        self.stack.setCurrentWidget(self.page_main)

    def _clear_option_slot(self, idx: int):
        slot = self.option_slots.get(idx)
        if not slot:
            return
        slot["name"].setText("")
        slot["price"].setText("")
        slot["img"].setPixmap(QPixmap())
        slot["img"].setText("")
        slot["row"] = None
        frame: QFrame = slot["frame"]  # type: ignore
        base = self.option_frame_base_styles.get(frame, "")
        frame.setStyleSheet(base)
        frame.setVisible(False)
        self.option_click_counts[idx] = 0

    def _fill_option_slot(self, idx: int, row: Dict):
        slot = self.option_slots.get(idx)
        if not slot:
            return

        name = row.get("kor_name", "").strip()
        price = row.get("noraml_drink", 0)
        img_file = row.get("_img_file")
        if not img_file:
            self._clear_option_slot(idx)
            return

        slot["name"].setText(name or "")
        if isinstance(price, int):
            slot["price"].setText(f"+{price:,}원" if price > 0 else "추가금 없음")
        else:
            slot["price"].setText(str(price))

        label: QLabel = slot["img"]  # type: ignore
        label.setPixmap(QPixmap())
        label.setText("")

        img_path = resource_path(f"img/option_img/{img_file}")
        if os.path.exists(img_path):
            pix = QPixmap(img_path)
            if not pix.isNull():
                size = label.size() if label.width() > 0 else QSize(120, 90)
                label.setPixmap(pix.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                label.setScaledContents(False)

        frame: QFrame = slot["frame"]  # type: ignore
        base = self.option_frame_base_styles.get(frame, "")
        frame.setStyleSheet(base)
        frame.setVisible(True)

        slot["row"] = row
        self.option_click_counts[idx] = 0

    def _render_options_for_menu(self, data: Dict):
        for i in range(1, 13):
            self._clear_option_slot(i)

        self.option_click_counts.clear()

        cat = (data.get("분류") or "").strip()
        eng_list = list(self.category_option_eng_map.get(cat, []))

        if cat == "스무디&프라페":
            name = (data.get("이름") or "").strip()
            if "커피" in name or "에스프레소" in name:
                for e in ["coffee_add_one", "coffee_add_two"]:
                    if e not in eng_list:
                        eng_list.append(e)

        option_rows: List[Dict] = []
        for eng in eng_list:
            row = self.drink_option_by_eng.get(eng)
            if not row:
                continue
            kor = (row.get("kor_name") or "").strip()
            img_file = self.option_image_map.get(kor)
            if not img_file:
                continue
            img_path = resource_path(f"img/option_img/{img_file}")
            if not os.path.exists(img_path):
                continue
            new_row = dict(row)
            new_row["_img_file"] = img_file
            option_rows.append(new_row)

        max_slots = len(self.option_slots)
        for idx in range(1, max_slots + 1):
            if idx <= len(option_rows):
                self._fill_option_slot(idx, option_rows[idx - 1])
            else:
                self._clear_option_slot(idx)

        self._update_selected_option_summary()
        self._update_detail_price()

    def _on_option_clicked(self, idx: int):
        slot = self.option_slots.get(idx)
        if not slot or not slot.get("row"):
            return
        self.option_click_counts[idx] = self.option_click_counts.get(idx, 0) + 1
        self._update_selected_option_summary()
        self._update_detail_price()

    def _update_selected_option_summary(self):
        if not self.detail_selected_opt_label:
            return

        parts = []
        for idx, count in self.option_click_counts.items():
            if count <= 0:
                continue
            slot = self.option_slots.get(idx)
            if not slot:
                continue
            row = slot.get("row")
            if not row:
                continue
            kor = (row.get("kor_name") or "").strip()
            if not kor:
                continue
            if count == 1:
                parts.append(kor)
            else:
                parts.append(f"{kor} x{count}")

        if not parts:
            self.detail_selected_opt_label.setText("선택한 옵션: 선택한 옵션 없음")
        else:
            joined = ", ".join(parts)
            self.detail_selected_opt_label.setText(f"선택한 옵션: {joined}")

    def _calculate_detail_total_price(self) -> int:
        total = self.detail_base_price
        for idx, cnt in self.option_click_counts.items():
            if cnt <= 0:
                continue
            slot = self.option_slots.get(idx)
            if not slot:
                continue
            row = slot.get("row")
            if not row:
                continue
            try:
                opt_price = int(row.get("noraml_drink") or 0)
            except ValueError:
                opt_price = 0
            total += opt_price * cnt
        return total

    def _update_detail_price(self):
        total = self._calculate_detail_total_price()
        if self.detail_price_label:
            self.detail_price_label.setText(f"{total:,}원")

    def _reset_detail_options(self):
        for idx in list(self.option_slots.keys()):
            self.option_click_counts[idx] = 0
            slot = self.option_slots[idx]
            frame: QFrame = slot["frame"]  # type: ignore
            base = self.option_frame_base_styles.get(frame, "")
            frame.setStyleSheet(base)

        self._update_selected_option_summary()
        if self.detail_price_label:
            self.detail_price_label.setText(f"{self.detail_base_price:,}원")

    # ====== 장바구니 ======
    def _detail_add_cart(self):
        if not self.current_detail_data:
            return

        menu_name = (self.current_detail_data.get("이름") or "").strip()
        menu_id = self.current_detail_data.get("카테고리번호", 0)

        # ✅ 품절이면 담기 금지
        if menu_name and menu_name in self.sold_out_menus:
            QMessageBox.information(self, "일시품절", "일시품절입니다.")
            return
        

        try:
            menu_id = int(menu_id)
        except (TypeError, ValueError):
            menu_id = 0

        base_price = self.detail_base_price
        total_price = self._calculate_detail_total_price()

        option_list: List[Dict] = []
        for idx, cnt in self.option_click_counts.items():
            if cnt <= 0:
                continue
            slot = self.option_slots.get(idx)
            if not slot:
                continue
            row = slot.get("row")
            if not row:
                continue
            kor = (row.get("kor_name") or "").strip()
            try:
                opt_price = int(row.get("noraml_drink") or 0)
            except ValueError:
                opt_price = 0
            option_list.append({
                "kor_name": kor,
                "count": cnt,
                "unit_price": opt_price,
                "total_price": opt_price * cnt,
            })

        cart_item = {
            "menu_name": menu_name,
            "menu_id": menu_id,
            "base_price": base_price,
            "options": option_list,
            "total_price": total_price,
        }

        self.cart_items.append(cart_item)
        self._recalc_cart_summary()
        self.stack.setCurrentWidget(self.page_main)

    def _recalc_cart_summary(self):
        total_price = sum(item.get("total_price", 0) for item in self.cart_items)
        total_count = len(self.cart_items)

        print(f"[CART] _recalc_cart_summary: 총 개수={total_count}, 총 금액={total_price}")

        if self.cart_total_label:
            self.cart_total_label.setText(f"총 {total_count}개 {total_price:,}원")

        if hasattr(self, "amount_label") and self.amount_label:
            self.amount_label.setText(f"결제금액: {total_price:,}원")

    # ====== 상세 페이지 데이터 채우기 ======
    def _show_detail(self, data: Dict):
        if not self.page_detail:
            return

        self.current_detail_data = data

        name = data.get("이름", "")
        price = data.get("가격", 0)
        try:
            self.detail_base_price = int(price)
        except (TypeError, ValueError):
            self.detail_base_price = 0

        desc = data.get("img_path", "") or ""

        if self.detail_name_label:
            self.detail_name_label.setText(name)
        if self.detail_price_label:
            self.detail_price_label.setText(f"{self.detail_base_price:,}원")
        if self.detail_desc_widget:
            if isinstance(self.detail_desc_widget, QTextBrowser):
                self.detail_desc_widget.setPlainText(desc)
            elif isinstance(self.detail_desc_widget, QLabel):
                self.detail_desc_widget.setText(desc)

        if self.detail_img_label:
            img_rel = self.menu_img_map.get(name, "")
            if img_rel:
                img_path = img_rel
                if not os.path.isabs(img_rel):
                    img_path = resource_path(img_rel)
                if os.path.exists(img_path):
                    pix = QPixmap(img_path)
                    if not pix.isNull():
                        # 품절인 경우 sold_out 오버레이
                        if name in self.sold_out_menus:
                            pix = self._apply_sold_out_overlay(pix)
                        size = self.detail_img_label.size() if self.detail_img_label.width() > 0 else QSize(220, 220)
                        self.detail_img_label.setPixmap(
                            pix.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        )
                        self.detail_img_label.setScaledContents(False)
                    else:
                        self.detail_img_label.setPixmap(QPixmap())
                else:
                    self.detail_img_label.setPixmap(QPixmap())
            else:
                self.detail_img_label.setPixmap(QPixmap())

        self._render_options_for_menu(data)
        self.stack.setCurrentWidget(self.page_detail)

    # ====== 주문/결제 페이지 ======
    def _init_order_page(self):
        self.order_stack: Optional[QStackedWidget] = self.page_order.findChild(QStackedWidget, "order_stack")

        # 주문확인 페이지
        self.order_check_page: Optional[QWidget] = self.page_order.findChild(QWidget, "order_check_page")
        self.order_table: Optional[QTableWidget] = self.page_order.findChild(QTableWidget, "order_table")
        self.oc_menu_btn: Optional[QPushButton] = self.page_order.findChild(QPushButton, "oc_menu_btn")
        self.oc_pay_btn: Optional[QPushButton] = self.page_order.findChild(QPushButton, "oc_pay_btn")

        # 결제수단 페이지
        self.payment_choose_page: Optional[QWidget] = self.page_order.findChild(QWidget, "payment_choose_page")
        self.amount_label: Optional[QLabel] = self.page_order.findChild(QLabel, "amount_label")
        self.pay_back_btn: Optional[QPushButton] = self.page_order.findChild(QPushButton, "pay_back_btn")
        self.pay_next_btn: Optional[QPushButton] = self.page_order.findChild(QPushButton, "pay_next_btn")

        # 결제버튼
        self.pay_card_btn: Optional[QPushButton] = self.page_order.findChild(QPushButton, "pay_card")
        self.pay_appcard_btn: Optional[QPushButton] = self.page_order.findChild(QPushButton, "pay_appcard")
        self.pay_npay_btn: Optional[QPushButton] = self.page_order.findChild(QPushButton, "pay_npay")
        self.pay_kakaopay_btn: Optional[QPushButton] = self.page_order.findChild(QPushButton, "pay_kakaopay")
        self.pay_kbpay_btn: Optional[QPushButton] = self.page_order.findChild(QPushButton, "pay_kbpay")

        # 결제 이미지 라벨
        self.pay_card_img: Optional[QLabel] = self.page_order.findChild(QLabel, "pay_card_img")
        self.pay_appcard_img: Optional[QLabel] = self.page_order.findChild(QLabel, "pay_appcard_img")
        self.pay_npay_img: Optional[QLabel] = self.page_order.findChild(QLabel, "pay_npay_img")
        self.pay_kakaopay_img: Optional[QLabel] = self.page_order.findChild(QLabel, "pay_kakaopay_img")
        self.pay_kbpay_img: Optional[QLabel] = self.page_order.findChild(QLabel, "pay_kbpay_img")

        # 결제 진행 페이지
        self.charge_page: Optional[QWidget] = self.page_order.findChild(QWidget, "charge_page")
        self.charge_done_btn: Optional[QPushButton] = self.page_order.findChild(QPushButton, "charge_done_btn")
        self.charge_msg: Optional[QLabel] = self.page_order.findChild(QLabel, "charge_msg")
        if self.charge_msg:
            self.charge_msg.installEventFilter(self)

        # 버튼 연결
        if self.oc_menu_btn:
            self.oc_menu_btn.clicked.connect(self._back_to_main_from_order)
        if self.oc_pay_btn and self.order_stack and self.payment_choose_page:
            self.oc_pay_btn.clicked.connect(lambda: self.order_stack.setCurrentWidget(self.payment_choose_page))

        if self.pay_back_btn and self.order_stack and self.order_check_page:
            self.pay_back_btn.clicked.connect(lambda: self.order_stack.setCurrentWidget(self.order_check_page))

        if self.pay_next_btn and self.charge_page and self.order_stack:
            self.pay_next_btn.clicked.connect(lambda: self._go_charge_page("직접선택"))

        if self.pay_card_btn:
            self.pay_card_btn.clicked.connect(lambda: self._go_charge_page("카드"))
        if self.pay_appcard_btn:
            self.pay_appcard_btn.clicked.connect(lambda: self._go_charge_page("앱카드"))
        if self.pay_npay_btn:
            self.pay_npay_btn.clicked.connect(lambda: self._go_charge_page("네이버페이"))
        if self.pay_kakaopay_btn:
            self.pay_kakaopay_btn.clicked.connect(lambda: self._go_charge_page("카카오페이"))
        if self.pay_kbpay_btn:
            self.pay_kbpay_btn.clicked.connect(lambda: self._go_charge_page("KB Pay"))

        if self.charge_done_btn:
            self.charge_done_btn.clicked.connect(self._go_opening)

        self._load_pay_images()

    def _load_pay_images(self):
        mapping = [
            (self.pay_card_img, "img/qt/card.png", "카드"),
            (self.pay_appcard_img, "img/qt/payment_phone.png", "앱카드"),
            (self.pay_npay_img, "img/qt/naverpay.png", "네이버페이"),
            (self.pay_kakaopay_img, "img/qt/kakaopay.jpg", "카카오페이"),
            (self.pay_kbpay_img, "img/qt/Kbpay.png", "KB Pay"),
        ]

        for label, rel, method in mapping:
            if not label:
                continue
            path = resource_path(rel)
            if os.path.exists(path):
                pix = QPixmap(path)
                if not pix.isNull():
                    size = label.size() if label.width() > 0 else QSize(120, 80)
                    label.setPixmap(pix.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                    label.setScaledContents(False)
            label.setCursor(Qt.PointingHandCursor)
            label.installEventFilter(self)
            self.pay_img_labels.append(label)
            self.pay_img_method_map[label] = method

    def _set_custom_cursor(self, img_rel: str):
        path = resource_path(img_rel)
        if os.path.exists(path):
            pix = QPixmap(path)
            if not pix.isNull():
                cur = QCursor(pix)
                QApplication.setOverrideCursor(cur)

    def _update_charge_page_visual(self):
        if not self.charge_msg:
            return

        is_card = (self.selected_pay_method == "카드")

        if is_card:
            msg_img = "img/pay/card.png"
            cursor_img = "img/pay/matercard.png"
        else:
            msg_img = "img/pay/qr.jpg"
            cursor_img = "img/pay/bacord.png"

        msg_path = resource_path(msg_img)
        if os.path.exists(msg_path):
            pix = QPixmap(msg_path)
            if not pix.isNull():
                size = self.charge_msg.size()
                # 처음 들어왔을 때 사이즈가 너무 작으면 기본값으로
                if size.width() < 200 or size.height() < 200:
                    size = QSize(500, 400)
                self.charge_msg.setPixmap(
                    pix.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
                self.charge_msg.setScaledContents(False)

        self._set_custom_cursor(cursor_img)

    def _go_charge_page(self, method: str):
        self.selected_pay_method = method
        if self.order_stack and self.charge_page:
            self._update_charge_page_visual()
            self.order_stack.setCurrentWidget(self.charge_page)

    def _back_to_main_from_order(self):
        self._reset_cursor()
        self.stack.setCurrentWidget(self.page_main)

    def _open_order_check_page(self):
        if not self.page_order or not self.order_stack or not self.order_table:
            return

        if self.order_check_page:
            self.order_stack.setCurrentWidget(self.order_check_page)

        self._populate_order_table()
        self._recalc_cart_summary()

        self.stack.setCurrentWidget(self.page_order)
        # ✅ 추가: 주문확인 페이지 진입 직후 자동 음성 질문 시작
        # ✅ 자동 진입일 때만 결제 질문
        if self.auto_enter_order_check:
            self.auto_enter_order_check = False  # 바로 꺼서 중복 방지
            QTimer.singleShot(0, self._start_voice_pay_confirm)


    def _start_voice_pay_confirm(self):
        """(자동 진입에서만 호출됨) 결제하시겠습니까? -> 네/아니요 음성 인식"""
        if self.pay_voice_busy:
            return
        self.pay_voice_busy = True
        print("[PAY] 결제 여부 음성 확인 시작")

        t = threading.Thread(target=self._voice_pay_confirm_flow, daemon=True)
        t.start()

    def _voice_pay_confirm_flow(self):
        try:
            # ✅ 질문 문구 변경
            question = "결제하시겠습니까?"
            self._speak_or_make_tts(question, "ask_pay.wav")

            # 네/아니요 최대 2회 시도
            for attempt in range(2):
                answer = self.voice_ai.record_and_stt(sec=3.0, in_filename="answer_pay.wav")
                print("📝 pay STT:", answer)

                if self._is_yes(answer):
                    print("✅ 결제 YES 인식")
                    self.requestPayDecision.emit(True)
                    return

                if self._is_no(answer):
                    print("✅ 결제 NO 인식")
                    self.requestPayDecision.emit(False)
                    return

                # ✅ 재질문 문구 변경
                retry = "죄송합니다. 다시한번 말씀해주세요."
                self._speak_or_make_tts(retry, "retry_pay.wav")

            print("⚠ 네/아니요 인식 실패: 자동 결제 플로우 종료")

        finally:
            self.pay_voice_busy = False


    def _on_pay_decision_from_voice(self, go_pay: bool):
        """음성에서 결제 여부가 결정되면 UI 전환은 메인스레드에서 수행"""
        if go_pay:
            if self.order_stack and self.payment_choose_page:
                self.order_stack.setCurrentWidget(self.payment_choose_page)
                # ✅ 여기 1줄 추가: 결제수단 음성 선택 시작
                QTimer.singleShot(0, self._start_voice_pay_method)
        else:
            self._back_to_main_from_order()


    def _populate_order_table(self):
        if not self.order_table:
            return

        print(f"[ORDER-TABLE] _populate_order_table 호출, cart_items 개수={len(self.cart_items)}")

        self.order_table.clearContents()
        rows = len(self.cart_items)
        self.order_table.setRowCount(rows)
        self.order_table.setColumnCount(5)

        for r, item in enumerate(self.cart_items):
            menu_name = item.get("menu_name", "")
            options = item.get("options", [])
            qty = 1  # 현재 구조상 LLM/상세페이지 모두 1개씩 append
            total_price = item.get("total_price", 0)

            if options:
                parts = []
                for opt in options:
                    kor = opt.get("kor_name", "")
                    cnt = opt.get("count", 0)
                    if not kor or cnt <= 0:
                        continue
                    if cnt == 1:
                        parts.append(kor)
                    else:
                        parts.append(f"{kor} x{cnt}")
                opt_text = ", ".join(parts)
            else:
                opt_text = ""

            print(f"[ORDER-TABLE] row {r}: menu={menu_name}, options={opt_text}, qty={qty}, total_price={total_price}")

            menu_item = QTableWidgetItem(menu_name)
            opt_item = QTableWidgetItem(opt_text)
            qty_item = QTableWidgetItem(str(qty))
            price_item = QTableWidgetItem(f"{total_price:,}원")

            self.order_table.setItem(r, 0, menu_item)
            self.order_table.setItem(r, 1, opt_item)
            self.order_table.setItem(r, 2, qty_item)
            self.order_table.setItem(r, 3, price_item)

            # 삭제 버튼
            btn = QPushButton("삭제")
            btn.setFixedHeight(36)
            btn.setStyleSheet(
                """
                QPushButton {
                    background-color: #FF6B6B;
                    border-radius: 10px;
                    padding: 6px 12px;
                    font-size: 14px;
                    font-weight: 600;
                    color: #ffffff;
                }
                QPushButton:hover {
                    background-color: #E05555;
                }
                """
            )
            btn.clicked.connect(lambda _=False, row_index=r: self._remove_cart_item(row_index))
            self.order_table.setCellWidget(r, 4, btn)

        self.order_table.resizeColumnsToContents()

    def _remove_cart_item(self, row_index: int):
        if 0 <= row_index < len(self.cart_items):
            del self.cart_items[row_index]
            self._recalc_cart_summary()
            self._populate_order_table()

    # ====== 결제 완료 + DB 저장 ======
    def _on_charge_msg_clicked(self):
        msg = QMessageBox(self)
        msg.setWindowTitle("결제 완료")
        msg.setText("결제가 완료되었습니다.\n이용해 주셔서 감사합니다.")
        msg.setIcon(QMessageBox.Information)
        msg.setStandardButtons(QMessageBox.NoButton)
        QTimer.singleShot(3000, msg.accept)
        msg.exec()

        if self.order_mode == "for_here":
            mode_text = "매장"
        elif self.order_mode == "to_go":
            mode_text = "포장"
        else:
            mode_text = self.order_mode or ""

        try:
            save_order(mode_text, self.selected_pay_method or "", self.cart_items)
        except Exception as e:
            print("주문 저장 중 오류:", e)

        self.cart_items.clear()
        self._recalc_cart_summary()
        self._reset_cursor()
        self._go_opening()

    # ------------------------------------------------------------------
    # 일시품절 상태를 외부(관리자창)에서 바꿀 때 쓰는 메서드
    # ------------------------------------------------------------------
    def set_menu_sold_out(self, menu_name: str, sold_out: bool):
        if sold_out:
            self.sold_out_menus.add(menu_name)
        else:
            self.sold_out_menus.discard(menu_name)

        # 현재 탭/상세페이지 이미지 다시 그리기
        self._render_current_tab()
        if self.current_detail_data and self.current_detail_data.get("이름") == menu_name:
            self._show_detail(self.current_detail_data)

    # ------------------------------------------------------------------
    # 이벤트 필터
    # ------------------------------------------------------------------
    def eventFilter(self, obj, event):
        if hasattr(self, "menu_frames") and obj in getattr(self, "menu_frames", []):
            et = event.type()
            if et == QEvent.Enter:
                self._enlarge_card(obj)
            elif et == QEvent.Leave:
                self._restore_card(obj)
            elif et == QEvent.MouseButtonPress:
                idx = self.menu_frame_index.get(obj)
                if idx is not None:
                    self._on_menu_clicked(idx)
            return False

        if hasattr(self, "option_frame_index") and obj in self.option_frame_index:
            if event.type() == QEvent.MouseButtonPress:
                idx = self.option_frame_index.get(obj)
                if idx is not None:
                    self._on_option_clicked(idx)
            return False

        if obj in self.pay_img_labels:
            if event.type() == QEvent.MouseButtonPress:
                method = self.pay_img_method_map.get(obj, "이미지")
                self._go_charge_page(method)
            return False

        if obj is getattr(self, "charge_msg", None):
            if event.type() == QEvent.MouseButtonPress:
                self._on_charge_msg_clicked()
            return False

        return super().eventFilter(obj, event)

    def _enlarge_card(self, frame: QFrame):
        orig = self.menu_frame_sizes.get(frame)
        if orig is None or orig.width() == 0 or orig.height() == 0:
            orig = frame.size()

        hover_size = QSize(int(orig.width() * 1.06), int(orig.height() * 1.06))
        frame.setMinimumSize(hover_size)
        frame.setMaximumSize(hover_size)
        frame.raise_()

    def _restore_card(self, frame: QFrame):
        orig = self.menu_frame_sizes.get(frame)
        if not orig:
            return
        frame.setMinimumSize(orig)
        frame.setMaximumSize(orig)

    def _on_menu_clicked(self, idx: int):
        if not self.category_tab:
            return
        tab_text = self.category_tab.tabText(self.category_tab.currentIndex())
        items = self.menu_by_tab.get(tab_text, [])
        list_index = idx - 1
        if 0 <= list_index < len(items):
            data = items[list_index]
            menu_name = (data.get("이름") or "").strip()

        # ✅ 품절이면 상세페이지 진입 막고 알림만 띄움
        if menu_name and menu_name in self.sold_out_menus:
            QMessageBox.information(self, "일시품절", "일시품절입니다.")
            return

        self._show_detail(data)

    # ------------------------------------------------------------------
    # 기타 이벤트
    # ------------------------------------------------------------------
    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.stack.currentWidget() is self.page_opening:
            self._show_logo()
            self._next_ad(initial=True)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            QApplication.quit()
        else:
            super().keyPressEvent(e)


def main():
    app = QApplication(sys.argv)
    w = KioskMain()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
