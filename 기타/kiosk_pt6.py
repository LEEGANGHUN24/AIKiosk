# kiosk.py (PySide6 / Qt6)
import os, sys
from typing import List, Optional
from PySide6.QtCore import Qt, QTimer, QFile, QSize
from PySide6.QtGui import QPixmap, QAction
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QToolButton, QMessageBox, QDialog, QMenu, QStackedWidget
)
from PySide6.QtUiTools import QUiLoader


def resource_path(rel: str) -> str:
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, rel)


class KioskMain(QMainWindow):
    def __init__(self):
        super().__init__()
        # 화면 고정 크기 (사용자 해상도 1536x864에 맞춰 세로 864로 고정)
        self.setFixedSize(768, 864)

        # === QStackedWidget ===
        self.stack = QStackedWidget(self)
        self.setCentralWidget(self.stack)

        self.loader = QUiLoader()

        # === 두 페이지 로드 ===
        self.page_opening = self._load_ui("ui/first_page_v2.ui")
        self.page_main    = self._load_ui("ui/main_page.ui")
        if not self.page_opening or not self.page_main:
            raise RuntimeError("필수 UI(first_page_v2.ui, main_page.ui) 로드 실패")

        # === 스택에 추가 ===
        self.stack.addWidget(self.page_opening)  # index 0
        self.stack.addWidget(self.page_main)     # index 1
        self.stack.setCurrentIndex(0)

        # === Opening 핸들 ===
        self.logo_label: Optional[QLabel] = self.page_opening.findChild(QLabel, "logo_label")
        self.ad_label:   Optional[QLabel] = self.page_opening.findChild(QLabel, "ad_label")
        self.btn_eat_here: Optional[QPushButton] = self.page_opening.findChild(QPushButton, "eat_here_btn")
        self.btn_to_go:   Optional[QPushButton]  = self.page_opening.findChild(QPushButton, "to_go_btn")
        self.btn_settings: Optional[QToolButton] = self.page_opening.findChild(QToolButton, "settings_btn")

        # === Main 핸들 ===
        self.main_mode_badge: Optional[QLabel]    = self.page_main.findChild(QLabel, "mode_badge")
        self.btn_back: Optional[QToolButton]      = self.page_main.findChild(QToolButton, "back_btn")  # QToolButton!

        # === 설정 메뉴 ===
        if self.btn_settings:
            menu = QMenu(self)
            act_admin = QAction("관리자 모드", self)
            menu.addAction(act_admin)
            self.btn_settings.setMenu(menu)
            self.btn_settings.setPopupMode(QToolButton.InstantPopup)
            act_admin.triggered.connect(self._open_manager)

        # === Opening 광고 슬라이드 ===
        self.ad_images = self._collect_ad_images()
        self.ad_index = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._next_ad)
        self.timer.start(3000)

        # === 버튼 연결 ===
        if self.btn_eat_here:
            self.btn_eat_here.clicked.connect(lambda: self._go_main("for_here"))
        if self.btn_to_go:
            self.btn_to_go.clicked.connect(lambda: self._go_main("to_go"))
        if self.btn_back:
            self.btn_back.clicked.connect(self._go_opening)

        # === 첫 그리기 ===
        QTimer.singleShot(0, self._show_logo)
        QTimer.singleShot(0, lambda: self._next_ad(initial=True))

        # === 상태 ===
        self.order_mode: Optional[str] = None  # "for_here" | "to_go"

    # ---------- 공통 로더 ----------
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

    # ---------- Opening ----------
    def _show_logo(self):
        path = resource_path("img/mega_logo.jpg")
        if not (self.logo_label and os.path.exists(path)):
            return
        pix = QPixmap(path)
        if pix.isNull():
            return
        # 크기는 UI에서 정함(중앙 정렬). 비율 유지 스케일만 적용.
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
        # Opening 페이지에서만 갤러리 동작
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

    # ---------- 전환 ----------
    def _go_main(self, mode: str):
        """Opening → Main"""
        self.order_mode = mode
        if self.timer.isActive():
            self.timer.stop()
        if self.main_mode_badge:
            self.main_mode_badge.setText("매장" if mode == "for_here" else "포장")
        self.stack.setCurrentWidget(self.page_main)

    def _go_opening(self):
        """Main → Opening (뒤로가기)"""
        if not self.timer.isActive():
            self.timer.start(3000)
        self.stack.setCurrentWidget(self.page_opening)
        # 돌아오면 로고/광고 다시 맞춤
        QTimer.singleShot(0, self._show_logo)
        QTimer.singleShot(0, lambda: self._next_ad(initial=True))

    # ---------- 기타 ----------
    def _open_manager(self):
        ui = resource_path("ui/manager_page.ui")
        if not os.path.exists(ui):
            QMessageBox.information(self, "관리자 모드", "manager_page.ui가 아직 없습니다. (ui/manager_page.ui)")
            return
        f = QFile(ui)
        if not f.open(QFile.ReadOnly):
            QMessageBox.warning(self, "관리자 모드", "관리자 UI를 열 수 없습니다.")
            return
        dlg = self.loader.load(f, self)
        f.close()
        if isinstance(dlg, QDialog):
            dlg.exec()

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
