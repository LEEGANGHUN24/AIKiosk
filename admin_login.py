# admin_login.py

import os
import sys
from PySide6.QtCore import QFile, Qt
from PySide6.QtWidgets import (
    QDialog, QLineEdit, QPushButton, QVBoxLayout, QMessageBox
)
from PySide6.QtUiTools import QUiLoader

from orders_db import get_conn


def resource_path(rel: str) -> str:
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, rel)


class AdminLoginDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("관리자 로그인")

        loader = QUiLoader()
        ui_path = resource_path("ui/admin_login.ui")

        ui_file = QFile(ui_path)
        if not ui_file.open(QFile.ReadOnly):
            QMessageBox.critical(self, "오류",
                                 f"admin_login.ui 파일을 열 수 없습니다.\n{ui_path}")
            self.reject()
            return

        # .ui 로드 (루트가 QDialog 이지만 child 로 로드)
        inner = loader.load(ui_file, self)
        ui_file.close()

        if not inner:
            QMessageBox.critical(self, "오류",
                                 "admin_login.ui 로드에 실패했습니다.")
            self.reject()
            return

        # ⭐ QDialog → 일반 위젯처럼 보이게 플래그 변경
        inner.setWindowFlags(Qt.Widget)

        # 다이얼로그 레이아웃에 inner 추가
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(inner)

        self.ui = inner

        # ---- ui 안의 위젯 찾기 ----
        self.id_edit: QLineEdit = self.ui.findChild(QLineEdit, "id_edit")
        self.pw_edit: QLineEdit = self.ui.findChild(QLineEdit, "pw_edit")
        self.login_btn: QPushButton = self.ui.findChild(QPushButton, "login_btn")
        self.cancel_btn: QPushButton = self.ui.findChild(QPushButton, "cancel_btn")

        if not all([self.id_edit, self.pw_edit, self.login_btn, self.cancel_btn]):
            QMessageBox.critical(
                self,
                "오류",
                "admin_login.ui 의 objectName(id_edit, pw_edit, "
                "login_btn, cancel_btn)을 확인해주세요."
            )
            self.reject()
            return

        # 다이얼로그 크기 고정
        self.setFixedSize(460, 260)

        # 비밀번호 가리기(안전용, ui에도 설정돼 있지만 한 번 더)
        self.pw_edit.setEchoMode(QLineEdit.Password)

        # 시그널 연결
        self.login_btn.clicked.connect(self.try_login)
        self.cancel_btn.clicked.connect(self.reject)

    # ---- 로그인 처리 ----
    def try_login(self):
        user_id = self.id_edit.text().strip()
        pw = self.pw_edit.text().strip()

        if not user_id or not pw:
            QMessageBox.warning(self, "로그인", "아이디와 비밀번호를 입력하세요.")
            return

        if self._check_admin(user_id, pw):
            QMessageBox.information(self, "로그인", "관리자 로그인 성공")
            self.accept()
        else:
            QMessageBox.warning(self, "로그인 실패",
                                "아이디 또는 비밀번호가 올바르지 않습니다.")

    def _check_admin(self, login_id: str, password: str) -> bool:
        try:
            conn = get_conn()
            cur = conn.cursor()
            cur.execute(
                "SELECT id FROM admins WHERE login_id=? AND password=?",
                (login_id, password),
            )
            row = cur.fetchone()
            conn.close()
            return row is not None
        except Exception as e:
            print("관리자 조회 오류:", e)
            return False
