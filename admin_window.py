# admin_window.py

import os
from typing import List, Dict

from PySide6.QtCore import Qt, QFile
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QTableWidget, QTableWidgetItem,
    QPushButton, QAbstractItemView, QMessageBox
)
from PySide6.QtUiTools import QUiLoader

from orders_db import get_conn
# kiosk.py 와 같은 resource_path 함수 하나 따로 만들어 둠
import sys


def resource_path(rel: str) -> str:
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, rel)


class AdminWindow(QMainWindow):
    """
    관리자 메인 창
    - 메뉴 품절 관리
    - 주문 내역 조회
    """

    def __init__(self, kiosk, parent=None):
        super().__init__(parent)
        self.kiosk = kiosk  # KioskMain 인스턴스 참조

        self.setWindowTitle("관리자 모드")

        loader = QUiLoader()
        ui_path = resource_path("ui/admin_main.ui")
        f = QFile(ui_path)
        if not f.exists():
            QMessageBox.critical(self, "오류", f"admin_main.ui 를 찾을 수 없습니다.\n경로: {ui_path}")
            self.setCentralWidget(QWidget())
            return
        f.open(QFile.ReadOnly)
        self.ui = loader.load(f, self)
        f.close()

        if not isinstance(self.ui, QWidget):
            QMessageBox.critical(self, "오류", "admin_main.ui 루트 위젯이 QWidget 이 아닙니다.")
            self.setCentralWidget(QWidget())
            return

        self.setCentralWidget(self.ui)

        # ====== 위젯 찾기 ======
        self.menu_table: QTableWidget = self.ui.findChild(QTableWidget, "menu_table")
        self.btn_set_on: QPushButton = self.ui.findChild(QPushButton, "btn_set_on")
        self.btn_set_soldout: QPushButton = self.ui.findChild(QPushButton, "btn_set_soldout")

        self.order_table: QTableWidget = self.ui.findChild(QTableWidget, "order_table")
        self.btn_refresh_orders: QPushButton = self.ui.findChild(QPushButton, "btn_refresh_orders")

        # 메뉴 테이블 셋업
        if self.menu_table:
            self.menu_table.setColumnCount(3)
            self.menu_table.setHorizontalHeaderLabels(["메뉴명", "상태", "비고"])
            self.menu_table.setSelectionBehavior(QAbstractItemView.SelectRows)
            self.menu_table.setSelectionMode(QAbstractItemView.MultiSelection)

        # 시그널 연결
        if self.btn_set_on:
            self.btn_set_on.clicked.connect(self._set_on_clicked)
        if self.btn_set_soldout:
            self.btn_set_soldout.clicked.connect(self._set_soldout_clicked)
        if self.btn_refresh_orders:
            self.btn_refresh_orders.clicked.connect(self.refresh_orders)

        # 데이터 로드
        self._menu_rows: List[Dict] = []
        self.refresh_menu_table()
        self.refresh_orders()

        # 크기: 키오스크보다 약간 작게
        self.resize(720, 820)

    # ------------------------------------------------------------------
    # 메뉴 품절 관리 탭
    # ------------------------------------------------------------------
    def refresh_menu_table(self):
        """kiosk.menu_by_tab 에서 전체 메뉴 목록을 읽어와 테이블에 표시"""
        if not self.menu_table or not hasattr(self.kiosk, "menu_by_tab"):
            return

        self._menu_rows.clear()

        # menu_by_tab: {탭이름: [row, row, ...]}
        for tab_name, items in self.kiosk.menu_by_tab.items():
            for data in items:
                name = (data.get("이름") or "").strip()
                if not name:
                    continue
                self._menu_rows.append(
                    {
                        "menu_name": name,
                        "tab_name": tab_name,
                    }
                )

        self.menu_table.setRowCount(len(self._menu_rows))

        for row, info in enumerate(self._menu_rows):
            name = info["menu_name"]
            is_soldout = name in getattr(self.kiosk, "sold_out_menus", set())

            name_item = QTableWidgetItem(name)
            state_item = QTableWidgetItem("일시품절" if is_soldout else "판매중")
            note_item = QTableWidgetItem(info["tab_name"])  # 비고에 카테고리 적어둠

            # 읽기 전용
            for it in (name_item, state_item, note_item):
                it.setFlags(it.flags() & ~Qt.ItemIsEditable)

            self.menu_table.setItem(row, 0, name_item)
            self.menu_table.setItem(row, 1, state_item)
            self.menu_table.setItem(row, 2, note_item)

        self.menu_table.resizeColumnsToContents()

    def _selected_menu_names(self) -> List[str]:
        """선택된 행들의 메뉴명 리스트 반환"""
        if not self.menu_table:
            return []
        rows = {idx.row() for idx in self.menu_table.selectedIndexes()}
        names: List[str] = []
        for r in rows:
            item = self.menu_table.item(r, 0)
            if item:
                names.append(item.text())
        return names

    def _set_soldout_clicked(self):
        """선택 메뉴들을 일시품절로"""
        names = self._selected_menu_names()
        if not names:
            QMessageBox.information(self, "알림", "일시품절로 변경할 메뉴를 선택하세요.")
            return

        for name in names:
            self.kiosk.set_menu_sold_out(name, True)

        self.refresh_menu_table()

    def _set_on_clicked(self):
        """선택 메뉴들을 판매중으로"""
        names = self._selected_menu_names()
        if not names:
            QMessageBox.information(self, "알림", "판매중으로 변경할 메뉴를 선택하세요.")
            return

        for name in names:
            self.kiosk.set_menu_sold_out(name, False)

        self.refresh_menu_table()

    # ------------------------------------------------------------------
    # 주문 내역 탭
    # ------------------------------------------------------------------
    def refresh_orders(self):
        """orders 테이블에서 주문 내역 읽어와 표시"""
        if not self.order_table:
            return

        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT order_no, order_time, mode, menu_name, menu_options, price, pay_method
            FROM orders
            ORDER BY id DESC
            """
        )
        rows = cur.fetchall()
        conn.close()

        self.order_table.setRowCount(len(rows))
        self.order_table.setColumnCount(7)
        self.order_table.setHorizontalHeaderLabels(
            ["주문번호", "시간", "구분", "메뉴", "옵션", "금액", "결제수단"]
        )

        for r, row in enumerate(rows):
            order_no, order_time, mode, menu_name, menu_options, price, pay_method = row
            data = [
                order_no,
                order_time,
                mode,
                menu_name,
                menu_options or "",
                f"{price:,}원",
                pay_method,
            ]
            for c, val in enumerate(data):
                item = QTableWidgetItem(str(val))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.order_table.setItem(r, c, item)

        self.order_table.resizeColumnsToContents()

    # ------------------------------------------------------------------
    def closeEvent(self, e: QCloseEvent):
        """창 닫을 때 그냥 닫기만 (키오스크는 계속 실행)"""
        e.accept()
