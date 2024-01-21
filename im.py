from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QLineEdit
from PyQt5.QtCore import Qt


class Login(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.setObjectName("Login")
        self.setFixedSize(500, 70)
        self.setWindowFlags(Qt.WindowMaximizeButtonHint | Qt.WindowStaysOnTopHint | Qt.Window)
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(0)
        self.layout_box = QHBoxLayout()
        self.setWindowTitle("Участник")
        self.push_button = QPushButton(self)
        self.push_button.setText("Сохранить")
        self.layout.addLayout(self.layout_box)
        self.name = QLineEdit(self)
        self.name.setPlaceholderText("ФИО")
        self.layout_box.addWidget(self.name)
        self.passport = QLineEdit(self)
        self.passport.setPlaceholderText("паспорт")
        self.layout_box.addWidget(self.passport)
        self.size_box = QLineEdit(self)
        self.size_box.setPlaceholderText("размер ориентира")
        self.layout_box.addWidget(self.size_box)
        self.layout.addWidget(self.push_button)
        self.layout.setStretch(0, 1)
        self.layout.setStretch(1, 1)

