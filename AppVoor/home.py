import sys

from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow

from jsonInfo.welcome import WelcomeMessenger


class Window(QMainWindow):
    def __init__(self, window: str):
        super().__init__()
        uic.loadUi(window, self)
        self.center_screen()
        self.change_message()

    def change_message(self):
        messenger = WelcomeMessenger(file_path=".\\jsonInfo\\welcomeMessage.json")
        text = str(messenger)
        self.lbl_phrase.setText(text)

    def center_screen(self):
        frame_gm = self.frameGeometry()
        screen = QApplication.desktop().screenNumber(QApplication.desktop().cursor().pos())
        center_point = QApplication.desktop().screenGeometry(screen).center()
        frame_gm.moveCenter(center_point)
        self.move(frame_gm.topLeft())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window_path = ".\\forms\\QT_Voorspelling_Home.ui"
    gui = Window(window_path)
    gui.show()
    sys.exit(app.exec_())
