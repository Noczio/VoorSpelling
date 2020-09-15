import sys
from PyQt5 import uic
from PyQt5.QtWidgets import  QMainWindow, QApplication

class MainForm(QMainWindow):

    def __init__(self):
        super().__init__()
        uic.loadUi("appgui.ui",self)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    GUI = MainForm()
    GUI.show()
    sys.exit(app.exec_())