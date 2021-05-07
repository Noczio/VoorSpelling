import sys

from PyQt5.QtCore import QSize
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QStackedWidget

from resources.backend_scripts.global_vars import GlobalVariables
from resources.forms.form_resources import qInitResources
from resources.integration.other.ui_path import ui_icons
from resources.integration.other.min_max_size import app_size


class MainInitializer:
    app: QApplication
    widget: QStackedWidget
    variables: GlobalVariables

    def __init__(self) -> None:
        qInitResources()
        self._initialize_app()
        self._initialize_widget()
        self._initialize_variables()

    def _initialize_app(self) -> None:
        self.app = QApplication(sys.argv)
        self.app.setApplicationName("Voorspelling")
        icons = QIcon()
        for data in ui_icons.values():
            icons.addFile(data["Path"], QSize(data["Size"], data["Size"]))
        self.app.setWindowIcon(icons)

    def _initialize_widget(self) -> None:
        self.widget = QStackedWidget()
        self.widget.setMaximumSize(app_size["Max"]["Width"], app_size["Max"]["Height"])
        self.widget.setMinimumSize(app_size["Min"]["Width"], app_size["Min"]["Height"])

    def _initialize_variables(self) -> None:
        self.variables = GlobalVariables.get_instance()

    def program_resources(self) -> tuple:
        return self.app, self.widget, self.variables
