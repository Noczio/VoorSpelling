from PyQt5.QtCore import QSize
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QStackedWidget

from resources.backend_scripts.global_vars import GlobalVariables
from resources.forms import form_resources
from resources.integration.other.ui_path import ui_icons


class AppMain:
    app: QApplication
    widget: QStackedWidget
    variables: GlobalVariables
    _resources = form_resources

    def __init__(self) -> None:
        self._initialize_app()
        self._initialize_widget()
        self._initialize_variables()

    def _initialize_app(self) -> None:
        self.app = QApplication(sys.argv)
        self.app.setApplicationName("Voorspelling")
        icons = QIcon()
        for data in ui_icons.values():
            icons.addFile(data[0], QSize(data[-1], data[-1]))
        self.app.setWindowIcon(icons)
        self._resources.qInitResources()

    def _initialize_widget(self) -> None:
        self.widget = QStackedWidget()
        self.widget.setMaximumSize(1440, 1024)
        self.widget.setMinimumSize(1440, 1024)

    def _initialize_variables(self) -> None:
        self.variables = GlobalVariables.get_instance()

    def program_resources(self) -> tuple:
        return self.app, self.widget, self.variables
