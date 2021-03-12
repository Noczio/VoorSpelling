from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow

from jsonInfo.help import HelpMessage
from frontend_scripts.pop_up import PopUp, InfoPopUp, WarningPopUp
from version import __version__ as version


class Window(QMainWindow):
    """Base Window class that inherits from QMainWindow"""

    def __init__(self, window: str, help_message_path: str = ".\\jsonInfo\\helpMessage.json") -> None:
        super().__init__()
        uic.loadUi(window, self)
        self._help_message = HelpMessage(file_path=help_message_path)
        self.on_load()

    def useful_info_pop_up(self, key: str) -> None:
        """Show useful info in a pop. A form may use this or not"""
        # get help message info from HelpMessage object
        title, body, example, url = self._help_message[key]
        if example is not "":
            body = body + "\n\n" + "Ejemplo:" + "\n\n" + example
        if url is not "":
            url = "Para más información visitar:" + " " + url
        # call general_info_pop_up with useful_info_pop_up info
        pop_up: PopUp = InfoPopUp()
        pop_up.open_pop_up(title, body, url)

    def last_warning_pop_up(self) -> bool:
        """Show warning info in a pop. A form may use this method, overwrite it or not use it at all"""
        pop_up: PopUp = WarningPopUp()
        title = "Listo para entrenar"
        body = "¿Estas seguro que deseas continuar?"
        additional = "La aplicación iniciará inmediatamente con el proceso de entrenamiento"
        answer = pop_up.open_pop_up(title, body, additional)
        return answer

    def next(self, *args, **kwargs) -> None:
        """Go to next form"""
        pass

    def back(self, *args, **kwargs) -> None:
        """Go to last form"""
        pass

    def close_window(self) -> None:
        """Close actual form"""
        self.close()

    def handle_error(self, *args, **kwargs) -> None:
        """Handle error arguments and the use them to display a custom message"""
        pass

    def on_load(self) -> None:
        """Additional behaviour on load"""
        if hasattr(self, "lbl_left_4"):
            info = f"Version {version}\nLicencia BSD 3"
            self.lbl_left_4.setText(info)
