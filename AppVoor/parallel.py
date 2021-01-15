import sys
from typing import Callable

from PyQt5.QtCore import pyqtSignal, pyqtSlot, QRunnable, QObject, QThread


class WorkerSignals(QObject):
    finished = pyqtSignal()
    program_error = pyqtSignal(object)
    result = pyqtSignal(object)


class Worker(QRunnable):
    """
    Worker thread
    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.
    :param callback: The function callback to run on this worker thread. Supplied args and
            kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function
    """

    def __init__(self, func: Callable, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        """
        Initialise the runner function with passed args, kwargs.
        """
        # Retrieve args/kwargs here; and fire processing using them
        try:
            output = self.func(*self.args, **self.kwargs)
            self.signals.result.emit(output)  # Return the result of the processing
        except Exception as e:
            self.signals.program_error.emit(e)
        else:
            self.signals.finished.emit()


class EmittingStream(QObject):
    textWritten = pyqtSignal(str)

    def __init__(self, text_destiny):
        super().__init__()
        self.textWritten = text_destiny

    def write(self, text):
        self.textWritten.emit(str(text))

    def __del__(self):
        sys.stdout = sys.__stdout__
