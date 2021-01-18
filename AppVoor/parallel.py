import sys

from PyQt5.QtCore import *


class WorkerSignals(QObject):
    """PyQt signals custom class"""
    program_finished = pyqtSignal()
    program_error = pyqtSignal(BaseException)
    result = pyqtSignal(object)

    def __init__(self):
        super().__init__()


class LongWorker(QRunnable):
    """
    Worker thread
    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.
    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function
    """
    signals = WorkerSignals()

    def __init__(self, func=None, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def set_params(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    @pyqtSlot()
    def run(self):
        """Run method of Worker class. Tries to execute a given function and emits a signal"""
        try:
            if len(self.args) > 0 and len(self.kwargs) > 0:
                output = self.func(*self.args, **self.kwargs)
            elif len(self.args) > 0 and len(self.kwargs) == 0:
                output = self.func(*self.args)
            elif len(self.args) == 0 and len(self.kwargs) > 0:
                output = self.func(**self.kwargs)
            else:
                output = self.func()
            self.signals.program_finished.emit()
        except Exception as e:
            self.signals.program_error.emit(e)
        else:
            self.signals.result.emit(output)


class EmittingStream(QObject):
    """Custom class that catches sys.stdout info and gives it back to a function"""
    textWritten = pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))

    def __del__(self):
        sys.stdout = sys.__stdout__
