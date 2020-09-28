import pandas as pd


def is_dict(data) -> bool:
    if isinstance(data, dict):
        return True
    return False


def is_tuple(data) -> bool:
    if isinstance(data, tuple):
        return True
    return False


def is_list(data) -> bool:
    if isinstance(data, list):
        return True
    return False


def is_str(data) -> bool:
    if isinstance(data, str):
        return True
    return False


def is_float(data) -> bool:
    if isinstance(data, float):
        return True
    return False


def is_int(data) -> bool:
    if isinstance(data, int):
        return True
    return False


def is_dataframe(data, min_data=100) -> bool:
    if isinstance(data, pd.DataFrame):
        x_shape, y_shape = data.shape
        if x_shape <= min_data or y_shape == 1:
            return False
        return True
    return False
