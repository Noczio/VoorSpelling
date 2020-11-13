from abc import abstractmethod, ABC


class MachineLearningModel(ABC):
    pass


class SimpleModel(MachineLearningModel):
    pass


class OnlyFeatureSelectionModel(MachineLearningModel):
    pass


class OnlyParameterSearchModel(MachineLearningModel):
    pass


class FeatureAndParameterSearch(MachineLearningModel):
    pass


class ModelTypeCreator:
    __instance = None
    _types: dict = {"SM": SimpleModel(), "FSM": OnlyFeatureSelectionModel(),
                    "PSM": OnlyParameterSearchModel(), "AM": FeatureAndParameterSearch()}

    @staticmethod
    def get_instance() -> "ModelTypeCreator":
        """Static access method."""
        if ModelTypeCreator.__instance is None:
            ModelTypeCreator()
        return ModelTypeCreator.__instance

    def __init__(self) -> None:
        """Virtually private constructor."""
        if ModelTypeCreator.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            ModelTypeCreator.__instance = self

    def create_model(self, feature_selection: bool, parameter_search: bool) -> MachineLearningModel:
        if not feature_selection and not parameter_search:
            return self._types["SM"]
        elif feature_selection and not parameter_search:
            return self._types["FSM"]
        elif not feature_selection and parameter_search:
            return self._types["PSM"]
        else:
            return self._types["AM"]

    def get_available_types(self) -> tuple:
        available_types = [k for k in self._types.keys()]
        types = tuple(available_types)
        return types
