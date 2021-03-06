load_dataset_errors = {FileNotFoundError: {"Body": "No se encontró el archivo de datos para realizar el entrenamiento",
                                           "Additional": ""},
                       TypeError: {"Body": "El archivo suministrado no cumple con los requerimientos",
                                   "Additional": "El archivo debe tener más de cien muestras y por lo menos una "
                                                 "característica y la predicción. Además, es importante "
                                                 "seleccionar correctamente si el archivo está separado por "
                                                 "coma (CSV) o tabulación (TSV)"},
                       ValueError: {
                           "Body": "El archivo seleccionado no cumple con los requerimientos para ser considerado "
                                   "un archivo de texto con las extensiones permitidas",
                           "Additional": ""},
                       OSError: {"Body": "No se encontró el archivo de datos para realizar el entrenamiento",
                                 "Additional": ""},
                       Exception: {
                           "Body": "El archivo seleccionado no cumple con los requerimientos para ser utilizado para "
                                   "entrenar un modelo de inteligencia artificial",
                           "Additional": ""}
                       }
