load_dataset_errors = {FileNotFoundError: {"body": "No se encontró el archivo de datos para realizar el entrenamiento",
                                           "additional": ""},
                       TypeError: {"body": "El archivo suministrado no cumple con los requerimientos",
                                   "additional": "El archivo debe tener más de cien muestras y por lo menos una "
                                                 "característica y la predicción. Además, es importante "
                                                 "seleccionar correctamente si el archivo está separado por "
                                                 "coma (CSV) o tabulación (TSV)"},
                       ValueError: {
                           "body": "El archivo seleccionado no cumple con los requerimientos para ser considerado "
                                   "un archivo de texto con las extensiones permitidas",
                           "additional": ""},
                       OSError: {"body": "No se encontró el archivo de datos para realizar el entrenamiento",
                                 "additional": ""},
                       Exception: {
                           "body": "El archivo seleccionado no cumple con los requerimientos para ser utilizado para "
                                   "entrenar un modelo de inteligencia artificial",
                           "additional": ""}
                       }
