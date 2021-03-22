import json
import os

from mdutils.mdutils import MdUtils


class FCreator:
    folder_path: str

    def __init__(self, location: str, obj: str = "SBS_ML") -> None:
        self._object = obj
        self._location = location
        self.create_folder()

    def create_folder(self) -> None:
        try:
            sub_value = 1
            while True:
                final_folder_name = self._object + "_" + str(sub_value)
                path = os.path.join(self._location, final_folder_name)
                if os.path.exists(path):
                    sub_value += 1
                else:
                    os.mkdir(path)
                    self.folder_path = path
                    break
        except Exception as e:
            print(e)


class SBSResult:

    @staticmethod
    def estimator_info(options: dict, features: list, initial_params: dict, final_params: dict,
                       performance: str, path: str) -> bool:
        f_title = "Resultados_del_estimador_paso_a_paso"
        f_name = path + "\\" + f_title + ".md"
        try:
            md_file = MdUtils(file_name=f_name, title=f_title)
            md_file.new_header(level=1, title="Selección")

            md_file.new_header(level=2, title="Opciones")
            md_file.new_table(columns=options["columns"], rows=options["rows"],
                              text=options["info"], text_align='center')

            md_file.new_header(level=1, title="Resultados")

            md_file.new_header(level=2, title="Caraterísticas")
            list_to_string = ', '.join(map(str, features))
            md_file.new_paragraph(list_to_string)

            md_file.new_header(level=2, title="Parámetros iniciales")
            initial_params_as_string = str(initial_params)
            md_file.new_line(initial_params_as_string)

            md_file.new_header(level=2, title="Parámetros finales")
            final_params_as_string = str(final_params)
            md_file.new_line(final_params_as_string)

            md_file.new_header(level=2, title="Rendimiento")
            md_file.new_line(performance)

            md_file.new_table_of_contents(table_title="Contenido", depth=2)
            md_file.create_md_file()
            return True
        except Exception as e:
            print(e)
            return False

    @staticmethod
    def console_info(info: list, path: str) -> bool:
        f_title = "Logs_del_estimador_paso_a_paso"
        f_name = path + "\\" + f_title + ".md"
        try:
            md_file = MdUtils(file_name=f_name, title=f_title)
            md_file.new_header(level=1, title="Logs")
            max_num_lines = len(info)
            print(f"Total lines to copy {max_num_lines}")
            for counter, line in enumerate(info, start=1):
                md_file.new_line(line)
                print(f"{counter} out of {max_num_lines} lines")
            md_file.create_md_file()
            return True
        except Exception as e:
            print(e)
            return False
