file_loaded_style = u"QPushButton{\n" \
                    "border-color: rgb(60,179,113);\n" \
                    "}\n" \
                    u"QPushButton:hover{\n" \
                    "background-color: rgb(235, 225, 240);\n" \
                    "}\n" \
                    "QPushButton:pressed{\n" \
                    "background-color: rgb(220, 211, 230);\n" \
                    "border-left-color: rgb(60,179,113);\n" \
                    "border-top-color: rgb(60,179,113);\n" \
                    "border-bottom-color: rgb(85, 194, 132);\n" \
                    "border-right-color: rgb(85, 194, 132);\n" \
                    "}"

file_dragged_style = u"image: url(:/file/bx-file 1.svg);\n" \
                     "padding-top: 20px;\n" \
                     "padding-bottom: 80px;\n" \
                     "border-color: rgb(60,179,113);\n" \
                     ""

file_not_loaded_style = u"QPushButton:hover{\n" \
                        "background-color: rgb(235, 225, 240);\n" \
                        "}\n" \
                        "QPushButton:pressed{\n" \
                        "background-color: rgb(220, 211, 230);\n" \
                        "border-left-color: rgb(190, 185, 220);\n" \
                        "border-top-color: rgb(190, 185, 220);\n" \
                        "border-bottom-color: rgb(215, 200, 239);\n" \
                        "border-right-color: rgb(215, 200, 239);\n" \
                        "}"

file_not_dragged_style = u"image: url(:/file/bx-file 1.svg);\n" \
                         "padding-top: 20px;\n" \
                         "padding-bottom: 80px;\n" \
                         "border-color: rgb(220, 220, 220);\n" \
                         ""

load_buttons_style = {"Load": (file_loaded_style, file_not_dragged_style),
                      "Drag": (file_not_loaded_style, file_dragged_style),
                      "Any": (file_not_loaded_style, file_not_dragged_style)}
