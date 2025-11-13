from enum import Enum

class OutputColumn(Enum):
    OFFSET = "Offset % a partir do neutro"
    TENSAO = "Tensão % nas linhas"
    CARGA = "% de carga nas ancoras"

# Usage example:
# OutputColumn.OFFSET.value
# OutputColumn.TENSAO.value
# OutputColumn.CARGA.value
