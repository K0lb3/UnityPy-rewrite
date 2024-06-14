from struct import Struct
from numpy import dtype

# struct
BOOL = Struct("?")

U8 = Struct("B")
U16_LE = Struct("<H")
U16_BE = Struct(">H")
U32_LE = Struct("<I")
U32_BE = Struct(">I")
U64_LE = Struct("<Q")
U64_BE = Struct(">Q")

I8 = Struct("b")
I16_LE = Struct("<h")
I16_BE = Struct(">h")
I32_LE = Struct("<i")
I32_BE = Struct(">i")
I64_LE = Struct("<q")
I64_BE = Struct(">q")

F16_LE = Struct("<e")
F16_BE = Struct(">e")
F32_LE = Struct("<f")
F32_BE = Struct(">f")
F64_LE = Struct("<d")
F64_BE = Struct(">d")

# numpy
BOOL_NP = dtype("?")

U8_NP = dtype("B")
U16_LE_NP = dtype("<H")
U16_BE_NP = dtype(">H")
U32_LE_NP = dtype("<I")
U32_BE_NP = dtype(">I")
U64_LE_NP = dtype("<Q")
U64_BE_NP = dtype(">Q")

I8_NP = dtype("b")
I16_LE_NP = dtype("<h")
I16_BE_NP = dtype(">h")
I32_LE_NP = dtype("<i")
I32_BE_NP = dtype(">i")
I64_LE_NP = dtype("<q")
I64_BE_NP = dtype(">q")

F16_LE_NP = dtype("<e")
F16_BE_NP = dtype(">e")
F32_LE_NP = dtype("<f")
F32_BE_NP = dtype(">f")
F64_LE_NP = dtype("<d")
F64_BE_NP = dtype(">d")
