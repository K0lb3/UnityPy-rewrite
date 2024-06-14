import math
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ..objects.math import Quaternionf

if TYPE_CHECKING:
    from ..objects import PackedBitVector


def unpack_floats(
    packed: PackedBitVector,
    itemCountInChunk: int,
    chunkStride: int,
    start: int = 0,
    numChunks: int = -1,
) -> npt.NDArray[np.float_]:
    assert (
        packed.m_BitSize is not None
        and packed.m_Range is not None
        and packed.m_Start is not None
    )

    bitPos = packed.m_BitSize * start
    indexPos = bitPos // 8
    bitPos %= 8

    scale: float = (1.0 / packed.m_Range) if packed.m_Range else float("inf")
    if numChunks == -1:
        numChunks = packed.m_NumItems // itemCountInChunk
    end = int(chunkStride * numChunks / 4)
    data = np.zeros(
        packed.m_NumItems, dtype=np.float32 if packed.m_BitSize < 32 else np.float64
    )

    def uint(num: int):
        if num < 0 or num > 4294967295:
            return num % 4294967296
        return num

    for i in range(0, end, chunkStride // 4):
        for j in range(itemCountInChunk):
            x = 0  # uint
            bits = 0
            while bits < packed.m_BitSize:
                x |= uint((packed.m_Data[indexPos] >> bitPos) << bits)
                num = min(packed.m_BitSize - bits, 8 - bitPos)
                bitPos += num
                bits += num
                if bitPos == 8:  #
                    indexPos += 1
                    bitPos = 0

            x &= uint((1 << packed.m_BitSize) - 1)  # (uint)(1 << m_BitSize) - 1u
            denomi = scale * ((1 << packed.m_BitSize) - 1)

            value = x / denomi if denomi else float("inf") + packed.m_Start
            data[i + j] = value

    return data


def unpack_ints(
    packed: PackedBitVector,
) -> npt.NDArray[np.int_]:
    assert packed.m_BitSize is not None

    dtype: npt.DTypeLike
    # TODO: unsure about this, might also be <=
    if packed.m_BitSize < 8:
        dtype = np.int8
    elif packed.m_BitSize < 16:
        dtype = np.int16
    elif packed.m_BitSize < 32:
        dtype = np.int32
    else:
        dtype = np.int64

    data = np.zeros(packed.m_NumItems, dtype=dtype)

    indexPos = 0
    bitPos = 0
    for i in range(packed.m_NumItems):
        bits = 0
        data[i] = 0
        while bits < packed.m_BitSize:
            data[i] |= (packed.m_Data[indexPos] >> bitPos) << bits
            num = min(packed.m_BitSize - bits, 8 - bitPos)
            bitPos += num
            bits += num
            if bitPos == 8:
                indexPos += 1
                bitPos = 0
        data[i] &= (1 << packed.m_BitSize) - 1
    return data


def unpack_quaternions(packed: PackedBitVector) -> npt.NDArray[np.float32]:
    m_Data = packed.m_Data
    data = np.zeros((packed.m_NumItems, 4), dtype=Quaternionf)
    indexPos = 0
    bitPos = 0

    for i in range(packed.m_NumItems):
        flags = 0
        bits = 0
        while bits < 3:
            flags |= (m_Data[indexPos] >> bitPos) << bits  # unit
            num = min(3 - bits, 8 - bitPos)
            bitPos += num
            bits += num
            if bitPos == 8:  #
                indexPos += 1
                bitPos = 0
        flags &= 7

        # sum = 0
        def decode_component(j: int) -> float:
            nonlocal indexPos, bitPos
            if (flags & 3) != j:
                bitSize = 9 if ((flags & 3) + 1) % 4 == j else 10
                x = 0

                bits = 0
                while bits < bitSize:
                    x |= (m_Data[indexPos] >> bitPos) << bits  # uint
                    num = min(bitSize - bits, 8 - bitPos)
                    bitPos += num
                    bits += num
                    if bitPos == 8:  #
                        indexPos += 1
                        bitPos = 0
                x &= (1 << bitSize) - 1  # unit
                return x / (0.5 * ((1 << bitSize) - 1)) - 1
            else:
                return 0.0

        data[i] = q = Quaternionf(
            x=decode_component(0),
            y=decode_component(1),
            z=decode_component(2),
            w=decode_component(3),
        )

        lastComponent = flags & 3  # int
        q[lastComponent] = math.sqrt(1 - (q * q).sum())  # float
        if (flags & 4) != 0:  # 0u
            q[lastComponent] = -q[lastComponent]

    return data
