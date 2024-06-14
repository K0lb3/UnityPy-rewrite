from io import BytesIO
from typing import BinaryIO, ByteString

from attrs import define, field
import numpy as np
import numpy.typing as npt

from .structs import *


@define
class EndianBinaryWriter:
    stream: BytesIO = field(default=BinaryIO)
    endian: str = "<"

    def get_bytes(self) -> ByteString:
        return self.stream.getvalue()

    def tell(self) -> int:
        return self.stream.tell()

    def seek(self, offset: int, whence: int = 0) -> int:
        return self.stream.seek(offset, whence)

    def write(self, value: ByteString) -> int:
        return self.stream.write(value)

    def write_count(self, value: int) -> int:
        return self.write_u32(value)

    def write_bool(self, value: bool) -> int:
        return self.write(BOOL.pack(value))

    def write_u8(self, value: int) -> int:
        return self.write(U8.pack(value))

    def write_u16(self, value: int) -> int:
        return (
            self.write_u16_le(value) if self.endian == "<" else self.write_u16_be(value)
        )

    def write_u16_le(self, value: int) -> int:
        return self.write(U16_LE.pack(value))

    def write_u16_be(self, value: int) -> int:
        return self.write(U16_BE.pack(value))

    def write_u32(self, value: int) -> int:
        return (
            self.write_u32_le(value) if self.endian == "<" else self.write_u32_be(value)
        )

    def write_u32_le(self, value: int) -> int:
        return self.write(U32_LE.pack(value))

    def write_u32_be(self, value: int) -> int:
        return self.write(U32_BE.pack(value))

    def write_u64(self, value: int) -> int:
        return (
            self.write_u64_le(value) if self.endian == "<" else self.write_u64_be(value)
        )

    def write_u64_le(self, value: int) -> int:
        return self.write(U64_LE.pack(value))

    def write_u64_be(self, value: int) -> int:
        return self.write(U64_BE.pack(value))

    def write_i8(self, value: int) -> int:
        return self.write(I8.pack(value))

    def write_i16(self, value: int) -> int:
        return (
            self.write_i16_le(value) if self.endian == "<" else self.write_i16_be(value)
        )

    def write_i16_le(self, value: int) -> int:
        return self.write(I16_LE.pack(value))

    def write_i16_be(self, value: int) -> int:
        return self.write(I16_BE.pack(value))

    def write_i32(self, value: int) -> int:
        return (
            self.write_i32_le(value) if self.endian == "<" else self.write_i32_be(value)
        )

    def write_i32_le(self, value: int) -> int:
        return self.write(I32_LE.pack(value))

    def write_i32_be(self, value: int) -> int:
        return self.write(I32_BE.pack(value))

    def write_i64(self, value: int) -> int:
        return (
            self.write_i64_le(value) if self.endian == "<" else self.write_i64_be(value)
        )

    def write_i64_le(self, value: int) -> int:
        return self.write(I64_LE.pack(value))

    def write_i64_be(self, value: int) -> int:
        return self.write(I64_BE.pack(value))

    def write_f16(self, value: float) -> int:
        return (
            self.write_f16_le(value) if self.endian == "<" else self.write_f16_be(value)
        )

    def write_f16_le(self, value: float) -> int:
        return self.write(F16_LE.pack(value))

    def write_f16_be(self, value: float) -> int:
        return self.write(F16_BE.pack(value))

    def write_f32(self, value: float) -> int:
        return (
            self.write_f32_le(value) if self.endian == "<" else self.write_f32_be(value)
        )

    def write_f32_le(self, value: float) -> int:
        return self.write(F32_LE.pack(value))

    def write_f32_be(self, value: float) -> int:
        return self.write(F32_BE.pack(value))

    def write_f64(self, value: float) -> int:
        return (
            self.write_f64_le(value) if self.endian == "<" else self.write_f64_be(value)
        )

    def write_f64_le(self, value: float) -> int:
        return self.write(F64_LE.pack(value))

    def write_f64_be(self, value: float) -> int:
        return self.write(F64_BE.pack(value))

    def _write_array(
        self, data: ByteString, count: int, write_count: bool = True
    ) -> int:
        if write_count:
            self.write_count(count)
        return self.write(data)

    def write_bytes(self, value: ByteString, write_count: bool = True) -> int:
        return self._write_array(value, len(value), write_count)

    def write_string(self, value: str, write_count: bool = True) -> int:
        return self.write_bytes(value.encode("utf8"), write_count)

    def write_cstring(self, value: str) -> int:
        return self.write(b"".join([value.encode("utf8"), b"\x00"]))

    def write_cstr_aligned(self, value: str) -> int:
        self.write(value.encode("utf8"))
        return self.align_stream()

    def align_stream(self, alignment: int = 4) -> int:
        pad = b"\x00" * ((alignment - self.tell() % alignment) % alignment)
        return self.write(pad)

    def write_bool_array(
        self, values: list[bool] | npt.NDArray[np.bool_], write_count: bool = True
    ) -> int:
        data = np.array(values, dtype=BOOL_NP).tobytes()
        return self._write_array(data, len(values), write_count)

    def write_u8_array(
        self, values: list[int] | npt.NDArray[np.uint8], write_count: bool = True
    ) -> int:
        data = np.array(values, dtype=U8_NP).tobytes()
        return self._write_array(data, len(values), write_count)

    def write_u16_array(
        self, values: list[int] | npt.NDArray[np.uint16], write_count: bool = True
    ) -> int:
        return (
            self.write_u16_array_le(values, write_count)
            if self.endian == "<"
            else self.write_u16_array_be(values, write_count)
        )

    def write_u16_array_le(
        self, values: list[int] | npt.NDArray[np.uint16], write_count: bool = True
    ) -> int:
        data = np.array(values, dtype=U16_LE_NP).tobytes()
        return self._write_array(data, len(values), write_count)

    def write_u16_array_be(
        self, values: list[int] | npt.NDArray[np.uint16], write_count: bool = True
    ) -> int:
        data = np.array(values, dtype=U16_BE_NP).tobytes()
        return self._write_array(data, len(values), write_count)

    def write_u32_array(
        self, values: list[int] | npt.NDArray[np.uint32], write_count: bool = True
    ) -> int:
        return (
            self.write_u32_array_le(values, write_count)
            if self.endian == "<"
            else self.write_u32_array_be(values, write_count)
        )

    def write_u32_array_le(
        self, values: list[int] | npt.NDArray[np.uint32], write_count: bool = True
    ) -> int:
        data = np.array(values, dtype=U32_LE_NP).tobytes()
        return self._write_array(data, len(values), write_count)

    def write_u32_array_be(
        self, values: list[int] | npt.NDArray[np.uint32], write_count: bool = True
    ) -> int:
        data = np.array(values, dtype=U32_BE_NP).tobytes()
        return self._write_array(data, len(values), write_count)

    def write_u64_array(
        self, values: list[int] | npt.NDArray[np.uint64], write_count: bool = True
    ) -> int:
        return (
            self.write_u64_array_le(values, write_count)
            if self.endian == "<"
            else self.write_u64_array_be(values, write_count)
        )

    def write_u64_array_le(
        self, values: list[int] | npt.NDArray[np.uint64], write_count: bool = True
    ) -> int:
        data = np.array(values, dtype=U64_LE_NP).tobytes()
        return self._write_array(data, len(values), write_count)

    def write_u64_array_be(
        self, values: list[int] | npt.NDArray[np.uint64], write_count: bool = True
    ) -> int:
        data = np.array(values, dtype=U64_BE_NP).tobytes()
        return self._write_array(data, len(values), write_count)

    def write_i8_array(
        self, values: list[int] | npt.NDArray[np.uint8], write_count: bool = True
    ) -> int:
        data = np.array(values, dtype=I8_NP).tobytes()
        return self._write_array(data, len(values), write_count)

    def write_i16_array(
        self, values: list[int] | npt.NDArray[np.uint16], write_count: bool = True
    ) -> int:
        return (
            self.write_i16_array_le(values, write_count)
            if self.endian == "<"
            else self.write_i16_array_be(values, write_count)
        )

    def write_i16_array_le(
        self, values: list[int] | npt.NDArray[np.uint16], write_count: bool = True
    ) -> int:
        data = np.array(values, dtype=I16_LE_NP).tobytes()
        return self._write_array(data, len(values), write_count)

    def write_i16_array_be(
        self, values: list[int] | npt.NDArray[np.uint16], write_count: bool = True
    ) -> int:
        data = np.array(values, dtype=I16_BE_NP).tobytes()
        return self._write_array(data, len(values), write_count)

    def write_i32_array(
        self, values: list[int] | npt.NDArray[np.uint32], write_count: bool = True
    ) -> int:
        return (
            self.write_i32_array_le(values, write_count)
            if self.endian == "<"
            else self.write_i32_array_be(values, write_count)
        )

    def write_i32_array_le(
        self, values: list[int] | npt.NDArray[np.uint32], write_count: bool = True
    ) -> int:
        data = np.array(values, dtype=I32_LE_NP).tobytes()
        return self._write_array(data, len(values), write_count)

    def write_i32_array_be(
        self, values: list[int] | npt.NDArray[np.uint32], write_count: bool = True
    ) -> int:
        data = np.array(values, dtype=I32_BE_NP).tobytes()
        return self._write_array(data, len(values), write_count)

    def write_i64_array(
        self, values: list[int] | npt.NDArray[np.uint64], write_count: bool = True
    ) -> int:
        return (
            self.write_i64_array_le(values, write_count)
            if self.endian == "<"
            else self.write_i64_array_be(values, write_count)
        )

    def write_i64_array_le(
        self, values: list[int] | npt.NDArray[np.uint64], write_count: bool = True
    ) -> int:
        data = np.array(values, dtype=I64_LE_NP).tobytes()
        return self._write_array(data, len(values), write_count)

    def write_i64_array_be(
        self, values: list[int] | npt.NDArray[np.uint64], write_count: bool = True
    ) -> int:
        data = np.array(values, dtype=I64_BE_NP).tobytes()
        return self._write_array(data, len(values), write_count)

    def write_f16_array(
        self,
        values: list[float] | npt.NDArray[np.uint16],
        write_count: bool = True,
    ) -> int:
        return (
            self.write_f16_array_le(values, write_count)
            if self.endian == "<"
            else self.write_f16_array_be(values, write_count)
        )

    def write_f16_array_le(
        self,
        values: list[float] | npt.NDArray[np.uint16],
        write_count: bool = True,
    ) -> int:
        data = np.array(values, dtype=F16_LE_NP).tobytes()
        return self._write_array(data, len(values), write_count)

    def write_f16_array_be(
        self,
        values: list[float] | npt.NDArray[np.uint16],
        write_count: bool = True,
    ) -> int:
        data = np.array(values, dtype=F16_BE_NP).tobytes()
        return self._write_array(data, len(values), write_count)

    def write_f32_array(
        self,
        values: list[float] | npt.NDArray[np.uint32],
        write_count: bool = True,
    ) -> int:
        return (
            self.write_f32_array_le(values, write_count)
            if self.endian == "<"
            else self.write_f32_array_be(values, write_count)
        )

    def write_f32_array_le(
        self,
        values: list[float] | npt.NDArray[np.uint32],
        write_count: bool = True,
    ) -> int:
        data = np.array(values, dtype=F32_LE_NP).tobytes()
        return self._write_array(data, len(values), write_count)

    def write_f32_array_be(
        self,
        values: list[float] | npt.NDArray[np.uint32],
        write_count: bool = True,
    ) -> int:
        data = np.array(values, dtype=F32_BE_NP).tobytes()
        return self._write_array(data, len(values), write_count)

    def write_f64_array(
        self,
        values: list[float] | npt.NDArray[np.uint64],
        write_count: bool = True,
    ) -> int:
        return (
            self.write_f64_array_le(values, write_count)
            if self.endian == "<"
            else self.write_f64_array_be(values, write_count)
        )

    def write_f64_array_le(
        self,
        values: list[float] | npt.NDArray[np.uint64],
        write_count: bool = True,
    ) -> int:
        data = np.array(values, dtype=F64_LE_NP).tobytes()
        return self._write_array(data, len(values), write_count)

    def write_f64_array_be(
        self,
        values: list[float] | npt.NDArray[np.uint64],
        write_count: bool = True,
    ) -> int:
        data = np.array(values, dtype=F64_BE_NP).tobytes()
        return self._write_array(data, len(values), write_count)
