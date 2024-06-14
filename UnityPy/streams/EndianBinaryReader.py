from __future__ import annotations
from abc import ABCMeta, abstractmethod
from io import UnsupportedOperation
import struct
from typing import Any, BinaryIO, ByteString, Optional

from attrs import define
import numpy as np
import numpy.typing as npt

from .structs import *


@define
class EndianBinaryReader(metaclass=ABCMeta):
    """An abstract class for reading binary streams with endian support.

    Attributes
    ----------
    endian : str
        The endian of the stream.
        Either ">" for big endian or "<" for little endian.
    length : int
        The length of the stream.
    offset : int
        The offset of the stream.
    """

    endian: str
    length: int
    offset: int

    @abstractmethod
    def __init__(
        self,
        inp: Any,
        endian: str = ">",
        offset: int = 0,
        length: int = -1,
    ) -> None:
        """Initializes the stream.

        Args:
            inp (Any): The input for the specific EndianBinaryReader implementation.
            endian (str, optional): Endianess of the stream. Defaults to ">".
            length (int, optional): Length of the stream. If -1, the length is determined via the input. Defaults to -1.
            offset (int], optional): Offset of the stream. Defaults to 0.

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError()

    @abstractmethod
    def tell(self) -> int: ...

    @abstractmethod
    def seek(self, offset: int, whence: int = 0) -> int: ...

    @abstractmethod
    def create_sub_reader(self, offset: int, length: int) -> EndianBinaryReader:
        """Creates a sub stream of the current stream that shares the same stream base,
        but only acts on a specific part of the stream and has its own endianness.

        Args:
            offset (int)
            length (int)

        Returns:
            EndianBinaryReader
        """
        ...

    @abstractmethod
    def get_bytes(self) -> ByteString: ...

    @abstractmethod
    def read(self, count: int) -> ByteString: ...

    def unpack(self, format: str | bytes | struct.Struct) -> Any:
        if not isinstance(format, struct.Struct):
            format = struct.Struct(f"{self.endian}{format}")
        return format.unpack(self.read(format.size))[0]

    def unpack_iter(self, format: str | bytes | struct.Struct) -> Any:
        if not isinstance(format, struct.Struct):
            format = struct.Struct(f"{self.endian}{format}")
        while True:
            try:
                yield format.unpack(self.read(format.size))[0]
            except EOFError:
                break

    def unpack_array(
        self, format: str | bytes | struct.Struct, count: Optional[int] = None
    ) -> list[Any]:
        if count is None:
            count = self.read_count()
        if not isinstance(format, struct.Struct):
            format = struct.Struct(f"{self.endian}{format}")
        return list(format.iter_unpack(self.read(format.size * count)))

    def read_strict(self, count: int) -> ByteString:
        v = self.read(count)
        if len(v) != count:
            raise EOFError()
        return v

    def read_count(self) -> int:
        return self.read_u32()

    def read_bool(self) -> bool:
        return BOOL.unpack(self.read_strict(1))[0]

    def read_u8(self) -> int:
        return U8.unpack(self.read_strict(1))[0]

    def read_u16(self) -> int:
        return self.read_u16_le() if self.endian == "<" else self.read_u16_be()

    def read_u16_le(self) -> int:
        return U16_LE.unpack(self.read_strict(2))[0]

    def read_u16_be(self) -> int:
        return U16_BE.unpack(self.read_strict(2))[0]

    def read_u32(self) -> int:
        return self.read_u32_le() if self.endian == "<" else self.read_u32_be()

    def read_u32_le(self) -> int:
        return U32_LE.unpack(self.read_strict(4))[0]

    def read_u32_be(self) -> int:
        return U32_BE.unpack(self.read_strict(4))[0]

    def read_u64(self) -> int:
        return self.read_u64_le() if self.endian == "<" else self.read_u64_be()

    def read_u64_le(self) -> int:
        return U64_LE.unpack(self.read_strict(8))[0]

    def read_u64_be(self) -> int:
        return U64_BE.unpack(self.read_strict(8))[0]

    def read_i8(self) -> int:
        return I8.unpack(self.read_strict(1))[0]

    def read_i16(self) -> int:
        return self.read_i16_le() if self.endian == "<" else self.read_i16_be()

    def read_i16_le(self) -> int:
        return I16_LE.unpack(self.read_strict(2))[0]

    def read_i16_be(self) -> int:
        return I16_BE.unpack(self.read_strict(2))[0]

    def read_i32(self) -> int:
        return self.read_i32_le() if self.endian == "<" else self.read_i32_be()

    def read_i32_le(self) -> int:
        return I32_LE.unpack(self.read_strict(4))[0]

    def read_i32_be(self) -> int:
        return I32_BE.unpack(self.read_strict(4))[0]

    def read_i64(self) -> int:
        return self.read_i64_le() if self.endian == "<" else self.read_i64_be()

    def read_i64_le(self) -> int:
        return I64_LE.unpack(self.read_strict(8))[0]

    def read_i64_be(self) -> int:
        return I64_BE.unpack(self.read_strict(8))[0]

    def read_f16(self) -> float:
        return self.read_f16_le() if self.endian == "<" else self.read_f16_be()

    def read_f16_le(self) -> float:
        return F16_LE.unpack(self.read_strict(2))[0]

    def read_f16_be(self) -> float:
        return F16_BE.unpack(self.read_strict(2))[0]

    def read_f32(self) -> float:
        return self.read_f32_le() if self.endian == "<" else self.read_f32_be()

    def read_f32_le(self) -> float:
        return F32_LE.unpack(self.read_strict(4))[0]

    def read_f32_be(self) -> float:
        return F32_BE.unpack(self.read_strict(4))[0]

    def read_f64(self) -> float:
        return self.read_f64_le() if self.endian == "<" else self.read_f64_be()

    def read_f64_le(self) -> float:
        return F64_LE.unpack(self.read_strict(8))[0]

    def read_f64_be(self) -> float:
        return F64_BE.unpack(self.read_strict(8))[0]

    def read_bytes(self, count: Optional[int] = None) -> bytes:
        if count is None:
            count = self.read_count()
        return bytes(self.read_strict(count))

    def read_string(self, count: Optional[int] = None) -> str:
        return self.read_bytes(count).decode("utf8", errors="surrogateescape")

    def read_aligned_string(self, count: Optional[int] = None) -> str:
        string = self.read_string(count)
        self.align_stream()
        return string

    def read_cstring(self, max_count: int = 32767) -> str:
        string: list[ByteString] = []
        read = 1
        v = self.read(1)

        while v and v != b"\x00" and read < max_count:
            string.append(v)
            v = self.read(1)
            read += 1

        return b"".join(string).decode("utf8", errors="surrogateescape")

    def align_stream(self, alignment: int = 4) -> int:
        pos = self.tell()
        pos += (alignment - pos % alignment) % alignment
        self.seek(pos)
        return pos

    def try_align_stream(self, alignment: int = 4) -> int:
        # only aligns if the aligned section is empty
        pos = self.tell()
        to_read = (alignment - pos % alignment) % alignment
        data = self.read(to_read)
        if any(data):
            self.seek(pos)
        return pos

    def read_bool_array(self, count: Optional[int] = None) -> npt.NDArray[np.bool_]:
        if count is None:
            count = self.read_count()
        return np.frombuffer(self.read_strict(count), BOOL_NP)

    def read_u8_array(self, count: Optional[int] = None) -> npt.NDArray[np.uint8]:
        if count is None:
            count = self.read_count()
        return np.frombuffer(self.read_strict(count), U8_NP)

    def read_u16_array(self, count: Optional[int] = None) -> npt.NDArray[np.uint16]:
        return (
            self.read_u16_array_le(count)
            if self.endian == "<"
            else self.read_u16_array_be(count)
        )

    def read_u16_array_le(self, count: Optional[int] = None) -> npt.NDArray[np.uint16]:
        if count is None:
            count = self.read_count()
        return np.frombuffer(self.read_strict(count * 2), U16_LE_NP)

    def read_u16_array_be(self, count: Optional[int] = None) -> npt.NDArray[np.uint16]:
        if count is None:
            count = self.read_count()
        return np.frombuffer(self.read_strict(count * 2), U16_BE_NP)

    def read_u32_array(self, count: Optional[int] = None) -> npt.NDArray[np.uint32]:
        return (
            self.read_u32_array_le(count)
            if self.endian == "<"
            else self.read_u32_array_be(count)
        )

    def read_u32_array_le(self, count: Optional[int] = None) -> npt.NDArray[np.uint32]:
        if count is None:
            count = self.read_count()
        return np.frombuffer(self.read_strict(count * 4), U32_LE_NP)

    def read_u32_array_be(self, count: Optional[int] = None) -> npt.NDArray[np.uint32]:
        if count is None:
            count = self.read_count()
        return np.frombuffer(self.read_strict(count * 4), U32_BE_NP)

    def read_u64_array(self, count: Optional[int] = None) -> npt.NDArray[np.uint64]:
        return (
            self.read_u64_array_le(count)
            if self.endian == "<"
            else self.read_u64_array_be(count)
        )

    def read_u64_array_le(self, count: Optional[int] = None) -> npt.NDArray[np.uint64]:
        if count is None:
            count = self.read_count()
        return np.frombuffer(self.read_strict(count * 8), U64_LE_NP)

    def read_u64_array_be(self, count: Optional[int] = None) -> npt.NDArray[np.uint64]:
        if count is None:
            count = self.read_count()
        return np.frombuffer(self.read_strict(count * 8), U64_BE_NP)

    def read_i8_array(self, count: Optional[int]) -> npt.NDArray[np.int8]:
        if count is None:
            count = self.read_count()
        return np.frombuffer(self.read_strict(count), I8_NP)

    def read_i16_array(self, count: Optional[int] = None) -> npt.NDArray[np.uint16]:
        return (
            self.read_i16_array_le(count)
            if self.endian == "<"
            else self.read_i16_array_be(count)
        )

    def read_i16_array_le(self, count: Optional[int] = None) -> npt.NDArray[np.uint16]:
        if count is None:
            count = self.read_count()
        return np.frombuffer(self.read_strict(count * 2), I16_LE_NP)

    def read_i16_array_be(self, count: Optional[int] = None) -> npt.NDArray[np.uint16]:
        if count is None:
            count = self.read_count()
        return np.frombuffer(self.read_strict(count * 2), I16_BE_NP)

    def read_i32_array(self, count: Optional[int] = None) -> npt.NDArray[np.uint32]:
        return (
            self.read_i32_array_le(count)
            if self.endian == "<"
            else self.read_i32_array_be(count)
        )

    def read_i32_array_le(self, count: Optional[int] = None) -> npt.NDArray[np.uint32]:
        if count is None:
            count = self.read_count()
        return np.frombuffer(self.read_strict(count * 4), I32_LE_NP)

    def read_i32_array_be(self, count: Optional[int] = None) -> npt.NDArray[np.uint32]:
        if count is None:
            count = self.read_count()
        return np.frombuffer(self.read_strict(count * 4), I32_BE_NP)

    def read_i64_array(self, count: Optional[int] = None) -> npt.NDArray[np.uint64]:
        return (
            self.read_i64_array_le(count)
            if self.endian == "<"
            else self.read_i64_array_be(count)
        )

    def read_i64_array_le(self, count: Optional[int] = None) -> npt.NDArray[np.uint64]:
        if count is None:
            count = self.read_count()
        return np.frombuffer(self.read_strict(count * 8), I64_LE_NP)

    def read_i64_array_be(self, count: Optional[int] = None) -> npt.NDArray[np.uint64]:
        if count is None:
            count = self.read_count()
        return np.frombuffer(self.read_strict(count * 8), I64_BE_NP)

    def read_f16_array(self, count: Optional[int] = None) -> npt.NDArray[np.uint16]:
        return (
            self.read_f16_array_le(count)
            if self.endian == "<"
            else self.read_f16_array_be(count)
        )

    def read_f16_array_le(self, count: Optional[int] = None) -> npt.NDArray[np.uint16]:
        if count is None:
            count = self.read_count()
        return np.frombuffer(self.read_strict(count * 2), F16_LE_NP)

    def read_f16_array_be(self, count: Optional[int] = None) -> npt.NDArray[np.uint16]:
        if count is None:
            count = self.read_count()
        return np.frombuffer(self.read_strict(count * 2), F16_BE_NP)

    def read_f32_array(self, count: Optional[int] = None) -> npt.NDArray[np.uint32]:
        return (
            self.read_f32_array_le(count)
            if self.endian == "<"
            else self.read_f32_array_be(count)
        )

    def read_f32_array_le(self, count: Optional[int] = None) -> npt.NDArray[np.uint32]:
        if count is None:
            count = self.read_count()
        return np.frombuffer(self.read_strict(count * 4), F32_LE_NP)

    def read_f32_array_be(self, count: Optional[int] = None) -> npt.NDArray[np.uint32]:
        if count is None:
            count = self.read_count()
        return np.frombuffer(self.read_strict(count * 4), F32_BE_NP)

    def read_f64_array(self, count: Optional[int] = None) -> npt.NDArray[np.uint64]:
        return (
            self.read_f64_array_le(count)
            if self.endian == "<"
            else self.read_f64_array_be(count)
        )

    def read_f64_array_le(self, count: Optional[int] = None) -> npt.NDArray[np.uint64]:
        if count is None:
            count = self.read_count()
        return np.frombuffer(self.read_strict(count * 8), F64_LE_NP)

    def read_f64_array_be(self, count: Optional[int] = None) -> npt.NDArray[np.uint64]:
        if count is None:
            count = self.read_count()
        return np.frombuffer(self.read_strict(count * 8), F64_BE_NP)

    def real_offset(self) -> int:
        """Returns offset in the underlying file.
        (Not working with unpacked streams.)
        """
        return self.offset + self.tell()

    def read_the_rest(self, obj_start: int, obj_size: int) -> bytes:
        """Returns the rest of the current reader bytes."""
        return self.read_bytes(obj_size - (self.tell() - obj_start))


@define
class EndianBinaryReaderStream(EndianBinaryReader):
    stream: BinaryIO
    endian: str
    length: int
    offset: int

    def __init__(
        self, inp: BinaryIO, endian: str = ">", offset: int = 0, length: int = -1
    ):
        self.stream = inp
        self.endian = endian
        self.offset = offset
        if length == -1:
            pos = inp.tell()
            inp.seek(0, 2)
            length = inp.tell() - offset
            inp.seek(pos)
        self.length = length

    def get_bytes(self) -> ByteString:
        pos = self.stream.tell()
        self.stream.seek(self.offset)
        data = self.stream.read(self.length)
        self.stream.seek(pos)
        return data

    def read(self, count: int) -> ByteString:
        return self.stream.read(count)

    def tell(self) -> int:
        return self.stream.tell() - self.offset

    def seek(self, offset: int, whence: int = 0) -> int:
        # io.BytesIO is inconsistent with file io
        if whence == 2 and offset > 0:
            raise UnsupportedOperation("can't do nonzero end-relative seeks")
        return self.stream.seek(offset + self.offset, whence)

    def create_sub_reader(self, offset: int, length: int) -> EndianBinaryReader:
        return EndianBinaryReaderStream(
            self.stream, self.endian, self.offset + offset, length
        )


@define
class EndianBinaryReaderMemory(EndianBinaryReader):
    view: ByteString
    endian: str
    position: int
    length: int
    offset: int

    def __init__(
        self, inp: ByteString, endian: str = ">", offset: int = 0, length: int = -1
    ):
        self.view = inp
        self.endian = endian
        self.offset = offset
        self.position = 0
        self.length = len(inp) - offset if length == -1 else length

    def get_bytes(self) -> ByteString:
        return self.view[self.offset : self.offset + self.length]

    def read(self, count: int) -> ByteString:
        start = self.position
        self.position += count
        return self.view[start : self.position]

    def tell(self) -> int:
        return self.position - self.offset

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            # absolute
            self.position = offset + self.offset
        elif whence == 1:
            # relative
            self.position += offset
        elif whence == 2:
            # relative to end
            if offset > 0:
                # error thrown in file io
                raise UnsupportedOperation("can't do nonzero end-relative seeks")
            self.position = self.length + self.offset - offset
        else:
            raise ValueError(f"Invalid whence: {whence}")
        return self.position

    def create_sub_reader(self, offset: int, length: int) -> EndianBinaryReader:
        return EndianBinaryReaderMemory(
            self.view, self.endian, self.offset + offset, length
        )
