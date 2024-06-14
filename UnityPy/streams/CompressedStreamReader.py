from enum import IntEnum
from struct import unpack
from typing import ByteString, List, Tuple

from attrs import define

from .EndianBinaryReader import EndianBinaryReader, EndianBinaryReaderMemory

import lzma
import lz4.block


class CompressionFlags(IntEnum):
    NONE = 0
    LZMA = 1
    LZ4 = 2
    LZ4HC = 3
    LZHAM = 4


@define
class BlockInfo:
    flags: int
    compressed_size: int
    decompressed_size: int
    offset: int


@define
class CompressedBlockEndianBinaryReader(EndianBinaryReader):
    compressed_stream: EndianBinaryReader
    block_infos: List[BlockInfo]
    endian: str
    length: int
    offset: int
    current_decompressed_stream: EndianBinaryReader
    current_block: BlockInfo
    current_block_index: int

    def __init__(
        self,
        inp: Tuple[EndianBinaryReader, List[BlockInfo]],
        endian: str = "<",
        offset: int = 0,
        length: int = -1,
    ):
        self.compressed_stream, self.block_infos = inp
        self.endian = endian
        self.offset = offset
        self.length = length
        if length == -1:
            self.length = sum(
                block_info.compressed_size for block_info in self.block_infos
            )
        self.set_active_block(0)

    def set_active_block(self, index: int):
        if index >= len(self.block_infos):
            raise EOFError("End of file reached")
        self.current_block = self.block_infos[index]
        self.current_block_index = index
        self.compressed_stream.seek(self.current_block.offset)
        compressed_data = self.compressed_stream.read(
            self.current_block.compressed_size
        )
        decompressed_data = decompress_block(compressed_data, self.current_block)
        self.current_decompressed_stream = EndianBinaryReaderMemory(
            decompressed_data, self.endian
        )

    def tell(self) -> int:
        return (
            self.current_block.offset
            + self.current_decompressed_stream.tell()
            - self.offset
        )

    def read(self, count: int) -> ByteString:
        data = self.current_decompressed_stream.read(count)
        if len(data) < count:
            self.set_active_block(self.current_block_index + 1)
            data = b"".join([data, self.read(count - len(data))])
        return data

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            offset += self.offset
        elif whence == 1:
            offset += self.tell()
        elif whence == 2:
            offset += self.length + self.offset
            if offset > self.length + self.offset:
                raise ValueError("Can't seek beyond end of file")
        if (
            offset < self.current_block.offset
            or offset > self.current_block.offset + self.current_block.decompressed_size
        ):
            for i, block_info in enumerate(self.block_infos):
                if block_info.offset > offset:
                    self.set_active_block(i - 1)
                    break
            else:
                self.set_active_block(len(self.block_infos) - 1)

        rel_offset = offset - self.current_block.offset
        
        if rel_offset > self.current_block.decompressed_size:
            raise EOFError("Trying to seek beyond end of the compressed blocks")
        
        self.current_decompressed_stream.seek(rel_offset)
        return rel_offset

    def create_sub_reader(self, offset: int, length: int) -> EndianBinaryReader:
        return CompressedBlockEndianBinaryReader(
            (self.compressed_stream, self.block_infos),
            self.endian,
            offset + self.offset,
            length,
        )

    def get_bytes(self) -> bytes | bytearray | memoryview:
        pos = self.tell()
        self.seek(0)
        data = self.read(self.length)
        self.seek(pos)
        return data


def decompress_block(
    compressed_data: ByteString, block_info: BlockInfo
) -> ByteString:
    compression_flags = block_info.flags & 0x3F
    match compression_flags:
        case CompressionFlags.NONE:
            return compressed_data
        case CompressionFlags.LZMA:
            props, dict_size = unpack("<BI", compressed_data[:5])
            lc = props % 9
            props = props // 9
            pb = props // 5
            lp = props % 5
            dec = lzma.LZMADecompressor(
                format=lzma.FORMAT_RAW,
                filters=[
                    {
                        "id": lzma.FILTER_LZMA1,
                        "dict_size": dict_size,
                        "lc": lc,
                        "lp": lp,
                        "pb": pb,
                    }
                ],
            )
            return dec.decompress(compressed_data[5:])
        case CompressionFlags.LZ4 | CompressionFlags.LZ4HC:
            return lz4.block.decompress(compressed_data, block_info.decompressed_size) # type: ignore
        case CompressionFlags.LZHAM:
            raise NotImplementedError("LZHAM decompression is not implemented")
        case _:
            raise NotImplementedError(f"Unknown compression format {compression_flags}!")
