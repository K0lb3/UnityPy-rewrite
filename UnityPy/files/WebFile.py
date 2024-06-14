from typing import Optional, cast, Literal, List, Tuple, Self

import brotli
import gzip

from ..streams import EndianBinaryWriter

from .File import ContainerFile, DirectoryInfo, parseable_filetype
from ..streams.EndianBinaryReader import EndianBinaryReader, EndianBinaryReaderMemory

GZIP_MAGIC: bytes = b"\x1f\x8b"
BROTLI_MAGIC: bytes = b"brotli"

UNITY_WEB_DATA_SIGNATURE = "UnityWebData1.0"
TCompressionType = Literal["none", "gzip", "brotli"]


@parseable_filetype
class WebFile(ContainerFile):
    """A package which can hold other WebFiles, Bundles and SerialiedFiles.
    It may be compressed via gzip or brotli.

    files -- list of all files in the WebFile
    """

    signature: str
    compression: TCompressionType

    @classmethod
    def probe(cls, reader: EndianBinaryReader) -> bool:
        start_pos = reader.tell()
        if reader.read_bytes(2) == GZIP_MAGIC:
            return True

        reader.seek(0x20)
        if reader.read_bytes(6) == BROTLI_MAGIC:
            return True

        reader.seek(start_pos)
        if reader.read_bytes(12) == b"UnityWebData":
            return True

        return False

    def parse(self, reader: Optional[EndianBinaryReader] = None) -> Self:
        reader = self._opt_get_set_reader(reader)

        # check compression
        magic = reader.read_bytes(2)
        reader.seek(0)

        if magic == GZIP_MAGIC:
            self.compression = "gzip"
            compressed_data = reader.get_bytes()
            decompressed_data = gzip.decompress(compressed_data)
            reader = EndianBinaryReaderMemory(decompressed_data, endian="<")
        else:
            reader.seek(0x20)
            magic = reader.read_bytes(6)
            reader.seek(0)
            if BROTLI_MAGIC == magic:
                self.compression = "brotli"
                compressed_data = reader.read(reader.length)
                # no type hint for brotli.decompress
                decompressed_data = cast(bytes, brotli.decompress(compressed_data))  # type: ignore
                reader = EndianBinaryReaderMemory(decompressed_data, endian="<")
            else:
                self.compression = "none"
                reader.endian = "<"

        # signature check
        self.signature = reader.read_cstring()
        if self.signature != UNITY_WEB_DATA_SIGNATURE:
            raise NotImplementedError(f"WebFile - {self.signature}")

        # read header -> contains file headers
        file_header_end = reader.read_i32()
        self.directory_infos = []
        while reader.tell() < file_header_end:
            self.directory_infos.append(
                DirectoryInfo(
                    offset=reader.read_i32(),
                    size=reader.read_i32(),
                    path=reader.read_string(),
                )
            )
        self.directory_reader = reader
        return self

    def dump(self, writer: Optional[EndianBinaryWriter] = None) -> EndianBinaryWriter:
        if writer is None:
            writer = EndianBinaryWriter(endian="<")
        else:
            raise NotImplementedError("WebFile - dump with writer")

        # write empty header to not having to keep the dumped files in memory
        header_length = (
            # signature - ending with \0
            len(UNITY_WEB_DATA_SIGNATURE)
            + 1
            # file header end
            + 4
            # directory infos
            + sum(
                # 4 - offset, 4 - size, 4 - string length, len(path) - string
                12 + len(child.path)
                for child in self.childs
            )
        )
        start_offset = writer.tell()
        writer.write_bytes(b"\0" * header_length)

        child_offset_sizes: List[Tuple[int, int]] = []
        for child in self.childs:
            child_data = child.dump().get_bytes()
            child_offset_sizes.append((writer.tell(), len(child_data)))
            writer.write_bytes(child_data)

        # write header
        writer.seek(start_offset)
        writer.write_cstring(UNITY_WEB_DATA_SIGNATURE)
        writer.write_i32(writer.tell() + header_length)
        for child, (offset, size) in zip(self.childs, child_offset_sizes):
            writer.write_i32(offset)
            writer.write_i32(size)
            writer.write_string(child.path)

        writer.seek(0, 2)

        if self.compression == "gzip":
            compressed_data = gzip.compress(writer.get_bytes())
            writer = EndianBinaryWriter(endian="<")
            writer.write_bytes(compressed_data)
        elif self.compression == "brotli":
            compressed_data = cast(bytes, brotli.compress(writer.get_bytes()))  # type: ignore
            writer = EndianBinaryWriter(endian="<")
            writer.write_bytes(compressed_data)

        return writer
