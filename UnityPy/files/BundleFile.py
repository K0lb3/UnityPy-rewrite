# TODO: implement encryption for saving files
from abc import ABC, ABCMeta
from enum import IntEnum, IntFlag
from typing import Union, Optional, List, Self, Annotated
from .File import ContainerFile, parseable_filetype, DirectoryInfo
from ..typetree.Tpk import UnityVersion
from ..streams import EndianBinaryReader, EndianBinaryWriter
from ..streams.EndianBinaryReader import EndianBinaryReaderMemory
from ..streams.CompressedStreamReader import (
    BlockInfo,
    CompressedBlockEndianBinaryReader,
)


class CompressionFlags(IntEnum):
    NONE = 0
    LZMA = 1
    LZ4 = 2
    LZ4HC = 3
    LZHAM = 4


class ArchiveFlagsOld(IntFlag):
    CompressionTypeMask = 0x3F
    BlocksAndDirectoryInfoCombined = 0x40
    BlocksInfoAtTheEnd = 0x80
    OldWebPluginCompatibility = 0x100
    UsesAssetBundleEncryption = 0x200


class ArchiveFlags(IntFlag):
    CompressionTypeMask = 0x3F
    BlocksAndDirectoryInfoCombined = 0x40
    BlocksInfoAtTheEnd = 0x80
    OldWebPluginCompatibility = 0x100
    BlockInfoNeedPaddingAtStart = 0x200
    UsesAssetBundleEncryption = 0x400


class BundleFile(ContainerFile, ABC, metaclass=ABCMeta):
    signature: str
    stream_version: int
    unity_version: str
    minimum_revision: str
    block_infos = List[BlockInfo]
    block_reader: EndianBinaryReader

    def parse(self, reader: Optional[EndianBinaryReader] = None) -> Self:
        reader = self._opt_get_set_reader(reader)

        self.signature = reader.read_cstring()
        self.stream_version = reader.read_u32()
        if self.stream_version > 0x1000000:
            reader.seek(reader.tell() - 4)
            reader.endian = "<" if reader.endian == ">" else ">"
            self.stream_version = reader.read_u32()

        self.unity_version = reader.read_cstring()
        self.minimum_revision = reader.read_cstring()
        return self

    # def get_objects(self) -> List:
    #     raise NotImplementedError("BundleFile - get_objects")

    # def get_serialized_files(self) -> List:
    #     return [
    #         child for child in self.childs if not isinstance(child, EndianBinaryReader)
    #     ]


@parseable_filetype
class BundleFileArchive(BundleFile):
    @classmethod
    def probe(cls, reader: EndianBinaryReader) -> bool:
        signature = reader.read_cstring()

        return signature == "UnityArchive"

    def parse(self, reader: Optional[EndianBinaryReader] = None) -> Self:
        super().parse(reader)
        raise NotImplementedError("BundleFile - UnityArchive")

    def dump(self, writer: Optional[EndianBinaryWriter] = None) -> EndianBinaryWriter:
        raise NotImplementedError("BundleFile - UnityArchive")


@parseable_filetype
class BundleFileFS(BundleFile):
    size: int
    dataflags: Union[ArchiveFlags, ArchiveFlagsOld]
    decompressed_data_hash: Annotated[bytes, 16]
    uses_block_alignment: bool = False
    # decryptor: Optional[ArchiveStorageManager.ArchiveStorageDecryptor] = None

    @classmethod
    def probe(cls, reader: EndianBinaryReader) -> bool:
        signature = reader.read_cstring()
        version = reader.read_u32_be()

        return signature == "UnityFS" or (signature == "UnityRaw" and version == 6)

    def parse(self, reader: Optional[EndianBinaryReader] = None) -> Self:
        reader = self._opt_get_set_reader(reader)

        super().parse(reader)
        self.size = reader.read_i64()

        # header
        compressedSize = reader.read_u32()
        uncompressedSize = reader.read_u32()
        dataflags = reader.read_u32()

        # https://issuetracker.unity3d.com/issues/files-within-assetbundles-do-not-start-on-aligned-boundaries-breaking-patching-on-nintendo-switch
        # Unity CN introduced encryption before the alignment fix was introduced.
        # Unity CN used the same flag for the encryption as later on the alignment fix,
        # so we have to check the version to determine the correct flag set.
        version = UnityVersion.fromString(self.minimum_revision)
        if (
            # < 2000, < 2020.3.34, < 2021.3.2, < 2022.1.1
            version < UnityVersion.fromList(2020)
            or (version.major == 2020 and version < UnityVersion.fromList(2020, 3, 34))
            or (version.major == 2021 and version < UnityVersion.fromList(2021, 3, 2))
            or (version.major == 2022 and version < UnityVersion.fromList(2022, 1, 1))
        ):
            self.dataflags = ArchiveFlagsOld(dataflags)
        else:
            self.dataflags = ArchiveFlags(dataflags)

        # if self.dataflags & self.dataflags.UsesAssetBundleEncryption:
        # self.decryptor = ArchiveStorageManager.ArchiveStorageDecryptor(reader)

        # check if we need to align the reader
        # - align to 16 bytes and check if all are 0
        # - if not, reset the reader to the previous position
        if self.stream_version >= 7:
            reader.align_stream(16)
            self.uses_block_alignment = True
        elif version >= UnityVersion.fromList(2019, 4):
            pre_align = reader.tell()
            align_data = reader.read((16 - pre_align % 16) % 16)
            if any(align_data):
                reader.seek(pre_align)
            else:
                self.uses_block_alignment = True

        seek_back = -1
        if (
            self.dataflags & ArchiveFlags.BlocksInfoAtTheEnd
        ):  # kArchiveBlocksInfoAtTheEnd
            seek_back = reader.tell()
            reader.seek(reader.length - compressedSize)
        # else:  # 0x40 kArchiveBlocksAndDirectoryInfoCombined

        blocksInfoReader = CompressedBlockEndianBinaryReader(
            (
                reader,
                [
                    BlockInfo(
                        compressed_size=compressedSize,
                        decompressed_size=uncompressedSize,
                        flags=self.dataflags & self.dataflags.CompressionTypeMask,
                        offset=reader.tell(),
                    )
                ],
            ),
            endian=reader.endian,
        )

        self.decompressed_data_hash = blocksInfoReader.read_bytes(16)

        blocksInfoCount = blocksInfoReader.read_i32()
        assert blocksInfoCount > 0, "blocksInfoCount <= 0"
        block_offset = 0

        def offset(compressed_size: int):
            nonlocal block_offset
            old_offset = block_offset
            block_offset += compressed_size
            return old_offset

        self.block_infos = [
            BlockInfo(
                flags,
                compressed_size,
                decompressed_size,
                offset(compressed_size),
            )
            for (
                decompressed_size,
                compressed_size,
                flags,
            ) in blocksInfoReader.unpack_array("IIH", blocksInfoCount)
        ]

        nodesCount = blocksInfoReader.read_i32()
        self.directory_infos = [
            DirectoryInfo(
                offset=blocksInfoReader.read_i64(),  # offset
                size=blocksInfoReader.read_i64(),  # size
                flags=blocksInfoReader.read_u32(),  # flags
                path=blocksInfoReader.read_cstring(),  # path
            )
            for _ in range(nodesCount)
        ]

        if seek_back != -1:
            reader.seek(seek_back)

        if (
            isinstance(self.dataflags, ArchiveFlags)
            and self.dataflags & ArchiveFlags.BlockInfoNeedPaddingAtStart
        ):
            reader.align_stream(16)

        self.directory_reader = CompressedBlockEndianBinaryReader(
            (reader.create_sub_reader(reader.tell(), -1), self.block_infos)
        )
        # TODO: still some issues with the compressed reader,
        # so we directly decompress everything and create a new reader around it
        self.directory_reader = EndianBinaryReaderMemory(
            self.directory_reader.read(self.directory_reader.length),
            endian=self.directory_reader.endian,
        )
        return self

    def dump(self, writer: Optional[EndianBinaryWriter] = None) -> EndianBinaryWriter:
        raise NotImplementedError("BundleFile - UnityFS")


@parseable_filetype
class BundleFileWeb(BundleFile):
    byteStart: int
    numberOfLevelsToDownloadBeforeStreaming: int
    hash: Optional[Annotated[bytes, 16]]
    crc: Optional[int]
    completeFileSize = Optional[int]
    fileInfoHeaderSize = Optional[int]

    @classmethod
    def probe(cls, reader: EndianBinaryReader) -> bool:
        signature = reader.read_cstring()
        version = reader.read_u32_be()

        return signature == "UnityWeb" or (signature == "UnityRaw" and version != 6)

    def parse(self, reader: Optional[EndianBinaryReader] = None) -> Self:
        reader = self._opt_get_set_reader(reader)

        super().parse(reader)

        version = self.stream_version
        if version >= 4:
            self.hash = reader.read_bytes(16)
            self.crc = reader.read_u32()

        self.byteStart = reader.read_u32()
        headerSize = reader.read_u32()
        self.numberOfLevelsToDownloadBeforeStreaming = reader.read_u32()
        levelCount = reader.read_i32()
        self.block_infos = [
            BlockInfo(
                compressed_size=reader.read_u32(),  # compressedSize
                decompressed_size=reader.read_u32(),  # uncompressedSize
                flags=0,
                offset=0,
            )
            for _ in range(levelCount)
        ]
        block_offset = 0
        for block in self.block_infos:
            block.offset = block_offset
            block_offset += block.compressed_size

        if version >= 2:
            self.completeFileSize = reader.read_u32()

        if version >= 3:
            self.fileInfoHeaderSize = reader.read_u32()

        reader.seek(headerSize)

        directory_reader = self.directory_reader = CompressedBlockEndianBinaryReader(
            (reader, self.block_infos), offset=reader.tell()
        )
        nodesCount = directory_reader.read_i32()
        self.directory_infos = [
            DirectoryInfo(
                path=directory_reader.read_cstring(),  # path
                offset=directory_reader.read_u32(),  # offset
                size=directory_reader.read_u32(),  # size
            )
            for _ in range(nodesCount)
        ]
        return self

    def dump(self, writer: Optional[EndianBinaryWriter] = None) -> EndianBinaryWriter:
        raise NotImplementedError("BundleFile - UnityFS")
