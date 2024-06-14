from __future__ import annotations
from enum import IntEnum, IntFlag
from struct import Struct
import os
from io import BytesIO
import re
from typing import List, Tuple, Any, Dict, BinaryIO

from attrs import define

from .TypeTreeNode import TypeTreeNode

NODE_CACHE: Dict[Tuple[int, UnityVersion], TypeTreeNode] = {}
COMMONSTRING_CACHE: Dict[UnityVersion | None, Dict[int, str]] = {}
_tpk_data_blob: TpkTypeTreeBlob | None = None


def get_tpktypetree() -> TpkTypeTreeBlob:
    global _tpk_data_blob
    if _tpk_data_blob is None:
        with open(
            os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "resources",
                "uncompressed.tpk",
            ),
            "rb",
        ) as f:
            blob = TpkFile(f).GetDataBlob()
            if not isinstance(blob, TpkTypeTreeBlob):
                raise ValueError("Invalid TPK file")
            _tpk_data_blob = blob

    return _tpk_data_blob


def get_typetree_node(class_id: int, version: UnityVersion) -> TypeTreeNode:
    global NODE_CACHE
    key = (class_id, version)
    if key in NODE_CACHE:
        return NODE_CACHE[key]

    class_info = (
        get_tpktypetree()
        .ClassInformation[class_id]
        .getVersionedClass(version)
    )

    node = convert_tpk_to_unitypy_node(class_info)
    NODE_CACHE[key] = node
    return node


def convert_tpk_to_unitypy_node(class_info: TpkUnityClass) -> TypeTreeNode:
    tpk_typetree = get_tpktypetree()
    fake_root = TypeTreeNode(-1, "dummy", "dummy", -1, -1, -1)
    NODES = tpk_typetree.NodeBuffer.Nodes
    stack = [(class_info.ReleaseRootNode, fake_root)]
    index = 0
    while stack:
        node_id, parent = stack.pop(0)
        if node_id is None:
            raise ValueError("No node found for class info")
        tpk_node = NODES[node_id]
        node = TypeTreeNode(
            m_ByteSize=tpk_node.ByteSize,
            m_Index=index,
            m_Version=tpk_node.Version,
            m_MetaFlag=tpk_node.MetaFlag,
            m_Level=parent.m_Level + 1,
            m_Type=tpk_typetree.StringBuffer.Strings[tpk_node.TypeName],
            m_Name=tpk_typetree.StringBuffer.Strings[tpk_node.Name],
            m_TypeFlags=tpk_node.TypeFlags,
        )
        parent.m_Children.append(node)
        stack = [(tpk_node_id, node) for tpk_node_id in tpk_node.SubNodes] + stack
        index += 1
    return fake_root.m_Children[0]


######################################################################################
#
#   Enums
#
######################################################################################


class TpkCompressionType(IntEnum):
    NONE = 0
    Lz4 = 1
    Lzma = 2
    Brotli = 3


class UnityVersionType(IntEnum):
    Alpha = 0
    Beta = 1
    China = 2
    Final = 3
    Patch = 4
    Experimental = 5


class TpkDataType(IntEnum):
    TypeTreeInformation = 0
    Collection = 1
    FileSystem = 2
    Json = 3
    ReferenceAssemblies = 4
    EngineAssets = 5

    def ToBlob(self, stream: BinaryIO) -> TpkDataBlob:
        if self.value == TpkDataType.TypeTreeInformation.value:
            return TpkTypeTreeBlob(stream)
        elif self.value == TpkDataType.Collection.value:
            return TpkCollectionBlob(stream)
        elif self.value == TpkDataType.FileSystem.value:
            return TpkFileSystemBlob(stream)
        elif self.value == TpkDataType.Json.value:
            return TpkJsonBlob(stream)
        else:
            raise Exception("Unimplemented TpkDataType -> Blob conversion")


class TpkUnityClassFlags(IntFlag):
    NONE = 0
    IsAbstract = 1
    IsSealed = 2
    IsEditorOnly = 4
    IsReleaseOnly = 8
    IsStripped = 16
    Reserved = 32
    HasEditorRootNode = 64
    HasReleaseRootNode = 128


######################################################################################
#
#   Main Class
#
######################################################################################

TPK_FILE_STRUCT: Struct = Struct("<IbbbbIII")
TPK_MAGIC_BYTES: int = 0x2A4B5054  # b"TPK*"
TPK_UNITY_CLASS_STRUCT: Struct = Struct("<HHb")
TPK_UNITY_NODE_STRUCT: Struct = Struct("<HHihbIH")
TPK_VERSION_NUMBER: int = 1


@define
class TpkFile:
    CompressionType: TpkCompressionType
    DataType: TpkDataType
    CompressedSize: int
    UncompressedSize: int
    CompressedBytes: bytes

    def __init__(self, stream: BinaryIO):
        (
            magic,
            versionNumber,
            compressionType,
            dataType,
            _,
            _,
            self.CompressedSize,
            self.UncompressedSize,
        ) = TPK_FILE_STRUCT.unpack(stream.read(TPK_FILE_STRUCT.size))
        if magic != TPK_MAGIC_BYTES:
            raise Exception("Invalid TPK magic bytes")
        if versionNumber != TPK_VERSION_NUMBER:
            raise Exception("Invalid TPK version number")
        self.CompressionType = TpkCompressionType(compressionType)
        self.DataType = TpkDataType(dataType)
        self.CompressedBytes = stream.read(self.CompressedSize)
        if len(self.CompressedBytes) != self.CompressedSize:
            raise Exception("Invalid compressed size")

    def GetDataBlob(self) -> TpkDataBlob:
        decompressed: bytes
        if self.CompressionType == TpkCompressionType.NONE:
            decompressed = self.CompressedBytes

        elif self.CompressionType == TpkCompressionType.Lz4:
            import lz4.block

            decompressed = (
                lz4.block.decompress(self.CompressedBytes, self.UncompressedSize)  # type: ignore
                is bytes
            )

        elif self.CompressionType == TpkCompressionType.Lzma:
            # import lzma

            raise Exception("LZMA compression not implemented")

        elif self.CompressionType == TpkCompressionType.Brotli:
            import brotli

            decompressed = brotli.decompress(self.CompressedBytes) is bytes  # type: ignore

        else:
            raise Exception("Invalid compression type")

        return self.DataType.ToBlob(BytesIO(decompressed))


######################################################################################
#
#   Blobs
#
######################################################################################


@define
class TpkDataBlob:
    DataType: TpkDataType

    def __init__(self, stream: BinaryIO) -> None:
        raise NotImplementedError("TpkDataBlob is an abstract class")


@define
class TpkTypeTreeBlob(TpkDataBlob):
    CreationTime: int
    Versions: List[UnityVersion]
    ClassInformation: Dict[int, TpkClassInformation]
    CommonString: TpkCommonString
    NodeBuffer: TpkUnityNodeBuffer
    StringBuffer: TpkStringBuffer
    DataType: TpkDataType = TpkDataType.TypeTreeInformation

    def __init__(self, stream: BinaryIO) -> None:
        (self.CreationTime,) = INT64.unpack(stream.read(INT64.size))
        (versionCount,) = INT32.unpack(stream.read(INT32.size))
        self.Versions = [UnityVersion.fromStream(stream) for _ in range(versionCount)]
        (classCount,) = INT32.unpack(stream.read(INT32.size))
        self.ClassInformation = {
            x.id: x for x in (TpkClassInformation(stream) for _ in range(classCount))
        }
        self.CommonString = TpkCommonString(stream)
        self.NodeBuffer = TpkUnityNodeBuffer(stream)
        self.StringBuffer = TpkStringBuffer(stream)


@define
class TpkCollectionBlob(TpkDataBlob):
    Blobs: List[Tuple[str, TpkDataBlob]]

    def __init__(self, stream: BinaryIO) -> None:
        (count,) = INT32.unpack(stream.read(INT32.size))
        self.Blobs = [
            # relativePath, data
            (
                read_string(stream),
                TpkDataType(BYTE.unpack(stream.read(1))[0]).ToBlob(stream),
            )
            for _ in range(count)
        ]


@define
class TpkFileSystemBlob(TpkDataBlob):
    Files: List[Tuple[str, bytes]]

    def __init__(self, stream: BinaryIO) -> None:
        (count,) = INT32.unpack(stream.read(INT32.size))
        self.Files = [
            # relativePath, data
            (read_string(stream), read_data(stream))
            for _ in range(count)
        ]


@define
class TpkJsonBlob(TpkDataBlob):
    Text: str
    DataType: TpkDataType = TpkDataType.Json

    def __init__(self, stream: BinaryIO) -> None:
        self.Text = read_string(stream)


######################################################################################
#
#   Unity
#
######################################################################################


class UnityVersion(int):
    # https://github.com/AssetRipper/VersionUtilities/blob/master/VersionUtilities/UnityVersion.cs
    """
    use following static methos instead of the constructor(__init__):
        UnityVersion.fromStream(stream: BinaryIO)
        UnityVersion.fromString(version: str)
        UnityVersion.fromList(major: int, minor: int, patch: int, build: int)
    """

    @staticmethod
    def fromStream(stream: BinaryIO) -> UnityVersion:
        (m_data,) = UINT64.unpack(stream.read(UINT64.size))
        return UnityVersion(m_data)

    @staticmethod
    def fromString(version: str) -> UnityVersion:
        return UnityVersion.fromList(*map(int, re.findall(r"(\d+)", version)[:4]))

    @staticmethod
    def fromList(
        major: int = 0, minor: int = 0, patch: int = 0, build: int = 0
    ) -> UnityVersion:
        return UnityVersion(major << 48 | minor << 32 | patch << 16 | build)

    @property
    def major(self) -> int:
        return (self >> 48) & 0xFFFF

    @property
    def minor(self) -> int:
        return (self >> 32) & 0xFFFF

    @property
    def build(self) -> int:
        return (self >> 16) & 0xFFFF

    @property
    def type(self) -> int:
        return UnityVersionType(self >> 8) & 0xFF

    @property
    def type_number(self) -> int:
        return self & 0xFF

    def __repr__(self) -> str:
        return f"UnityVersion {self.major}.{self.minor}.{self.build}.{self.type_number}"


@define
class TpkUnityClass:
    Name: int
    Base: int
    Flags: TpkUnityClassFlags
    EditorRootNode: int | None
    ReleaseRootNode: int | None

    def __init__(self, stream: BinaryIO) -> None:
        self.Name, self.Base, Flags = TPK_UNITY_CLASS_STRUCT.unpack(
            stream.read(TPK_UNITY_CLASS_STRUCT.size)
        )
        self.Flags = TpkUnityClassFlags(Flags)
        self.EditorRootNode = self.ReleaseRootNode = None
        if self.Flags & TpkUnityClassFlags.HasEditorRootNode:
            (self.EditorRootNode,) = UINT16.unpack(stream.read(UINT16.size))
        if self.Flags & TpkUnityClassFlags.HasReleaseRootNode:
            (self.ReleaseRootNode,) = UINT16.unpack(stream.read(UINT16.size))


@define
class TpkClassInformation:
    id: int
    Classes: List[Tuple[UnityVersion, TpkUnityClass | None]]

    def __init__(self, stream: BinaryIO) -> None:
        (self.id,) = INT32.unpack(stream.read(INT32.size))
        (count,) = INT32.unpack(stream.read(INT32.size))
        self.Classes = [
            (
                UnityVersion.fromStream(stream),
                TpkUnityClass(stream) if stream.read(1)[0] else None,
            )
            for _ in range(count)
        ]

    def getVersionedClass(self, version: UnityVersion) -> TpkUnityClass:
        return get_item_for_version(version, self.Classes)


@define
class TpkUnityNodeBuffer:
    Nodes: List[TpkUnityNode]

    def __init__(self, stream: BinaryIO) -> None:
        (count,) = INT32.unpack(stream.read(INT32.size))
        self.Nodes = [TpkUnityNode(stream) for _ in range(count)]

    def __getitem__(self, index: int) -> TpkUnityNode:
        return self.Nodes[index]


@define
class TpkUnityNode:
    TypeName: int
    Name: int
    ByteSize: int
    Version: int
    TypeFlags: int
    MetaFlag: int
    SubNodes: List[int]

    def __init__(self, stream: BinaryIO) -> None:
        (
            self.TypeName,
            self.Name,
            self.ByteSize,
            self.Version,
            self.TypeFlags,
            self.MetaFlag,
            count,
        ) = TPK_UNITY_NODE_STRUCT.unpack(stream.read(TPK_UNITY_NODE_STRUCT.size))

        SubNodeStruct = Struct(f"<{count}H")
        self.SubNodes = list(SubNodeStruct.unpack(stream.read(SubNodeStruct.size)))


######################################################################################
#
#   Strings
#
######################################################################################


@define
class TpkStringBuffer:
    Strings: List[str]

    def __init__(self, stream: BinaryIO) -> None:
        self.Strings = [
            read_string(stream) for _ in range(INT32.unpack(stream.read(INT32.size))[0])
        ]

    @property
    def Count(self) -> int:
        return len(self.Strings)


@define
class TpkCommonString:
    VersionInformation: List[Tuple[UnityVersion, int]]
    StringBufferIndices: tuple[int]

    def __init__(self, stream: BinaryIO) -> None:
        (versionCount,) = INT32.unpack(stream.read(INT32.size))
        self.VersionInformation = [
            (UnityVersion.fromStream(stream), stream.read(1)[0])
            for _ in range(versionCount)
        ]
        (indicesCount,) = INT32.unpack(stream.read(INT32.size))
        indicesStruct = Struct(f"<{indicesCount}H")
        self.StringBufferIndices = indicesStruct.unpack(stream.read(indicesStruct.size))

    def GetStrings(self, buffer: TpkStringBuffer) -> List[str]:
        return [buffer.Strings[i] for i in self.StringBufferIndices]

    def GetCount(self, exactVersion: UnityVersion) -> int:
        return get_item_for_version(exactVersion, self.VersionInformation)


######################################################################################
#
# helper functions
#
######################################################################################

BYTE = Struct("b")
UINT16 = Struct("<H")
INT32 = Struct("<i")
INT64 = Struct("<q")
UINT64 = Struct("<Q")


def read_string(stream: BinaryIO) -> str:
    # varint
    shift = 0
    length = 0
    while True:
        (i,) = stream.read(1)
        length |= (i & 0x7F) << shift
        shift += 7
        if not (i & 0x80):
            break
    # string
    return stream.read(length).decode("utf-8")


def read_data(stream: BinaryIO) -> bytes:
    return stream.read(INT32.unpack(stream.read(INT32.size))[0])


def get_item_for_version(
    exactVersion: UnityVersion, items: List[Tuple[UnityVersion, Any]]
) -> Any:
    ret = None
    for version, item in items:
        if exactVersion >= version:
            ret = item
        else:
            break
    if ret:
        return ret
    else:
        raise ValueError("Could not find exact version")
