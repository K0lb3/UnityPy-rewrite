from __future__ import annotations
from typing import Optional, List, Dict, Annotated, Self, Any, cast

from attrs import define

from .File import File, parseable_filetype
from ..streams import EndianBinaryReader, EndianBinaryWriter
from ..typetree.TypeTreeNode import TypeTreeNode
from ..typetree.Tpk import UnityVersion
from .ObjectInfo import ObjectInfo


@define
class SerializedFileHeader:
    m_MetadataSize: int
    m_FileSize: int
    m_Version: int
    m_DataOffset: int
    m_Endianess: int = 0
    m_Reserved: Optional[Annotated[bytes, 3]] = None
    m_Unknown1: Optional[int] = None

    @classmethod
    def parse(cls, reader: EndianBinaryReader) -> SerializedFileHeader:
        header = cls(*reader.read_u32_array(4))

        if header.m_Version > 0x1000000:
            reader.seek(reader.tell() - 16)
            reader.endian = "<" if reader.endian == ">" else ">"
            header = cls(*reader.read_u32_array(4))

        if header.m_Version >= 9:
            header.m_Endianess = reader.read_u8()
            header.m_Reserved = reader.read_bytes(3)

            if header.m_Version >= 22:
                header.m_MetadataSize = reader.read_u32()
                header.m_FileSize = reader.read_i64()
                header.m_DataOffset = reader.read_i64()
                header.m_Unknown1 = reader.read_i64()
        else:
            reader.seek(header.m_FileSize - header.m_MetadataSize)
            header.m_Endianess = reader.read_u8()

        return header


@define
class LocalSerializedObjectIdentifier:  # script type
    local_serialized_file_index: int
    local_identifier_in_file: int

    @classmethod
    def parse(
        cls, reader: EndianBinaryReader, version: int
    ) -> LocalSerializedObjectIdentifier:
        local_serialized_file_index = reader.read_i32()
        if version < 14:
            local_identifier_in_file = reader.read_i32()
        else:
            reader.align_stream()
            local_identifier_in_file = reader.read_i64()
        return cls(local_serialized_file_index, local_identifier_in_file)

    def dump(self, writer: EndianBinaryWriter, version: int):
        writer.write_i32(self.local_serialized_file_index)
        if version < 14:
            writer.write_i32(self.local_identifier_in_file)
        else:
            writer.align_stream()
            writer.write_i64(self.local_identifier_in_file)


@define
class FileIdentifier:
    path: str
    guid: Optional[bytes] = None
    type: Optional[int] = None
    temp_empty: Optional[str] = None
    # enum { kNonAssetType = 0, kDeprecatedCachedAssetType = 1, kSerializedAssetType = 2, kMetaAssetType = 3 };

    @classmethod
    def parse(cls, reader: EndianBinaryReader, version: int) -> FileIdentifier:
        guid = None
        type = None
        temp_empty = None
        if version >= 6:
            temp_empty = reader.read_cstring()
        if version >= 5:
            guid = reader.read_bytes(16)
            type = reader.read_i32()
        path = reader.read_cstring()
        return cls(path, guid, type, temp_empty)

    def dump(self, writer: EndianBinaryWriter, version: int):
        if version >= 6:
            assert self.temp_empty is not None
            writer.write_cstring(self.temp_empty)
        if version >= 5:
            assert self.guid is not None and self.type is not None
            writer.write_bytes(self.guid)
            writer.write_i32(self.type)
        writer.write_cstring(self.path)


@define
class BuildType:
    build_type: str

    @property
    def IsAlpha(self) -> bool:
        return self.build_type == "a"

    @property
    def IsPatch(self) -> bool:
        return self.build_type == "p"


@define
class SerializedType:
    m_ClassID: int
    m_IsStrippedType: Optional[bool] = None
    m_ScriptTypeIndex: Optional[int] = None
    m_Type: Optional[TypeTreeNode] = None
    m_ScriptID: Optional[Annotated[bytes, 16]] = None
    m_OldTypeHash: Optional[Annotated[bytes, 16]] = None
    m_TypeDependencies: Optional[List[int]] = None
    m_ClassName: Optional[str] = None
    m_NameSpace: Optional[str] = None
    m_AsmName: Optional[str] = None

    @classmethod
    def parse(
        cls,
        reader: EndianBinaryReader,
        version: int,
        enable_typetree: bool,
        is_ref_type: bool,
    ) -> SerializedType:
        m_ClassID = reader.read_i32()
        type = cls(m_ClassID)

        if version >= 16:
            type.m_IsStrippedType = reader.read_bool()

        if version >= 17:
            type.m_ScriptTypeIndex = reader.read_i16()

        if version >= 13:
            if (
                (is_ref_type and type.m_ScriptTypeIndex not in [None, -1])
                or (version < 16 and type.m_ClassID < 0)
                or (version >= 16 and type.m_ClassID == 114)
            ):
                type.m_ScriptID = reader.read_bytes(16)

            type.m_OldTypeHash = reader.read_bytes(16)

        if enable_typetree:
            if version >= 12 or version == 10:
                type.m_Type = TypeTreeNode.parse_blob(reader, version)
            else:
                type.m_Type = TypeTreeNode.parse(reader, version)

            if version >= 21:
                if is_ref_type:
                    type.m_ClassName = reader.read_cstring()
                    type.m_NameSpace = reader.read_cstring()
                    type.m_AsmName = reader.read_cstring()
                else:
                    type.m_TypeDependencies = list(reader.read_i32_array())

        return type

    def dump(
        self,
        writer: EndianBinaryWriter,
        version: int,
        enable_typetree: bool,
        is_ref_type: bool,
    ):
        writer.write_i32(self.m_ClassID)

        if version >= 16:
            assert self.m_IsStrippedType is not None
            writer.write_bool(self.m_IsStrippedType)

        if version >= 17:
            assert self.m_ScriptTypeIndex is not None
            writer.write_i16(self.m_ScriptTypeIndex)

        if version >= 13:
            if (
                (is_ref_type and self.m_ScriptTypeIndex not in [None, -1])
                or (version < 16 and self.m_ClassID < 0)
                or (version >= 16 and self.m_ClassID == 114)
            ):
                assert self.m_ScriptID is not None
                writer.write_bytes(self.m_ScriptID)

            assert self.m_OldTypeHash is not None
            writer.write_bytes(self.m_OldTypeHash)

        if enable_typetree:
            assert self.m_Type is not None
            if version >= 12 or version == 10:
                self.m_Type.dump_blob(writer, version)
            else:
                self.m_Type.dump(writer, version)

            if version >= 21:
                if is_ref_type:
                    assert (
                        self.m_ClassName is not None
                        and self.m_NameSpace is not None
                        and self.m_AsmName is not None
                    )
                    writer.write_cstring(self.m_ClassName)
                    writer.write_cstring(self.m_NameSpace)
                    writer.write_cstring(self.m_AsmName)
                else:
                    assert self.m_TypeDependencies is not None
                    writer.write_i32_array(self.m_TypeDependencies)


@parseable_filetype
class SerializedFile(File):
    m_Header: SerializedFileHeader
    m_UnityVersion: str
    m_TargetPlatform: int
    m_EnableTypeTree: bool = True
    m_Types: List[SerializedType]
    m_BigIDEnabled: int = 0
    m_Objects: Dict[int, ObjectInfo[Any]]
    m_ScriptTypes: List[LocalSerializedObjectIdentifier]
    m_Externals: List[FileIdentifier]
    m_RefTypes: List[SerializedType]
    m_UserInformation: str = ""

    @classmethod
    def probe(cls, reader: EndianBinaryReader) -> bool:
        header = SerializedFileHeader.parse(reader)

        # check if header is valid
        if (
            header.m_MetadataSize < 0
            or header.m_FileSize < 0
            or header.m_DataOffset < 0
            or header.m_Endianess not in [0, 1]
            or header.m_FileSize > reader.length
            or header.m_DataOffset > reader.length
            or header.m_MetadataSize > reader.length
            or header.m_MetadataSize > header.m_FileSize
            or header.m_DataOffset > header.m_FileSize
            or not (0 < header.m_Version < 30)
        ):
            return False
        return True

    def parse(self, reader: Optional[EndianBinaryReader] = None) -> Self:
        reader = self._opt_get_set_reader(reader)

        self.m_Header = header = SerializedFileHeader.parse(reader)

        reader.endian = "<" if header.m_Endianess == 0 else ">"
        version: int = header.m_Version

        unity_version = ""
        if version >= 7:
            unity_version = reader.read_cstring()
        if unity_version in [None, "", "0.0.0"]:
            raise ValueError(f"Invalid Unity version {unity_version}")

        self.m_UnityVersion = unity_version

        if version >= 8:
            self.m_TargetPlatform = reader.read_i32()

        if version >= 13:
            self.m_EnableTypeTree = reader.read_bool()

        # ReadTypes
        type_count = reader.read_i32()
        self.m_Types = [
            SerializedType.parse(reader, version, self.m_EnableTypeTree, False)
            for _ in range(type_count)
        ]

        if 7 <= version < 14:
            self.m_BigIDEnabled = reader.read_i32()

        # ReadObjects
        object_count = reader.read_i32()
        self.m_Objects = {
            obj.m_PathID: obj
            for obj in (
                cast(ObjectInfo[Any], ObjectInfo(reader, self))
                for _ in range(object_count)
            )
        }

        # Read Scripts
        if version >= 11:
            script_count = reader.read_i32()
            self.m_ScriptTypes = [
                LocalSerializedObjectIdentifier.parse(reader, version)
                for _ in range(script_count)
            ]

        # Read Externals
        externals_count = reader.read_i32()
        self.m_Externals = [
            FileIdentifier.parse(reader, version) for _ in range(externals_count)
        ]

        if version >= 20:
            ref_type_count = reader.read_i32()
            self.m_RefTypes = [
                SerializedType.parse(reader, version, self.m_EnableTypeTree, True)
                for _ in range(ref_type_count)
            ]

        if version >= 5:
            self.userInformation = reader.read_cstring()

        return self

    def dump(self, writer: Optional[EndianBinaryWriter] = None) -> EndianBinaryWriter:
        raise NotImplementedError("SerializedFile.dump is not implemented")

    def get_objects(self) -> List[ObjectInfo[Any]]:
        return list(self.m_Objects.values())

    def get_containers(self) -> Dict[str, List[ObjectInfo[Any]]]:
        raise NotImplementedError("SerializedFile.get_containers is not implemented")

    def get_unityversion(self) -> UnityVersion:
        return UnityVersion.fromString(self.m_UnityVersion)
