from __future__ import annotations
from typing import Optional, Any, TYPE_CHECKING, TypeVar, Generic, cast

from ..typetree import TypeTreeHelper
from ..typetree.Tpk import get_typetree_node

if TYPE_CHECKING:
    from .SerializedFile import SerializedFile, SerializedType
    from ..streams import EndianBinaryReader
    from ..typetree.TypeTreeNode import TypeTreeNode

T = TypeVar("T")


class ObjectInfo(Generic[T]):
    assetsfile: SerializedFile
    m_PathID: int
    m_TypeID: int
    m_ClassID: int
    byteStart: int
    byteSize: int
    # platform: BuildTarget
    m_Version: int
    m_IsDestroyed: Optional[int] = None
    m_Stripped: Optional[int] = None
    serializedType: Optional[SerializedType] = None

    # saves where the parser stopped
    # in case that not all data is read
    # and the obj.data is changed, the unknown data can be added again
    def __init__(
        self,
        reader: EndianBinaryReader,
        assetsfile: SerializedFile,
    ):
        self.assetsfile = assetsfile
        self.reader = reader

        version = assetsfile.m_Header.m_Version

        if assetsfile.m_BigIDEnabled:
            self.m_PathID = reader.read_i64()
        elif version < 14:
            self.m_PathID = reader.read_i32()
        else:
            reader.align_stream()
            self.m_PathID = reader.read_i64()

        if version >= 22:
            self.byte_start = reader.read_i64()
        else:
            self.byte_start = reader.read_u32()

        self.byte_start += assetsfile.m_Header.m_DataOffset
        self.byte_size = reader.read_u32()
        self.m_TypeID = reader.read_i32()

        if version < 16:
            self.m_ClassID = reader.read_u16()
            self.serializedType = next(
                (typ for typ in assetsfile.m_Types if typ.m_ClassID == self.m_TypeID),
                None,
            )
        else:
            self.serializedType = assetsfile.m_Types[self.m_TypeID]
            self.m_ClassID = self.serializedType.m_ClassID

        if version < 11:
            self.m_IsDestroyed = reader.read_u16()

        if 11 <= version < 17:
            script_type_index = reader.read_i16()
            if self.serializedType:
                self.serializedType.m_ScriptTypeIndex = script_type_index

        if version == 15 or version == 16:
            self.m_Stripped = reader.read_u8()

    def get_typetree_node(self) -> TypeTreeNode:
        if self.serializedType and self.serializedType.m_Type:
            return self.serializedType.m_Type
        else:
            return get_typetree_node(self.m_ClassID, self.assetsfile.get_unityversion())

    def parse_as_object(self) -> T:
        self.reader.seek(self.byte_start)
        ret = TypeTreeHelper.read_typetree(
            self.get_typetree_node(),
            self.reader,
            as_dict=False,
            expected_read=self.byte_size,
            assetsfile=self.assetsfile,
        )
        assert not isinstance(ret, dict)
        ret.set_object_info(self)
        return cast(T, ret)

    def parse_as_dict(self) -> dict[str, Any]:
        self.reader.seek(self.byte_start)
        ret = TypeTreeHelper.read_typetree(
            self.get_typetree_node(),
            self.reader,
            as_dict=True,
            expected_read=self.byte_size,
        )
        assert isinstance(ret, dict)
        return ret

    @property
    def class_name(self) -> str:
        try:
            return self.get_typetree_node().m_Type
        except Exception:
            return "Unknown"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(PathID:{self.m_PathID}, Class: {self.class_name}({self.m_ClassID})>"
