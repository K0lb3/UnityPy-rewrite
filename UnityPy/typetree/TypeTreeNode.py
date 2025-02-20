from __future__ import annotations
from struct import Struct
from typing import Optional, List, Tuple, TYPE_CHECKING, Dict, Iterator

from attrs import define, field

from ..streams.EndianBinaryWriter import EndianBinaryWriter

if TYPE_CHECKING:
    from .Tpk import UnityVersion

from ..streams import EndianBinaryReader
from ..streams.EndianBinaryReader import EndianBinaryReaderMemory


@define
class TypeTreeNode:
    m_Level: int
    m_Type: str
    m_Name: str
    m_ByteSize: int
    m_TypeFlags: int
    m_Version: int
    m_Children: List[TypeTreeNode] = field(factory=list)
    m_VariableCount: Optional[int] = None
    m_Index: Optional[int] = None
    m_MetaFlag: Optional[int] = None
    m_RefTypeHash: Optional[int] = None

    def traverse(self) -> Iterator[TypeTreeNode]:
        stack: list[TypeTreeNode] = [self]
        while stack:
            node = stack.pop()
            stack.extend(reversed(node.m_Children))
            yield node

    @classmethod
    def parse(cls, reader: EndianBinaryReader, version: int) -> TypeTreeNode:
        # stack approach is way faster than recursion
        # using a fake root node to avoid special case for root node
        dummy_node = TypeTreeNode(-1, "", "", 0, 0, 0, [])
        dummy_root = TypeTreeNode(-1, "", "", 0, 0, 0, [dummy_node])

        stack: List[Tuple[TypeTreeNode, int]] = [(dummy_root, 1)]
        while stack:
            parent, count = stack[-1]
            if count == 1:
                stack.pop()
            else:
                stack[-1] = (parent, count - 1)

            node = TypeTreeNode(
                m_Level=parent.m_Level + 1,
                m_Type=reader.read_cstring(),
                m_Name=reader.read_cstring(),
                m_ByteSize=reader.read_i32(),
                m_VariableCount=reader.read_i32() if version == 2 else None,
                m_Index=reader.read_i32() if version != 3 else None,
                m_TypeFlags=reader.read_i32(),
                m_Version=reader.read_i32(),
                m_MetaFlag=reader.read_i32() if version != 3 else None,
            )
            parent.m_Children[-count] = node
            children_count = reader.read_i32()
            if children_count > 0:
                node.m_Children = [dummy_node] * children_count
                stack.append((node, children_count))
        return dummy_root.m_Children[0]

    @classmethod
    def parse_blob(cls, reader: EndianBinaryReader, version: int) -> TypeTreeNode:
        node_count = reader.read_i32()
        stringbuffer_size = reader.read_i32()

        node_struct, keys = _get_blob_node_struct(reader.endian, version)
        struct_data = reader.read(node_struct.size * node_count)
        stringbuffer_reader = EndianBinaryReaderMemory(
            reader.read(stringbuffer_size), reader.endian
        )

        CommonString = get_common_strings()

        def read_string(reader: EndianBinaryReader, value: int) -> str:
            is_offset = (value & 0x80000000) == 0
            if is_offset:
                reader.seek(value)
                return reader.read_cstring()

            offset = value & 0x7FFFFFFF
            return CommonString.get(offset, str(offset))

        fake_root: TypeTreeNode = TypeTreeNode(-1, "", "", 0, 0, 0, [])
        stack: List[TypeTreeNode] = [fake_root]
        parent = fake_root
        prev = fake_root

        for raw_node in node_struct.iter_unpack(struct_data):
            node = TypeTreeNode(
                **dict(zip(keys[:3], raw_node[:3])),
                **dict(zip(keys[5:], raw_node[5:])),
                m_Type=read_string(stringbuffer_reader, raw_node[3]),
                m_Name=read_string(stringbuffer_reader, raw_node[4]),
            )

            if node.m_Level > prev.m_Level:
                stack.append(parent)
                parent = prev
            elif node.m_Level < prev.m_Level:
                while node.m_Level <= parent.m_Level:
                    parent = stack.pop()

            parent.m_Children.append(node)
            prev = node

        return fake_root.m_Children[0]

    def dump(self, writer: EndianBinaryWriter, version: int):
        stack: list[TypeTreeNode] = [self]
        while stack:
            node = stack.pop()

            writer.write_cstring(self.m_Type)
            writer.write_cstring(self.m_Name)
            writer.write_i32(self.m_ByteSize)
            if version == 2:
                assert self.m_VariableCount is not None
                writer.write_i32(self.m_VariableCount)
            if version != 3:
                assert self.m_Index is not None
                writer.write_i32(self.m_Index)
            writer.write_i32(self.m_TypeFlags)
            writer.write_i32(self.m_Version)
            if version != 3:
                assert self.m_MetaFlag is not None
                writer.write_i32(self.m_MetaFlag)

            writer.write_i32(len(self.m_Children))

            stack.extend(reversed(node.m_Children))

    def dump_blob(self, writer: EndianBinaryWriter, version: int):
        node_writer = EndianBinaryWriter(endian=writer.endian)
        string_writer = EndianBinaryWriter()

        # string buffer setup
        CommonStringOffsetMap = {
            string: offset for offset, string in get_common_strings().items()
        }

        string_offsets: dict[str, int] = {}

        def write_string(string: str) -> int:
            offset = string_offsets.get(string)
            if offset is None:
                common_offset = CommonStringOffsetMap.get(string)
                if common_offset:
                    offset = common_offset | 0x80000000
                else:
                    offset = string_writer.tell()
                    string_writer.write_cstring(string)
                string_offsets[string] = offset
            return offset

        # node buffer setup
        node_struct, keys = _get_blob_node_struct(writer.endian, version)

        def write_node(node: TypeTreeNode):
            node_writer.write(
                node_struct.pack(
                    *[getattr(node, key) for key in keys[:3]],
                    write_string(node.m_Type),
                    write_string(node.m_Name),
                    *[getattr(node, key) for key in keys[5:]],
                )
            )

        # write nodes
        node_count = len([write_node(node) for node in self.traverse()])

        # write blob
        writer.write_i32(node_count)
        writer.write_i32(string_writer.tell())
        writer.write(node_writer.get_bytes())
        writer.write(string_writer.get_bytes())


COMMONSTRING_CACHE: Dict[Optional[UnityVersion], Dict[int, str]] = {}


def get_common_strings(version: Optional[UnityVersion] = None) -> Dict[int, str]:
    if version in COMMONSTRING_CACHE:
        return COMMONSTRING_CACHE[version]

    from .Tpk import get_tpktypetree

    tree = get_tpktypetree()
    common_string = tree.CommonString
    strings = common_string.GetStrings(tree.StringBuffer)
    if version:
        count = common_string.GetCount(version)
        strings = strings[:count]

    ret: Dict[int, str] = {}
    offset = 0
    for string in strings:
        ret[offset] = string
        offset += len(string) + 1

    COMMONSTRING_CACHE[version] = ret
    return ret


def _get_blob_node_struct(endian: str, version: int) -> tuple[Struct, list[str]]:
    struct_type = f"{endian}hBBIIiii"
    keys = [
        "m_Version",
        "m_Level",
        "m_TypeFlags",
        "m_TypeStrOffset",
        "m_NameStrOffset",
        "m_ByteSize",
        "m_Index",
        "m_MetaFlag",
    ]
    if version >= 19:
        struct_type += "Q"
        keys.append("m_RefTypeHash")

    return Struct(struct_type), keys
