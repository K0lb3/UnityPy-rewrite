from __future__ import annotations
import re
from typing import Optional, Any, Union, TYPE_CHECKING

from .TypeTreeNode import TypeTreeNode
from ..streams import EndianBinaryReader
from .. import objects

if TYPE_CHECKING:
    from ..files.SerializedFile import SerializedFile

kAlignBytes = 0x4000


def read_typetree(
    root_node: TypeTreeNode,
    reader: EndianBinaryReader,
    as_dict: bool = True,
    expected_read: Optional[int] = None,
    assetsfile: Optional[SerializedFile] = None,
) -> Union[dict[str, Any], objects.Object]:
    """Reads the typetree of the object contained in the reader via the node list.

    Parameters
    ----------
    nodes : list
        List of nodes/nodes
    reader : EndianBinaryReader
        Reader of the object to be parsed

    Returns
    -------
    dict | objects.Object
        The parsed typtree
    """
    pos = reader.tell()
    obj = read_value(root_node, reader, as_dict, assetsfile)

    read = reader.tell() - pos
    if expected_read is not None and read != expected_read:
        raise ValueError(
            f"Expected to read {expected_read} bytes, but read {read} bytes"
        )

    return obj


def read_value(node: TypeTreeNode, reader: EndianBinaryReader, as_dict: bool, assetsfile: Optional[SerializedFile]) -> Any:
    # print(reader.tell(), node.m_Name, node.m_Type, node.m_MetaFlag)
    align = metaflag_is_aligned(node.m_MetaFlag)

    match node.m_Type:
        case "SInt8":
            value = reader.read_i8()
        case "UInt8" | "char":
            value = reader.read_u8()
        case "short" | "SInt16":
            value = reader.read_i16()
        case "UInt16" | "unsigned short":
            value = reader.read_u16()
        case "int" | "SInt32":
            value = reader.read_i32()
        case "UInt32" | "unsigned int" | "Type*":
            value = reader.read_u32()
        case "long long" | "SInt64":
            value = reader.read_i64()
        case "UInt64" | "unsigned long long" | "FileSize":
            value = reader.read_u64()
        case "float":
            value = reader.read_f32()
        case "double":
            value = reader.read_f64()
        case "bool":
            value = reader.read_bool()
        case "string":
            value = reader.read_aligned_string()
        case "TypelessData":
            value = reader.read_bytes()
        case "pair":
            first = read_value(node.m_Children[0], reader, as_dict, assetsfile)
            second = read_value(node.m_Children[1], reader, as_dict, assetsfile)
            value = (first, second)
        case _:
            # Vector
            if node.m_Children[0].m_Type == "Array":
                if metaflag_is_aligned(node.m_Children[0].m_MetaFlag):
                    align = True

                size = (
                    reader.read_count()
                )  # read_value(node.m_Children[0].m_Children[0], reader, as_dict)
                subtype = node.m_Children[0].m_Children[1]
                if metaflag_is_aligned(subtype.m_MetaFlag):
                    value = read_value_array(subtype, reader, as_dict, size)
                else:
                    value = [read_value(subtype, reader, as_dict, assetsfile) for _ in range(size)]

            else:  # Class
                value = {
                    child.m_Name: read_value(child, reader, as_dict, assetsfile)
                    for child in node.m_Children
                }

                if not as_dict:
                    if node.m_Type.startswith("PPtr<"):
                        value = objects.PPtr[Any](
                            assetsfile=assetsfile,
                            m_FileID=value["m_FileID"],
                            m_PathID=value["m_PathID"],
                            type=node.m_Type[6:-1],
                        )
                    else:
                        value = getattr(
                            objects,
                            node.m_Type,
                            objects.Object,
                        )(**{clean_name(key): value for key, value in value.items()})

    if align:
        reader.align_stream()
    
    return value


def read_value_array(
    node: TypeTreeNode, reader: EndianBinaryReader, as_dict: bool, size: int, assetsfile: Optional[SerializedFile] = None,
) -> Any:
    align = metaflag_is_aligned(node.m_MetaFlag)

    match node.m_Type:
        case "SInt8":
            value = reader.read_i8_array(size)
        case "UInt8" | "char":
            value = reader.read_u8_array(size)
        case "short" | "SInt16":
            value = reader.read_i16_array(size)
        case "UInt16" | "unsigned short":
            value = reader.read_u16_array(size)
        case "int" | "SInt32":
            value = reader.read_i32_array(size)
        case "UInt32" | "unsigned int" | "Type*":
            value = reader.read_u32_array(size)
        case "long long" | "SInt64":
            value = reader.read_i64_array(size)
        case "UInt64" | "unsigned long long" | "FileSize":
            value = reader.read_u64_array(size)
        case "float":
            value = reader.read_f32_array(size)
        case "double":
            value = reader.read_f64_array(size)
        case "bool":
            value = reader.read_bool_array(size)
        case "string":
            value = [reader.read_aligned_string() for _ in range(size)]
        case "TypelessData":
            value = [reader.read_bytes() for _ in range(size)]
        case "pair":
            value = [
                (
                    read_value(node.m_Children[0], reader, as_dict, assetsfile),
                    read_value(node.m_Children[1], reader, as_dict, assetsfile),
                )
                for _ in range(size)
            ]
        case _:
            # Vector
            if node.m_Children[0].m_Type == "Array":
                if metaflag_is_aligned(node.m_Children[0].m_MetaFlag):
                    align = True
                subtype = node.m_Children[0].m_Children[1]
                if metaflag_is_aligned(subtype.m_MetaFlag):
                    value = [
                        read_value_array(subtype, reader, as_dict, reader.read_count())
                        for _ in range(size)
                    ]
                else:
                    value = [
                        [
                            read_value(subtype, reader, as_dict, assetsfile)
                            for _ in range(reader.read_count())
                        ]
                        for _ in range(size)
                    ]
            else:  # Class
                if as_dict:
                    value = [
                        {
                            child.m_Name: read_value(child, reader, as_dict, assetsfile)
                            for child in node.m_Children
                        }
                        for _ in range(size)
                    ]
                else:
                    if node.m_Type.startswith("PPtr<"):
                        value = [
                            objects.PPtr[Any](
                                assetsfile=assetsfile,
                                type=node.m_Type[6:-1],
                                **{
                                    child.m_Name: read_value(child, reader, as_dict, assetsfile)
                                    for child in node.m_Children
                                }
                            )
                            for _ in range(size)
                        ]
                    else:
                        clz = getattr(
                            objects,
                            "PPtr" if node.m_Type.startswith("PPtr") else node.m_Type,
                            objects.Object,
                        )
                        clean_names = [
                            clean_name(child.m_Name) for child in node.m_Children
                        ]
                        value = [
                            clz(
                                **{
                                    name: read_value(child, reader, as_dict, assetsfile)
                                    for name, child in zip(clean_names, node.m_Children)
                                }
                            )
                            for _ in range(size)
                        ]

    if align:
        reader.align_stream()
    return value

def metaflag_is_aligned(meta_flag: int | None) -> bool:
    return ((meta_flag or 0) & kAlignBytes) != 0

def clean_name(name: str) -> str:
    if name.startswith("(int&)"):
        name = name[6:]
    if name.endswith("?"):
        name = name[:-1]
    name = re.sub(r"[ \.:\-\[\]]", "_", name)
    if name in ["pass", "from"]:
        name += "_"
    if name[0].isdigit():
        name = f"x{name}"
    return name
