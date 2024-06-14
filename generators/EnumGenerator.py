"""Extract the enums from UnityCsReference and generate a python file with them."""
from collections import defaultdict
from dataclasses import dataclass
import os
import re
from io import BytesIO, TextIOBase
from urllib.request import urlopen
from zipfile import ZipFile

URL = "https://github.com/Unity-Technologies/UnityCsReference/archive/refs/heads/master.zip"
STRUCTURE = {}
reEnum = re.compile(r"\s+enum (\w+?)\s+\{(.+?)\}", re.DOTALL | re.MULTILINE)
reEnumField = re.compile(r"^\s+(\w+?)(\s*=\s*(.+?))?\s*[,\n\/]", re.MULTILINE)
reAttribute = re.compile(r"\s*\[(.+?)\]")


@dataclass
class Enum:
    name: str
    fields: list[tuple[str, str]]
    attributes: list[str]
    file: str

    def write(self, stream: TextIOBase, indent: str = ""):
        enum_cls = "IntFlag" if "Flags" in self.attributes else "IntEnum"
        stream.write(f"{indent}class {self.name}({enum_cls}):\n")
        prevValue = 0
        dependentValues: list[tuple[str, str]] = []
        for field, value in self.fields:
            if field in ["None", "True", "False"]:
                field = field.upper()
            if value:
                try:
                    value = int(value)
                except ValueError:
                    dependentValues.append((field, value))
                    continue
                if value < 0:  # negative values are usually obsolete
                    value = abs(value)
            else:
                value = int(prevValue) + 1

            stream.write(f"{indent}    {field} = {value}\n")
            prevValue = value

        for field, value in dependentValues:
            stream.write(f"{indent}    {field} = {value}\n")

        stream.write("\n")


def parseFile(text: str, file_name: str):
    enums: list[Enum] = []
    for match in reEnum.finditer(text):
        name, body = match.groups()
        attrs: list[str] = []
        for prevLine in text[: match.start()].rsplit("\n", 5)[1:-1][::-1]:
            attrMatch = reAttribute.match(prevLine)
            if attrMatch:
                attrs.append(attrMatch.group(1))
            else:
                break
        enums.append(
            Enum(
                name=name,
                fields=[(name, value) for name, _, value in reEnumField.findall(body)],
                attributes=attrs,
                file=file_name,
            )
        )
    return enums


def generate_enums_from_cs_reference(fp: str):
    # data = open("UnityCsReference.zip", "rb").read()
    data = urlopen(URL).read()
    stream = BytesIO(data)
    zip = ZipFile(stream)
    enum_dict: dict[str, list[Enum]] = {}
    enums_names: dict[str, list[Enum]] = defaultdict(list)
    for file in zip.namelist():
        if file.startswith("UnityCsReference-master/Runtime/Export/") and file.endswith(
            ".cs"
        ):
            with zip.open(file) as f:
                text = f.read().decode("utf-8")
                enums = parseFile(text, file)
                if enums:
                    enum_dict[file] = enums
                    for enum in enums:
                        enums_names[enum.name].append(enum)
    zip.close()

    # handling duplicate enum names and enums
    for _, enums in enums_names.items():
        if len(enums) > 1:
            for enum in enums:
                base_file_name = enum.file.rsplit("/", 1)[1].split(".", 1)[0]
                enum.name = f"{base_file_name}_{enum.name}"

    with open(fp, "w") as f:
        f.write("from enum import IntEnum, IntFlag\n\n")
        for file, enums in enum_dict.items():
            f.write(f"# {file}\n")
            for enum in enums:
                enum.write(f)


if __name__ == "__main__":
    local = os.path.dirname(__file__)
    dst = os.path.join(local, "..", "UnityPy", "enums", "cs_reference.py")
    generate_enums_from_cs_reference(dst)
