# UnityPy (rewrite / dev)

The rewrite is still very much work in progress,
with breaking changes bound to take place.

Currently there is still no top level environment handler,
so assets have to be parsed in following way atm:

```python
from UnityPy.streams.EndianBinaryReader import EndianBinaryReaderStream
from UnityPy.files import parse_file
from UnityPy.object_handlers.Mesh import MeshHandler

fp = "tests/resources/prefab"
file = parse_file(
    EndianBinaryReaderStream(open(fp, "rb")),
    os.path.basename(fp),
    fp,
)
assert file is not None
for obj in file.get_objects():
    # filter by obj class type
    # obj.m_ClassID is a pure int now, that gets resolved via the tpk,
    # so the property .classname has to be able to get the object class without parsing
    # - bound to change, propably by exporting a map from the tpk, or generating a corresponding enum
    if obj.class_name != "Mesh":
        continue
    try:
        print(obj.m_PathID)
        # parse the object content as dict, comparable to .read_typetree()
        obj_dict = obj.parse_as_dict()
        # parse the object as object class, comparable to .read(),
        # but now it also uses the typetrees and doesn't do any post processing automatically anymore
        obj_inst = obj.parse_as_object()
        # the exporters are now handlers, to allow reimporting later on
        # any previous automatic post processing of the asset data will be done here instead
        handler = MeshHandler(obj_inst)
        handler.process()
        pass
    except Exception as e:
        print(obj.m_PathID, e)
        exit()
```

## Similar Projects

TODO

## Credits

- [Perfare/AssetStudio](https://github.com/Perfare/AssetStudio) - a very big part of the code originated from there
- [Jerome Leclanche](https://github.com/jleclanche) - for his [HearthSim/UnityPack](https://github.com/HearthSim/UnityPack) and his contributions to and creation of modules that also get used by UnityPy
- [Jeremy Pritts](https://github.com/ds5678) - for his [TypeTreeDumps](https://github.com/AssetRipper/TypeTreeDumps), which are the base of the whole object class system of UnityPy
- [Razmoth](https://github.com/Razmoth) for figuring out and sharing Unity CN's AssetBundle decryption ([src](https://github.com/Razmoth/PGRStudio))
- to everyone who contributed to the project
