from __future__ import annotations
from typing import TypeVar, Generic, TYPE_CHECKING, Optional, Any, cast

from attr import define

if TYPE_CHECKING:
    from ..files.ObjectInfo import ObjectInfo
    from ..files.SerializedFile import SerializedFile

T = TypeVar("T")


@define
class PPtr(Generic[T]):
    m_FileID: int
    m_PathID: int
    assetsfile: Optional[SerializedFile] = None
    type: Optional[str] = None

    def deref(self, assetsfile: Optional[SerializedFile] = None) -> ObjectInfo[T]:
        assetsfile = assetsfile or self.assetsfile
        if assetsfile is None:
            raise ValueError("PPtr can't deref without an assetsfile!")

        if self.m_FileID == 0:
            pass
        else:
            # resolve file id to external name
            external_id = self.m_FileID - 1
            if external_id >= len(assetsfile.m_Externals):
                raise FileNotFoundError("Failed to resolve pointer - invalid m_FileID!")
            external = assetsfile.m_Externals[external_id]

            # resolve external name to assetsfile
            container = assetsfile.parent
            if container is None:
                # TODO - use default fs
                raise FileNotFoundError(
                    f"PPtr points to {external.path} but no container is set!"
                )

            for child in container.childs:
                if isinstance(child, SerializedFile) and child.name == external.path:
                    assetsfile = child
                    break
            else:
                raise FileNotFoundError(
                    f"Failed to resolve pointer - {external.path} not found!"
                )

        return cast(ObjectInfo[T], assetsfile.m_Objects[self.m_PathID])

    def deref_parse_as_object(self, assetsfile: Optional[SerializedFile] = None) -> T:
        return self.deref(assetsfile).parse_as_object()

    def deref_parse_as_dict(
        self, assetsfile: Optional[SerializedFile] = None
    ) -> dict[str, Any]:
        ret = self.deref(assetsfile).parse_as_dict()
        assert isinstance(ret, dict)
        return ret
