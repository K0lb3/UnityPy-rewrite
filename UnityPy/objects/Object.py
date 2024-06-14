from __future__ import annotations
from abc import ABC, ABCMeta
from typing import TYPE_CHECKING, Dict, Any, Optional, Self

if TYPE_CHECKING:
    from ..files.ObjectInfo import ObjectInfo


class Object(ABC, metaclass=ABCMeta):
    object_info: Optional[ObjectInfo[Self]] = None

    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        self.__dict__.update(**kwargs)

    def set_object_info(self, object_info: ObjectInfo[Any]):
        self.object_info = object_info

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"
