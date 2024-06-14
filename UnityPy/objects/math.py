"""
Definitions for math related classes.
As most calculations involving them are done in numpy,
we define them here as subtypes of np.ndarray, so that casting won't be necessary.
"""

import numpy as np
import nptyping as npt
from typing import Self, TypeAlias


class Vector2f(np.ndarray[npt.Shape["2"], npt.Float32]):
    def __new__(cls, x: float = 0, y: float = 0) -> Self:
        return super().__new__(cls, shape=(2,), dtype=np.float32, buffer=np.array([x, y], dtype=np.float32))  # type: ignore

    @property
    def x(self) -> float:
        return self[0]

    @x.setter
    def x(self, value: float):
        self[0] = value

    @property
    def y(self) -> float:
        return self[1]

    @y.setter
    def y(self, value: float):
        self[1] = value

    def __repr__(self) -> str:
        return f"Vector2f({self.x}, {self.y})"


class Vector3f(np.ndarray[npt.Shape["3"], npt.Float32]):
    def __new__(cls, x: float = 0, y: float = 0, z: float = 0) -> Self:
        return super().__new__(cls, shape=(3,), dtype=np.float32, buffer=np.array([x, y, z], dtype=np.float32))  # type: ignore

    @property
    def x(self) -> float:
        return self[0]

    @x.setter
    def x(self, value: float):
        self[0] = value

    @property
    def y(self) -> float:
        return self[1]

    @y.setter
    def y(self, value: float):
        self[1] = value

    @property
    def z(self) -> float:
        return self[2]

    @z.setter
    def z(self, value: float):
        self[2] = value

    def __repr__(self) -> str:
        return f"Vector3f({self.x}, {self.y}, {self.z})"


class Vector4f(np.ndarray[npt.Shape["4"], npt.Float32]):
    def __new__(cls, x: float = 0, y: float = 0, z: float = 0, w: float = 0) -> Self:
        return super().__new__(cls, shape=(4,), dtype=np.float32, buffer=np.array([x, y, z, w], dtype=np.float32))  # type: ignore

    @property
    def x(self) -> float:
        return self[0]

    @x.setter
    def x(self, value: float):
        self[0] = value

    @property
    def y(self) -> float:
        return self[1]

    @y.setter
    def y(self, value: float):
        self[1] = value

    @property
    def z(self) -> float:
        return self[2]

    @z.setter
    def z(self, value: float):
        self[2] = value

    @property
    def w(self) -> float:
        return self[3]

    @w.setter
    def w(self, value: float):
        self[3] = value

    def __repr__(self) -> str:
        return f"Vector4f({self.x}, {self.y}, {self.z}, {self.w})"


float3: TypeAlias = Vector3f
float4: TypeAlias = Vector4f


class Quaternionf(Vector4f):
    # TODO: Implement quaternion operations
    def __repr__(self) -> str:
        return f"Quaternion({self.x}, {self.y}, {self.z}, {self.w})"


class Matrix3x4f(np.ndarray[npt.Shape["3, 4"], npt.Float32]):
    e00: float
    e01: float
    e02: float
    e03: float
    e10: float
    e11: float
    e12: float
    e13: float
    e20: float
    e21: float
    e22: float
    e23: float

    def __new__(cls, *args: float, **kwargs: float) -> Self:
        values: tuple[float, ...]
        if args:
            values = args
        elif kwargs:
            values = tuple(
                kwargs.get(f"e{x}{y}", 0.0) for x in range(3) for y in range(4)
            )
        else:
            values = (0.0,) * 12

        return super().__new__(
            cls,
            shape=(3, 4),
            dtype=np.float32,
            buffer=np.array(values, dtype=np.float32),
        )

    def __getattr__(self, name: str) -> float:
        assert name in self.__annotations__, f"Matrix3x4f has no attribute {name}"
        x = int(name[1])
        y = int(name[2])
        return self[x, y]

    def __setattr__(self, name: str, value: float):
        assert name in self.__annotations__, f"Matrix3x4f has no attribute {name}"
        x = int(name[1])
        y = int(name[2])
        self[x, y] = value


class Matrix4x4f(np.ndarray[npt.Shape["4, 4"], npt.Float32]):
    e00: float
    e01: float
    e02: float
    e03: float
    e10: float
    e11: float
    e12: float
    e13: float
    e20: float
    e21: float
    e22: float
    e23: float
    e30: float
    e31: float
    e32: float
    e33: float

    def __new__(cls, *args: float, **kwargs: float) -> Self:
        values: tuple[float, ...]
        if args:
            values = args
        elif kwargs:
            values = tuple(
                kwargs.get(f"e{x}{y}", 0.0) for x in range(4) for y in range(4)
            )
        else:
            values = (0.0,) * 12

        return super().__new__(
            cls,
            shape=(4, 4),
            dtype=np.float32,
            buffer=np.array(values, dtype=np.float32),
        )

    def __getattr__(self, name: str) -> float:
        assert name in self.__annotations__, f"Matrix3x4f has no attribute {name}"
        x = int(name[1])
        y = int(name[2])
        return self[x, y]

    def __setattr__(self, name: str, value: float):
        assert name in self.__annotations__, f"Matrix3x4f has no attribute {name}"
        x = int(name[1])
        y = int(name[2])
        self[x, y] = value


class ColorRGBA(np.ndarray[npt.Shape["4"], npt.Float32]):
    r: float
    g: float
    b: float
    a: float

    def __new__(
        cls, r: float = 0, g: float = 0, b: float = 0, a: float = 1, rgba: int = -1
    ) -> Self:
        if rgba != -1:
            r = ((rgba >> 24) & 0xFF) / 255
            g = ((rgba >> 16) & 0xFF) / 255
            b = ((rgba >> 8) & 0xFF) / 255
            a = ((rgba & 0xFF)) / 255
        return super().__new__(
            cls,
            shape=(4,),
            dtype=np.float32,
            buffer=np.array([r, g, b, a], dtype=np.float32),
        )

    @property
    def rgba(self) -> int:
        return (self * 255).astype(np.uint8).view(np.uint32)[0]

    @rgba.setter
    def rgba(self, value: int):
        self[:] = [
            (value >> 24) & 0xFF,
            (value >> 16) & 0xFF,
            (value >> 8) & 0xFF,
            (value & 0xFF),
        ]
        self /= 255
