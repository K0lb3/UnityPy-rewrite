# relvant background info:
# BC: https://www.reedbeta.com/blog/understanding-bcn-texture-compression-formats/
# BC PIL decoder args: https://github.com/python-pillow/Pillow/blob/114e01701ac92d6bd1df7df771a2be8abcc0ae33/src/PIL/DdsImagePlugin.py#L385
from abc import ABC, ABCMeta, abstractmethod
from typing import (
    Optional,
    TYPE_CHECKING,
    Any,
    Literal,
    ByteString,
    TypeVar,
    Generic,
    BinaryIO,
    Callable,
)

from attr import define
import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image
import texture2ddecoder_rs as t2dd

from UnityPy.objects import Texture2D
from ..enums import TextureFormat

if TYPE_CHECKING:
    from ..objects import Texture2D
    from ..typetree.Tpk import UnityVersion


def get_texture_data(texture: Texture2D) -> ByteString:
    if len(texture.image_data) > 0:
        return texture.image_data
    if texture.m_StreamData:
        raise NotImplementedError(
            "Texture2D.m_StreamData resolving is not implemented yet"
        )
    raise ValueError("Texture2D has no image data!")

E = TypeVar("E", bound=np.generic, covariant=True)


class TextureDataHandler(ABC, metaclass=ABCMeta):
    bpp: float
    block_size: tuple[int, int]
    channel_count: int

    @abstractmethod
    def decode_pil(self, texture: Texture2D) -> Image.Image:
        pass

    @abstractmethod
    def decode_raw(self, texture: Texture2D) -> npt.NDArray[np.generic]:
        pass

    @abstractmethod
    def patch_texture(
        self, image: Image.Image, texture: Texture2D, delta: bool = False
    ) -> None:
        """
        Patch the texture with the given image.
        If delta is true, then only the changed blocks are updated."""
        pass

    @abstractmethod
    def save_fp(self, texture: Texture2D, path: str) -> None:
        pass

    @abstractmethod
    def save_stream(self, texture: Texture2D, stream: BinaryIO, ext: str) -> None:
        pass

    def get_level_data(self, texture: Texture2D, level: int) -> ByteString:
        raw_data = get_texture_data(texture)

        width = texture.m_Width
        height = texture.m_Height

        offset = 0
        for _ in range(level):
            offset += width * height * self.bpp // 8
            width = max(1, width // 2)
            height = max(1, height // 2)
        level_size = width * height * self.bpp // 8

        return raw_data[offset : offset + level_size]


@define(kw_only=True, slots=True, frozen=True)
class PILRawTextureDataHandler(TextureDataHandler):
    bpp: float
    block_size: tuple[int, int] = (1, 1)
    channel_count: int
    mode: Literal["L", "LA", "RGB", "RGBA"]
    decoder_name: Literal["raw", "bcn"] = "raw"
    decoder_args: Optional[tuple[Any, ...]] = None
    band_reorder: Optional[tuple[int, ...]] = None

    def decode_pil(self, texture: Texture2D) -> Image.Image:
        image = Image.frombytes(self.mode, (texture.m_Width, texture.m_Height), texture.image_data, self.decoder_name, *self.decoder_args)  # type: ignore - PIL.Image.frombytes has no type hint

        if self.band_reorder:
            bands = image.split()
            image = Image.merge(self.mode, [bands[i] for i in self.band_reorder])  # type: ignore - PIL.Image.merge has no type hint

        assert isinstance(image, Image.Image)
        return image

    def decode_raw(self, texture: Texture2D) -> np.ndarray[Any, np.dtype[np.uint8]]:
        image = self.decode_pil(texture)
        return np.array(image)

    def save_fp(self, texture: Texture2D, path: str) -> None:
        image = self.decode_pil(texture)
        image.save(path)  # type: ignore - PIL.Image.save has no type hint

    def save_stream(self, texture: Texture2D, stream: BinaryIO, ext: str) -> None:
        image = self.decode_pil(texture)
        image.save(stream, ext)  # type: ignore - PIL.Image.save has no type hint

    def patch_texture(
        self, image: Image.Image, texture: Texture2D, delta: bool = False
    ) -> None:
        raise NotImplementedError(
            "PILRawTextureDataHandler.patch_texture is not implemented"
        )


@define(kw_only=True, slots=True, frozen=True)
class OpenCVTextureDataHandler(TextureDataHandler, Generic[E]):
    bpp: float
    block_size: tuple[int, int] = (1, 1)
    channel_count: int
    dtype: npt.DTypeLike
    channel_reorder: Optional[tuple[int, ...]] = None

    def decode_pil(self, texture: Texture2D) -> Image.Image:
        image_data = self.decode_raw(texture)
        dtype = image_data.dtype
        if dtype == np.uint8:
            pass
        elif dtype == np.float16 or dtype == np.float32 or dtype == np.float64:
            image_data = np.multiply(image_data, 255).astype(np.uint8)
        elif image_data.dtype == np.uint16:
            image_data = np.divide(image_data, 256).astype(np.uint8)
        else:
            raise ValueError(f"Unsupported dtype {dtype}")
        
        if self.channel_reorder:
            image_data = image_data[..., self.channel_reorder]
        
        return Image.fromarray(image_data, mode="RGBA") # type: ignore - PIL.Image.fromarray has no type hint

    def decode_raw(self, texture: Texture2D) -> npt.NDArray[E]:
        data = self.get_level_data(texture, 0)
        image_data = np.frombuffer(data, dtype=self.dtype).reshape(
            texture.m_Height, texture.m_Width, self.channel_count
        )

        return image_data

    def save_fp(self, texture: Texture2D, path: str) -> None:
        image = self.decode_raw(texture)
        cv2.imwrite(path, image)

    def save_stream(self, texture: Texture2D, stream: BinaryIO, ext: str) -> None:
        image = self.decode_raw(texture)
        result, data = cv2.imencode(ext, image)
        assert result, cv2.error("Failed to encode image")
        stream.write(data.tobytes())

    def patch_texture(
        self, image: Image.Image, texture: Texture2D, delta: bool = False
    ) -> None:
        raise NotImplementedError(
            "PILRawTextureDataHandler.patch_texture is not implemented"
        )


@define(kw_only=True, slots=True, frozen=True)
class Texture2DDecoderTextureDataHandler(TextureDataHandler):
    bpp: float
    block_size: tuple[int, int] = (1, 1)
    channel_count: int
    function: Callable[[bytes, int, int], bytes]

    def decode_data(self, texture: Texture2D) -> bytes:
        data = self.get_level_data(texture, 0)
        return self.function(bytes(data), texture.m_Width, texture.m_Height)

    def decode_pil(self, texture: Texture2D) -> Image.Image:
        image_data = self.decode_data(texture)
        image = Image.frombytes("RGBA", (texture.m_Width, texture.m_Height), image_data, "raw", "BGRA")  # type: ignore - PIL.Image.frombytes has no type hint
        match self.channel_count:
            case 1:
                image = image.convert("L")
            case 2:
                image = image.convert("LA")
            case 3:
                image = image.convert("RGB")
            case 4:
                pass
            case _:
                raise ValueError("Invalid channel count")
        return image
        
    def decode_raw(self, texture: Texture2D) -> np.ndarray[Any, np.dtype[np.uint8]]:
        image = self.decode_pil(texture)
        return np.array(image)

    def save_fp(self, texture: Texture2D, path: str) -> None:
        image = self.decode_pil(texture)
        image.save(path)  # type: ignore - PIL.Image.save has no type hint

    def save_stream(self, texture: Texture2D, stream: BinaryIO, ext: str) -> None:
        image = self.decode_pil(texture)
        image.save(stream, ext)  # type: ignore - PIL.Image.save has no type hint

    def patch_texture(
        self, image: Image.Image, texture: Texture2D, delta: bool = False
    ) -> None:
        raise NotImplementedError(
            "PILRawTextureDataHandler.patch_texture is not implemented"
        )

@define(kw_only=True, slots=True, frozen=True)
class Texture2DDecoderCrunchTextureDataHandler(Texture2DDecoderTextureDataHandler):
    bpp: float = -1
    block_size: tuple[int, int] = (1, 1)
    channel_count: int = -1
    function: Callable[[bytes, int, int], bytes] = t2dd.decode_crunch
    
    def decode_data(self, texture: Texture2D) -> bytes:
        object_info = texture.object_info
        if object_info is None:
            raise ValueError("Texture2D with crunch format has no object info!")
        version = object_info.assetsfile.get_unityversion()
        if version > UnityVersion.fromList(2017, 3) or texture.m_TextureFormat == TextureFormat.ETC_RGB4Crunched or texture.m_TextureFormat == TextureFormat.ETC2_RGBA8Crunched:
            function = t2dd.decode_unity_crunch
        else:
            function = t2dd.decode_crunch
        
        return function(texture.image_data, texture.m_Width, texture.m_Height)


class OpenCVRGBE9EFTextureDataHandler(OpenCVTextureDataHandler[np.float32]):
    bpp: int = 32
    channel_count: int = 3
    dtype: npt.DTypeLike = np.float32

    def decode_raw(self, texture: Texture2D) -> npt.NDArray[np.float32]:
        return self._decode_rgb9e5float(texture)

    def _decode_rgb9e5float(self, texture: Texture2D) -> npt.NDArray[np.float32]:
        data = self.get_level_data(texture, 0)
        # unsure if it's really int32 and not uint32
        entries = np.frombuffer(data, dtype=np.int32)
        res = np.zeros((texture.m_Height, texture.m_Width, 3), dtype=np.float32)
        for entry, target in zip(entries.reshape(-1, 3), res):
            scale = entry >> 27 & 0x1F
            scalef = 2 ** (scale - 24)
            target[:] = (
                # blue
                (entry >> 18 & 0x1FF),
                # green
                (entry >> 9 & 0x1FF),
                # red
                (entry & 0x1FF),
            )
            target *= scalef

        return res

@define(kw_only=True, slots=True, frozen=True)
class PILYUY2TextureDataHandler(PILRawTextureDataHandler):
    def decode_yuy2(self, texture: Texture2D) -> npt.NDArray[np.uint8]:
        # see: https://github.com/mafaca/Yuy2/blob/master/Yuy2/Yuy2Decoder.cs
        output = np.zeros((texture.m_Height, texture.m_Width, 3), dtype=np.int16)
        for p, o in zip(range(0, texture.m_Height * texture.m_Width // 2, 4), range(0, len(output), 6)):
            y0, u0, y1, v0 = texture.image_data[p:p+4]
            # c = y0 - 16
            c1 = 298 * (y0 - 16)
            c2 = 298 * (y1 - 16)
            d = u0 - 128
            e = v0 - 128
            
            output[o:o+6] = [
                # pixel 1
                (c1 + 409 * e + 128) >> 8, # red
                (c1 - 100 * d - 208 * e + 128) >> 8, # green
                (c1 + 516 * d + 128) >> 8, # blue,
                # pixel 2
                (c2 + 409 * e + 128) >> 8, # red
                (c2 - 100 * d - 208 * e + 128) >> 8, # green
                (c2 + 516 * d + 128) >> 8, # blue,
            ]
        
        output = output.clip(0, 255)
        return output.astype(np.uint8)
    
    def decode_pil(self, texture: Texture2D) -> Image.Image:
        image_data = self.decode_yuy2(texture)
        return Image.fromarray(image_data, mode="RGB") # type: ignore - PIL.Image.fromarray has no type hint

TEXTURE_FORMAT_INFOS: dict[TextureFormat, TextureDataHandler] = {
    TextureFormat.Alpha8: PILRawTextureDataHandler(
        bpp=8,
        channel_count=1,
        mode="L",
        decoder_args=("L",),
    ),
    TextureFormat.ARGB4444: PILRawTextureDataHandler(
        bpp=16,
        channel_count=4,
        mode="RGBA",
        decoder_args=("RGBA;4B",),
        band_reorder=(2, 1, 0, 3),
    ),
    TextureFormat.RGB24: PILRawTextureDataHandler(
        bpp=24,
        channel_count=3,
        mode="RGB",
        decoder_args=("RGB",),
    ),
    TextureFormat.RGBA32: PILRawTextureDataHandler(
        bpp=32,
        channel_count=4,
        mode="RGBA",
        decoder_args=("RGBA",),
    ),
    TextureFormat.ARGB32: PILRawTextureDataHandler(
        bpp=32,
        channel_count=4,
        mode="RGBA",
        decoder_args=("ARGB",),
    ),
    TextureFormat.RGB565: PILRawTextureDataHandler(
        # ref: https://github.com/python-pillow/Pillow/issues/2160
        # to be tested
        bpp=16,
        channel_count=3,
        mode="RGB",
        decoder_args=("RGB;16",),
    ),
    TextureFormat.R16: OpenCVTextureDataHandler(
        bpp=16,
        channel_count=1,
        dtype=np.uint16,
    ),
    TextureFormat.DXT1: PILRawTextureDataHandler(
        bpp=4,
        channel_count=3,
        mode="RGBA",
        decoder_name="bcn",
        decoder_args=(1,"DXT1"),
    ),
    TextureFormat.DXT5: PILRawTextureDataHandler(
        bpp=8,
        channel_count=4,
        mode="RGBA",
        decoder_name="bcn",
        decoder_args=(3,"DXT5"),
    ),
    TextureFormat.RGBA4444: PILRawTextureDataHandler(
        bpp=16,
        channel_count=4,
        mode="RGBA",
        decoder_args=("RGBA;4B",),
    ),
    TextureFormat.BGRA32: PILRawTextureDataHandler(
        bpp=32,
        channel_count=4,
        mode="RGBA",
        decoder_args=("BGRA",),
    ),
    TextureFormat.RHalf: OpenCVTextureDataHandler(
        bpp=16,
        dtype=np.float16,
        channel_count=1,
    ),
    TextureFormat.RGHalf: OpenCVTextureDataHandler(
        bpp=32,
        dtype=np.float16,
        channel_count=2,
    ),
    TextureFormat.RGBAHalf: OpenCVTextureDataHandler(
        bpp=48,
        dtype=np.float16,
        channel_count=3,
    ),
    TextureFormat.RFloat: OpenCVTextureDataHandler(
        bpp=32,
        dtype=np.float32,
        channel_count=1,
    ),
    TextureFormat.RGFloat: OpenCVTextureDataHandler(
        bpp=64,
        dtype=np.float32,
        channel_count=2,
    ),
    TextureFormat.RGBAFloat: OpenCVTextureDataHandler(
        bpp=96,
        dtype=np.float32,
        channel_count=3,
    ),
    TextureFormat.YUY2: PILYUY2TextureDataHandler(
        bpp = 16,
        channel_count = 3,
        mode = "RGB",
    ),
    TextureFormat.RGB9e5Float: OpenCVRGBE9EFTextureDataHandler(
        bpp=32,
        channel_count=3,
        dtype=np.float32,
    ),
    TextureFormat.BC4: PILRawTextureDataHandler(
        bpp=8,
        channel_count=1,
        mode="L",
        decoder_name="bcn",
        decoder_args=(4,"BC4"),
    ),
    TextureFormat.BC5: PILRawTextureDataHandler(
        bpp=16,
        channel_count=2,
        mode="RGB",
        decoder_name="bcn",
        decoder_args=(5,"BC5"),
    ),
    TextureFormat.BC6H: PILRawTextureDataHandler(
        bpp=16/3,
        channel_count=3,
        mode="RGB",
        decoder_name="bcn",
        decoder_args=(6,"BC6H"),
    ),
    TextureFormat.BC7: PILRawTextureDataHandler(
        bpp=8,
        channel_count=4,
        mode="RGBA",
        decoder_name="bcn",
        decoder_args=(7,"BC7"),
    ),
    TextureFormat.DXT1Crunched: Texture2DDecoderCrunchTextureDataHandler(),
    TextureFormat.DXT5Crunched: Texture2DDecoderCrunchTextureDataHandler(),
    TextureFormat.PVRTC_RGB2: Texture2DDecoderTextureDataHandler(
        bpp=2,
        channel_count=3,
        function=t2dd.decode_pvrtc_2bpp,
    ),
    TextureFormat.PVRTC_RGBA2: Texture2DDecoderTextureDataHandler(
        bpp=2,
        channel_count=4,
        function=t2dd.decode_pvrtc_2bpp,
    ),
    TextureFormat.PVRTC_RGB4: Texture2DDecoderTextureDataHandler(
        bpp=2,
        channel_count=3,
        function=t2dd.decode_pvrtc_4bpp,
    ),
    TextureFormat.PVRTC_RGBA4: Texture2DDecoderTextureDataHandler(
        bpp=2,
        channel_count=4,
        function=t2dd.decode_pvrtc_4bpp,
    ),
    TextureFormat.ETC_RGB4: Texture2DDecoderTextureDataHandler(
        bpp=4,
        block_size=(4, 4),
        channel_count=3,
        function=t2dd.decode_etc1,
    ),
    TextureFormat.EAC_R: Texture2DDecoderTextureDataHandler(
        bpp=4,
        block_size=(4, 4),
        channel_count=1,
        function=t2dd.decode_eacr,
    ),
    TextureFormat.EAC_R_SIGNED: Texture2DDecoderTextureDataHandler(
        bpp=4,
        block_size=(4, 4),
        channel_count=1,
        function=t2dd.decode_eacr_signed,
    ),
    TextureFormat.EAC_RG: Texture2DDecoderTextureDataHandler(
        bpp=8,
        block_size=(4, 4),
        channel_count=2,
        function=t2dd.decode_eacrg,
    ),
    TextureFormat.EAC_RG_SIGNED: Texture2DDecoderTextureDataHandler(
        bpp=8,
        block_size=(4, 4),
        channel_count=2,
        function=t2dd.decode_eacrg_signed,
    ),
    TextureFormat.ETC2_RGB: Texture2DDecoderTextureDataHandler(
        bpp=4,
        block_size=(4, 4),
        channel_count=3,
        function=t2dd.decode_etc2_rgb,
    ),
    TextureFormat.ETC2_RGBA1: Texture2DDecoderTextureDataHandler(
        bpp=4,
        block_size=(4, 4),
        channel_count=4,
        function=t2dd.decode_etc2_rgba1,
    ),
    TextureFormat.ETC2_RGBA8: Texture2DDecoderTextureDataHandler(
        bpp=8,
        block_size=(4, 4),
        channel_count=4,
        function=t2dd.decode_etc2_rgba8,
    ),
    TextureFormat.ASTC_4x4: Texture2DDecoderTextureDataHandler(
        bpp=8,
        block_size=(4, 4),
        channel_count=4,
        function=t2dd.decode_astc_4_4,
    ),
    TextureFormat.ASTC_5x5: Texture2DDecoderTextureDataHandler(
        bpp=5.12,
        block_size=(5, 5),
        channel_count=4,
        function=t2dd.decode_astc_5_5,
    ),
    TextureFormat.ASTC_6x6: Texture2DDecoderTextureDataHandler(
        bpp=3.56,
        block_size=(6, 6),
        channel_count=4,
        function=t2dd.decode_astc_6_6,
    ),
    TextureFormat.ASTC_8x8: Texture2DDecoderTextureDataHandler(
        bpp=2,
        block_size=(8, 8),
        channel_count=4,
        function=t2dd.decode_astc_8_8,
    ),
    TextureFormat.ASTC_10x10: Texture2DDecoderTextureDataHandler(
        bpp=1.28,
        block_size=(10, 10),
        channel_count=4,
        function=t2dd.decode_astc_10_10,
    ),
    TextureFormat.ASTC_12x12: Texture2DDecoderTextureDataHandler(
        bpp=0.89,
        block_size=(12, 12),
        channel_count=4,
        function=t2dd.decode_astc_12_12,
    ),
    TextureFormat.ETC_RGB4_3DS: Texture2DDecoderTextureDataHandler(
        bpp=4,
        block_size=(4, 4),
        channel_count=3,
        function=t2dd.decode_etc1,
    ),
    TextureFormat.ETC_RGBA8_3DS: Texture2DDecoderTextureDataHandler(
        bpp=8,
        block_size=(4, 4),
        channel_count=4,
        function=t2dd.decode_etc1,
    ),
    TextureFormat.RG16: PILRawTextureDataHandler(
        bpp=16,
        channel_count=2,
        mode="LA",
        decoder_args=("LA",),
    ),
    TextureFormat.R8: PILRawTextureDataHandler(
        bpp=8,
        channel_count=1,
        mode="L",
        decoder_args=("L",),
    ),
    TextureFormat.ETC_RGB4Crunched: Texture2DDecoderCrunchTextureDataHandler(),
    TextureFormat.ETC2_RGBA8Crunched: Texture2DDecoderCrunchTextureDataHandler(),
    TextureFormat.ASTC_HDR_4x4: Texture2DDecoderTextureDataHandler(
        bpp=8,
        block_size=(4, 4),
        channel_count=4,
        function=t2dd.decode_astc_4_4,
    ),
    TextureFormat.ASTC_HDR_5x5: Texture2DDecoderTextureDataHandler(
        bpp=5.12,
        block_size=(5, 5),
        channel_count=4,
        function=t2dd.decode_astc_5_5,
    ),
    TextureFormat.ASTC_HDR_6x6: Texture2DDecoderTextureDataHandler(
        bpp=3.56,
        block_size=(6, 6),
        channel_count=4,
        function=t2dd.decode_astc_6_6,
    ),
    TextureFormat.ASTC_HDR_8x8: Texture2DDecoderTextureDataHandler(
        bpp=2,
        block_size=(8, 8),
        channel_count=4,
        function=t2dd.decode_astc_8_8,
    ),
    TextureFormat.ASTC_HDR_10x10: Texture2DDecoderTextureDataHandler(
        bpp=1.28,
        block_size=(10, 10),
        channel_count=4,
        function=t2dd.decode_astc_10_10,
    ),
    TextureFormat.ASTC_HDR_12x12: Texture2DDecoderTextureDataHandler(
        bpp=0.89,
        block_size=(12, 12),
        channel_count=4,
        function=t2dd.decode_astc_12_12,
    ),
    TextureFormat.RG32: OpenCVTextureDataHandler(
        bpp=32,
        channel_count=2,
        dtype=np.uint16,
    ),
    TextureFormat.RGB48: OpenCVTextureDataHandler(
        bpp=48,
        channel_count=3,
        dtype=np.uint16,
    ),
    TextureFormat.RGBA64: OpenCVTextureDataHandler(
        bpp=64,
        channel_count=4,
        dtype=np.uint16,
    ),
    TextureFormat.R8_SIGNED: OpenCVTextureDataHandler(
        bpp=8,
        channel_count=1,
        dtype=np.int8,
    ),
    TextureFormat.RG16_SIGNED: OpenCVTextureDataHandler(
        bpp=16,
        channel_count=2,
        dtype=np.int8,
    ),
    TextureFormat.RGB24_SIGNED: OpenCVTextureDataHandler(
        bpp=24,
        channel_count=3,
        dtype=np.int8,
    ),
    TextureFormat.RGBA32_SIGNED: OpenCVTextureDataHandler(
        bpp=32,
        channel_count=4,
        dtype=np.int8,
    ),
    TextureFormat.R16_SIGNED: OpenCVTextureDataHandler(
        bpp=16,
        channel_count=1,
        dtype=np.int16,
    ),
    TextureFormat.RG32_SIGNED: OpenCVTextureDataHandler(
        bpp=32,
        channel_count=2,
        dtype=np.int16,
    ),
    TextureFormat.RGB48_SIGNED: OpenCVTextureDataHandler(
        bpp=48,
        channel_count=3,
        dtype=np.int16,
    ),
    TextureFormat.RGBA64_SIGNED: OpenCVTextureDataHandler(
        bpp=64,
        channel_count=4,
        dtype=np.int16,
    ),
    TextureFormat.ASTC_RGB_4x4: Texture2DDecoderTextureDataHandler(
        bpp=8,
        block_size=(4, 4),
        channel_count=3,
        function=t2dd.decode_astc_4_4,
    ),
    TextureFormat.ASTC_RGB_5x5: Texture2DDecoderTextureDataHandler(
        bpp=5.12,
        block_size=(5, 5),
        channel_count=3,
        function=t2dd.decode_astc_5_5,
    ),
    TextureFormat.ASTC_RGB_6x6: Texture2DDecoderTextureDataHandler(
        bpp=3.56,
        block_size=(6, 6),
        channel_count=3,
        function=t2dd.decode_astc_6_6,
    ),
    TextureFormat.ASTC_RGB_8x8: Texture2DDecoderTextureDataHandler(
        bpp=2,
        block_size=(8, 8),
        channel_count=3,
        function=t2dd.decode_astc_8_8,
    ),
    TextureFormat.ASTC_10x10: Texture2DDecoderTextureDataHandler(
        bpp=1.28,
        block_size=(10, 10),
        channel_count=3,
        function=t2dd.decode_astc_10_10,
    ),
    TextureFormat.ASTC_RGB_12x12: Texture2DDecoderTextureDataHandler(
        bpp=0.89,
        block_size=(12, 12),
        channel_count=3,
        function=t2dd.decode_astc_12_12,
    ),
    TextureFormat.ASTC_RGBA_4x4: Texture2DDecoderTextureDataHandler(
        bpp=8,
        block_size=(4, 4),
        channel_count=4,
        function=t2dd.decode_astc_4_4,
    ),
    TextureFormat.ASTC_RGBA_5x5: Texture2DDecoderTextureDataHandler(
        bpp=5.12,
        block_size=(5, 5),
        channel_count=4,
        function=t2dd.decode_astc_5_5,
    ),
    TextureFormat.ASTC_RGBA_6x6: Texture2DDecoderTextureDataHandler(
        bpp=3.56,
        block_size=(6, 6),
        channel_count=4,
        function=t2dd.decode_astc_6_6,
    ),
    TextureFormat.ASTC_RGBA_8x8: Texture2DDecoderTextureDataHandler(
        bpp=2,
        block_size=(8, 8),
        channel_count=4,
        function=t2dd.decode_astc_8_8,
    ),
    TextureFormat.ASTC_RGBA_10x10: Texture2DDecoderTextureDataHandler(
        bpp=1.28,
        block_size=(10, 10),
        channel_count=4,
        function=t2dd.decode_astc_10_10,
    ),
    TextureFormat.ASTC_RGBA_12x12: Texture2DDecoderTextureDataHandler(
        bpp=0.89,
        block_size=(12, 12),
        channel_count=4,
        function=t2dd.decode_astc_12_12,
    ),
}


def get_texture_data_handler(texture: Texture2D) -> TextureDataHandler:
    return TEXTURE_FORMAT_INFOS[TextureFormat(texture.m_TextureFormat)]

def export_as_dds(texture: Texture2D) -> bytes:
    raise NotImplementedError("export_as_dds is not implemented yet")

def export_as_ktx2(texture: Texture2D) -> bytes:
    raise NotImplementedError("export_as_ktx2 is not implemented yet")
