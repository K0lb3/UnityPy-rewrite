from __future__ import annotations
from enum import IntEnum
from typing import Optional, Any
import numpy as np
import numpy.typing as npt
from ..typetree.Tpk import UnityVersion

from ..objects import Mesh, ChannelInfo, StreamInfo
from .PacketBitVector import unpack_floats, unpack_ints

class MeshHandler:
    mesh: Mesh
    version: UnityVersion

    m_Vertices: Optional[npt.NDArray[np.float32]] = None
    m_Normals: Optional[npt.NDArray[np.float32]] = None
    m_Colors: Optional[npt.NDArray[np.float32]] = None
    m_UV0: Optional[npt.NDArray[np.float32]] = None
    m_UV1: Optional[npt.NDArray[np.float32]] = None
    m_UV2: Optional[npt.NDArray[np.float32]] = None
    m_UV3: Optional[npt.NDArray[np.float32]] = None
    m_UV4: Optional[npt.NDArray[np.float32]] = None
    m_UV5: Optional[npt.NDArray[np.float32]] = None
    m_UV6: Optional[npt.NDArray[np.float32]] = None
    m_UV7: Optional[npt.NDArray[np.float32]] = None
    m_Tangents: Optional[npt.NDArray[np.float32]] = None
    m_BoneIndices: Optional[npt.NDArray[np.uint32]] = None
    m_BoneWeights: Optional[npt.NDArray[np.float32]] = None


    def __init__(self, mesh: Mesh, version: Optional[UnityVersion] = None) -> None:
        self.mesh = mesh
        if version != None:
            self.version = version
        else:
            object_info = mesh.object_info
            if object_info is None:
                raise ValueError("Mesh object info is None")
            self.version = object_info.assetsfile.get_unityversion()

    def process(self):
        vertex_data = self.mesh.m_VertexData
        assert vertex_data is not None

        m_Channels: list[ChannelInfo]
        m_Streams: list[StreamInfo]

        if self.version.major < 4:
            assert (
                vertex_data.m_Streams_0_ is not None
                and vertex_data.m_Streams_1_ is not None
                and vertex_data.m_Streams_2_ is not None
                and vertex_data.m_Streams_3_ is not None
            )
            m_Streams = [
                vertex_data.m_Streams_0_,
                vertex_data.m_Streams_1_,
                vertex_data.m_Streams_2_,
                vertex_data.m_Streams_3_,
            ]
            assert all(stream is not None for stream in m_Streams)
            m_Channels = self.get_channels(m_Streams)
        elif self.version.major == 4:
            assert (
                vertex_data.m_Streams is not None and vertex_data.m_Channels is not None
            )
            m_Streams = vertex_data.m_Streams
            m_Channels = vertex_data.m_Channels
        else:
            assert vertex_data.m_Channels is not None
            m_Channels = vertex_data.m_Channels
            m_Streams = self.get_streams(m_Channels, vertex_data.m_VertexCount)

        steam_data = self.mesh.m_StreamData
        if steam_data and steam_data.path:
            vertex_data = self.mesh.m_VertexData
            if vertex_data and vertex_data.m_VertexCount > 0:
                raise NotImplementedError("External data is not supported")
                # resourceReader = new ResourceReader(m_StreamData.path, assetsFile, m_StreamData.offset, m_StreamData.size)
                # m_VertexData.m_DataSize = resourceReader.GetData()

        if self.version >= UnityVersion.fromList(3, 5):
            self.read_vertex_data(m_Channels, m_Streams)

        if self.version >= UnityVersion.fromList(2, 6):
            self.decompress_compressed_mesh()

        # self.get_triangles()

    def get_streams(
        self, m_Channels: list[ChannelInfo], m_VertexCount: int
    ) -> list[StreamInfo]:
        streamCount = 1
        if m_Channels:
            streamCount += max(x.stream for x in m_Channels)

        m_Streams: list[StreamInfo] = []
        offset = 0
        for s in range(streamCount):
            chnMask = 0
            stride = 0
            for chn, m_Channel in enumerate(m_Channels):
                if m_Channel.stream == s:
                    if m_Channel.dimension > 0:
                        chnMask |= 1 << chn
                        component_size = self.get_channel_component_size(m_Channel)
                        stride += m_Channel.dimension * component_size

            m_Streams.append(
                StreamInfo(
                    channelMask=chnMask,
                    offset=offset,
                    stride=stride,
                    dividerOp=0,
                    frequency=0,
                )
            )
            offset += m_VertexCount * stride
            offset = (offset + (16 - 1)) & ~(16 - 1)
        return m_Streams

    def get_channels(self, m_Streams: list[StreamInfo]) -> list[ChannelInfo]:
        m_Channels = [
            ChannelInfo(
                dimension=0,
                format=0,
                offset=0,
                stream=0,
            )
            for _ in range(6)
        ]
        for s, m_Stream in enumerate(m_Streams):
            channelMask = bytearray(m_Stream.channelMask)  # BitArray
            offset = 0
            for i in range(6):
                if channelMask[i]:
                    m_Channel = m_Channels[i]
                    m_Channel.stream = s
                    m_Channel.offset = offset
                    if i in [0, 1]:
                        # 0 - kShaderChannelVertex
                        # 1 - kShaderChannelNormal
                        m_Channel.format = 0  # kChannelFormatFloat
                        m_Channel.dimension = 3
                    elif i == 2:  # kShaderChannelColor
                        m_Channel.format = 2  # kChannelFormatColor
                        m_Channel.dimension = 4
                    elif i in [3, 4]:
                        # 3 - kShaderChannelTexCoord0
                        # 4 - kShaderChannelTexCoord1
                        m_Channel.format = 0  # kChannelFormatFloat
                        m_Channel.dimension = 2
                    elif i == 5:  # kShaderChannelTangent
                        m_Channel.format = 0  # kChannelFormatFloat
                        m_Channel.dimension = 4
                    
                    component_size = self.get_channel_component_size(m_Channel)
                    offset += m_Channel.dimension * component_size
        
        return m_Channels

    def read_vertex_data(
        self, m_Channels: list[ChannelInfo], m_Streams: list[StreamInfo]
    ) -> None:
        m_VertexData = self.mesh.m_VertexData
        if m_VertexData is None:
            raise ValueError("Vertex data is None")

        m_VertexCount = m_VertexData.m_VertexCount
        m_VertexDataRaw = np.frombuffer(m_VertexData.m_DataSize, dtype=np.uint8)

        for chn, m_Channel in enumerate(m_Channels):
            if m_Channel.dimension == 0:
                continue

            m_Stream = m_Streams[m_Channel.stream]
            m_StreamData = m_VertexDataRaw[
                m_Stream.offset : m_Stream.offset + m_VertexCount * m_Stream.stride
            ].reshape(m_VertexCount, m_Stream.stride)

            channelMask = bin(m_Stream.channelMask)[::-1]
            if channelMask[chn] == "1":
                if (
                    self.version.major < 2018 and chn == 2 and m_Channel.format == 2
                ):  # kShaderChannelColor && kChannelFormatColor
                    # new instance to not modify the original
                    m_Channel = ChannelInfo(
                        dimension=4,
                        format=2,
                        offset=m_Channel.offset,
                        stream=m_Channel.stream,
                    )

                component_dtype = self.get_channel_dtype(m_Channel)
                component_byte_size = np.array([], dtype=component_dtype).itemsize
                channel_byte_size = m_Channel.dimension * component_byte_size
                component_data = m_StreamData[
                    :,
                    m_Channel.offset : m_Channel.offset + channel_byte_size,
                ].view(dtype=component_dtype)
                print(component_dtype)
                self.assign_channel_vertex_data(chn, component_data)
                
    def assign_channel_vertex_data(self, channel: int, component_data: npt.NDArray[Any]):
        if self.version.major >= 2018:
            match channel:
                case 0: # kShaderChannelVertex
                    self.m_Vertices = component_data
                case 1: # kShaderChannelNormal
                    self.m_Normals = component_data
                case 2: # kShaderChannelTangent
                    self.m_Tangents = component_data
                case 3: # kShaderChannelColor
                    self.m_Colors = component_data
                case 4: # kShaderChannelTexCoord0
                    self.m_UV0 = component_data
                case 5: # kShaderChannelTexCoord1
                    self.m_UV1 = component_data
                case 6: # kShaderChannelTexCoord2
                    self.m_UV2 = component_data
                case 7: # kShaderChannelTexCoord3
                    self.m_UV3 = component_data
                case 8: # kShaderChannelTexCoord4
                    self.m_UV4 = component_data
                case 9: # kShaderChannelTexCoord5
                    self.m_UV5 = component_data
                case 10: # kShaderChannelTexCoord6
                    self.m_UV6 = component_data
                case 11: # kShaderChannelTexCoord7
                    self.m_UV7 = component_data
                # 2018.2 and up
                case 12: # kShaderChannelBlendWeight
                    self.m_BoneWeights = component_data
                case 13: # kShaderChannelBlendIndices
                    self.m_BoneIndices = component_data
                case _:
                    raise ValueError(f"Unknown channel {channel}")
        else:
            match channel:
                case 0: # kShaderChannelVertex
                    self.m_Vertices = component_data
                case 1: # kShaderChannelNormal
                    self.m_Normals = component_data
                case 2: # kShaderChannelColor
                    self.m_Colors = component_data
                case 3: # kShaderChannelTexCoord0
                    self.m_UV0 = component_data
                case 4: # kShaderChannelTexCoord1
                    self.m_UV1 = component_data
                case 5:
                    if self.version.major >= 5: # kShaderChannelTexCoord2
                        self.m_UV2 = component_data
                    else: # kShaderChannelTangent
                        self.m_Tangents = component_data
                case 6: # kShaderChannelTexCoord3
                    self.m_UV3 = component_data
                case 7: # kShaderChannelTangent
                    self.m_Tangents = component_data
                case _:
                    raise ValueError(f"Unknown channel {channel}")
    
    def get_channel_dtype(self, m_Channel: ChannelInfo):
        if self.version.major < 2017:
            format = VertexChannelFormat(m_Channel.format)
            component_dtype = VERTEX_CHANNEL_FORMAT_DTYPE_MAP[format]
        elif self.version.major < 2019:
            format = VertexFormat2017(m_Channel.format)
            component_dtype = VERTEX_FORMAT_2017_DTYPE_MAP[format]
        else:
            format = VertexFormat(m_Channel.format)
            component_dtype = VERTEX_FORMAT_DTYPE_MAP[format]

        return component_dtype
    
    def get_channel_component_size(self, m_Channel: ChannelInfo):
        dtype = self.get_channel_dtype(m_Channel)
        return np.array([], dtype=dtype).itemsize

    def decompress_compressed_mesh(self):
        assert self.mesh.m_CompressedMesh is not None
        # Vertex
        version = self.version
        m_CompressedMesh = self.mesh.m_CompressedMesh
        if m_CompressedMesh.m_Vertices.m_NumItems <= 0:
            return # no vertices, doesn't make sense to continue
        
        m_VertexCount = int(m_CompressedMesh.m_Vertices.m_NumItems / 3)
        self.m_Vertices = unpack_floats(m_CompressedMesh.m_Vertices, 3, 3 * 4).reshape(-1, 3)

        # UV
        if m_CompressedMesh.m_UV.m_NumItems > 0:
            m_UVInfo = m_CompressedMesh.m_UVInfo
            if m_UVInfo is not None and m_UVInfo != 0:
                kInfoBitsPerUV = 4
                kUVDimensionMask = 3
                kUVChannelExists = 4
                kMaxTexCoordShaderChannels = 8

                uvSrcOffset = 0

                for uv_channel in range(kMaxTexCoordShaderChannels):
                    texCoordBits = m_UVInfo >> (uv_channel * kInfoBitsPerUV)
                    texCoordBits &= (1 << kInfoBitsPerUV) - 1
                    if (texCoordBits & kUVChannelExists) != 0:
                        uvDim = 1 + int(texCoordBits & kUVDimensionMask)
                        m_UV = unpack_floats(m_CompressedMesh.m_UV,
                            uvDim, uvDim * 4, uvSrcOffset, m_VertexCount
                        ).reshape(-1, uvDim)
                        setattr(self, f"m_UV{uv_channel}", m_UV)
            else:
                self.m_UV0 = unpack_floats(m_CompressedMesh.m_UV,
                    2, 2 * 4, 0, m_VertexCount
                ).reshape(-1, 2)
                if m_CompressedMesh.m_UV.m_NumItems >= m_VertexCount * 4:
                    self.m_UV1 = unpack_floats(m_CompressedMesh.m_UV,
                        2, 2 * 4, m_VertexCount * 2, m_VertexCount
                    ).reshape(-1, 2)

        # BindPose
        if version.major < 5:  # 5.0 down
            m_BindPoses = m_CompressedMesh.m_BindPoses
            if m_BindPoses and m_BindPoses.m_NumItems > 0: 
                self.m_BindPose = unpack_floats(m_BindPoses,16, 4 * 16).reshape(-1, 4, 4)

        # Normal
        if m_CompressedMesh.m_Normals.m_NumItems > 0:
            normalData = unpack_floats(m_CompressedMesh.m_Normals, 2, 4 * 2).reshape(-1, 2)
            signs = unpack_ints(m_CompressedMesh.m_NormalSigns)
            
            self.m_Normals = np.zeros((m_CompressedMesh.m_Normals.m_NumItems // 2, 3), dtype=normalData.dtype)
            for srcNrm, sign, dstNrm in zip(normalData, signs, self.m_Normals):
                x,y  = srcNrm
                zsqr = 1 - x * x - y * y
                if zsqr >= 0:
                    z = np.sqrt(zsqr)
                    dstNrm[:] = x, y, z
                else:
                    z = 0
                    dstNrm[:] = x, y, z
                    dstNrm /= np.linalg.norm(dstNrm)
                if sign == 0:
                    dstNrm[2] *= -1
        
        # Tangent
        if m_CompressedMesh.m_Tangents.m_NumItems > 0:
            tangentData = unpack_floats(m_CompressedMesh.m_Tangents, 2, 4 * 2)
            signs = unpack_ints(m_CompressedMesh.m_TangentSigns)
            self.m_Tangents = np.zeros((m_CompressedMesh.m_Tangents.m_NumItems // 2, 4), dtype=tangentData.dtype)
            for srcTan, (sign_z, sign_w), dstTan in zip(tangentData, signs, self.m_Tangents):
                x,y = srcTan
                zsqr = 1 - x * x - y * y
                z = 0
                w = 0
                if zsqr >= 0:
                    z = np.sqrt(zsqr)
                else:
                    vec3 = np.array([x, y, 0], dtype=tangentData.dtype)
                    vec3 /= np.linalg.norm(vec3)
                    x, y, z = vec3
                if sign_z == 0:
                    z = -z
                w = 1.0 if sign_w > 0 else -1.0
                dstTan[:] = x, y, z, w

        # FloatColor
        if version.major >= 5:  # 5.0 and up
            m_FloatColors = m_CompressedMesh.m_FloatColors
            if m_FloatColors and m_FloatColors.m_NumItems > 0:
                self.m_Colors = unpack_floats(m_FloatColors, 1, 4).reshape(-1, 4)
        # Skin
        if m_CompressedMesh.m_Weights.m_NumItems > 0:
            weightsData = unpack_ints(m_CompressedMesh.m_Weights).astype(np.float32)/31
            boneIndicesData = unpack_ints(m_CompressedMesh.m_BoneIndices)
            
            vertexIndex = 0
            boneIndecesIndex = 0
            j = 0
            sum = 0

            self.m_BoneWeights = np.zeros((m_CompressedMesh.m_Weights.m_NumItems // 4, 4), dtype=np.float32)
            self.m_BoneIndices = np.zeros((m_CompressedMesh.m_Weights.m_NumItems // 4, 4), dtype=np.uint32)
            
            
            for weight in weightsData:
                # read bone index and weight
                self.m_BoneWeights[vertexIndex, j] = weight
                self.m_BoneIndices[vertexIndex, j] = boneIndicesData[boneIndecesIndex]

                boneIndecesIndex += 1
                j += 1
                sum += weight

                # the weights add up to one, continue with the next vertex.
                if sum >= 1.0:
                    j = 4

                    vertexIndex += 1
                    j = 0
                    sum = 0
                # we read three weights, but they don't add up to one. calculate the fourth one, and read
                # missing bone index. continue with next vertex.
                elif j == 3:  #
                    self.m_BoneWeights[vertexIndex, j] = (1 - sum)
                    self.m_BoneIndices[vertexIndex, j] = boneIndicesData[boneIndecesIndex]
                    
                    boneIndecesIndex += 1
                    vertexIndex += 1
                    j = 0
                    sum = 0
        
        # IndexBuffer
        if m_CompressedMesh.m_Triangles.m_NumItems > 0:  #
            self.m_IndexBuffer = unpack_ints(m_CompressedMesh.m_Triangles)
        # Color
        if m_CompressedMesh.m_Colors and m_CompressedMesh.m_Colors.m_NumItems > 0:
            self.m_Colors = unpack_ints(m_CompressedMesh.m_Colors).view(np.uint8).reshape(-1, 4).astype(np.float32) / 255.0


class VertexChannelFormat(IntEnum):
    kChannelFormatFloat = 0
    kChannelFormatFloat16 = 1
    kChannelFormatColor = 2
    kChannelFormatByte = 3
    kChannelFormatUInt32 = 4


class VertexFormat2017(IntEnum):
    kVertexFormatFloat = 0
    kVertexFormatFloat16 = 1
    kVertexFormatColor = 2
    kVertexFormatUNorm8 = 3
    kVertexFormatSNorm8 = 4
    kVertexFormatUNorm16 = 5
    kVertexFormatSNorm16 = 6
    kVertexFormatUInt8 = 7
    kVertexFormatSInt8 = 8
    kVertexFormatUInt16 = 9
    kVertexFormatSInt16 = 10
    kVertexFormatUInt32 = 11
    kVertexFormatSInt32 = 12


class VertexFormat(IntEnum):
    kVertexFormatFloat = 0
    kVertexFormatFloat16 = 1
    kVertexFormatUNorm8 = 2
    kVertexFormatSNorm8 = 3
    kVertexFormatUNorm16 = 4
    kVertexFormatSNorm16 = 5
    kVertexFormatUInt8 = 6
    kVertexFormatSInt8 = 7
    kVertexFormatUInt16 = 8
    kVertexFormatSInt16 = 9
    kVertexFormatUInt32 = 10
    kVertexFormatSInt32 = 11


VERTEX_CHANNEL_FORMAT_DTYPE_MAP = {
    VertexChannelFormat.kChannelFormatFloat: np.float32,
    VertexChannelFormat.kChannelFormatFloat16: np.float16,
    VertexChannelFormat.kChannelFormatColor: np.uint8,
    VertexChannelFormat.kChannelFormatByte: np.uint8,
    VertexChannelFormat.kChannelFormatUInt32: np.uint32,
}

VERTEX_FORMAT_2017_DTYPE_MAP = {
    VertexFormat2017.kVertexFormatFloat: np.float32,
    VertexFormat2017.kVertexFormatFloat16: np.float16,
    VertexFormat2017.kVertexFormatColor: np.uint8,
    VertexFormat2017.kVertexFormatUNorm8: np.uint8,
    VertexFormat2017.kVertexFormatSNorm8: np.int8,
    VertexFormat2017.kVertexFormatUNorm16: np.uint16,
    VertexFormat2017.kVertexFormatSNorm16: np.int16,
    VertexFormat2017.kVertexFormatUInt8: np.uint8,
    VertexFormat2017.kVertexFormatSInt8: np.int8,
    VertexFormat2017.kVertexFormatUInt16: np.uint16,
    VertexFormat2017.kVertexFormatSInt16: np.int16,
    VertexFormat2017.kVertexFormatUInt32: np.uint32,
    VertexFormat2017.kVertexFormatSInt32: np.int32,
}

VERTEX_FORMAT_DTYPE_MAP = {
    VertexFormat.kVertexFormatFloat: np.float32,
    VertexFormat.kVertexFormatFloat16: np.float16,
    VertexFormat.kVertexFormatUNorm8: np.uint8,
    VertexFormat.kVertexFormatSNorm8: np.int8,
    VertexFormat.kVertexFormatUNorm16: np.uint16,
    VertexFormat.kVertexFormatSNorm16: np.int16,
    VertexFormat.kVertexFormatUInt8: np.uint8,
    VertexFormat.kVertexFormatSInt8: np.int8,
    VertexFormat.kVertexFormatUInt16: np.uint16,
    VertexFormat.kVertexFormatSInt16: np.int16,
    VertexFormat.kVertexFormatUInt32: np.uint32,
    VertexFormat.kVertexFormatSInt32: np.int32,
}
