from enum import IntEnum, IntFlag

# UnityCsReference-master/Runtime/Export/Analytics/AnalyticsCommon.cs
class SendEventOptions(IntFlag):
    kAppendNone = 0
    kAppendBuildGuid = 1 << 0
    kAppendBuildTarget = 1 << 1

# UnityCsReference-master/Runtime/Export/Animation/AnimationCurve.bindings.cs
class WeightedMode(IntEnum):
    NONE = 0
    In = 1 << 0
    Out = 1 << 1
    Both = In | Out

class WrapMode(IntEnum):
    Once = 1
    Loop = 2
    PingPong = 4
    Default = 0
    ClampForever = 8
    Clamp = 1

# UnityCsReference-master/Runtime/Export/Annotations/JetBrains.Annotations.cs
class ImplicitUseKindFlags(IntFlag):
    Access = 1
    Assign = 2
    InstantiatedWithFixedConstructorSignature = 4
    InstantiatedNoFixedConstructorSignature = 8
    Default = Access | Assign | InstantiatedWithFixedConstructorSignature

class ImplicitUseTargetFlags(IntFlag):
    Itself = 1
    Members = 2
    Default = Itself
    WithMembers = Itself | Members

class CollectionAccessType(IntFlag):
    NONE = 0
    Read = 1
    ModifyExistingContent = 2
    UpdatedContent = ModifyExistingContent | 4

class AssertionConditionType(IntEnum):
    IS_TRUE = 0
    IS_FALSE = 1
    IS_NULL = 2
    IS_NOT_NULL = 3

# UnityCsReference-master/Runtime/Export/Apple/FrameCaptureMetal.bindings.cs
class FrameCaptureDestination(IntEnum):
    DevTools = 1
    GPUTraceDocument = 2

# UnityCsReference-master/Runtime/Export/Application/Application.cs
class ApplicationMemoryUsage(IntEnum):
    Unknown = 0
    Low = 1
    Medium = 2
    High = 3
    Critical = 4

class StackTraceLogType(IntEnum):
    NONE = 1
    ScriptOnly = 2
    Full = 3

class NetworkReachability(IntEnum):
    NotReachable = 0
    ReachableViaCarrierDataNetwork = 1
    ReachableViaLocalAreaNetwork = 2

class UserAuthorization(IntEnum):
    WebCam = 1
    Microphone = 2

class ApplicationInstallMode(IntEnum):
    Unknown = 0
    Store = 1
    DeveloperBuild = 2
    Adhoc = 3
    Enterprise = 4
    Editor = 5

class ApplicationSandboxType(IntEnum):
    Unknown = 0
    NotSandboxed = 1
    Sandboxed = 2
    SandboxBroken = 3

# UnityCsReference-master/Runtime/Export/Audio/AudioType.cs
class AudioType(IntEnum):
    UNKNOWN = 0
    ACC = 1
    AIFF = 2
    IT = 10
    MOD = 12
    MPEG = 13
    OGGVORBIS = 14
    S3M = 17
    WAV = 20
    XM = 21
    XMA = 22
    VAG = 23
    AUDIOQUEUE = 24

# UnityCsReference-master/Runtime/Export/BaseClass.cs
class SendMessageOptions(IntEnum):
    RequireReceiver = 0
    DontRequireReceiver = 1

class PrimitiveType(IntEnum):
    Sphere = 0
    Capsule = 1
    Cylinder = 2
    Cube = 3
    Plane = 4
    Quad = 5

class Space(IntEnum):
    World = 0
    Self = 1

class RuntimePlatform(IntEnum):
    OSXEditor = 0
    OSXPlayer = 1
    WindowsPlayer = 2
    OSXWebPlayer = 3
    OSXDashboardPlayer = 4
    WindowsWebPlayer = 5
    WindowsEditor = 7
    IPhonePlayer = 8
    XBOX360 = 10
    PS3 = 9
    Android = 11
    NaCl = 12
    FlashPlayer = 15
    LinuxPlayer = 13
    LinuxEditor = 16
    WebGLPlayer = 17
    MetroPlayerX86 = 18
    WSAPlayerX86 = 18
    MetroPlayerX64 = 19
    WSAPlayerX64 = 19
    MetroPlayerARM = 20
    WSAPlayerARM = 20
    WP8Player = 21
    BB10Player = 22
    BlackBerryPlayer = 22
    TizenPlayer = 23
    PSP2 = 24
    PS4 = 25
    PSM = 26
    XboxOne = 27
    SamsungTVPlayer = 28
    WiiU = 30
    tvOS = 31
    Switch = 32
    Lumin = 33
    Stadia = 34
    CloudRendering = 1
    LinuxHeadlessSimulation = 35
    GameCoreScarlett = 1
    GameCoreXboxSeries = 36
    GameCoreXboxOne = 37
    PS5 = 38
    EmbeddedLinuxArm64 = 39
    EmbeddedLinuxArm32 = 40
    EmbeddedLinuxX64 = 41
    EmbeddedLinuxX86 = 42
    LinuxServer = 43
    WindowsServer = 44
    OSXServer = 45
    QNXArm32 = 46
    QNXArm64 = 47
    QNXX64 = 48
    QNXX86 = 49
    VisionOS = 50

class SystemLanguage(IntEnum):
    Afrikaans = 0
    Arabic = 1
    Basque = 2
    Belarusian = 3
    Bulgarian = 4
    Catalan = 5
    Chinese = 6
    Czech = 7
    Danish = 8
    Dutch = 9
    English = 10
    Estonian = 11
    Faroese = 12
    Finnish = 13
    French = 14
    German = 15
    Greek = 16
    Hebrew = 17
    Hugarian = 18
    Icelandic = 19
    Indonesian = 20
    Italian = 21
    Japanese = 22
    Korean = 23
    Latvian = 24
    Lithuanian = 25
    Norwegian = 26
    Polish = 27
    Portuguese = 28
    Romanian = 29
    Russian = 30
    SerboCroatian = 31
    Slovak = 32
    Slovenian = 33
    Spanish = 34
    Swedish = 35
    Thai = 36
    Turkish = 37
    Ukrainian = 38
    Vietnamese = 39
    ChineseSimplified = 40
    ChineseTraditional = 41
    Hindi = 42
    Unknown = 43
    Hungarian = 18

class LogType(IntEnum):
    Error = 0
    Assert = 1
    Warning = 2
    Log = 3
    Exception = 4

class LogOption(IntEnum):
    NONE = 0
    NoStacktrace = 1

class ThreadPriority(IntEnum):
    Low = 0
    BelowNormal = 1
    Normal = 2
    High = 4

# UnityCsReference-master/Runtime/Export/Burst/BurstCompilerService.cs
class BurstLogType(IntEnum):
    Info = 1
    Warning = 2
    Error = 3

# UnityCsReference-master/Runtime/Export/Camera/Camera.bindings.cs
class FieldOfViewAxis(IntEnum):
    Vertical = 1

class StereoscopicEye(IntEnum):
    Left = 1

class MonoOrStereoscopicEye(IntEnum):
    Left = 1

class SceneViewFilterMode(IntEnum):
    Off = 0
    ShowFiltered = 1

class RenderRequestMode(IntEnum):
    NONE = 0
    ObjectId = 1
    Depth = 2
    VertexNormal = 3
    WorldPosition = 4
    EntityId = 5
    BaseColor = 6
    SpecularColor = 7
    Metallic = 8
    Emission = 9
    Normal = 10
    Smoothness = 11
    Occlusion = 12
    DiffuseColor = 13

class RenderRequestOutputSpace(IntEnum):
    ScreenSpace = 1
    UV0 = 0
    UV1 = 1
    UV2 = 2
    UV3 = 3
    UV4 = 4
    UV5 = 5
    UV6 = 6
    UV7 = 7
    UV8 = 8

# UnityCsReference-master/Runtime/Export/Camera/CullingGroup.bindings.cs
class CullingQueryOptions(IntEnum):
    Normal = 0
    IgnoreVisibility = 1
    IgnoreDistance = 2

# UnityCsReference-master/Runtime/Export/Camera/ReflectionProbe.bindings.cs
class ReflectionProbeEvent(IntEnum):
    ReflectionProbeAdded = 0
    ReflectionProbeRemoved = 1

# UnityCsReference-master/Runtime/Export/Collections/NativeCollectionEnums.bindings.cs
class Allocator(IntEnum):
    Invalid = 0
    NONE = 1
    Temp = 2
    TempJob = 3
    Persistent = 4
    AudioKernel = 5
    Domain = 6
    FirstUserIndex = 64

class NativeLeakDetectionMode(IntEnum):
    Disabled = 1
    Enabled = 2
    EnabledWithStackTrace = 3

class LeakCategory(IntEnum):
    Invalid = 0
    Malloc = 1
    TempJob = 2
    Persistent = 3
    LightProbesQuery = 4
    NativeTest = 5
    MeshDataArray = 6
    TransformAccessArray = 7
    NavMeshQuery = 8

# UnityCsReference-master/Runtime/Export/DedicatedServer/Arguments.bindings.cs
class ArgumentErrorPolicy(IntEnum):
    Ignore = 1
    Warn = 2
    Fatal = 3

# UnityCsReference-master/Runtime/Export/DiagnosticSwitch/DiagnosticSwitch.cs
class DiagnosticSwitch_Flags(IntFlag):
    NONE = 0
    CanChangeAfterEngineStart = (1 << 0)
    PropagateToAssetImportWorkerProcess = (1 << 1)

# UnityCsReference-master/Runtime/Export/Diagnostics/DiagnosticsUtils.bindings.cs
class ForcedCrashCategory(IntEnum):
    AccessViolation = 0
    FatalError = 1
    Abort = 2
    PureVirtualFunction = 3
    MonoAbort = 4

# UnityCsReference-master/Runtime/Export/Director/FrameData.cs
class FrameData_Flags(IntFlag):
    Evaluate = 1
    SeekOccured = 2
    Loop = 4
    Hold = 8
    EffectivePlayStateDelayed = 16
    EffectivePlayStatePlaying = 32

class EvaluationType(IntEnum):
    Evaluate = 1
    Playback = 2

# UnityCsReference-master/Runtime/Export/Director/Playable.cs
class DirectorWrapMode(IntEnum):
    Hold = 0
    Loop = 1
    NONE = 2

# UnityCsReference-master/Runtime/Export/Director/PlayableBindings.cs
class DataStreamType(IntEnum):
    Animation = 0
    Audio = 1
    Texture = 2
    NONE = 3

# UnityCsReference-master/Runtime/Export/Director/PlayableExtensions.cs
class PlayableTraversalMode(IntEnum):
    Mix = 0
    Passthrough = 1

# UnityCsReference-master/Runtime/Export/Director/PlayableGraph.bindings.cs
class DirectorUpdateMode(IntEnum):
    DSPClock = 0
    GameTime = 1
    UnscaledGameTime = 2
    Manual = 3

# UnityCsReference-master/Runtime/Export/Director/PlayableHandle.bindings.cs
class PlayState(IntEnum):
    Paused = 0
    Playing = 1
    Delayed = 2

# UnityCsReference-master/Runtime/Export/ExpressionEvaluator.cs
class Op(IntEnum):
    Add = 1
    Neg = 2
    Pow = 3
    Sin = 4
    Floor = 5
    Rand = 6

class Associativity(IntEnum):
    Left = 1

# UnityCsReference-master/Runtime/Export/File/ArchiveFile.bindings.cs
class ArchiveStatus(IntEnum):
    InProgress = 1
    Complete = 2
    Failed = 3

# UnityCsReference-master/Runtime/Export/File/AsyncReadManager.bindings.cs
class FileState(IntEnum):
    Absent = 0
    Exists = 1

class FileStatus(IntEnum):
    Closed = 0
    Pending = 1
    Open = 2
    OpenFailed = 3

class AssetLoadingSubsystem(IntEnum):
    Other = 0
    Texture = 1
    VirtualTexture = 2
    Mesh = 3
    Audio = 4
    Scripts = 5
    EntitiesScene = 6
    EntitiesStreamBinaryReader = 7
    FileInfo = 8
    ContentLoading = 9

class ReadStatus(IntEnum):
    Complete = 0
    InProgress = 1
    Failed = 2
    Truncated = 4
    Canceled = 5

class Priority(IntEnum):
    PriorityLow = 0
    PriorityHigh = 1

class ProcessingState(IntEnum):
    Unknown = 0
    InQueue = 1
    Reading = 2
    Completed = 3
    Failed = 4
    Canceled = 5

class FileReadType(IntEnum):
    Sync = 0
    Async = 1

class AsyncReadManager_Flags(IntFlag):
    NONE = 0
    ClearOnRead = 1 << 0

# UnityCsReference-master/Runtime/Export/File/BuildCompression.cs
class CompressionType(IntEnum):
    NONE = 1
    Lzma = 2
    Lz4 = 3
    Lz4HC = 4

class CompressionLevel(IntEnum):
    NONE = 1
    Fastest = 2
    Fast = 3
    Normal = 4
    High = 5
    Maximum = 6

# UnityCsReference-master/Runtime/Export/File/File.bindings.cs
class ThreadIORestrictionMode(IntEnum):
    Allowed = 0
    TreatAsError = 1

# UnityCsReference-master/Runtime/Export/GI/GIDebugVisualisation.bindings.cs
class GITextureType(IntEnum):
    Charting = 1
    Albedo = 2
    Emissive = 3
    Irradiance = 4
    Directionality = 5
    Baked = 6
    BakedDirectional = 7
    InputWorkspace = 8
    BakedShadowMask = 9
    BakedAlbedo = 10
    BakedEmissive = 11
    BakedCharting = 12
    BakedTexelValidity = 13
    BakedUVOverlap = 14
    BakedLightmapCulling = 15

# UnityCsReference-master/Runtime/Export/GI/LightingSettings.bindings.cs
class Lightmapper(IntEnum):
    Enlighten = 0
    ProgressiveCPU = 1
    ProgressiveGPU = 2

class Sampling(IntEnum):
    Auto = 0
    Fixed = 1

class LightingSettings_FilterMode(IntEnum):
    NONE = 0
    Auto = 1
    Advanced = 2

class DenoiserType(IntEnum):
    NONE = 0
    Optix = 1
    OpenImage = 2
    RadeonPro = 3

class FilterType(IntEnum):
    Gaussian = 0
    ATrous = 1
    NONE = 2

# UnityCsReference-master/Runtime/Export/Graphics/GPUFence.deprecated.cs
class SynchronisationStage(IntEnum):
    VertexProcessing = 0
    PixelProcessing = 1

# UnityCsReference-master/Runtime/Export/Graphics/Graphics.bindings.cs
class EnabledOrientation(IntEnum):
    kAutorotateToPortrait = 1
    kAutorotateToPortraitUpsideDown = 2
    kAutorotateToLandscapeLeft = 4
    kAutorotateToLandscapeRight = 8

class FullScreenMode(IntEnum):
    ExclusiveFullScreen = 0
    FullScreenWindow = 1
    MaximizedWindow = 2
    Windowed = 3

class MemorylessMode(IntEnum):
    Unused = 1
    Forced = 2
    Automatic = 3

class ComputeBufferMode(IntEnum):
    Immutable = 0
    Dynamic = 1
    Circular = 2
    StreamOut = 3
    SubUpdates = 4

class D3DHDRDisplayBitDepth(IntEnum):
    D3DHDRDisplayBitDepth10 = 0
    D3DHDRDisplayBitDepth16 = 1

class WaitForPresentSyncPoint(IntEnum):
    BeginFrame = 0
    EndFrame = 1

class GraphicsJobsSyncPoint(IntEnum):
    EndOfFrame = 0
    AfterScriptUpdate = 1
    AfterScriptLateUpdate = 2
    WaitForPresent = 3

# UnityCsReference-master/Runtime/Export/Graphics/Graphics.deprecated.cs
class LightmapsModeLegacy(IntEnum):
    Single = 0
    Dual = 1
    Directional = 2

class ShaderHardwareTier(IntEnum):
    Tier1 = 0
    Tier2 = 1
    Tier3 = 2

# UnityCsReference-master/Runtime/Export/Graphics/GraphicsBuffer.bindings.cs
class Target(IntFlag):
    Vertex = 1 << 0
    Index = 1 << 1
    CopySource = 1 << 2
    CopyDestination = 1 << 3
    Structured = 1 << 4
    Raw = 1 << 5
    Append = 1 << 6
    Counter = 1 << 7
    IndirectArguments = 1 << 8
    Constant = 1 << 9

class UsageFlags(IntFlag):
    NONE = 0
    LockBufferForWrite = 1 << 0

# UnityCsReference-master/Runtime/Export/Graphics/GraphicsComponents.bindings.cs
class LightShadowCasterMode(IntEnum):
    Default = 1
    NonLightmappedOnly = 2
    Everything = 3

# UnityCsReference-master/Runtime/Export/Graphics/GraphicsEnums.cs
class RenderingPath(IntEnum):
    UsePlayerSettings = 1
    VertexLit = 0
    Forward = 1
    DeferredShading = 3

class TransparencySortMode(IntEnum):
    Default = 0
    Perspective = 1
    Orthographic = 2
    CustomAxis = 3

class StereoTargetEyeMask(IntEnum):
    NONE = 0
    Left = 1 << 0
    Right = 1 << 1
    Both = Left | Right

class CameraType(IntFlag):
    Game = 1
    SceneView = 2
    Preview = 4
    VR = 8
    Reflection = 16

class ComputeBufferType(IntFlag):
    Default = 0
    Raw = 1
    Append = 2
    Counter = 4
    Constant = 8
    Structured = 16
    DrawIndirect = 256
    IndirectArguments = 256
    GPUMemory = 512

class LightType(IntEnum):
    Spot = 0
    Directional = 1
    Point = 2
    Area = 3
    Rectangle = 3
    Disc = 4
    Pyramid = 5
    Box = 6
    Tube = 7

class LightShape(IntEnum):
    Cone = 0
    Pyramid = 1
    Box = 2

class LightRenderMode(IntEnum):
    Auto = 0
    ForcePixel = 1
    ForceVertex = 2

class LightShadows(IntEnum):
    NONE = 0
    Hard = 1
    Soft = 2

class FogMode(IntEnum):
    Linear = 1
    Exponential = 2
    ExponentialSquared = 3

class LightmapBakeType(IntFlag):
    Realtime = 4
    Baked = 2
    Mixed = 1

class MixedLightingMode(IntEnum):
    IndirectOnly = 0
    Shadowmask = 2
    Subtractive = 1

class ReceiveGI(IntEnum):
    Lightmaps = 1
    LightProbes = 2

class LightmapCompression(IntEnum):
    NONE = 0
    LowQuality = 1
    NormalQuality = 2
    HighQuality = 3

class QualityLevel(IntEnum):
    Fastest = 0
    Fast = 1
    Simple = 2
    Good = 3
    Beautiful = 4
    Fantastic = 5

class ShadowProjection(IntEnum):
    CloseFit = 0
    StableFit = 1

class ShadowQuality(IntEnum):
    Disable = 0
    HardOnly = 1
    All = 2

# class ShadowResolution(IntEnum):
#     Low = UnityEngine.Rendering.LightShadowResolution.Low
#     Medium = UnityEngine.Rendering.LightShadowResolution.Medium
#     High = UnityEngine.Rendering.LightShadowResolution.High
#     VeryHigh = UnityEngine.Rendering.LightShadowResolution.VeryHigh

class ShadowmaskMode(IntEnum):
    Shadowmask = 0
    DistanceShadowmask = 1

class ShadowObjectsFilter(IntEnum):
    AllObjects = 0
    DynamicOnly = 1
    StaticOnly = 2

class CameraClearFlags(IntEnum):
    Skybox = 1
    Color = 2
    SolidColor = 2
    Depth = 3
    Nothing = 4

class DepthTextureMode(IntFlag):
    NONE = 0
    Depth = 1
    DepthNormals = 2
    MotionVectors = 4

class TexGenMode(IntEnum):
    NONE = 0
    SphereMap = 1
    Object = 2
    EyeLinear = 3
    CubeReflect = 4
    CubeNormal = 5

class AnisotropicFiltering(IntEnum):
    Disable = 0
    Enable = 1
    ForceEnable = 2

class BlendWeights(IntEnum):
    OneBone = 1
    TwoBones = 2
    FourBones = 4

class SkinWeights(IntEnum):
    NONE = 0
    OneBone = 1
    TwoBones = 2
    FourBones = 4
    Unlimited = 255

class MeshTopology(IntEnum):
    Triangles = 0
    Quads = 2
    Lines = 3
    LineStrip = 4
    Points = 5

class SkinQuality(IntEnum):
    Auto = 0
    Bone1 = 1
    Bone2 = 2
    Bone4 = 4

class ColorSpace(IntEnum):
    Uninitialized = 1
    Gamma = 0
    Linear = 1

class ColorGamut(IntEnum):
    sRGB = 0
    Rec709 = 1
    Rec2020 = 2
    DisplayP3 = 3
    HDR10 = 4
    DolbyHDR = 5
    P3D65G22 = 6

class ColorPrimaries(IntEnum):
    Unknown = 1
    Rec709 = 0
    Rec2020 = 1
    P3 = 2

class WhitePoint(IntEnum):
    Unknown = 1
    D65 = 0

class TransferFunction(IntEnum):
    Unknown = 1
    sRGB = 0
    BT1886 = 1
    PQ = 2
    Linear = 3
    Gamma22 = 4

class ScreenOrientation(IntEnum):
    Portrait = 1
    PortraitUpsideDown = 2
    LandscapeLeft = 3
    LandscapeRight = 4
    AutoRotation = 5

class GraphicsEnums_FilterMode(IntEnum):
    Point = 0
    Bilinear = 1
    Trilinear = 2

class TextureWrapMode(IntEnum):
    Repeat = 0
    Clamp = 1
    Mirror = 2
    MirrorOnce = 3

class NPOTSupport(IntEnum):
    NONE = 0
    Restricted = 1
    Full = 2

class TextureFormat(IntEnum):
    Alpha8 = 1
    ARGB4444 = 2
    RGB24 = 3
    RGBA32 = 4
    ARGB32 = 5
    RGB565 = 7
    R16 = 9
    DXT1 = 10
    DXT5 = 12
    RGBA4444 = 13
    BGRA32 = 14
    RHalf = 15
    RGHalf = 16
    RGBAHalf = 17
    RFloat = 18
    RGFloat = 19
    RGBAFloat = 20
    YUY2 = 21
    RGB9e5Float = 22
    BC4 = 26
    BC5 = 27
    BC6H = 24
    BC7 = 25
    DXT1Crunched = 28
    DXT5Crunched = 29
    PVRTC_RGB2 = 30
    PVRTC_RGBA2 = 31
    PVRTC_RGB4 = 32
    PVRTC_RGBA4 = 33
    ETC_RGB4 = 34
    EAC_R = 41
    EAC_R_SIGNED = 42
    EAC_RG = 43
    EAC_RG_SIGNED = 44
    ETC2_RGB = 45
    ETC2_RGBA1 = 46
    ETC2_RGBA8 = 47
    ASTC_4x4 = 48
    ASTC_5x5 = 49
    ASTC_6x6 = 50
    ASTC_8x8 = 51
    ASTC_10x10 = 52
    ASTC_12x12 = 53
    ETC_RGB4_3DS = 60
    ETC_RGBA8_3DS = 61
    RG16 = 62
    R8 = 63
    ETC_RGB4Crunched = 64
    ETC2_RGBA8Crunched = 65
    ASTC_HDR_4x4 = 66
    ASTC_HDR_5x5 = 67
    ASTC_HDR_6x6 = 68
    ASTC_HDR_8x8 = 69
    ASTC_HDR_10x10 = 70
    ASTC_HDR_12x12 = 71
    RG32 = 72
    RGB48 = 73
    RGBA64 = 74
    R8_SIGNED = 75
    RG16_SIGNED = 76
    RGB24_SIGNED = 77
    RGBA32_SIGNED = 78
    R16_SIGNED = 79
    RG32_SIGNED = 80
    RGB48_SIGNED = 81
    RGBA64_SIGNED = 82
    ASTC_RGB_4x4 = 48
    ASTC_RGB_5x5 = 49
    ASTC_RGB_6x6 = 50
    ASTC_RGB_8x8 = 51
    ASTC_RGB_10x10 = 52
    ASTC_RGB_12x12 = 53
    ASTC_RGBA_4x4 = 54
    ASTC_RGBA_5x5 = 55
    ASTC_RGBA_6x6 = 56
    ASTC_RGBA_8x8 = 57
    ASTC_RGBA_10x10 = 58
    ASTC_RGBA_12x12 = 59

class TextureColorSpace(IntEnum):
    Linear = 0
    sRGB = 1

class CubemapFace(IntEnum):
    Unknown = 1
    PositiveX = 0
    NegativeX = 1
    PositiveY = 2
    NegativeY = 3
    PositiveZ = 4
    NegativeZ = 5

class RenderTextureFormat(IntEnum):
    ARGB32 = 0
    Depth = 1
    ARGBHalf = 2
    Shadowmap = 3
    RGB565 = 4
    ARGB4444 = 5
    ARGB1555 = 6
    Default = 7
    ARGB2101010 = 8
    DefaultHDR = 9
    ARGB64 = 10
    ARGBFloat = 11
    RGFloat = 12
    RGHalf = 13
    RFloat = 14
    RHalf = 15
    R8 = 16
    ARGBInt = 17
    RGInt = 18
    RInt = 19
    BGRA32 = 20
    RGB111110Float = 22
    RG32 = 23
    RGBAUShort = 24
    RG16 = 25
    BGRA10101010_XR = 26
    BGR101010_XR = 27
    R16 = 28

class VRTextureUsage(IntEnum):
    NONE = 1
    OneEye = 2
    TwoEyes = 3
    DeviceSpecific = 4

class RenderTextureCreationFlags(IntFlag):
    MipMap = 1 << 0
    AutoGenerateMips = 1 << 1
    SRGB = 1 << 2
    EyeTexture = 1 << 3
    EnableRandomWrite = 1 << 4
    CreatedFromScript = 1 << 5
    AllowVerticalFlip = 1 << 7
    NoResolvedColorSurface = 1 << 8
    DynamicallyScalable = 1 << 10
    BindMS = 1 << 11
    DynamicallyScalableExplicit = 1 << 16

class RenderTextureReadWrite(IntEnum):
    Default = 0
    Linear = 1
    sRGB = 2

class RenderTextureMemoryless(IntFlag):
    NONE = 0
    Color = 1
    Depth = 2
    MSAA = 4

class HDRDisplaySupportFlags(IntFlag):
    NONE = 0
    Supported = 1 << 0
    RuntimeSwitchable = 1 << 1
    AutomaticTonemapping = 1 << 2

class HDRDisplayBitDepth(IntEnum):
    BitDepth10 = 0
    BitDepth16 = 1

class BlendShapeBufferLayout(IntEnum):
    PerShape = 1
    PerVertex = 2

class RayTracingAccelerationStructureBuildFlags(IntFlag):
    NONE = 0
    PreferFastTrace = 1 << 0
    PreferFastBuild = 1 << 1
    MinimizeMemory = 1 << 2

class TextureCreationFlags(IntFlag):
    NONE = 0
    MipChain = 1 << 0
    DontInitializePixels = 1 << 2
    Crunch = 1 << 6
    DontUploadUponCreate = 1 << 10
    IgnoreMipmapLimit = 1 << 11

class FormatUsage(IntEnum):
    Sample = 0
    Linear = 1
    Sparse = 2
    Render = 4
    Blend = 5
    GetPixels = 6
    SetPixels = 7
    SetPixels32 = 8
    ReadPixels = 9
    LoadStore = 10
    MSAA2x = 11
    MSAA4x = 12
    MSAA8x = 13
    StencilSampling = 16

class GraphicsFormatUsage(IntFlag):
    NONE = 0
    Sample = 1 << 0
    Linear = 1 << 1
    Sparse = 1 << 2
    Render = 1 << 4
    Blend = 1 << 5
    GetPixels = 1 << 6
    SetPixels = 1 << 7
    SetPixels32 = 1 << 8
    ReadPixels = 1 << 9
    LoadStore = 1 << 10
    MSAA2x = 1 << 11
    MSAA4x = 1 << 12
    MSAA8x = 1 << 13
    StencilSampling = 1 << 16

class DefaultFormat(IntEnum):
    LDR = 1
    HDR = 2
    DepthStencil = 3
    Shadow = 4
    Video = 5

class GraphicsFormat(IntEnum):
    NONE = 0
    R8_SRGB = 1
    R8G8_SRGB = 2
    R8G8B8_SRGB = 3
    R8G8B8A8_SRGB = 4
    R8_UNorm = 5
    R8G8_UNorm = 6
    R8G8B8_UNorm = 7
    R8G8B8A8_UNorm = 8
    R8_SNorm = 9
    R8G8_SNorm = 10
    R8G8B8_SNorm = 11
    R8G8B8A8_SNorm = 12
    R8_UInt = 13
    R8G8_UInt = 14
    R8G8B8_UInt = 15
    R8G8B8A8_UInt = 16
    R8_SInt = 17
    R8G8_SInt = 18
    R8G8B8_SInt = 19
    R8G8B8A8_SInt = 20
    R16_UNorm = 21
    R16G16_UNorm = 22
    R16G16B16_UNorm = 23
    R16G16B16A16_UNorm = 24
    R16_SNorm = 25
    R16G16_SNorm = 26
    R16G16B16_SNorm = 27
    R16G16B16A16_SNorm = 28
    R16_UInt = 29
    R16G16_UInt = 30
    R16G16B16_UInt = 31
    R16G16B16A16_UInt = 32
    R16_SInt = 33
    R16G16_SInt = 34
    R16G16B16_SInt = 35
    R16G16B16A16_SInt = 36
    R32_UInt = 37
    R32G32_UInt = 38
    R32G32B32_UInt = 39
    R32G32B32A32_UInt = 40
    R32_SInt = 41
    R32G32_SInt = 42
    R32G32B32_SInt = 43
    R32G32B32A32_SInt = 44
    R16_SFloat = 45
    R16G16_SFloat = 46
    R16G16B16_SFloat = 47
    R16G16B16A16_SFloat = 48
    R32_SFloat = 49
    R32G32_SFloat = 50
    R32G32B32_SFloat = 51
    R32G32B32A32_SFloat = 52
    B8G8R8_SRGB = 56
    B8G8R8A8_SRGB = 57
    B8G8R8_UNorm = 58
    B8G8R8A8_UNorm = 59
    B8G8R8_SNorm = 60
    B8G8R8A8_SNorm = 61
    B8G8R8_UInt = 62
    B8G8R8A8_UInt = 63
    B8G8R8_SInt = 64
    B8G8R8A8_SInt = 65
    R4G4B4A4_UNormPack16 = 66
    B4G4R4A4_UNormPack16 = 67
    R5G6B5_UNormPack16 = 68
    B5G6R5_UNormPack16 = 69
    R5G5B5A1_UNormPack16 = 70
    B5G5R5A1_UNormPack16 = 71
    A1R5G5B5_UNormPack16 = 72
    E5B9G9R9_UFloatPack32 = 73
    B10G11R11_UFloatPack32 = 74
    A2B10G10R10_UNormPack32 = 75
    A2B10G10R10_UIntPack32 = 76
    A2B10G10R10_SIntPack32 = 77
    A2R10G10B10_UNormPack32 = 78
    A2R10G10B10_UIntPack32 = 79
    A2R10G10B10_SIntPack32 = 80
    A2R10G10B10_XRSRGBPack32 = 81
    A2R10G10B10_XRUNormPack32 = 82
    R10G10B10_XRSRGBPack32 = 83
    R10G10B10_XRUNormPack32 = 84
    A10R10G10B10_XRSRGBPack32 = 85
    A10R10G10B10_XRUNormPack32 = 86
    D16_UNorm = 90
    D24_UNorm = 91
    D24_UNorm_S8_UInt = 92
    D32_SFloat = 93
    D32_SFloat_S8_UInt = 94
    S8_UInt = 95
    RGB_DXT1_SRGB = 96
    RGBA_DXT1_SRGB = 96
    RGB_DXT1_UNorm = 97
    RGBA_DXT1_UNorm = 97
    RGBA_DXT3_SRGB = 98
    RGBA_DXT3_UNorm = 99
    RGBA_DXT5_SRGB = 100
    RGBA_DXT5_UNorm = 101
    R_BC4_UNorm = 102
    R_BC4_SNorm = 103
    RG_BC5_UNorm = 104
    RG_BC5_SNorm = 105
    RGB_BC6H_UFloat = 106
    RGB_BC6H_SFloat = 107
    RGBA_BC7_SRGB = 108
    RGBA_BC7_UNorm = 109
    RGB_PVRTC_2Bpp_SRGB = 110
    RGB_PVRTC_2Bpp_UNorm = 111
    RGB_PVRTC_4Bpp_SRGB = 112
    RGB_PVRTC_4Bpp_UNorm = 113
    RGBA_PVRTC_2Bpp_SRGB = 114
    RGBA_PVRTC_2Bpp_UNorm = 115
    RGBA_PVRTC_4Bpp_SRGB = 116
    RGBA_PVRTC_4Bpp_UNorm = 117
    RGB_ETC_UNorm = 118
    RGB_ETC2_SRGB = 119
    RGB_ETC2_UNorm = 120
    RGB_A1_ETC2_SRGB = 121
    RGB_A1_ETC2_UNorm = 122
    RGBA_ETC2_SRGB = 123
    RGBA_ETC2_UNorm = 124
    R_EAC_UNorm = 125
    R_EAC_SNorm = 126
    RG_EAC_UNorm = 127
    RG_EAC_SNorm = 128
    RGBA_ASTC4X4_SRGB = 129
    RGBA_ASTC4X4_UNorm = 130
    RGBA_ASTC5X5_SRGB = 131
    RGBA_ASTC5X5_UNorm = 132
    RGBA_ASTC6X6_SRGB = 133
    RGBA_ASTC6X6_UNorm = 134
    RGBA_ASTC8X8_SRGB = 135
    RGBA_ASTC8X8_UNorm = 136
    RGBA_ASTC10X10_SRGB = 137
    RGBA_ASTC10X10_UNorm = 138
    RGBA_ASTC12X12_SRGB = 139
    RGBA_ASTC12X12_UNorm = 140
    YUV2 = 141
    RGBA_ASTC4X4_UFloat = 145
    RGBA_ASTC5X5_UFloat = 146
    RGBA_ASTC6X6_UFloat = 147
    RGBA_ASTC8X8_UFloat = 148
    RGBA_ASTC10X10_UFloat = 149
    RGBA_ASTC12X12_UFloat = 150
    D16_UNorm_S8_UInt = 151

class RayTracingMode(IntEnum):
    Off = 0
    Static = 1
    DynamicTransform = 2
    DynamicGeometry = 3

class LightmapsMode(IntFlag):
    NonDirectional = 0
    CombinedDirectional = 1
    SeparateDirectional = 2
    Single = 0
    Dual = 1
    Directional = 2

class MaterialGlobalIlluminationFlags(IntFlag):
    NONE = 0
    RealtimeEmissive = 1 << 0
    BakedEmissive = 1 << 1
    EmissiveIsBlack = 1 << 2
    AnyEmissive = RealtimeEmissive | BakedEmissive

class MaterialSerializedProperty(IntEnum):
    NONE = 0
    LightmapFlags = 1 << 1
    EnableInstancingVariants = 1 << 2
    DoubleSidedGI = 1 << 3
    CustomRenderQueue = 1 << 4

class ResolutionMode(IntEnum):
    Automatic = 0
    Custom = 1

class BoundingBoxMode(IntEnum):
    AutomaticLocal = 0
    AutomaticWorld = 1
    Custom = 2

class ProbePositionMode(IntEnum):
    CellCorner = 0
    CellCenter = 1

class RefreshMode(IntEnum):
    Automatic = 0
    EveryFrame = 1
    ViaScripting = 2

class QualityMode(IntEnum):
    Low = 0
    Normal = 1

class DataFormat(IntEnum):
    HalfFloat = 0
    Float = 1

class CustomRenderTextureInitializationSource(IntEnum):
    TextureAndColor = 0
    Material = 1

class CustomRenderTextureUpdateMode(IntEnum):
    OnLoad = 0
    Realtime = 1
    OnDemand = 2

class CustomRenderTextureUpdateZoneSpace(IntEnum):
    Normalized = 0
    Pixel = 1

class MotionVectorGenerationMode(IntEnum):
    Camera = 0
    Object = 1
    ForceNoMotion = 2

class LineTextureMode(IntEnum):
    Stretch = 0
    Tile = 1
    DistributePerSegment = 2
    RepeatPerSegment = 3
    Static = 4

class LineAlignment(IntEnum):
    View = 0
    Local = 1
    TransformZ = 1

class TextureMipmapLimitBiasMode(IntEnum):
    OffsetGlobalLimit = 0
    OverrideGlobalLimit = 1

class IndexFormat(IntEnum):
    UInt16 = 0
    UInt32 = 1

class MeshUpdateFlags(IntFlag):
    Default = 0
    DontValidateIndices = 1 << 0
    DontResetBoneBounds = 1 << 1
    DontNotifyMeshUsers = 1 << 2
    DontRecalculateBounds = 1 << 3

class VertexAttributeFormat(IntEnum):
    Float32 = 0
    Float16 = 1
    UNorm8 = 2
    SNorm8 = 3
    UNorm16 = 4
    SNorm16 = 5
    UInt8 = 6
    SInt8 = 7
    UInt16 = 8
    SInt16 = 9
    UInt32 = 10
    SInt32 = 11

class VertexAttribute(IntEnum):
    Position = 0
    Normal = 1
    Tangent = 2
    Color = 3
    TexCoord0 = 4
    TexCoord1 = 5
    TexCoord2 = 6
    TexCoord3 = 7
    TexCoord4 = 8
    TexCoord5 = 9
    TexCoord6 = 10
    TexCoord7 = 11
    BlendWeight = 12
    BlendIndices = 13

class ShaderParamType(IntEnum):
    Float = 0
    Int = 1
    Bool = 2
    Half = 3
    Short = 4
    UInt = 5

class ShaderConstantType(IntEnum):
    Vector = 0
    Matrix = 1
    Struct = 2

class OpaqueSortMode(IntEnum):
    Default = 0
    FrontToBack = 1
    NoDistanceSort = 2

class RenderQueue(IntEnum):
    Background = 1000
    Geometry = 2000
    AlphaTest = 2450
    GeometryLast = 2500
    Transparent = 3000
    Overlay = 4000

class RenderBufferLoadAction(IntEnum):
    Load = 0
    Clear = 1
    DontCare = 2

class RenderBufferStoreAction(IntEnum):
    Store = 0
    Resolve = 1
    StoreAndResolve = 2
    DontCare = 3

class FastMemoryFlags(IntFlag):
    NONE = 0
    SpillTop = (1 << 0)
    SpillBottom = (1 << 1)

class BlendMode(IntEnum):
    Zero = 0
    One = 1
    DstColor = 2
    SrcColor = 3
    OneMinusDstColor = 4
    SrcAlpha = 5
    OneMinusSrcColor = 6
    DstAlpha = 7
    OneMinusDstAlpha = 8
    SrcAlphaSaturate = 9
    OneMinusSrcAlpha = 10

class BlendOp(IntEnum):
    Add = 0
    Subtract = 1
    ReverseSubtract = 2
    Min = 3
    Max = 4
    LogicalClear = 5
    LogicalSet = 6
    LogicalCopy = 7
    LogicalCopyInverted = 8
    LogicalNoop = 9
    LogicalInvert = 10
    LogicalAnd = 11
    LogicalNand = 12
    LogicalOr = 13
    LogicalNor = 14
    LogicalXor = 15
    LogicalEquivalence = 16
    LogicalAndReverse = 17
    LogicalAndInverted = 18
    LogicalOrReverse = 19
    LogicalOrInverted = 20
    Multiply = 21
    Screen = 22
    Overlay = 23
    Darken = 24
    Lighten = 25
    ColorDodge = 26
    ColorBurn = 27
    HardLight = 28
    SoftLight = 29
    Difference = 30
    Exclusion = 31
    HSLHue = 32
    HSLSaturation = 33
    HSLColor = 34
    HSLLuminosity = 35

class CompareFunction(IntEnum):
    Disabled = 0
    Never = 1
    Less = 2
    Equal = 3
    LessEqual = 4
    Greater = 5
    NotEqual = 6
    GreaterEqual = 7
    Always = 8

class CullMode(IntEnum):
    Off = 0
    Front = 1
    Back = 2

class ColorWriteMask(IntFlag):
    Alpha = 1
    Blue = 2
    Green = 4
    Red = 8
    All = 15

class StencilOp(IntEnum):
    Keep = 0
    Zero = 1
    Replace = 2
    IncrementSaturate = 3
    DecrementSaturate = 4
    Invert = 5
    IncrementWrap = 6
    DecrementWrap = 7

class AmbientMode(IntEnum):
    Skybox = 0
    Trilight = 1
    Flat = 3
    Custom = 4

class DefaultReflectionMode(IntEnum):
    Skybox = 0
    Custom = 1

class ReflectionCubemapCompression(IntEnum):
    Uncompressed = 0
    Compressed = 1
    Auto = 2

class CameraEvent(IntEnum):
    BeforeDepthTexture = 0
    AfterDepthTexture = 1
    BeforeDepthNormalsTexture = 2
    AfterDepthNormalsTexture = 3
    BeforeGBuffer = 4
    AfterGBuffer = 5
    BeforeLighting = 6
    AfterLighting = 7
    BeforeFinalPass = 8
    AfterFinalPass = 9
    BeforeForwardOpaque = 10
    AfterForwardOpaque = 11
    BeforeImageEffectsOpaque = 12
    AfterImageEffectsOpaque = 13
    BeforeSkybox = 14
    AfterSkybox = 15
    BeforeForwardAlpha = 16
    AfterForwardAlpha = 17
    BeforeImageEffects = 18
    AfterImageEffects = 19
    AfterEverything = 20
    BeforeReflections = 21
    AfterReflections = 22
    BeforeHaloAndLensFlares = 23
    AfterHaloAndLensFlares = 24

class LightEvent(IntEnum):
    BeforeShadowMap = 0
    AfterShadowMap = 1
    BeforeScreenspaceMask = 2
    AfterScreenspaceMask = 3
    BeforeShadowMapPass = 4
    AfterShadowMapPass = 5

class ShadowMapPass(IntFlag):
    PointlightPositiveX = 1 << 0
    PointlightNegativeX = 1 << 1
    PointlightPositiveY = 1 << 2
    PointlightNegativeY = 1 << 3
    PointlightPositiveZ = 1 << 4
    PointlightNegativeZ = 1 << 5
    DirectionalCascade0 = 1 << 6
    DirectionalCascade1 = 1 << 7
    DirectionalCascade2 = 1 << 8
    DirectionalCascade3 = 1 << 9
    Spotlight = 1 << 10
    AreaLight = 1 << 11
    Pointlight = PointlightPositiveX | PointlightNegativeX | PointlightPositiveY | PointlightNegativeY | PointlightPositiveZ | PointlightNegativeZ
    Directional = DirectionalCascade0 | DirectionalCascade1 | DirectionalCascade2 | DirectionalCascade3
    All = Pointlight | Spotlight | Directional

class BuiltinRenderTextureType(IntEnum):
    PropertyName = 4
    BufferPtr = 3
    RenderTexture = 2
    BindableTexture = 1
    NONE = 0
    CurrentActive = 1
    CameraTarget = 2
    Depth = 3
    DepthNormals = 4
    ResolvedDepth = 5
    GBuffer0 = 10
    GBuffer1 = 11
    GBuffer2 = 12
    GBuffer3 = 13
    Reflections = 14
    MotionVectors = 15
    GBuffer4 = 16
    GBuffer5 = 17
    GBuffer6 = 18
    GBuffer7 = 19

class PassType(IntEnum):
    Normal = 0
    Vertex = 1
    VertexLM = 2
    VertexLMRGBM = 3
    ForwardBase = 4
    ForwardAdd = 5
    ShadowCaster = 8
    Deferred = 10
    Meta = 11
    MotionVectors = 12
    ScriptableRenderPipeline = 13
    ScriptableRenderPipelineDefaultUnlit = 14
    GrabPass = 15

class ShadowCastingMode(IntEnum):
    Off = 0
    On = 1
    TwoSided = 2
    ShadowsOnly = 3

class LightShadowResolution(IntEnum):
    FromQualitySettings = 1
    Low = 0
    Medium = 1
    High = 2
    VeryHigh = 3

class LightUnit(IntEnum):
    Lumen = 0
    Candela = 1
    Lux = 2
    Nits = 3
    Ev100 = 4

class GraphicsDeviceType(IntEnum):
    OpenGL2 = 0
    Direct3D9 = 1
    Direct3D11 = 2
    PlayStation3 = 3
    Null = 4
    Xbox360 = 6
    OpenGLES2 = 8
    OpenGLES3 = 11
    PlayStationVita = 12
    PlayStation4 = 13
    XboxOne = 14
    PlayStationMobile = 15
    Metal = 16
    OpenGLCore = 17
    Direct3D12 = 18
    N3DS = 19
    Vulkan = 21
    Switch = 22
    XboxOneD3D12 = 23
    GameCoreXboxOne = 24
    GameCoreScarlett = 1
    GameCoreXboxSeries = 25
    PlayStation5 = 26
    PlayStation5NGGC = 27
    WebGPU = 28

class GraphicsTier(IntEnum):
    Tier1 = 0
    Tier2 = 1
    Tier3 = 2

class FormatSwizzle(IntEnum):
    FormatSwizzleR = 1
    FormatSwizzleG = 2
    FormatSwizzleB = 3
    FormatSwizzleA = 4
    FormatSwizzle0 = 5
    FormatSwizzle1 = 6

class RenderTargetFlags(IntFlag):
    NONE = 0
    ReadOnlyDepth = 1 << 0
    ReadOnlyStencil = 1 << 1
    ReadOnlyDepthStencil = ReadOnlyDepth | ReadOnlyStencil

class ReflectionProbeUsage(IntEnum):
    Off = 1
    BlendProbes = 2
    BlendProbesAndSkybox = 3
    Simple = 4

class ReflectionProbeType(IntEnum):
    Cube = 0
    Card = 1

class ReflectionProbeClearFlags(IntEnum):
    Skybox = 1
    SolidColor = 2

class ReflectionProbeMode(IntEnum):
    Baked = 0
    Realtime = 1
    Custom = 2

class ReflectionProbeRefreshMode(IntEnum):
    OnAwake = 0
    EveryFrame = 1
    ViaScripting = 2

class ReflectionProbeTimeSlicingMode(IntEnum):
    AllFacesAtOnce = 0
    IndividualFaces = 1
    NoTimeSlicing = 2

class ShadowSamplingMode(IntEnum):
    CompareDepths = 0
    RawDepth = 1
    NONE = 2

class LightProbeUsage(IntEnum):
    Off = 0
    BlendProbes = 1
    UseProxyVolume = 2
    CustomProvided = 4

class BuiltinShaderType(IntEnum):
    DeferredShading = 0
    DeferredReflections = 1
    ScreenSpaceShadows = 3
    DepthNormals = 4
    MotionVectors = 5
    LightHalo = 6
    LensFlare = 7

class BuiltinShaderMode(IntEnum):
    Disabled = 0
    UseBuiltin = 1
    UseCustom = 2

class BuiltinShaderDefine(IntEnum):
    UNITY_NO_DXT5nm = 1
    UNITY_NO_RGBM = 2
    UNITY_ENABLE_REFLECTION_BUFFERS = 3
    UNITY_FRAMEBUFFER_FETCH_AVAILABLE = 4
    UNITY_ENABLE_NATIVE_SHADOW_LOOKUPS = 5
    UNITY_METAL_SHADOWS_USE_POINT_FILTERING = 6
    UNITY_NO_CUBEMAP_ARRAY = 7
    UNITY_NO_SCREENSPACE_SHADOWS = 8
    UNITY_USE_DITHER_MASK_FOR_ALPHABLENDED_SHADOWS = 9
    UNITY_PBS_USE_BRDF1 = 10
    UNITY_PBS_USE_BRDF2 = 11
    UNITY_PBS_USE_BRDF3 = 12
    UNITY_SPECCUBE_BOX_PROJECTION = 13
    UNITY_SPECCUBE_BLENDING = 14
    UNITY_ENABLE_DETAIL_NORMALMAP = 15
    SHADER_API_MOBILE = 16
    SHADER_API_DESKTOP = 17
    UNITY_HARDWARE_TIER1 = 18
    UNITY_HARDWARE_TIER2 = 19
    UNITY_HARDWARE_TIER3 = 20
    UNITY_COLORSPACE_GAMMA = 21
    UNITY_LIGHT_PROBE_PROXY_VOLUME = 22
    UNITY_LIGHTMAP_DLDR_ENCODING = 23
    UNITY_LIGHTMAP_RGBM_ENCODING = 24
    UNITY_LIGHTMAP_FULL_HDR = 25
    UNITY_VIRTUAL_TEXTURING = 26
    UNITY_PRETRANSFORM_TO_DISPLAY_ORIENTATION = 27
    UNITY_ASTC_NORMALMAP_ENCODING = 28
    SHADER_API_GLES30 = 29
    UNITY_UNIFIED_SHADER_PRECISION_MODEL = 30
    UNITY_PLATFORM_SUPPORTS_WAVE_32 = 31
    UNITY_PLATFORM_SUPPORTS_WAVE_64 = 32
    UNITY_NEEDS_RENDERPASS_FBFETCH_FALLBACK = 33

class VideoShadersIncludeMode(IntEnum):
    Never = 0
    Referenced = 1
    Always = 2

class TextureDimension(IntEnum):
    Unknown = 1
    NONE = 0
    Any = 1
    Tex2D = 2
    Tex3D = 3
    Cube = 4
    Tex2DArray = 5
    CubeArray = 6

class CopyTextureSupport(IntFlag):
    NONE = 0
    Basic = (1 << 0)
    Copy3D = (1 << 1)
    DifferentTypes = (1 << 2)
    TextureToRT = (1 << 3)
    RTToTexture = (1 << 4)

class CameraHDRMode(IntEnum):
    FP16 = 1
    R11G11B10 = 2

class RealtimeGICPUUsage(IntEnum):
    Low = 25
    Medium = 50
    High = 75
    Unlimited = 100

class ComputeQueueType(IntEnum):
    Default = 0
    Background = 1
    Urgent = 2

class SinglePassStereoMode(IntEnum):
    NONE = 0
    SideBySide = 1
    Instancing = 2
    Multiview = 3

class FoveatedRenderingCaps(IntFlag):
    NONE = 0
    FoveationImage = 1 << 0
    NonUniformRaster = 1 << 1
    ModeChangeOnlyBeforeRenderTargetSet = 1 << 2

class FoveatedRenderingMode(IntEnum):
    Disabled = 0
    Enabled = 1

class CommandBufferExecutionFlags(IntEnum):
    NONE = 0
    AsyncCompute = 1 << 1

class RTClearFlags(IntFlag):
    NONE = 0
    Color = 1 << 0
    Depth = 1 << 1
    Stencil = 1 << 2
    All = Color | Depth | Stencil
    DepthStencil = Depth | Stencil
    ColorDepth = Color | Depth
    ColorStencil = Color | Stencil

class RenderTextureSubElement(IntEnum):
    Color = 0
    Depth = 1
    Stencil = 2
    Default = 3

class RenderingThreadingMode(IntEnum):
    Direct = 0
    SingleThreaded = 1
    MultiThreaded = 2
    LegacyJobified = 3
    NativeGraphicsJobs = 4
    NativeGraphicsJobsWithoutRenderThread = 5
    NativeGraphicsJobsSplitThreading = 6

class CameraLateLatchMatrixType(IntEnum):
    View = 0
    InverseView = 1
    ViewProjection = 2
    InverseViewProjection = 3

class OpenGLESVersion(IntEnum):
    NONE = 1
    OpenGLES20 = 2
    OpenGLES30 = 3
    OpenGLES31 = 4
    OpenGLES31AEP = 5
    OpenGLES32 = 6

class CustomMarkerCallbackFlags(IntFlag):
    CustomMarkerCallbackDefault = 0
    CustomMarkerCallbackForceInvalidateStateTracking = 1 << 2

class LightmapType(IntEnum):
    NoLightmap = 1
    StaticLightmap = 0
    DynamicLightmap = 1

# UnityCsReference-master/Runtime/Export/Graphics/GraphicsFence.bindings.cs
class SynchronisationStageFlags(IntEnum):
    VertexProcessing = 1
    PixelProcessing = 2
    ComputeProcessing = 4
    AllGPUOperations = VertexProcessing | PixelProcessing | ComputeProcessing

class GraphicsFenceType(IntEnum):
    AsyncQueueSynchronisation = 0
    CPUSynchronisation = 1

# UnityCsReference-master/Runtime/Export/Graphics/GraphicsManagers.bindings.cs
class TerrainQualityOverrides(IntFlag):
    NONE = 0
    PixelError = 1
    BasemapDistance = 2
    DetailDensity = 4
    DetailDistance = 8
    TreeDistance = 16
    BillboardStart = 32
    FadeLength = 64
    MaxTrees = 128

# UnityCsReference-master/Runtime/Export/Graphics/GraphicsTexture.bindings.cs
class GraphicsTextureDescriptorFlags(IntFlag):
    NONE = 0
    RenderTarget = 1 << 0
    RandomWriteTarget = 1 << 1

class GraphicsTextureState(IntEnum):
    Constructed = 0
    Initializing = 1
    InitializedOnRenderThread = 2
    DestroyQueued = 3
    Destroyed = 4

# UnityCsReference-master/Runtime/Export/Graphics/IRenderPipelineResources.cs
class SearchType(IntEnum):
    ProjectPath = 1
    BuiltinPath = 2
    BuiltinExtraPath = 3
    ShaderName = 4

# UnityCsReference-master/Runtime/Export/Graphics/LOD.bindings.cs
class LODFadeMode(IntEnum):
    NONE = 0
    CrossFade = 1
    SpeedTree = 2

# UnityCsReference-master/Runtime/Export/Graphics/Light.deprecated.cs
class LightmappingMode(IntEnum):
    Realtime = 4
    Baked = 2
    Mixed = 1

# UnityCsReference-master/Runtime/Export/Graphics/Mesh.bindings.cs
class SafetyHandleIndex(IntEnum):
    BonesPerVertexArray = 1
    BonesWeightsArray = 2
    BindposeArray = 3

# UnityCsReference-master/Runtime/Export/Graphics/RayTracingAccelerationStructure.bindings.cs
class RayTracingSubMeshFlags(IntFlag):
    Disabled = 0
    Enabled = (1 << 0)
    ClosestHitOnly = (1 << 1)
    UniqueAnyHitCalls = (1 << 2)

class RayTracingInstanceCullingFlags(IntFlag):
    NONE = 0
    EnableSphereCulling = (1 << 0)
    EnablePlaneCulling = (1 << 1)
    EnableLODCulling = (1 << 2)
    ComputeMaterialsCRC = (1 << 3)
    IgnoreReflectionProbes = (1 << 4)

class RayTracingModeMask(IntFlag):
    Nothing = 0
    Static = (1 << RayTracingMode.Static)
    DynamicTransform = (1 << RayTracingMode.DynamicTransform)
    DynamicGeometry = (1 << RayTracingMode.DynamicGeometry)
    Everything = (Static | DynamicTransform | DynamicGeometry)

class ManagementMode(IntEnum):
    Manual = 0
    Automatic = 1

# UnityCsReference-master/Runtime/Export/Graphics/SplashScreen.bindings.cs
class StopBehavior(IntEnum):
    StopImmediate = 0
    FadeOut = 1

# UnityCsReference-master/Runtime/Export/Graphics/SupportedOnRenderPipeline.cs
class SupportedMode(IntEnum):
    Unsupported = 1
    Supported = 2
    SupportedByBaseClass = 3

# UnityCsReference-master/Runtime/Export/Graphics/Texture.cs
class EXRFlags(IntFlag):
    NONE = 0
    OutputAsFloat = 1 << 0
    CompressZIP = 1 << 1
    CompressRLE = 1 << 2
    CompressPIZ = 1 << 3

# UnityCsReference-master/Runtime/Export/Handheld/Handheld.bindings.cs
class FullScreenMovieControlMode(IntEnum):
    Full = 0
    Minimal = 1
    CancelOnInput = 2
    Hidden = 3

class FullScreenMovieScalingMode(IntEnum):
    NONE = 0
    AspectFit = 1
    AspectFill = 2
    Fill = 3

class AndroidActivityIndicatorStyle(IntEnum):
    DontShow = 1
    Large = 0
    InversedLarge = 1
    Small = 2
    InversedSmall = 3

# UnityCsReference-master/Runtime/Export/Input/Cursor.bindings.cs
class CursorMode(IntEnum):
    Auto = 0
    ForceSoftware = 1

class CursorLockMode(IntEnum):
    NONE = 0
    Locked = 1
    Confined = 2

# UnityCsReference-master/Runtime/Export/Input/KeyCode.cs
class KeyCode(IntEnum):
    NONE = 0
    Backspace = 8
    Delete = 127
    Tab = 9
    Clear = 12
    Return = 13
    Pause = 19
    Escape = 27
    Space = 32
    Keypad0 = 256
    Keypad1 = 257
    Keypad2 = 258
    Keypad3 = 259
    Keypad4 = 260
    Keypad5 = 261
    Keypad6 = 262
    Keypad7 = 263
    Keypad8 = 264
    Keypad9 = 265
    KeypadPeriod = 266
    KeypadDivide = 267
    KeypadMultiply = 268
    KeypadMinus = 269
    KeypadPlus = 270
    KeypadEnter = 271
    KeypadEquals = 272
    UpArrow = 273
    DownArrow = 274
    RightArrow = 275
    LeftArrow = 276
    Insert = 277
    Home = 278
    End = 279
    PageUp = 280
    PageDown = 281
    F1 = 282
    F2 = 283
    F3 = 284
    F4 = 285
    F5 = 286
    F6 = 287
    F7 = 288
    F8 = 289
    F9 = 290
    F10 = 291
    F11 = 292
    F12 = 293
    F13 = 294
    F14 = 295
    F15 = 296
    Alpha0 = 48
    Alpha1 = 49
    Alpha2 = 50
    Alpha3 = 51
    Alpha4 = 52
    Alpha5 = 53
    Alpha6 = 54
    Alpha7 = 55
    Alpha8 = 56
    Alpha9 = 57
    Exclaim = 33
    DoubleQuote = 34
    Hash = 35
    Dollar = 36
    Percent = 37
    Ampersand = 38
    Quote = 39
    LeftParen = 40
    RightParen = 41
    Asterisk = 42
    Plus = 43
    Comma = 44
    Minus = 45
    Period = 46
    Slash = 47
    Colon = 58
    Semicolon = 59
    Less = 60
    Equals = 61
    Greater = 62
    Question = 63
    At = 64
    LeftBracket = 91
    Backslash = 92
    RightBracket = 93
    Caret = 94
    Underscore = 95
    BackQuote = 96
    A = 97
    B = 98
    C = 99
    D = 100
    E = 101
    F = 102
    G = 103
    H = 104
    I = 105
    J = 106
    K = 107
    L = 108
    M = 109
    N = 110
    O = 111
    P = 112
    Q = 113
    R = 114
    S = 115
    T = 116
    U = 117
    V = 118
    W = 119
    X = 120
    Y = 121
    Z = 122
    LeftCurlyBracket = 123
    Pipe = 124

# UnityCsReference-master/Runtime/Export/Jobs/AtomicSafetyHandle.bindings.cs
class EnforceJobResult(IntEnum):
    AllJobsAlreadySynced = 0
    DidSyncRunningJobs = 1
    HandleWasAlreadyDeallocated = 2

class AtomicSafetyErrorType(IntEnum):
    Deallocated = 0
    DeallocatedFromJob = 1
    NotAllocatedFromJob = 2

# UnityCsReference-master/Runtime/Export/Math/Gradient.bindings.cs
class GradientMode(IntEnum):
    Blend = 0
    Fixed = 1
    PerceptualBlend = 2

# UnityCsReference-master/Runtime/Export/Misc/ObjectDispatcher.bindings.cs
class TransformTrackingType(IntEnum):
    GlobalTRS = 1
    LocalTRS = 2
    Hierarchy = 3

class TypeTrackingFlags(IntFlag):
    SceneObjects = 1
    Assets = 2
    EditorOnlyObjects = 4
    Default = SceneObjects | Assets
    All = SceneObjects | Assets | EditorOnlyObjects

# UnityCsReference-master/Runtime/Export/NativeArray/NativeArray.cs
class NativeArrayOptions(IntEnum):
    UninitializedMemory = 0
    ClearMemory = 1

# UnityCsReference-master/Runtime/Export/Networking/PlayerConnection/ConnectionApi.cs
class ConnectionTarget(IntEnum):
    NONE = 1
    Player = 2
    Editor = 3

# UnityCsReference-master/Runtime/Export/PlayerConnection/PlayerConnectionInternal.bindings.cs
class MulticastFlags(IntFlag):
    kRequestImmediateConnect = 1 << 0
    kSupportsProfile = 1 << 1
    kCustomMessage = 1 << 2
    kUseAlternateIP = 1 << 3

# UnityCsReference-master/Runtime/Export/RenderPipeline/AttachmentDescriptor.cs
class SubPassFlags(IntFlag):
    NONE = 0
    ReadOnlyDepth = 1 << 1
    ReadOnlyStencil = 1 << 2
    ReadOnlyDepthStencil = ReadOnlyDepth | ReadOnlyStencil

# UnityCsReference-master/Runtime/Export/RenderPipeline/CullingParameters.cs
class CullingOptions(IntFlag):
    NONE = 0
    ForceEvenIfCameraIsNotActive = 1 << 0
    OcclusionCull = 1 << 1
    NeedsLighting = 1 << 2
    NeedsReflectionProbes = 1 << 3
    Stereo = 1 << 4
    DisablePerObjectCulling = 1 << 5
    ShadowCasters = 1 << 6

# UnityCsReference-master/Runtime/Export/RenderPipeline/DrawRendererFlags.cs
class DrawRendererFlags(IntEnum):
    NONE = 0
    EnableDynamicBatching = (1 << 0)
    EnableInstancing = (1 << 1)

# UnityCsReference-master/Runtime/Export/RenderPipeline/GizmoSubset.cs
class GizmoSubset(IntEnum):
    PreImageEffects = 0
    PostImageEffects = 1

# UnityCsReference-master/Runtime/Export/RenderPipeline/PerObjectData.cs
class PerObjectData(IntFlag):
    NONE = 0
    LightProbe = (1 << 0)
    ReflectionProbes = (1 << 1)
    LightProbeProxyVolume = (1 << 2)
    Lightmaps = (1 << 3)
    LightData = (1 << 4)
    MotionVectors = (1 << 5)
    LightIndices = (1 << 6)
    ReflectionProbeData = (1 << 7)
    OcclusionProbe = (1 << 8)
    OcclusionProbeProxyVolume = (1 << 9)
    ShadowMask = (1 << 10)

# UnityCsReference-master/Runtime/Export/RenderPipeline/ReflectionProbeSortingCriteria.cs
class ReflectionProbeSortingCriteria(IntEnum):
    NONE = 1
    Importance = 2
    Size = 3
    ImportanceThenSize = 4

# UnityCsReference-master/Runtime/Export/RenderPipeline/RenderStateMask.cs
class RenderStateMask(IntFlag):
    Nothing = 0
    Blend = 1 << 0
    Raster = 1 << 1
    Depth = 1 << 2
    Stencil = 1 << 3
    Everything = Blend | Raster | Depth | Stencil

# UnityCsReference-master/Runtime/Export/RenderPipeline/RendererList.bindings.cs
class RendererListStatus(IntEnum):
    kRendererListInvalid = 2
    kRendererListProcessing = 1
    kRendererListEmpty = 0
    kRendererListPopulated = 1

# UnityCsReference-master/Runtime/Export/RenderPipeline/SortingCriteria.cs
class SortingCriteria(IntFlag):
    NONE = 0
    SortingLayer = (1 << 0)
    RenderQueue = (1 << 1)
    BackToFront = (1 << 2)
    QuantizedFrontToBack = (1 << 3)
    OptimizeStateChanges = (1 << 4)
    CanvasOrder = (1 << 5)
    RendererPriority = (1 << 6)
    CommonOpaque = SortingLayer | RenderQueue | QuantizedFrontToBack | OptimizeStateChanges | CanvasOrder
    CommonTransparent = SortingLayer | RenderQueue | BackToFront | OptimizeStateChanges

# UnityCsReference-master/Runtime/Export/RenderPipeline/SortingSettings.cs
class DistanceMetric(IntEnum):
    Perspective = 1
    Orthographic = 2
    CustomAxis = 3

# UnityCsReference-master/Runtime/Export/RenderPipeline/SupportedRenderingFeatures.cs
class ReflectionProbeModes(IntFlag):
    NONE = 0
    Rotation = 1

class LightmapMixedBakeModes(IntFlag):
    NONE = 0
    IndirectOnly = 1
    Subtractive = 2
    Shadowmask = 4

# UnityCsReference-master/Runtime/Export/RenderPipeline/UISubset.cs
class UISubset(IntFlag):
    UIToolkit_UGUI = (1 << 0)
    LowLevel = (1 << 1)
    All = ~0

# UnityCsReference-master/Runtime/Export/RenderPipeline/VisibleLightFlags.cs
class VisibleLightFlags(IntEnum):
    IntersectsNearPlane = 1 << 0
    IntersectsFarPlane = 1 << 1
    ForcedVisible = 1 << 2

# UnityCsReference-master/Runtime/Export/Rendering/GPUDrivenRendering.bindings.cs
class GPUDrivenBitOpType(IntEnum):
    And = 1
    Or = 2

# UnityCsReference-master/Runtime/Export/SceneManager/Scene.cs
class LoadingState(IntEnum):
    NotLoaded = 0
    Loading = 1
    Loaded = 2
    Unloading = 3

# UnityCsReference-master/Runtime/Export/SceneManager/SceneManager.cs
class LoadSceneMode(IntEnum):
    Single = 0
    Additive = 1

class LocalPhysicsMode(IntFlag):
    NONE = 0
    Physics2D = 1
    Physics3D = 2

class UnloadSceneOptions(IntFlag):
    NONE = 0
    UnloadAllEmbeddedSceneObjects = 1

# UnityCsReference-master/Runtime/Export/Scripting/ApiRestrictions.bindings.cs
class GlobalRestrictions(IntEnum):
    OBJECT_DESTROYIMMEDIATE = 0
    OBJECT_SENDMESSAGE = 1
    OBJECT_RENDERING = 2
    GLOBALCOUNT = 3

class ContextRestrictions(IntEnum):
    RENDERERSCENE_ADDREMOVE = 0
    OBJECT_ADDCOMPONENTTRANSFORM = 1
    CONTEXTCOUNT = 2

# UnityCsReference-master/Runtime/Export/Scripting/EnumDataUtility.cs
class CachedType(IntEnum):
    ExcludeObsolete = 1
    IncludeObsoleteExceptErrors = 2
    IncludeAllObsolete = 3

# UnityCsReference-master/Runtime/Export/Scripting/GarbageCollector.bindings.cs
class Mode(IntEnum):
    Disabled = 0
    Enabled = 1
    Manual = 2

# UnityCsReference-master/Runtime/Export/Scripting/InspectorOrderAttribute.cs
class InspectorSort(IntEnum):
    ByName = 1
    ByValue = 2

class InspectorSortDirection(IntEnum):
    Ascending = 1
    Descending = 2

# UnityCsReference-master/Runtime/Export/Scripting/RuntimeInitializeOnLoadAttribute.cs
class RuntimeInitializeLoadType(IntEnum):
    AfterSceneLoad = 0
    BeforeSceneLoad = 1
    AfterAssembliesLoaded = 2
    BeforeSplashScreen = 3
    SubsystemRegistration = 4

# UnityCsReference-master/Runtime/Export/Scripting/TextAsset.cs
class CreateOptions(IntEnum):
    NONE = 0
    CreateNativeObject = 1

# UnityCsReference-master/Runtime/Export/Scripting/UnityEngineObject.bindings.cs
class HideFlags(IntFlag):
    NONE = 0
    HideInHierarchy = 1
    HideInInspector = 2
    DontSaveInEditor = 4
    NotEditable = 8
    DontSaveInBuild = 16
    DontUnloadUnusedAsset = 32
    DontSave = DontSaveInEditor | DontSaveInBuild | DontUnloadUnusedAsset
    HideAndDontSave = HideInHierarchy | DontSaveInEditor | NotEditable | DontSaveInBuild | DontUnloadUnusedAsset

class FindObjectsSortMode(IntEnum):
    NONE = 0
    InstanceID = 1

class FindObjectsInactive(IntEnum):
    Exclude = 0
    Include = 1

# UnityCsReference-master/Runtime/Export/SearchService/SearchContextAttribute.cs
class SearchViewFlags(IntFlag):
    NONE = 0
    Debug = 1 << 4
    NoIndexing = 1 << 5
    Packages = 1 << 8
    OpenLeftSidePanel = 1 << 11
    OpenInspectorPreview = 1 << 12
    Centered = 1 << 13
    HideSearchBar = 1 << 14
    CompactView = 1 << 15
    ListView = 1 << 16
    GridView = 1 << 17
    TableView = 1 << 18
    EnableSearchQuery = 1 << 19
    DisableInspectorPreview = 1 << 20
    DisableSavedSearchQuery = 1 << 21
    OpenInBuilderMode = 1 << 22
    OpenInTextMode = 1 << 23
    DisableBuilderModeToggle = 1 << 24
    Borderless = 1 << 25
    DisableQueryHelpers = 1 << 26
    DisableNoResultTips = 1 << 27
    IgnoreSavedSearches = 1 << 28
    ObjectPicker = 1 << 29
    ObjectPickerAdvancedUI = 1 << 30
    ContextSwitchPreservedMask = Borderless | OpenLeftSidePanel | OpenInspectorPreview

# UnityCsReference-master/Runtime/Export/Shaders/Material.cs
class MaterialPropertyType(IntEnum):
    Float = 1
    Int = 2
    Vector = 3
    Matrix = 4
    Texture = 5
    ConstantBuffer = 6
    ComputeBuffer = 7

# UnityCsReference-master/Runtime/Export/Shaders/Shader.bindings.cs
class DisableBatchingType(IntEnum):
    FALSE = 1
    TRUE = 2
    WhenLODFading = 3

# UnityCsReference-master/Runtime/Export/Shaders/ShaderKeyword.bindings.cs
class ShaderKeywordType(IntEnum):
    NONE = 0
    BuiltinDefault = (1 << 1)
    BuiltinExtra = (1 << 2) | BuiltinDefault
    BuiltinAutoStripped = (1 << 3) | BuiltinDefault
    UserDefined = (1 << 4)
    Plugin = (1 << 5)

# UnityCsReference-master/Runtime/Export/Shaders/ShaderProperties.cs
class ShaderPropertyType(IntEnum):
    Color = 1
    Vector = 2
    Float = 3
    Range = 4
    Texture = 5
    Int = 6

class ShaderPropertyFlags(IntFlag):
    NONE = 0
    HideInInspector = 1 << 0
    PerRendererData = 1 << 1
    NoScaleOffset = 1 << 2
    Normal = 1 << 3
    HDR = 1 << 4
    Gamma = 1 << 5
    NonModifiableTextureData = 1 << 6
    MainTexture = 1 << 7
    MainColor = 1 << 8

# UnityCsReference-master/Runtime/Export/SystemInfo/SystemInfo.bindings.cs
class BatteryStatus(IntEnum):
    Unknown = 0
    Charging = 1
    Discharging = 2
    NotCharging = 3
    Full = 4

class OperatingSystemFamily(IntEnum):
    Other = 0
    MacOSX = 1
    Windows = 2
    Linux = 3

class DeviceType(IntEnum):
    Unknown = 0
    Handheld = 1
    Console = 2
    Desktop = 3

# UnityCsReference-master/Runtime/Export/TouchScreenKeyboard/TouchScreenKeyboard.bindings.cs
class Status(IntEnum):
    Visible = 0
    Done = 1
    Canceled = 2
    LostFocus = 3

# UnityCsReference-master/Runtime/Export/TouchScreenKeyboard/TouchScreenKeyboardType.cs
class TouchScreenKeyboardType(IntEnum):
    Default = 0
    ASCIICapable = 1
    NumbersAndPunctuation = 2
    URL = 3
    NumberPad = 4
    PhonePad = 5
    NamePhonePad = 6
    EmailAddress = 7
    NintendoNetworkAccount = 8
    Social = 9
    Search = 10
    DecimalPad = 11
    OneTimeCode = 12

# UnityCsReference-master/Runtime/Export/UnityEngineInternal/TypeInferenceRuleAttribute.cs
class TypeInferenceRules(IntEnum):
    TypeReferencedByFirstArgument = 1
    TypeReferencedBySecondArgument = 2
    ArrayOfTypeReferencedByFirstArgument = 3
    TypeOfFirstArgument = 4

# UnityCsReference-master/Runtime/Export/UnityEvent/UnityEvent.cs
class PersistentListenerMode(IntEnum):
    EventDefined = 1
    Void = 2
    Object = 3
    Int = 4
    Float = 5
    String = 6
    Bool = 7

class UnityEventCallState(IntEnum):
    Off = 0
    EditorAndRuntime = 1
    RuntimeOnly = 2

# UnityCsReference-master/Runtime/Export/WSA/WSAApplication.bindings.cs
class WindowActivationState(IntEnum):
    CodeActivated = 0
    Deactivated = 1
    PointerActivated = 2

# UnityCsReference-master/Runtime/Export/WSA/WSALauncher.bindings.cs
class Folder(IntEnum):
    Installation = 1
    Temporary = 2
    Local = 3
    Roaming = 4
    CameraRoll = 5
    DocumentsLibrary = 6
    HomeGroup = 7
    MediaServerDevices = 8
    MusicLibrary = 9
    PicturesLibrary = 10
    Playlists = 11
    RemovableDevices = 12
    SavedPictures = 13
    VideosLibrary = 14

# UnityCsReference-master/Runtime/Export/WSA/WSATiles.bindings.cs
class TileTemplate(IntEnum):
    TileSquare150x150Image = 0
    TileSquare150x150Block = 1
    TileSquare150x150Text01 = 2
    TileSquare150x150Text02 = 3
    TileSquare150x150Text03 = 4
    TileSquare150x150Text04 = 5
    TileSquare150x150PeekImageAndText01 = 6
    TileSquare150x150PeekImageAndText02 = 7
    TileSquare150x150PeekImageAndText03 = 8
    TileSquare150x150PeekImageAndText04 = 9
    TileWide310x150Image = 10
    TileWide310x150ImageCollection = 11
    TileWide310x150ImageAndText01 = 12
    TileWide310x150ImageAndText02 = 13
    TileWide310x150BlockAndText01 = 14
    TileWide310x150BlockAndText02 = 15
    TileWide310x150PeekImageCollection01 = 16
    TileWide310x150PeekImageCollection02 = 17
    TileWide310x150PeekImageCollection03 = 18
    TileWide310x150PeekImageCollection04 = 19
    TileWide310x150PeekImageCollection05 = 20
    TileWide310x150PeekImageCollection06 = 21
    TileWide310x150PeekImageAndText01 = 22
    TileWide310x150PeekImageAndText02 = 23
    TileWide310x150PeekImage01 = 24
    TileWide310x150PeekImage02 = 25
    TileWide310x150PeekImage03 = 26
    TileWide310x150PeekImage04 = 27
    TileWide310x150PeekImage05 = 28
    TileWide310x150PeekImage06 = 29
    TileWide310x150SmallImageAndText01 = 30
    TileWide310x150SmallImageAndText02 = 31
    TileWide310x150SmallImageAndText03 = 32
    TileWide310x150SmallImageAndText04 = 33
    TileWide310x150SmallImageAndText05 = 34
    TileWide310x150Text01 = 35
    TileWide310x150Text02 = 36
    TileWide310x150Text03 = 37
    TileWide310x150Text04 = 38
    TileWide310x150Text05 = 39
    TileWide310x150Text06 = 40
    TileWide310x150Text07 = 41
    TileWide310x150Text08 = 42
    TileWide310x150Text09 = 43
    TileWide310x150Text10 = 44
    TileWide310x150Text11 = 45
    TileSquare310x310BlockAndText01 = 46
    TileSquare310x310BlockAndText02 = 47
    TileSquare310x310Image = 48
    TileSquare310x310ImageAndText01 = 49
    TileSquare310x310ImageAndText02 = 50
    TileSquare310x310ImageAndTextOverlay01 = 51
    TileSquare310x310ImageAndTextOverlay02 = 52
    TileSquare310x310ImageAndTextOverlay03 = 53
    TileSquare310x310ImageCollectionAndText01 = 54
    TileSquare310x310ImageCollectionAndText02 = 55
    TileSquare310x310ImageCollection = 56
    TileSquare310x310SmallImagesAndTextList01 = 57
    TileSquare310x310SmallImagesAndTextList02 = 58
    TileSquare310x310SmallImagesAndTextList03 = 59
    TileSquare310x310SmallImagesAndTextList04 = 60
    TileSquare310x310Text01 = 61
    TileSquare310x310Text02 = 62
    TileSquare310x310Text03 = 63
    TileSquare310x310Text04 = 64
    TileSquare310x310Text05 = 65
    TileSquare310x310Text06 = 66
    TileSquare310x310Text07 = 67
    TileSquare310x310Text08 = 68
    TileSquare310x310TextList01 = 69
    TileSquare310x310TextList02 = 70
    TileSquare310x310TextList03 = 71
    TileSquare310x310SmallImageAndText01 = 72
    TileSquare310x310SmallImagesAndTextList05 = 73
    TileSquare310x310Text09 = 74
    TileSquare71x71IconWithBadge = 75
    TileSquare150x150IconWithBadge = 76
    TileWide310x150IconWithBadgeAndText = 77
    TileSquare71x71Image = 78
    TileTall150x310Image = 79
    TileSquare99x99IconWithBadge = 1000
    TileSquare210x210IconWithBadge = 1001
    TileWide432x210IconWithBadgeAndText = 1002

class ToastTemplate(IntEnum):
    ToastImageAndText01 = 0
    ToastImageAndText02 = 1
    ToastImageAndText03 = 2
    ToastImageAndText04 = 3
    ToastText01 = 4
    ToastText02 = 5
    ToastText03 = 6
    ToastText04 = 7

class TileForegroundText(IntEnum):
    Default = 1
    Dark = 0
    Light = 1

# UnityCsReference-master/Runtime/Export/Windows/PhotoCapture.bindings.cs
class PhotoCaptureFileOutputFormat(IntEnum):
    PNG = 0
    JPG = 1

class PhotoCapture_CaptureResultType(IntEnum):
    Success = 0
    UnknownError = 1

# UnityCsReference-master/Runtime/Export/Windows/Speech.cs
class ConfidenceLevel(IntEnum):
    High = 0
    Medium = 1
    Low = 2
    Rejected = 3

class SpeechSystemStatus(IntEnum):
    Stopped = 0
    Running = 1
    Failed = 2

class SpeechError(IntEnum):
    NoError = 0
    TopicLanguageNotSupported = 1
    GrammarLanguageMismatch = 2
    GrammarCompilationFailure = 3
    AudioQualityFailure = 4
    PauseLimitExceeded = 5
    TimeoutExceeded = 6
    NetworkFailure = 7
    MicrophoneUnavailable = 8
    UnknownError = 9

class DictationTopicConstraint(IntEnum):
    WebSearch = 1
    Form = 2
    Dictation = 3

class DictationCompletionCause(IntEnum):
    Complete = 1
    AudioQualityFailure = 2
    Canceled = 3
    TimeoutExceeded = 4
    PauseLimitExceeded = 5
    NetworkFailure = 6
    MicrophoneUnavailable = 7
    UnknownError = 8

# UnityCsReference-master/Runtime/Export/Windows/VideoCapture.bindings.cs
class VideoCapture_CaptureResultType(IntEnum):
    Success = 0
    UnknownError = 1

class AudioState(IntEnum):
    MicAudio = 0
    ApplicationAudio = 1
    ApplicationAndMicAudio = 2
    NONE = 3

# UnityCsReference-master/Runtime/Export/Windows/WebCam.bindings.cs
class CapturePixelFormat(IntEnum):
    BGRA32 = 0
    NV12 = 1
    JPEG = 2
    PNG = 3

class WebCamMode(IntEnum):
    NONE = 0
    PhotoMode = 1
    VideoMode = 2

# UnityCsReference-master/Runtime/Export/iOS/iOS.deprecated.cs
class iPhoneScreenOrientation(IntEnum):
    Unknown = 1
    Portrait = 2
    PortraitUpsideDown = 3
    LandscapeLeft = 4
    LandscapeRight = 5
    AutoRotation = 6
    Landscape = 7

class iPhoneNetworkReachability(IntEnum):
    NotReachable = 1
    ReachableViaCarrierDataNetwork = 2

class iPhoneGeneration(IntEnum):
    Unknown = 1
    iPhone = 2
    iPhone3G = 3
    iPhone3GS = 4
    iPodTouch1Gen = 5
    iPodTouch2Gen = 6
    iPodTouch3Gen = 7
    iPad1Gen = 8
    iPhone4 = 9
    iPodTouch4Gen = 10
    iPad2Gen = 11
    iPhone4S = 12
    iPad3Gen = 13
    iPhone5 = 14
    iPodTouch5Gen = 15
    iPadMini1Gen = 16
    iPad4Gen = 17
    iPhone5C = 18
    iPhone5S = 19
    iPhoneUnknown = 20
    iPadUnknown = 21
    iPodTouchUnknown = 22

class iPhoneTouchPhase(IntEnum):
    Began = 1
    Moved = 2
    Stationary = 3
    Ended = 4
    Canceled = 5

class iPhoneMovieControlMode(IntEnum):
    Full = 1
    Minimal = 2
    Hidden = 3

class iPhoneMovieScalingMode(IntEnum):
    NONE = 1
    AspectFit = 2
    AspectFill = 3
    Fill = 4

class iPhoneKeyboardType(IntEnum):
    Default = 1
    ASCIICapable = 2
    NumbersAndPunctuation = 3
    URL = 4
    NumberPad = 5
    PhonePad = 6
    NamePhonePad = 7
    EmailAddress = 8

class iPhoneOrientation(IntEnum):
    Unknown = 1
    Portrait = 2
    PortraitUpsideDown = 3
    LandscapeLeft = 4
    LandscapeRight = 5
    FaceUp = 6
    FaceDown = 7

class Layout(IntEnum):
    Top = 1
    Bottom = 2
    TopLeft = 0
    TopRight = 4
    TopCenter = 8
    BottomLeft = 1
    BottomRight = 5
    BottomCenter = 9
    CenterLeft = 2
    CenterRight = 6
    Center = 10
    Manual = 1

class Type(IntEnum):
    Banner = 1
    MediumRect = 2

# UnityCsReference-master/Runtime/Export/iOS/iOSDevice.bindings.cs
class iOSDevice_DeviceGeneration(IntEnum):
    Unknown = 0
    iPhone = 1
    iPhone3G = 2
    iPhone3GS = 3
    iPodTouch1Gen = 4
    iPodTouch2Gen = 5
    iPodTouch3Gen = 6
    iPad1Gen = 7
    iPhone4 = 8
    iPodTouch4Gen = 9
    iPad2Gen = 10
    iPhone4S = 11
    iPad3Gen = 12
    iPhone5 = 13
    iPodTouch5Gen = 14
    iPadMini1Gen = 15
    iPad4Gen = 16
    iPhone5C = 17
    iPhone5S = 18
    iPadAir1 = 19
    iPadMini2Gen = 20
    iPhone6 = 21
    iPhone6Plus = 22
    iPadMini3Gen = 23
    iPadAir2 = 24
    iPhone6S = 25
    iPhone6SPlus = 26
    iPadPro1Gen = 27
    iPadMini4Gen = 28
    iPhoneSE1Gen = 29
    iPadPro10Inch1Gen = 30
    iPhone7 = 31
    iPhone7Plus = 32
    iPodTouch6Gen = 33
    iPad5Gen = 34
    iPadPro2Gen = 35
    iPadPro10Inch2Gen = 36
    iPhone8 = 37
    iPhone8Plus = 38
    iPhoneX = 39
    iPhoneXS = 40
    iPhoneXSMax = 41
    iPhoneXR = 42
    iPadPro11Inch = 43
    iPadPro3Gen = 44
    iPad6Gen = 45
    iPadAir3Gen = 46
    iPadMini5Gen = 47
    iPhone11 = 48
    iPhone11Pro = 49
    iPhone11ProMax = 50
    iPodTouch7Gen = 51
    iPad7Gen = 52
    iPhoneSE2Gen = 53
    iPadPro11Inch2Gen = 54
    iPadPro4Gen = 55
    iPhone12Mini = 56
    iPhone12 = 57
    iPhone12Pro = 58
    iPhone12ProMax = 59
    iPad8Gen = 60
    iPadAir4Gen = 61
    iPad9Gen = 62
    iPadMini6Gen = 63
    iPhone13 = 64
    iPhone13Mini = 65
    iPhone13Pro = 66
    iPhone13ProMax = 67
    iPadPro5Gen = 68
    iPadPro11Inch3Gen = 69
    iPhoneSE3Gen = 70
    iPadAir5Gen = 71
    iPhone14 = 72
    iPhone14Plus = 73
    iPhone14Pro = 74
    iPhone14ProMax = 75
    iPadPro6Gen = 76
    iPadPro11Inch4Gen = 77
    iPad10Gen = 78
    iPhone15 = 79
    iPhone15Plus = 80
    iPhone15Pro = 81
    iPhone15ProMax = 82
    iPhoneUnknown = 10001
    iPadUnknown = 10002
    iPodTouchUnknown = 10003

class ActivityIndicatorStyle(IntEnum):
    DontShow = 1
    WhiteLarge = 0
    White = 1
    Gray = 2
    Medium = 100
    Large = 101

# UnityCsReference-master/Runtime/Export/iOS/tvOSDevice.bindings.cs
class tvOSDevice_DeviceGeneration(IntEnum):
    Unknown = 0
    AppleTVHD = 1001
    AppleTV4K = 1002
    AppleTV4K2Gen = 1003
    AppleTV4K3Gen = 1004

