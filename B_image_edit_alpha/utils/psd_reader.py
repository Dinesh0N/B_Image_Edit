# Standalone PSD (Adobe Photoshop) binary reader
# Reads multi-layer PSD files into raw pixel data without PIL/Pillow.
#
# Reference: Adobe Photoshop File Formats Specification
# Structural reference: psd ref.py (PIL PSD reader)

import struct
import numpy as np


# ----------------------------------------------------------------
# PSD blend mode key (4-byte ASCII) -> Blender blend mode
# Inverse of BLEND_MODE_MAP from psd_writer.py
# ----------------------------------------------------------------
PSD_BLEND_TO_BLENDER = {
    b'norm': 'MIX',
    b'dark': 'DARKEN',
    b'mul ': 'MULTIPLY',
    b'idiv': 'COLOR_BURN',
    b'lite': 'LIGHTEN',
    b'scrn': 'SCREEN',
    b'div ': 'COLOR_DODGE',
    b'lddg': 'ADD',        # Linear Dodge (Add)
    b'over': 'OVERLAY',
    b'sLit': 'SOFT_LIGHT',
    b'lLit': 'LINEAR_LIGHT',
    b'diff': 'DIFFERENCE',
    b'smud': 'EXCLUSION',
    b'fsub': 'SUBTRACT',
    b'fdiv': 'DIVIDE',
    b'hue ': 'HUE',
    b'sat ': 'SATURATION',
    b'colr': 'COLOR',
    b'lum ': 'VALUE',
    # Additional common PSD blend modes mapped to closest Blender equivalent
    b'pass': 'MIX',        # Pass Through (group default)
    b'diss': 'MIX',        # Dissolve -> MIX (no Blender equivalent)
    b'lbrn': 'COLOR_BURN', # Linear Burn -> Color Burn
    b'vLit': 'LINEAR_LIGHT',  # Vivid Light -> Linear Light
    b'pLit': 'LINEAR_LIGHT',  # Pin Light -> Linear Light
    b'hMix': 'MIX',        # Hard Mix -> MIX
}


# ----------------------------------------------------------------
# Binary helpers (big-endian)
# ----------------------------------------------------------------
def _i8(data, offset=0):
    return data[offset]


def _i16(data, offset=0):
    return struct.unpack_from('>H', data, offset)[0]


def _i32(data, offset=0):
    return struct.unpack_from('>I', data, offset)[0]


def _si16(data, offset=0):
    return struct.unpack_from('>h', data, offset)[0]


def _si32(data, offset=0):
    return struct.unpack_from('>i', data, offset)[0]


# ----------------------------------------------------------------
# PackBits (RLE) decompression
# ----------------------------------------------------------------
def _unpackbits(data, expected_size):
    """Decompress PackBits-encoded data.

    Args:
        data: bytes of PackBits-compressed data
        expected_size: expected number of output bytes

    Returns:
        bytes of decompressed data
    """
    result = bytearray()
    i = 0
    while i < len(data) and len(result) < expected_size:
        n = data[i]
        i += 1
        if n >= 128:
            # Run of repeated byte
            count = 257 - n
            if i < len(data):
                result.extend([data[i]] * count)
                i += 1
        elif n < 128:
            # Literal run
            count = n + 1
            result.extend(data[i:i + count])
            i += count
        # n == 128 is a no-op
    return bytes(result[:expected_size])


# ----------------------------------------------------------------
# PSD Layer data class
# ----------------------------------------------------------------
class PSDLayer:
    """Represents a single layer from a PSD file."""
    __slots__ = ('name', 'top', 'left', 'bottom', 'right',
                 'width', 'height', 'pixels',
                 'opacity', 'blend_mode', 'visible')

    def __init__(self):
        self.name = 'Layer'
        self.top = 0
        self.left = 0
        self.bottom = 0
        self.right = 0
        self.width = 0
        self.height = 0
        self.pixels = None  # (H, W, 4) float32 RGBA
        self.opacity = 1.0
        self.blend_mode = 'MIX'
        self.visible = True


class PSDFile:
    """Represents a parsed PSD file."""
    __slots__ = ('width', 'height', 'layers')

    def __init__(self):
        self.width = 0
        self.height = 0
        self.layers = []  # List of PSDLayer, bottom-to-top order


# ----------------------------------------------------------------
# Main reader
# ----------------------------------------------------------------
def read_psd(filepath):
    """Read a PSD file and return a PSDFile with layers.

    Args:
        filepath: path to .psd file

    Returns:
        PSDFile with canvas dimensions and list of PSDLayer objects.
        Layers are in bottom-to-top order (first = bottom).

    Raises:
        ValueError: if file is not a valid PSD or uses unsupported features
    """
    with open(filepath, 'rb') as f:
        data = f.read()

    psd = PSDFile()
    offset = 0

    # ---- Header (26 bytes) ----
    if len(data) < 26:
        raise ValueError("File too small to be a PSD")

    sig = data[0:4]
    if sig != b'8BPS':
        raise ValueError("Not a PSD file (bad signature)")

    version = _i16(data, 4)
    if version != 1:
        raise ValueError(f"Unsupported PSD version: {version}")

    channels = _i16(data, 12)
    height = _i32(data, 14)
    width = _i32(data, 18)
    bits = _i16(data, 22)
    color_mode = _i16(data, 24)

    psd.width = width
    psd.height = height

    if bits != 8:
        raise ValueError(f"Only 8-bit PSD files are supported (got {bits}-bit)")

    if color_mode != 3:  # 3 = RGB
        raise ValueError(f"Only RGB PSD files are supported (got mode {color_mode})")

    offset = 26

    # ---- Color Mode Data ----
    if offset + 4 > len(data):
        raise ValueError("Unexpected end of file in color mode section")
    cm_size = _i32(data, offset)
    offset += 4 + cm_size

    # ---- Image Resources ----
    if offset + 4 > len(data):
        raise ValueError("Unexpected end of file in image resources section")
    ir_size = _i32(data, offset)
    offset += 4 + ir_size

    # ---- Layer and Mask Information ----
    if offset + 4 > len(data):
        raise ValueError("Unexpected end of file in layer/mask section")
    lm_size = _i32(data, offset)
    offset += 4

    if lm_size == 0:
        # No layers
        return psd

    lm_end = offset + lm_size

    # Layer info sub-section
    if offset + 4 > len(data):
        return psd
    li_size = _i32(data, offset)
    offset += 4

    if li_size == 0:
        return psd

    li_end = offset + li_size

    # Layer count (can be negative — abs value is the count)
    layer_count = _si16(data, offset)
    offset += 2
    layer_count = abs(layer_count)

    if layer_count == 0:
        return psd

    # ---- Parse Layer Records ----
    layer_records = []

    for _ in range(layer_count):
        record = {}

        # Bounding box (top, left, bottom, right)
        record['top'] = _si32(data, offset)
        record['left'] = _si32(data, offset + 4)
        record['bottom'] = _si32(data, offset + 8)
        record['right'] = _si32(data, offset + 12)
        offset += 16

        # Number of channels
        num_channels = _i16(data, offset)
        offset += 2

        # Skip layers with too many channels (likely groups/adjustments)
        if num_channels > 4:
            # Read channel info to advance offset
            channel_info = []
            for _ in range(num_channels):
                ch_id = _si16(data, offset)
                ch_size = _i32(data, offset + 2)
                channel_info.append((ch_id, ch_size))
                offset += 6
            record['channels'] = channel_info
            record['skip'] = True

            # Skip blend mode signature + key + opacity + clipping + flags + filler = 12 bytes
            offset += 12

            # Extra data
            extra_size = _i32(data, offset)
            offset += 4 + extra_size

            layer_records.append(record)
            continue

        # Channel info
        channel_info = []
        for _ in range(num_channels):
            ch_id = _si16(data, offset)
            ch_size = _i32(data, offset + 2)
            channel_info.append((ch_id, ch_size))
            offset += 6

        record['channels'] = channel_info
        record['skip'] = False

        # Blend mode signature (should be '8BIM')
        # blend_sig = data[offset:offset + 4]
        offset += 4

        # Blend mode key (4 bytes)
        blend_key = data[offset:offset + 4]
        record['blend_mode'] = PSD_BLEND_TO_BLENDER.get(blend_key, 'MIX')
        offset += 4

        # Opacity (0-255)
        record['opacity'] = _i8(data, offset) / 255.0
        offset += 1

        # Clipping
        offset += 1

        # Flags: bit 1 = hidden (inverted visibility)
        flags = _i8(data, offset)
        record['visible'] = not bool(flags & 0x02)
        offset += 1

        # Filler
        offset += 1

        # Extra data
        extra_size = _i32(data, offset)
        offset += 4
        extra_end = offset + extra_size

        record['name'] = 'Layer'

        if extra_size > 0:
            # Layer mask data
            mask_size = _i32(data, offset)
            offset += 4
            if mask_size > 0:
                offset += mask_size

            # Layer blending ranges
            ranges_size = _i32(data, offset)
            offset += 4
            if ranges_size > 0:
                offset += ranges_size

            # Pascal string (layer name)
            if offset < extra_end:
                name_len = _i8(data, offset)
                offset += 1
                if name_len > 0:
                    name_bytes = data[offset:offset + name_len]
                    record['name'] = name_bytes.decode('latin-1', 'replace')

            # Jump to end of extra data (skip additional resources like luni, etc.)
            offset = extra_end

        layer_records.append(record)

    # ---- Read Channel Image Data ----
    for record in layer_records:
        if record.get('skip', False):
            # Still need to skip the channel data
            for ch_id, ch_size in record['channels']:
                offset += ch_size
            continue

        top = record['top']
        left = record['left']
        bottom = record['bottom']
        right = record['right']
        w = right - left
        h = bottom - top

        if w <= 0 or h <= 0:
            # Empty layer, skip channel data
            for ch_id, ch_size in record['channels']:
                offset += ch_size
            continue

        # Read each channel
        channel_data = {}  # ch_id -> (H, W) uint8 array

        for ch_id, ch_size in record['channels']:
            # ch_size includes the 2-byte compression field
            ch_start = offset
            ch_end = ch_start + ch_size

            if ch_size < 2:
                offset = ch_end
                continue

            compression = _i16(data, offset)
            offset += 2

            if compression == 0:
                # Raw data
                expected = w * h
                raw = data[offset:offset + expected]
                if len(raw) == expected:
                    channel_data[ch_id] = np.frombuffer(raw, dtype=np.uint8).reshape((h, w))
                else:
                    channel_data[ch_id] = np.zeros((h, w), dtype=np.uint8)

            elif compression == 1:
                # PackBits (RLE)
                # Read byte counts per scanline (2 bytes each)
                bytecount_size = h * 2
                bytecounts_data = data[offset:offset + bytecount_size]
                offset += bytecount_size

                # Total compressed size from bytecounts
                total_compressed = 0
                for row in range(h):
                    total_compressed += _i16(bytecounts_data, row * 2)

                compressed = data[offset:offset + total_compressed]

                # Decompress
                decompressed = _unpackbits(compressed, w * h)
                if len(decompressed) >= w * h:
                    channel_data[ch_id] = np.frombuffer(decompressed[:w * h], dtype=np.uint8).reshape((h, w))
                else:
                    # Pad if short
                    padded = decompressed + b'\x00' * (w * h - len(decompressed))
                    channel_data[ch_id] = np.frombuffer(padded, dtype=np.uint8).reshape((h, w))
            else:
                # Unsupported compression (ZIP etc.), fill zeros
                channel_data[ch_id] = np.zeros((h, w), dtype=np.uint8)

            # Always advance to exact end of this channel's data
            offset = ch_end

        # Assemble RGBA
        pixels = np.zeros((h, w, 4), dtype=np.float32)

        # Channel IDs: 0=R, 1=G, 2=B, -1=A
        if 0 in channel_data:
            pixels[:, :, 0] = channel_data[0].astype(np.float32) / 255.0
        if 1 in channel_data:
            pixels[:, :, 1] = channel_data[1].astype(np.float32) / 255.0
        if 2 in channel_data:
            pixels[:, :, 2] = channel_data[2].astype(np.float32) / 255.0
        if -1 in channel_data:
            pixels[:, :, 3] = channel_data[-1].astype(np.float32) / 255.0
        else:
            # No alpha channel — fully opaque
            pixels[:, :, 3] = 1.0

        # Create PSDLayer
        layer = PSDLayer()
        layer.name = record.get('name', 'Layer')
        layer.top = top
        layer.left = left
        layer.bottom = bottom
        layer.right = right
        layer.width = w
        layer.height = h
        layer.pixels = pixels
        layer.opacity = record.get('opacity', 1.0)
        layer.blend_mode = record.get('blend_mode', 'MIX')
        layer.visible = record.get('visible', True)

        psd.layers.append(layer)

    return psd
