# Standalone PSD (Adobe Photoshop) binary writer
# Writes multi-layer PSD files from raw pixel data without PIL/Pillow.
#
# Reference: Adobe Photoshop File Formats Specification
# Structural reference: psd ref.py (PIL PSD reader)

import os
import struct
import numpy as np


# ----------------------------------------------------------------
# Blender blend mode -> PSD blend mode key (4-byte ASCII)
# ----------------------------------------------------------------
BLEND_MODE_MAP = {
    'MIX':          b'norm',
    'DARKEN':       b'dark',
    'MULTIPLY':     b'mul ',
    'COLOR_BURN':   b'idiv',
    'LIGHTEN':      b'lite',
    'SCREEN':       b'scrn',
    'COLOR_DODGE':  b'div ',
    'ADD':          b'lddg',   # Linear Dodge (Add)
    'OVERLAY':      b'over',
    'SOFT_LIGHT':   b'sLit',
    'LINEAR_LIGHT': b'lLit',
    'DIFFERENCE':   b'diff',
    'EXCLUSION':    b'smud',
    'SUBTRACT':     b'fsub',
    'DIVIDE':       b'fdiv',
    'HUE':          b'hue ',
    'SATURATION':   b'sat ',
    'COLOR':        b'colr',
    'VALUE':        b'lum ',
}


def _pack_be_u16(val):
    return struct.pack('>H', int(val))


def _pack_be_u32(val):
    return struct.pack('>I', int(val))


def _pack_be_i16(val):
    return struct.pack('>h', int(val))


def _pack_be_i32(val):
    return struct.pack('>i', int(val))


def _pixels_to_channels_u8(pixels_rgba):
    """Convert float32 RGBA (H, W, 4) to separate uint8 channel arrays.

    Returns (R, G, B, A) each as (H, W) uint8 arrays.
    """
    clamped = np.clip(pixels_rgba, 0.0, 1.0)
    u8 = (clamped * 255.0 + 0.5).astype(np.uint8)
    return u8[:, :, 0], u8[:, :, 1], u8[:, :, 2], u8[:, :, 3]


def _write_channel_data_raw(channel):
    """Write a single channel as raw (uncompressed) data.

    Args:
        channel: (H, W) uint8 numpy array

    Returns:
        bytes - compression type (0) + raw scanline data
    """
    # Compression type 0 = raw
    data = _pack_be_u16(0)
    # Write scanlines top-to-bottom
    data += channel.tobytes()
    return data


def _write_pascal_string(name, pad_to=4):
    """Write a Pascal string padded to `pad_to` byte boundary.

    PSD uses Pascal strings: 1-byte length prefix + string bytes + padding.
    """
    encoded = name.encode('latin-1', 'replace')[:255]
    length = len(encoded)
    data = struct.pack('B', length) + encoded
    # Pad to `pad_to` boundary (including the length byte)
    total = len(data)
    remainder = total % pad_to
    if remainder:
        data += b'\x00' * (pad_to - remainder)
    return data


def _load_icc_profile():
    """Load the sRGB ICC profile from the addon's utils directory.

    Returns bytes of the ICC profile, or empty bytes if not found.
    """
    icc_path = os.path.join(os.path.dirname(__file__), 'sRGB-elle-V2-srgbtrc.icc')
    try:
        with open(icc_path, 'rb') as f:
            return f.read()
    except (OSError, IOError):
        return b''


def write_psd(filepath, layers_data, canvas_width, canvas_height):
    """Write a PSD file with multiple layers.

    Args:
        filepath: Output file path (str)
        layers_data: List of dicts, each with:
            'name': str - layer name
            'pixels': np.ndarray (H, W, 4) float32 RGBA
            'x': int - left edge position on canvas
            'y': int - top edge position on canvas (from top)
            'width': int
            'height': int
            'opacity': float 0-1
            'blend_mode': str - Blender blend mode key
            'visible': bool
        canvas_width: int - total canvas width
        canvas_height: int - total canvas height
    """
    with open(filepath, 'wb') as f:
        _write_header(f, canvas_width, canvas_height, channels=4)
        _write_color_mode_data(f)
        _write_image_resources(f)
        _write_layer_and_mask_info(f, layers_data, canvas_width, canvas_height)
        _write_merged_image_data(f, layers_data, canvas_width, canvas_height)


def _write_header(f, width, height, channels=4):
    """Write PSD file header (26 bytes)."""
    f.write(b'8BPS')                    # Signature
    f.write(_pack_be_u16(1))            # Version
    f.write(b'\x00' * 6)               # Reserved
    f.write(_pack_be_u16(channels))     # Number of channels
    f.write(_pack_be_u32(height))       # Height
    f.write(_pack_be_u32(width))        # Width
    f.write(_pack_be_u16(8))            # Bits per channel
    f.write(_pack_be_u16(3))            # Color mode: RGB


def _write_color_mode_data(f):
    """Write empty color mode data section."""
    f.write(_pack_be_u32(0))


def _write_image_resources(f):
    """Write image resources section with embedded sRGB ICC profile."""
    icc_data = _load_icc_profile()

    if not icc_data:
        # No ICC profile available, write empty section
        f.write(_pack_be_u32(0))
        return

    # Build ICC profile resource (ID 1039)
    resource = b''
    resource += b'8BIM'                     # Signature
    resource += _pack_be_u16(1039)          # Resource ID: ICC profile
    resource += b'\x00\x00'                 # Pascal string (empty name: length 0 + pad)
    resource += _pack_be_u32(len(icc_data)) # Data length
    resource += icc_data
    # Pad resource data to even length
    if len(icc_data) % 2:
        resource += b'\x00'

    # Write section length + resource
    f.write(_pack_be_u32(len(resource)))
    f.write(resource)


def _write_layer_and_mask_info(f, layers_data, canvas_width, canvas_height):
    """Write Layer and Mask Information section."""
    # Build the layer info block in memory first so we can get its size
    layer_info = _build_layer_info(layers_data, canvas_width, canvas_height)

    # Layer info sub-section length (must be even)
    layer_info_length = len(layer_info)

    # Layer and Mask info section length = 4 (for sub-section length field) + layer_info data
    section_length = 4 + layer_info_length
    # Section length itself must be even
    if section_length % 2:
        section_length += 1

    f.write(_pack_be_u32(section_length))

    # Layer info sub-section length
    f.write(_pack_be_u32(layer_info_length))

    # Layer info data
    f.write(layer_info)

    # Pad section to even length if needed
    written = 4 + layer_info_length
    if written < section_length:
        f.write(b'\x00' * (section_length - written))


def _build_layer_info(layers_data, canvas_width, canvas_height):
    """Build the layer info block.

    PSD stores layers bottom-to-top, so we reverse the input list
    (which comes in top-to-bottom order from Blender).

    Structure:
        - Layer count (i16)
        - Layer records (per-layer metadata)
        - Channel image data (per-layer pixel data)
    """
    import io
    buf = io.BytesIO()

    # Reverse: Blender gives top-to-bottom, PSD expects bottom-to-top
    psd_layers = list(reversed(layers_data))

    count = len(psd_layers)
    # Negative count means first alpha channel contains transparency data for merged result
    buf.write(_pack_be_i16(count))

    # --- Layer Records ---
    # Pre-compute channel data for each layer so we know sizes
    all_channel_data = []

    for layer in psd_layers:
        r, g, b, a = _pixels_to_channels_u8(layer['pixels'])
        ch_data_r = _write_channel_data_raw(r)
        ch_data_g = _write_channel_data_raw(g)
        ch_data_b = _write_channel_data_raw(b)
        ch_data_a = _write_channel_data_raw(a)
        all_channel_data.append((ch_data_a, ch_data_r, ch_data_g, ch_data_b))

    for i, layer in enumerate(psd_layers):
        ch_a, ch_r, ch_g, ch_b = all_channel_data[i]

        # Bounding box: top, left, bottom, right
        top = layer['y']
        left = layer['x']
        bottom = top + layer['height']
        right = left + layer['width']

        buf.write(_pack_be_i32(top))
        buf.write(_pack_be_i32(left))
        buf.write(_pack_be_i32(bottom))
        buf.write(_pack_be_i32(right))

        # Number of channels: 4 (R, G, B, A)
        buf.write(_pack_be_u16(4))

        # Channel info: (id, data_length) pairs
        # Channel IDs: -1=alpha, 0=red, 1=green, 2=blue
        buf.write(_pack_be_i16(-1))
        buf.write(_pack_be_u32(len(ch_a)))
        buf.write(_pack_be_i16(0))
        buf.write(_pack_be_u32(len(ch_r)))
        buf.write(_pack_be_i16(1))
        buf.write(_pack_be_u32(len(ch_g)))
        buf.write(_pack_be_i16(2))
        buf.write(_pack_be_u32(len(ch_b)))

        # Blend mode signature
        buf.write(b'8BIM')

        # Blend mode key
        bm = BLEND_MODE_MAP.get(layer.get('blend_mode', 'MIX'), b'norm')
        buf.write(bm)

        # Opacity (0-255)
        opacity_byte = int(np.clip(layer.get('opacity', 1.0), 0.0, 1.0) * 255)
        buf.write(struct.pack('B', opacity_byte))

        # Clipping: 0 = base
        buf.write(struct.pack('B', 0))

        # Flags: bit 1 = visible (inverted: 0 = visible, 1 = hidden in PSD)
        visible = layer.get('visible', True)
        flags = 0 if visible else 0x02
        buf.write(struct.pack('B', flags))

        # Filler
        buf.write(b'\x00')

        # Extra data length (layer name as pascal string)
        name = layer.get('name', 'Layer')
        pascal_name = _write_pascal_string(name, pad_to=4)

        # Extra data: mask data (0) + blending ranges (0) + pascal name
        extra_data = _pack_be_u32(0)  # Layer mask data length = 0
        extra_data += _pack_be_u32(0)  # Layer blending ranges length = 0
        extra_data += pascal_name

        buf.write(_pack_be_u32(len(extra_data)))
        buf.write(extra_data)

    # --- Channel Image Data ---
    for ch_a, ch_r, ch_g, ch_b in all_channel_data:
        buf.write(ch_a)
        buf.write(ch_r)
        buf.write(ch_g)
        buf.write(ch_b)

    result = buf.getvalue()

    # Pad to even length
    if len(result) % 2:
        result += b'\x00'

    return result


def _write_merged_image_data(f, layers_data, canvas_width, canvas_height):
    """Write merged (flattened) composite image data.

    This is what PSD viewers show when they don't parse layers.
    Only visible layers are included in the composite.
    """
    # Create composite by alpha-blending visible layers bottom to top
    composite = np.zeros((canvas_height, canvas_width, 4), dtype=np.float32)

    # layers_data is top-to-bottom from Blender, so reversed gives bottom-to-top
    for layer in reversed(layers_data):
        # Skip hidden layers
        if not layer.get('visible', True):
            continue

        x0 = layer['x']
        y0 = layer['y']
        x1 = x0 + layer['width']
        y1 = y0 + layer['height']

        # Clamp to canvas
        sx0 = max(0, -x0)
        sy0 = max(0, -y0)
        dx0 = max(0, x0)
        dy0 = max(0, y0)
        dx1 = min(canvas_width, x1)
        dy1 = min(canvas_height, y1)
        sx1 = sx0 + (dx1 - dx0)
        sy1 = sy0 + (dy1 - dy0)

        if dx1 <= dx0 or dy1 <= dy0:
            continue

        src = layer['pixels'][sy0:sy1, sx0:sx1]
        dst = composite[dy0:dy1, dx0:dx1]

        opacity = layer.get('opacity', 1.0)
        src_a = src[:, :, 3:4] * opacity
        inv_a = 1.0 - src_a

        out_a = src_a + dst[:, :, 3:4] * inv_a
        safe_a = np.where(out_a == 0, 1.0, out_a)

        dst[:, :, :3] = (src[:, :, :3] * src_a + dst[:, :, :3] * dst[:, :, 3:4] * inv_a) / safe_a
        dst[:, :, 3:4] = out_a

    # Convert to uint8 channels
    r, g, b, a = _pixels_to_channels_u8(composite)

    # Write compression type 0 (raw) for the merged image
    f.write(_pack_be_u16(0))

    # Write channels in order: R, G, B, A (planar)
    f.write(r.tobytes())
    f.write(g.tobytes())
    f.write(b.tobytes())
    f.write(a.tobytes())
