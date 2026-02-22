# Shared state and utilities for B_image_edit addon

import numpy as np


# ----------------------------
# Shared State
# ----------------------------
cursor_pos = None
show_cursor = False
cursor_pixel_scale = 1.0

# Gradient Tool preview state
gradient_preview_start = None  # (x, y) screen coordinates
gradient_preview_end = None    # (x, y) screen coordinates

# Crop Tool preview state
crop_preview_start = None  # (x, y) screen coordinates
crop_preview_end = None    # (x, y) screen coordinates

# Clone Tool state
clone_source_pos = None    # (x, y) screen coordinates - source point
clone_cursor_pos = None    # (x, y) screen coordinates - current brush position
clone_source_set = False   # Whether source has been set


def blend_pixel(pixels, idx, tr, tg, tb, ta, mode):
    """
    Blend source pixel (tr, tg, tb, ta) onto destination pixels[idx] using 'mode'.
    Modes correspond to Blender Brush Blend modes.
    """
    # Destination pixel
    dr = pixels[idx]
    dg = pixels[idx + 1]
    db = pixels[idx + 2]
    da = pixels[idx + 3]

    out_r, out_g, out_b, out_a = dr, dg, db, da

    if mode == 'MIX':
        # Standard Alpha Blending (Source Over)
        inv_ta = 1.0 - ta
        out_r = tr * ta + dr * inv_ta
        out_g = tg * ta + dg * inv_ta
        out_b = tb * ta + db * inv_ta
        out_a = ta + da * inv_ta

    elif mode == 'ADD':
        # Additive blending
        out_r = min(1.0, dr + tr * ta)
        out_g = min(1.0, dg + tg * ta)
        out_b = min(1.0, db + tb * ta)
        out_a = min(1.0, da + ta)

    elif mode == 'SUBTRACT':
        # Subtractive
        out_r = max(0.0, dr - tr * ta)
        out_g = max(0.0, dg - tg * ta)
        out_b = max(0.0, db - tb * ta)
        out_a = da

    elif mode == 'MULTIPLY':
        # Multiply
        inv_ta = 1.0 - ta
        out_r = dr * (inv_ta + tr * ta)
        out_g = dg * (inv_ta + tg * ta)
        out_b = db * (inv_ta + tb * ta)
        out_a = ta + da * inv_ta

    elif mode == 'LIGHTEN':
        # Lighten: max(dst, src)
        target_r = max(dr, tr)
        target_g = max(dg, tg)
        target_b = max(db, tb)
        inv_ta = 1.0 - ta
        out_r = target_r * ta + dr * inv_ta
        out_g = target_g * ta + dg * inv_ta
        out_b = target_b * ta + db * inv_ta
        out_a = ta + da * inv_ta

    elif mode == 'DARKEN':
        # Darken: min(dst, src)
        target_r = min(dr, tr)
        target_g = min(dg, tg)
        target_b = min(db, tb)
        inv_ta = 1.0 - ta
        out_r = target_r * ta + dr * inv_ta
        out_g = target_g * ta + dg * inv_ta
        out_b = target_b * ta + db * inv_ta
        out_a = ta + da * inv_ta
        
    elif mode == 'ERASE_ALPHA':
        # Erase Alpha: Reduce dst alpha by src alpha
        out_a = max(0.0, da - ta)
    
    elif mode == 'ADD_ALPHA':
        # Add Alpha
        out_a = min(1.0, da + ta)
    
    else:
        # Fallback to MIX
        inv_ta = 1.0 - ta
        out_r = tr * ta + dr * inv_ta
        out_g = tg * ta + dg * inv_ta
        out_b = tb * ta + db * inv_ta
        out_a = ta + da * inv_ta

    pixels[idx] = out_r
    pixels[idx + 1] = out_g
    pixels[idx + 2] = out_b
    pixels[idx + 3] = out_a


def blend_pixels_numpy(dst, src_r, src_g, src_b, src_a, mode='MIX'):
    """Vectorized pixel blending using NumPy arrays.
    
    Args:
        dst: NumPy array of shape (N, 4) — destination pixels (modified in-place)
        src_r, src_g, src_b, src_a: NumPy arrays of shape (N,) — source RGBA
        mode: Blend mode string
    """
    dr, dg, db, da = dst[:, 0], dst[:, 1], dst[:, 2], dst[:, 3]
    inv_ta = 1.0 - src_a

    if mode == 'MIX':
        dst[:, 0] = src_r * src_a + dr * inv_ta
        dst[:, 1] = src_g * src_a + dg * inv_ta
        dst[:, 2] = src_b * src_a + db * inv_ta
        dst[:, 3] = src_a + da * inv_ta
    elif mode == 'ADD':
        dst[:, 0] = np.minimum(1.0, dr + src_r * src_a)
        dst[:, 1] = np.minimum(1.0, dg + src_g * src_a)
        dst[:, 2] = np.minimum(1.0, db + src_b * src_a)
        dst[:, 3] = np.minimum(1.0, da + src_a)
    elif mode == 'SUBTRACT':
        dst[:, 0] = np.maximum(0.0, dr - src_r * src_a)
        dst[:, 1] = np.maximum(0.0, dg - src_g * src_a)
        dst[:, 2] = np.maximum(0.0, db - src_b * src_a)
        dst[:, 3] = da
    elif mode == 'MULTIPLY':
        dst[:, 0] = dr * (inv_ta + src_r * src_a)
        dst[:, 1] = dg * (inv_ta + src_g * src_a)
        dst[:, 2] = db * (inv_ta + src_b * src_a)
        dst[:, 3] = src_a + da * inv_ta
    elif mode == 'LIGHTEN':
        target_r = np.maximum(dr, src_r)
        target_g = np.maximum(dg, src_g)
        target_b = np.maximum(db, src_b)
        dst[:, 0] = target_r * src_a + dr * inv_ta
        dst[:, 1] = target_g * src_a + dg * inv_ta
        dst[:, 2] = target_b * src_a + db * inv_ta
        dst[:, 3] = src_a + da * inv_ta
    elif mode == 'DARKEN':
        target_r = np.minimum(dr, src_r)
        target_g = np.minimum(dg, src_g)
        target_b = np.minimum(db, src_b)
        dst[:, 0] = target_r * src_a + dr * inv_ta
        dst[:, 1] = target_g * src_a + dg * inv_ta
        dst[:, 2] = target_b * src_a + db * inv_ta
        dst[:, 3] = src_a + da * inv_ta
    elif mode == 'ERASE_ALPHA':
        dst[:, 3] = np.maximum(0.0, da - src_a)
    elif mode == 'ADD_ALPHA':
        dst[:, 3] = np.minimum(1.0, da + src_a)
    else:
        # Fallback to MIX
        dst[:, 0] = src_r * src_a + dr * inv_ta
        dst[:, 1] = src_g * src_a + dg * inv_ta
        dst[:, 2] = src_b * src_a + db * inv_ta
        dst[:, 3] = src_a + da * inv_ta



def composite_layer_numpy(dst, src, blend_mode='MIX', opacity=1.0):
    """Composite src (H,W,4) onto dst (H,W,4) in-place with blend mode and opacity.

    Steps:
        1. Scale src alpha by opacity.
        2. Compute blended RGB according to blend_mode.
        3. Alpha-composite blended result over dst using Porter-Duff Source Over.
    """
    sa = src[:, :, 3:4] * opacity  # effective source alpha
    da = dst[:, :, 3:4]
    sr = src[:, :, :3]
    dr = dst[:, :, :3]

    # Avoid div-by-zero for modes that divide
    eps = 1e-6

    # --- Compute blended RGB (before alpha composite) ---
    if blend_mode == 'MIX':
        blended = sr

    elif blend_mode == 'DARKEN':
        blended = np.minimum(dr, sr)

    elif blend_mode == 'MULTIPLY':
        blended = dr * sr

    elif blend_mode == 'COLOR_BURN':
        # 1 - (1 - dst) / src
        blended = 1.0 - np.minimum(1.0, (1.0 - dr) / np.maximum(sr, eps))

    elif blend_mode == 'LIGHTEN':
        blended = np.maximum(dr, sr)

    elif blend_mode == 'SCREEN':
        blended = 1.0 - (1.0 - dr) * (1.0 - sr)

    elif blend_mode == 'COLOR_DODGE':
        # dst / (1 - src)
        blended = np.minimum(1.0, dr / np.maximum(1.0 - sr, eps))

    elif blend_mode == 'ADD':
        blended = np.minimum(1.0, dr + sr)

    elif blend_mode == 'OVERLAY':
        # overlay(a,b) = 2ab  if a < 0.5  else 1 - 2(1-a)(1-b)
        lo = 2.0 * dr * sr
        hi = 1.0 - 2.0 * (1.0 - dr) * (1.0 - sr)
        blended = np.where(dr < 0.5, lo, hi)

    elif blend_mode == 'SOFT_LIGHT':
        # Pegtop formula: (1-2s)*d^2 + 2*s*d
        blended = (1.0 - 2.0 * sr) * dr * dr + 2.0 * sr * dr

    elif blend_mode == 'LINEAR_LIGHT':
        # 2*src + dst - 1, clamped
        blended = np.clip(2.0 * sr + dr - 1.0, 0.0, 1.0)

    elif blend_mode == 'DIFFERENCE':
        blended = np.abs(dr - sr)

    elif blend_mode == 'EXCLUSION':
        blended = dr + sr - 2.0 * dr * sr

    elif blend_mode == 'SUBTRACT':
        blended = np.maximum(0.0, dr - sr)

    elif blend_mode == 'DIVIDE':
        blended = np.minimum(1.0, dr / np.maximum(sr, eps))

    elif blend_mode in ('HUE', 'SATURATION', 'COLOR', 'VALUE'):
        blended = _hsl_blend(dr, sr, blend_mode)

    else:
        # Fallback to normal
        blended = sr

    # --- Porter-Duff Source Over with blended colour ---
    out_a = sa + da * (1.0 - sa)
    out_a_safe = np.where(out_a == 0, 1.0, out_a)

    dst[:, :, :3] = (blended * sa + dr * da * (1.0 - sa)) / out_a_safe
    dst[:, :, 3:4] = out_a


def _hsl_blend(dr, sr, mode):
    """HSL-family blend modes (HUE, SATURATION, COLOR, VALUE).

    Operates on (H,W,3) float32 arrays.  Converts to HSL, mixes the
    requested channel(s), converts back.
    """
    dh, ds, dl = _rgb_to_hsl(dr)
    sh, ss, sl = _rgb_to_hsl(sr)

    if mode == 'HUE':
        rh, rs, rl = sh, ds, dl
    elif mode == 'SATURATION':
        rh, rs, rl = dh, ss, dl
    elif mode == 'COLOR':
        rh, rs, rl = sh, ss, dl
    else:  # VALUE
        rh, rs, rl = dh, ds, sl

    return _hsl_to_rgb(rh, rs, rl)


def _rgb_to_hsl(rgb):
    """Convert (H,W,3) RGB [0-1] to separate H, S, L arrays."""
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    # Lightness
    l = (cmax + cmin) * 0.5

    # Saturation
    s = np.zeros_like(l)
    mask = delta > 1e-7
    low = l <= 0.5
    s[mask & low] = delta[mask & low] / np.maximum(cmax[mask & low] + cmin[mask & low], 1e-7)
    s[mask & ~low] = delta[mask & ~low] / np.maximum(2.0 - cmax[mask & ~low] - cmin[mask & ~low], 1e-7)

    # Hue
    h = np.zeros_like(l)
    delta_safe = np.where(delta == 0, 1.0, delta)
    mask_r = mask & (cmax == r)
    mask_g = mask & (cmax == g) & ~mask_r
    mask_b = mask & ~mask_r & ~mask_g
    h[mask_r] = ((g[mask_r] - b[mask_r]) / delta_safe[mask_r]) % 6.0
    h[mask_g] = (b[mask_g] - r[mask_g]) / delta_safe[mask_g] + 2.0
    h[mask_b] = (r[mask_b] - g[mask_b]) / delta_safe[mask_b] + 4.0
    h = h / 6.0  # normalise to 0-1

    return h, s, l


def _hsl_to_rgb(h, s, l):
    """Convert H, S, L arrays back to (H,W,3) RGB [0-1]."""
    c = (1.0 - np.abs(2.0 * l - 1.0)) * s
    h6 = h * 6.0
    x = c * (1.0 - np.abs(h6 % 2.0 - 1.0))
    m = l - c * 0.5

    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)

    idx = (h6 >= 0) & (h6 < 1)
    r[idx] = c[idx]; g[idx] = x[idx]
    idx = (h6 >= 1) & (h6 < 2)
    r[idx] = x[idx]; g[idx] = c[idx]
    idx = (h6 >= 2) & (h6 < 3)
    g[idx] = c[idx]; b[idx] = x[idx]
    idx = (h6 >= 3) & (h6 < 4)
    g[idx] = x[idx]; b[idx] = c[idx]
    idx = (h6 >= 4) & (h6 < 5)
    r[idx] = x[idx]; b[idx] = c[idx]
    idx = (h6 >= 5) & (h6 < 6)
    r[idx] = c[idx]; b[idx] = x[idx]

    return np.stack([r + m, g + m, b + m], axis=-1)


# ----------------------------
# Texture Refresh Helper
# ----------------------------
def force_texture_refresh(context, image):
    """Force the 3D viewport to refresh the texture after modification."""
    if not image:
        return

    image.update()
    
    # Tag all 3D viewports for redraw
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
