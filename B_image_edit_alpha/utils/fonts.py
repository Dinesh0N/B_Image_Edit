# Font management utilities for B_image_edit addon

import os
import math
import bpy
import gpu
import blf
from gpu_extras.batch import batch_for_shader
from mathutils import Vector
from gpu.types import GPUOffScreen


# ----------------------------
# Fonts
# ----------------------------

def get_custom_font_dirs():
    """Get custom font directories from addon preferences."""
    custom_dirs = []
    try:
        # Need to get the addon package name
        package_name = __package__.rsplit('.', 1)[0] if __package__ else "TextTex"
        if package_name in bpy.context.preferences.addons:
            prefs = bpy.context.preferences.addons[package_name].preferences
            if hasattr(prefs, 'custom_font_paths'):
                for item in prefs.custom_font_paths:
                    if item.path and os.path.exists(bpy.path.abspath(item.path)):
                        custom_dirs.append(bpy.path.abspath(item.path))
    except Exception:
        pass
    return custom_dirs

def load_custom_fonts_to_blender():
    """Scan custom directories and load fonts into bpy.data.fonts so they appear in the UI."""
    stats = {"loaded": 0, "existing": 0, "failed": 0}
    font_dirs = get_custom_font_dirs()
    
    existing_paths = {f.filepath for f in bpy.data.fonts}
    
    for font_dir in font_dirs:
        if os.path.exists(font_dir):
            for root, _, files in os.walk(font_dir):
                for f in files:
                    if f.lower().endswith((".ttf", ".otf")):
                        full_path = os.path.join(root, f)
                        if full_path in existing_paths:
                            stats["existing"] += 1
                            continue
                            
                        try:
                            bpy.data.fonts.load(full_path, check_existing=True)
                            stats["loaded"] += 1
                        except Exception as e:
                            print(f"Failed to load font {full_path}: {e}")
                            stats["failed"] += 1
    
    print(f"[TextTool] Custom fonts: {stats['loaded']} loaded, {stats['existing']} skipped, {stats['failed']} failed.")
    return stats


# ----------------------------
# Loaded Font Cache (blf font IDs)
# ----------------------------
_blf_font_cache = {}  # font_path -> font_id

def _get_blf_font_id(font_path):
    """Get or load a font using blf, returns font_id."""
    global _blf_font_cache
    
    if font_path in _blf_font_cache:
        return _blf_font_cache[font_path]
    
    font_id = 0  # Default font
    if font_path and os.path.exists(font_path):
        try:
            font_id = blf.load(font_path)
            if font_id == -1:
                font_id = 0
        except Exception as e:
            print(f"[TextTool] Failed to load font {font_path}: {e}")
            font_id = 0
    
    _blf_font_cache[font_path] = font_id
    return font_id


def reset_font_cache():
    """Clear the font cache so fonts are reloaded on next use."""
    global _blf_font_cache
    _blf_font_cache.clear()


# ----------------------------
# Native Blender Font Manager
# ----------------------------
class FontManager:
    @staticmethod
    def create_text_image(text, font_path, font_size, color, width=None, height=None, rotation_degrees=0.0, gradient_lut=None, outline_info=None, alignment='CENTER', line_spacing=1.2):
        """Render text to pixel buffer using Blender's native blf and GPUOffScreen.
        
        Args:
            text: Text string to render (can contain newlines for multi-line)
            font_path: Path to the font file
            font_size: Font size in pixels
            color: Base RGBA color tuple (used when gradient_lut is None)
            width: Optional canvas width
            height: Optional canvas height
            rotation_degrees: Rotation angle in degrees
            gradient_lut: Optional list of RGBA tuples (Look-Up Table) for gradient.
            outline_info: Optional dict with outline settings.
            alignment: Text alignment: 'LEFT', 'CENTER', 'RIGHT', or 'JUSTIFY'
            line_spacing: Line spacing multiplier (1.0 = normal, 1.5 = 150%)
        
        Returns: (pixels_list, (width, height)) or (None, None) on failure
        """
        if not text:
            return None, None
        
        try:
            font_id = _get_blf_font_id(font_path)
            
            # Set font size
            blf.size(font_id, font_size)
            
            # Split text into lines for multi-line support
            lines = text.split('\n')
            
            # Calculate dimensions for each line and find max width
            line_heights = []
            line_widths = []
            for line in lines:
                if line.strip():  # Non-empty line
                    w, h = blf.dimensions(font_id, line)
                else:  # Empty line - use height of a space character
                    _, h = blf.dimensions(font_id, " ")
                    w = 0
                line_widths.append(w)
                line_heights.append(h)
            
            # Use max width and sum of heights with line spacing
            single_line_height = max(line_heights) if line_heights else font_size
            text_width = max(line_widths) if line_widths else 0
            text_height = single_line_height * line_spacing * len(lines)
            
            # Store for later use in rendering
            _lines_data = {
                'lines': lines,
                'line_widths': line_widths,
                'single_line_height': single_line_height,
                'line_spacing': line_spacing,
                'alignment': alignment,
                'text_width': text_width
            }
            
            # Add padding - extra for outline if enabled
            outline_size = 0
            if outline_info and outline_info.get('enabled'):
                outline_size = outline_info.get('size', 2)
            
            padding = 10 + outline_size
            base_width = int(text_width + padding * 2)
            base_height = int(text_height + padding * 2)
            
            # For rotation, we need a larger canvas to fit rotated text
            if rotation_degrees != 0.0:
                angle_rad = math.radians(abs(rotation_degrees))
                # Calculate bounding box of rotated rectangle
                cos_a = abs(math.cos(angle_rad))
                sin_a = abs(math.sin(angle_rad))
                rotated_width = int(base_width * cos_a + base_height * sin_a) + padding * 2
                rotated_height = int(base_width * sin_a + base_height * cos_a) + padding * 2
                canvas_width = max(rotated_width, base_width)
                canvas_height = max(rotated_height, base_height)
            else:
                canvas_width = base_width
                canvas_height = base_height
            
            # Ensure minimum size
            canvas_width = max(2, canvas_width)
            canvas_height = max(2, canvas_height)
            
            # Create offscreen buffer
            offscreen = GPUOffScreen(canvas_width, canvas_height)
            
            text_pixels = []
            outline_pixels = []
            has_outline = outline_info and outline_info.get('enabled')
            
            with offscreen.bind():
                # Get framebuffer
                fb = gpu.state.active_framebuffer_get()
                
                # Setup 2D orthographic projection
                from mathutils import Matrix
                sx = 2.0 / canvas_width
                sy = 2.0 / canvas_height
                proj = Matrix((
                    (sx, 0, 0, -1),
                    (0, sy, 0, -1),
                    (0, 0, 1, 0),
                    (0, 0, 0, 1)
                ))
                
                gpu.matrix.push()
                gpu.matrix.push_projection()
                gpu.matrix.load_identity()
                gpu.matrix.load_projection_matrix(proj)
                gpu.state.blend_set('ALPHA')
                
                # Calculate geometry
                cx, cy = canvas_width / 2, canvas_height / 2
                if rotation_degrees != 0.0:
                    angle_rad = math.radians(rotation_degrees)
                    blf.enable(font_id, blf.ROTATION)
                    blf.rotation(font_id, angle_rad)
                    
                    cos_r = math.cos(angle_rad)
                    sin_r = math.sin(angle_rad)
                    offset_x = text_width / 2
                    offset_y = text_height / 2
                    rotated_offset_x = offset_x * cos_r - offset_y * sin_r
                    rotated_offset_y = offset_x * sin_r + offset_y * cos_r
                    base_x = cx - rotated_offset_x
                    base_y = cy - rotated_offset_y
                else:
                    base_x = (canvas_width - text_width) / 2
                    base_y = (canvas_height - text_height) / 2
                
                # --- PASS 1: Text Body ---
                fb.clear(color=(0.0, 0.0, 0.0, 0.0))
                
                # Set text color
                if gradient_lut:
                    blf.color(font_id, 1.0, 1.0, 1.0, 1.0)
                else:
                    r, g, b = color[0], color[1], color[2]
                    a = color[3] if len(color) > 3 else 1.0
                    blf.color(font_id, r, g, b, a)
                
                # Draw each line (multi-line support)
                line_height = _lines_data['single_line_height'] * _lines_data['line_spacing']
                total_text_width = _lines_data['text_width']
                align = _lines_data['alignment']
                
                for i, line in enumerate(_lines_data['lines']):
                    if not line.strip():
                        continue  # Skip empty lines (but they still take vertical space)
                    
                    # Calculate Y position for this line (top line first)
                    line_y = base_y + text_height - (i + 1) * line_height + line_height * 0.2
                    
                    # Calculate X position based on alignment
                    line_w = _lines_data['line_widths'][i]
                    if align == 'LEFT':
                        line_x = base_x
                    elif align == 'RIGHT':
                        line_x = base_x + total_text_width - line_w
                    elif align == 'JUSTIFY' and len(_lines_data['lines']) > 1 and i < len(_lines_data['lines']) - 1:
                        # Justify: stretch to fill width (except last line)
                        line_x = base_x  # Start from left, word spacing handled by blf
                    else:  # CENTER (default)
                        line_x = base_x + (total_text_width - line_w) / 2
                    
                    blf.position(font_id, line_x, line_y, 0)
                    blf.draw(font_id, line)
                
                buffer_text = fb.read_color(0, 0, canvas_width, canvas_height, 4, 0, 'FLOAT')
                # Convert to flat list
                for row in buffer_text:
                    for pixel in row:
                        text_pixels.extend(pixel)
                
                # --- PASS 2: Outline (only if enabled) ---
                if has_outline:
                    fb.clear(color=(0.0, 0.0, 0.0, 0.0))
                    
                    outline_color = outline_info.get('color', (0, 0, 0, 1))
                    outline_sz = outline_info.get('size', 2)
                    
                    or_, og, ob = outline_color[0], outline_color[1], outline_color[2]
                    oa = outline_color[3] if len(outline_color) > 3 else 1.0
                    blf.color(font_id, or_, og, ob, oa)
                    
                    # Draw outline for each line (multi-line support)
                    line_height = _lines_data['single_line_height'] * _lines_data['line_spacing']
                    total_text_width = _lines_data['text_width']
                    align = _lines_data['alignment']
                    
                    for i, line in enumerate(_lines_data['lines']):
                        if not line.strip():
                            continue
                        
                        line_y = base_y + text_height - (i + 1) * line_height + line_height * 0.2
                        line_w = _lines_data['line_widths'][i]
                        
                        # Calculate X position based on alignment
                        if align == 'LEFT':
                            line_x = base_x
                        elif align == 'RIGHT':
                            line_x = base_x + total_text_width - line_w
                        elif align == 'JUSTIFY' and len(_lines_data['lines']) > 1 and i < len(_lines_data['lines']) - 1:
                            line_x = base_x
                        else:  # CENTER (default)
                            line_x = base_x + (total_text_width - line_w) / 2
                        
                        # Draw outline circular pattern
                        for angle in range(0, 360, 30):
                            rad = math.radians(angle)
                            ox = math.cos(rad) * outline_sz
                            oy = math.sin(rad) * outline_sz
                            blf.position(font_id, line_x + ox, line_y + oy, 0)
                            blf.draw(font_id, line)
                        
                        # Cardinal directions
                        for ox, oy in [(outline_sz, 0), (-outline_sz, 0), (0, outline_sz), (0, -outline_sz)]:
                            blf.position(font_id, line_x + ox, line_y + oy, 0)
                            blf.draw(font_id, line)
                        
                    buffer_outline = fb.read_color(0, 0, canvas_width, canvas_height, 4, 0, 'FLOAT')
                    for row in buffer_outline:
                        for pixel in row:
                            outline_pixels.extend(pixel)

                # Cleanup
                if rotation_degrees != 0.0:
                    blf.disable(font_id, blf.ROTATION)
                
                gpu.state.blend_set('NONE')
                gpu.matrix.pop_projection()
                gpu.matrix.pop()
            
            offscreen.free()
            
            # --- Post-Processing ---
            
            # 1. Apply Gradient to Text Body
            if gradient_lut:
                text_pixels = FontManager._apply_gradient(
                    text_pixels, canvas_width, canvas_height, gradient_lut
                )
            
            # 2. Composite (Text over Outline)
            if has_outline and outline_pixels:
                final_pixels = FontManager._composite_layers(text_pixels, outline_pixels)
            else:
                final_pixels = text_pixels
            
            return final_pixels, (canvas_width, canvas_height)
            
        except Exception as e:
            print(f"[TextTool] Render error: {e}")
            import traceback
            traceback.print_exc()
            return None, None
            
    @staticmethod
    def _composite_layers(fg_pixels, bg_pixels):
        """Composite foreground (text) over background (outline)."""
        # Both are flat lists of RGBA floats
        count = len(fg_pixels)
        if len(bg_pixels) != count:
            return fg_pixels 
            
        result = [0.0] * count
        
        for i in range(0, count, 4):
            fr, fg, fb, fa = fg_pixels[i], fg_pixels[i+1], fg_pixels[i+2], fg_pixels[i+3]
            br, bg, bb, ba = bg_pixels[i], bg_pixels[i+1], bg_pixels[i+2], bg_pixels[i+3]
            
            inv_fa = 1.0 - fa
            
            # Standard "Over":
            out_a = fa + ba * inv_fa
            
            # Avoid divide by zero
            if out_a > 0:
                result[i]   = (fr * fa + br * ba * inv_fa) / out_a
                result[i+1] = (fg * fa + bg * ba * inv_fa) / out_a
                result[i+2] = (fb * fa + bb * ba * inv_fa) / out_a
                result[i+3] = out_a
            else:
                result[i] = 0.0
                result[i+1] = 0.0
                result[i+2] = 0.0
                result[i+3] = 0.0
                
        return result

    
    @staticmethod
    def _apply_gradient(pixels, width, height, gradient_data):
        """Apply gradient colors to rendered text pixels.
        
        Args:
            pixels: Flat list of RGBA pixel values
            width: Image width
            height: Image height
            gradient_data: Dict with {'type': str, 'lut': list of RGBA, 'angle': float, 'font_rotation': float}
        """
        gradient_type = gradient_data.get('type', 'LINEAR')
        lut = gradient_data.get('lut', [])
        gradient_angle = gradient_data.get('angle', 0.0)
        font_rotation = gradient_data.get('font_rotation', 0.0)
        lut_len = len(lut)
        
        if lut_len < 2:
            return pixels 
        
        # --- Step 1: Find actual text bounding box (non-transparent pixels) ---
        min_x, max_x = width, 0
        min_y, max_y = height, 0
        
        for y in range(height):
            for x in range(width):
                idx = (y * width + x) * 4
                if pixels[idx + 3] > 0:  # Has alpha
                    if x < min_x: min_x = x
                    if x > max_x: max_x = x
                    if y < min_y: min_y = y
                    if y > max_y: max_y = y
        
        # If no text pixels found, return unchanged
        if max_x < min_x or max_y < min_y:
            return pixels
            
        # Text bounding box center (in canvas coordinates)
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2
        
        # --- Step 2: Precompute rotation constants ---
        font_rad = math.radians(-font_rotation)
        font_cos = math.cos(font_rad)
        font_sin = math.sin(font_rad)
        
        # Now we need to find the text bounds in the UNROTATED space
        local_min_x, local_max_x = float('inf'), float('-inf')
        local_min_y, local_max_y = float('inf'), float('-inf')
        
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                idx = (y * width + x) * 4
                if pixels[idx + 3] > 0:
                    # Transform to local (unrotated) coordinates
                    dx = x - cx
                    dy = y - cy
                    local_x = dx * font_cos - dy * font_sin
                    local_y = dx * font_sin + dy * font_cos
                    
                    if local_x < local_min_x: local_min_x = local_x
                    if local_x > local_max_x: local_max_x = local_x
                    if local_y < local_min_y: local_min_y = local_y
                    if local_y > local_max_y: local_max_y = local_y
        
        # Local text dimensions
        local_width = local_max_x - local_min_x if local_max_x > local_min_x else 1.0
        local_height = local_max_y - local_min_y if local_max_y > local_min_y else 1.0
        local_cx = (local_min_x + local_max_x) / 2
        local_cy = (local_min_y + local_max_y) / 2
        
        # --- Step 3: Calculate gradient parameters in LOCAL space ---
        grad_rad = math.radians(gradient_angle)
        grad_cos = math.cos(grad_rad)
        grad_sin = math.sin(grad_rad)
        
        # For linear gradient, project local corners onto the gradient axis
        hw = local_width / 2
        hh = local_height / 2
        corners = [
            (-hw * grad_cos - -hh * grad_sin),
            ( hw * grad_cos - -hh * grad_sin),
            ( hw * grad_cos -  hh * grad_sin),
            (-hw * grad_cos -  hh * grad_sin),
        ]
        min_p = min(corners)
        max_p = max(corners)
        span = max_p - min_p if (max_p - min_p) > 0.001 else 1.0

        # Radial max dist
        max_dist = math.sqrt(hw * hw + hh * hh) if (hw > 0 and hh > 0) else 1.0

        # --- Step 4: Apply gradient ---
        result = list(pixels)
        
        for y in range(height):
            for x in range(width):
                idx = (y * width + x) * 4
                
                alpha = result[idx + 3]
                if alpha <= 0:
                    continue
                
                # Transform pixel to LOCAL coordinates (undo font rotation)
                dx = x - cx
                dy = y - cy
                local_x = dx * font_cos - dy * font_sin
                local_y = dx * font_sin + dy * font_cos
                
                # Offset from local center
                lx = local_x - local_cx
                ly = local_y - local_cy
                
                # Calculate gradient factor
                if gradient_type == 'LINEAR':
                    rot_x = lx * grad_cos + ly * grad_sin
                    t = (rot_x - min_p) / span
                else:  # RADIAL
                    dist = math.sqrt(lx * lx + ly * ly)
                    t = dist / max_dist
                
                # Clamp t
                t = max(0.0, min(1.0, t))
                
                # Sample LUT
                lut_index = int(t * (lut_len - 1))
                color = lut[lut_index]
                
                # Apply gradient color while preserving luminance and alpha from original
                orig_lum = result[idx]  # Original was rendered white
                
                result[idx] = color[0] * orig_lum
                result[idx + 1] = color[1] * orig_lum
                result[idx + 2] = color[2] * orig_lum
                # Alpha stays the same
        
        return result
