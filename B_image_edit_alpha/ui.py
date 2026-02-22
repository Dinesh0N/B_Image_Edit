import bpy
import os
import gpu
import blf
import math
from gpu_extras.batch import batch_for_shader
from bpy.types import Panel, WorkSpaceTool
from . import utils

# ----------------------------
# Text Preview Drawing
# ----------------------------

# Shader for drawing textured quads with alpha
_preview_shader = None

def _get_preview_shader():
    """Get or create the shader for drawing text preview with alpha."""
    global _preview_shader
    if _preview_shader is None:
        # Try available shaders in order of preference (Blender 5.0 names)
        shader_names = ['IMAGE', 'IMAGE_SCENE_LINEAR_TO_REC709_SRGB']
        for shader_name in shader_names:
            try:
                _preview_shader = gpu.shader.from_builtin(shader_name)
                break
            except ValueError:
                continue
    return _preview_shader


def _draw_text_preview_direct(x, y, for_image_editor=False):
    """Draw the text preview at cursor position using direct blf rendering."""
    context = bpy.context
    if not hasattr(context.scene, "text_tool_properties"):
        return False
    
    props = context.scene.text_tool_properties
    
    # Get text content from text block or simple property
    if props.use_text_block and props.text_block:
        text = props.text_block.as_string()
    else:
        text = props.text
    
    if not text:
        return False
    
    # Get font ID
    font_path = props.font_file.filepath if props.font_file else None
    font_id = utils._get_blf_font_id(font_path)
    
    # Set font size - for image editor, scale by zoom level
    font_size = props.font_size
    if for_image_editor:
        try:
            sima = context.space_data
            if sima.type == 'IMAGE_EDITOR' and sima.image:
                i_width, i_height = sima.image.size
                if i_width > 0:
                    region = context.region
                    cx = region.width / 2
                    cy = region.height / 2
                    v0 = region.view2d.region_to_view(cx, cy)
                    v1 = region.view2d.region_to_view(cx + 100, cy)
                    
                    dist_view_x = v1[0] - v0[0]
                    if abs(dist_view_x) > 0.000001:
                        img_subset_pixels = dist_view_x * i_width
                        scale = 100.0 / img_subset_pixels
                        font_size = int(props.font_size * scale)
        except Exception:
            pass
    
    # Clamp font size to reasonable bounds
    font_size = max(8, min(font_size, 500))
    
    blf.size(font_id, font_size)
    
    # Split text into lines for multi-line support
    lines = text.split('\n')
    
    # Calculate dimensions for multi-line text
    line_widths = []
    line_heights = []
    for line in lines:
        if line.strip():
            w, h = blf.dimensions(font_id, line)
        else:
            _, h = blf.dimensions(font_id, " ")
            w = 0
        line_widths.append(w)
        line_heights.append(h)
    
    line_spacing = props.line_spacing
    single_line_height = max(line_heights) if line_heights else font_size
    text_width = max(line_widths) if line_widths else 0
    text_height = single_line_height * line_spacing * len(lines)
    alignment = props.text_alignment
    
    # Calculate base position based on anchor settings
    # Horizontal anchor
    if props.anchor_horizontal == 'LEFT':
        draw_x = x
    elif props.anchor_horizontal == 'RIGHT':
        draw_x = x - text_width
    else:  # CENTER
        draw_x = x - text_width / 2
    
    # Vertical anchor
    if props.anchor_vertical == 'TOP':
        draw_y = y - text_height
    elif props.anchor_vertical == 'BOTTOM':
        draw_y = y
    else:  # CENTER
        draw_y = y - text_height / 2
    
    # Set text color
    if props.use_gradient:
        r, g, b, a = 1.0, 1.0, 1.0, 1.0
    else:
        r, g, b = props.color[0], props.color[1], props.color[2]
        a = props.color[3] if len(props.color) > 3 else 1.0
    blf.color(font_id, r, g, b, a)
    
    # Apply rotation if needed
    if props.rotation != 0.0:
        blf.enable(font_id, blf.ROTATION)
        blf.rotation(font_id, props.rotation)
    
    # Draw each line
    line_height = single_line_height * line_spacing
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        
        # Calculate Y position for this line (top line first)
        line_y = draw_y + text_height - (i + 1) * line_height + line_height * 0.2
        
        # Calculate X position based on alignment
        line_w = line_widths[i]
        if alignment == 'LEFT':
            line_x = draw_x
        elif alignment == 'RIGHT':
            line_x = draw_x + text_width - line_w
        elif alignment == 'JUSTIFY' and len(lines) > 1 and i < len(lines) - 1:
            line_x = draw_x
        else:  # CENTER (default)
            line_x = draw_x + (text_width - line_w) / 2
        
        if props.rotation != 0.0 and props.anchor_horizontal == 'CENTER' and props.anchor_vertical == 'CENTER':
            cos_r = math.cos(props.rotation)
            sin_r = math.sin(props.rotation)
            offset_x = line_x - x
            offset_y = line_y - y
            rotated_x = x + offset_x * cos_r - offset_y * sin_r
            rotated_y = y + offset_x * sin_r + offset_y * cos_r
            blf.position(font_id, rotated_x, rotated_y, 0)
        else:
            blf.position(font_id, line_x, line_y, 0)
        
        blf.draw(font_id, line)
    
    # Disable rotation after drawing
    if props.rotation != 0.0:
        blf.disable(font_id, blf.ROTATION)
    
    return True


def _draw_text_preview(x, y, for_image_editor=False):
    """Draw the text preview at cursor position."""
    # Use direct blf rendering for preview (more reliable than texture-based)
    return _draw_text_preview_direct(x, y, for_image_editor)


def _draw_crosshair(x, y, color, size=15):
    """Draw a simple crosshair at the given position."""
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    
    coords = [(x - size, y), (x + size, y)]
    batch = batch_for_shader(shader, 'LINES', {"pos": coords})
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)

    coords = [(x, y - size), (x, y + size)]
    batch = batch_for_shader(shader, 'LINES', {"pos": coords})
    batch.draw(shader)


def draw_cursor_callback_3d():
    """Draw text preview for 3D Viewport texture paint mode."""
    if not utils.show_cursor or not utils.cursor_pos:
        return
    
    x, y = utils.cursor_pos
    
    # Draw crosshair first (behind the preview)
    _draw_crosshair(x, y, (1.0, 1.0, 1.0, 0.5), size=20)
    
    # Draw text preview
    preview_drawn = _draw_text_preview(x, y, for_image_editor=False)
    
    # If preview failed, show text label as fallback
    if not preview_drawn:
        context = bpy.context
        if hasattr(context.scene, "text_tool_properties"):
            props = context.scene.text_tool_properties
            if props.text:
                font_id = 0
                blf.position(font_id, x + 25, y + 10, 0)
                blf.size(font_id, 14)
                blf.color(font_id, 1.0, 1.0, 0.0, 0.8)
                blf.draw(font_id, f"'{props.text}'")


def draw_cursor_callback_image():
    """Draw text preview for Image Editor paint mode."""
    if not utils.show_cursor or not utils.cursor_pos:
        return
    
    x, y = utils.cursor_pos
    
    # Draw crosshair first (behind the preview)
    _draw_crosshair(x, y, (1.0, 0.2, 0.2, 0.7), size=15)
    
    # Draw text preview
    preview_drawn = _draw_text_preview(x, y, for_image_editor=True)
    
    # If preview failed, show text label as fallback
    if not preview_drawn:
        context = bpy.context
        if hasattr(context.scene, "text_tool_properties"):
            props = context.scene.text_tool_properties
            if props.text:
                font_id = 0
                blf.position(font_id, x + 20, y + 8, 0)
                blf.size(font_id, 12)
                blf.color(font_id, 1.0, 0.5, 0.0, 0.9)
                blf.draw(font_id, f"'{props.text}'")


# ----------------------------
# Gradient Preview Drawing
# ----------------------------
def draw_gradient_preview_3d():
    """Draw gradient line preview for 3D Viewport."""
    if utils.gradient_preview_start is None or utils.gradient_preview_end is None:
        return
    
    x1, y1 = utils.gradient_preview_start
    x2, y2 = utils.gradient_preview_end
    
    # Draw line
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    coords = [(x1, y1), (x2, y2)]
    batch = batch_for_shader(shader, 'LINES', {"pos": coords})
    
    gpu.state.blend_set('ALPHA')
    gpu.state.line_width_set(2.0)
    shader.bind()
    shader.uniform_float("color", (1.0, 1.0, 0.0, 0.8))
    batch.draw(shader)
    
    # Draw start circle
    _draw_circle(x1, y1, 8, (0.0, 1.0, 0.0, 0.8))
    # Draw end circle
    _draw_circle(x2, y2, 8, (1.0, 0.0, 0.0, 0.8))
    
    gpu.state.line_width_set(1.0)
    gpu.state.blend_set('NONE')


def draw_gradient_preview_image():
    """Draw gradient line preview for Image Editor."""
    if utils.gradient_preview_start is None or utils.gradient_preview_end is None:
        return
    
    x1, y1 = utils.gradient_preview_start
    x2, y2 = utils.gradient_preview_end
    
    # Draw line
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    coords = [(x1, y1), (x2, y2)]
    batch = batch_for_shader(shader, 'LINES', {"pos": coords})
    
    gpu.state.blend_set('ALPHA')
    gpu.state.line_width_set(2.0)
    shader.bind()
    shader.uniform_float("color", (1.0, 0.5, 0.0, 0.9))
    batch.draw(shader)
    
    # Draw start/end circles
    _draw_circle(x1, y1, 6, (0.0, 1.0, 0.0, 0.9))
    _draw_circle(x2, y2, 6, (1.0, 0.0, 0.0, 0.9))
    
    gpu.state.line_width_set(1.0)
    gpu.state.blend_set('NONE')


def _draw_circle(x, y, radius, color, segments=16):
    """Draw a circle at the given position."""
    import math
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    
    coords = []
    for i in range(segments):
        angle = 2.0 * math.pi * i / segments
        coords.append((x + radius * math.cos(angle), y + radius * math.sin(angle)))
    coords.append(coords[0])  # Close the circle
    
    batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": coords})
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)


# ----------------------------
# Crop Preview Drawing
# ----------------------------
def draw_crop_preview_image():
    """Draw crop selection rectangle for Image Editor with rule of thirds."""
    if utils.crop_preview_start is None or utils.crop_preview_end is None:
        return
    
    x1, y1 = utils.crop_preview_start
    x2, y2 = utils.crop_preview_end
    
    # Normalize coordinates
    min_x, max_x = min(x1, x2), max(x1, x2)
    min_y, max_y = min(y1, y2), max(y1, y2)
    
    # Draw rectangle outline
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    
    # Rectangle corners
    coords = [
        (min_x, min_y), (max_x, min_y),  # Bottom
        (max_x, min_y), (max_x, max_y),  # Right
        (max_x, max_y), (min_x, max_y),  # Top
        (min_x, max_y), (min_x, min_y),  # Left
    ]
    
    batch = batch_for_shader(shader, 'LINES', {"pos": coords})
    
    gpu.state.blend_set('ALPHA')
    gpu.state.line_width_set(2.0)
    shader.bind()
    shader.uniform_float("color", (1.0, 1.0, 1.0, 0.9))
    batch.draw(shader)
    
    # Draw rule of thirds grid if enabled
    context = bpy.context
    if hasattr(context.scene, "text_tool_properties"):
        props = context.scene.text_tool_properties
        if props.crop_show_thirds:
            width = max_x - min_x
            height = max_y - min_y
            
            if width > 10 and height > 10:
                thirds_coords = []
                # Vertical lines (1/3 and 2/3)
                thirds_coords.extend([(min_x + width / 3, min_y), (min_x + width / 3, max_y)])
                thirds_coords.extend([(min_x + 2 * width / 3, min_y), (min_x + 2 * width / 3, max_y)])
                # Horizontal lines (1/3 and 2/3)
                thirds_coords.extend([(min_x, min_y + height / 3), (max_x, min_y + height / 3)])
                thirds_coords.extend([(min_x, min_y + 2 * height / 3), (max_x, min_y + 2 * height / 3)])
                
                gpu.state.line_width_set(1.0)
                thirds_batch = batch_for_shader(shader, 'LINES', {"pos": thirds_coords})
                shader.uniform_float("color", (1.0, 1.0, 1.0, 0.4))
                thirds_batch.draw(shader)
    
    gpu.state.line_width_set(2.0)
    
    # Draw corner handles
    handle_size = 6
    for cx, cy in [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]:
        handle_coords = [
            (cx - handle_size, cy - handle_size),
            (cx + handle_size, cy - handle_size),
            (cx + handle_size, cy + handle_size),
            (cx - handle_size, cy + handle_size),
        ]
        handle_batch = batch_for_shader(shader, 'TRI_FAN', {"pos": handle_coords})
        shader.uniform_float("color", (0.2, 0.6, 1.0, 0.9))
        handle_batch.draw(shader)
    
    # Draw edge handles (small rectangles at edge midpoints)
    edge_handle_size = 4
    edge_positions = [
        ((min_x + max_x) / 2, min_y),  # Bottom
        ((min_x + max_x) / 2, max_y),  # Top
        (min_x, (min_y + max_y) / 2),  # Left
        (max_x, (min_y + max_y) / 2),  # Right
    ]
    for ex, ey in edge_positions:
        edge_coords = [
            (ex - edge_handle_size, ey - edge_handle_size),
            (ex + edge_handle_size, ey - edge_handle_size),
            (ex + edge_handle_size, ey + edge_handle_size),
            (ex - edge_handle_size, ey + edge_handle_size),
        ]
        edge_batch = batch_for_shader(shader, 'TRI_FAN', {"pos": edge_coords})
        shader.uniform_float("color", (0.2, 0.8, 0.4, 0.9))
        edge_batch.draw(shader)
    
    gpu.state.line_width_set(1.0)
    gpu.state.blend_set('NONE')


# ----------------------------
# Clone Preview Drawing
# ----------------------------
def draw_clone_preview_image():
    """Draw clone tool brush cursor and source crosshair."""
    context = bpy.context
    if not hasattr(context.scene, "text_tool_properties"):
        return
    
    props = context.scene.text_tool_properties
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    
    gpu.state.blend_set('ALPHA')
    gpu.state.line_width_set(2.0)
    shader.bind()
    
    # Draw source crosshair if set
    if utils.clone_source_set and utils.clone_source_pos:
        sx, sy = utils.clone_source_pos
        size = 15
        
        # Crosshair lines
        cross_coords = [
            (sx - size, sy), (sx + size, sy),
            (sx, sy - size), (sx, sy + size),
        ]
        cross_batch = batch_for_shader(shader, 'LINES', {"pos": cross_coords})
        shader.uniform_float("color", (1.0, 0.3, 0.3, 0.9))
        cross_batch.draw(shader)
        
        # Small circle around crosshair
        _draw_circle(sx, sy, 8, (1.0, 0.3, 0.3, 0.7), segments=16)
    
    # Draw brush cursor at current position
    if utils.clone_cursor_pos:
        cx, cy = utils.clone_cursor_pos
        brush_size = props.clone_brush_size
        
        # Scale brush size based on image zoom
        try:
            sima = context.space_data
            if sima.type == 'IMAGE_EDITOR' and sima.image:
                i_width, i_height = sima.image.size
                if i_width > 0:
                    region = context.region
                    center_x = region.width / 2
                    v0 = region.view2d.region_to_view(center_x, 0)
                    v1 = region.view2d.region_to_view(center_x + 100, 0)
                    dist_view_x = v1[0] - v0[0]
                    if abs(dist_view_x) > 0.000001:
                        img_subset_pixels = dist_view_x * i_width
                        scale = 100.0 / img_subset_pixels
                        brush_size = int(props.clone_brush_size * scale)
        except Exception:
            pass
        
        # Draw brush outline
        _draw_circle(cx, cy, brush_size, (0.2, 0.8, 1.0, 0.8), segments=32)
        
        # Draw inner circle for hardness indication
        inner_radius = brush_size * props.clone_brush_hardness
        if inner_radius > 2:
            _draw_circle(cx, cy, inner_radius, (0.2, 0.8, 1.0, 0.4), segments=32)
    
    gpu.state.line_width_set(1.0)
    gpu.state.blend_set('NONE')


def draw_pen_preview(context):
    """Draw pen tool path preview with real-time fill/stroke (Atelier-Paint style)."""
    from .operators import pen_ops  # Import module directly to get live variable references
    from mathutils.geometry import delaunay_2d_cdt
    from mathutils import Vector
    
    if not pen_ops.pen_points and not pen_ops.pen_preview_pos:
        return
    
    gpu.state.blend_set('ALPHA')
    
    region = context.region
    view2d = region.view2d
    sima = context.space_data
    
    if not sima or not sima.image:
        return
    
    img_width, img_height = sima.image.size
    props = context.scene.text_tool_properties
    
    # Helper to convert image coords to screen coords
    def img_to_screen(ix, iy):
        uv_x = ix / img_width
        uv_y = iy / img_height
        return view2d.view_to_region(uv_x, uv_y, clip=False)
    
    # Generate bezier curve points for preview
    preview_points = []  # Screen coords
    preview_points_img = []  # Image coords for fill
    
    if len(pen_ops.pen_points) >= 2:
        for i in range(len(pen_ops.pen_points) - 1):
            p0 = pen_ops.pen_points[i]
            p1 = pen_ops.pen_points[i + 1]
            
            x0, y0 = p0[0], p0[1]
            x1, y1 = p0[4], p0[5]
            x2, y2 = p1[2], p1[3]
            x3, y3 = p1[0], p1[1]
            
            # Generate bezier curve samples
            for t in range(21):
                t_val = t / 20
                mt = 1 - t_val
                x = mt**3 * x0 + 3 * mt**2 * t_val * x1 + 3 * mt * t_val**2 * x2 + t_val**3 * x3
                y = mt**3 * y0 + 3 * mt**2 * t_val * y1 + 3 * mt * t_val**2 * y2 + t_val**3 * y3
                preview_points_img.append((x, y))
                sx, sy = img_to_screen(x, y)
                preview_points.append((sx, sy))
    
    # Draw filled polygon preview (if enabled and enough points)
    # Using TRI_FAN from centroid for simpler, more reliable preview
    if props.pen_use_fill and len(preview_points) >= 3 and pen_ops.pen_realtime_preview:
        try:
            # Calculate centroid
            cx = sum(p[0] for p in preview_points) / len(preview_points)
            cy = sum(p[1] for p in preview_points) / len(preview_points)
            
            # Create triangle fan from centroid
            fan_verts = [(cx, cy)]  # Center point first
            fan_verts.extend(preview_points)
            fan_verts.append(preview_points[0])  # Close the fan
            
            shader = gpu.shader.from_builtin('UNIFORM_COLOR')
            fill_color = (*props.pen_fill_color[:3], props.pen_fill_color[3] * 0.4)  # Semi-transparent preview
            shader.uniform_float("color", fill_color)
            batch = batch_for_shader(shader, 'TRI_FAN', {"pos": fan_verts})
            batch.draw(shader)
        except Exception:
            pass  # Fallback silently if fill fails
    
    # Draw stroke preview (curve outline)
    if len(preview_points) >= 2:
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        
        # Calculate stroke width - simpler approach
        stroke_width_preview = 2.0  # Default thin preview line
        if props.pen_use_stroke and pen_ops.pen_realtime_preview:
            stroke_color = (*props.pen_stroke_color[:3], 0.9)
            # Limit preview stroke width to reasonable range (1-8 pixels on screen)
            stroke_width_preview = max(1.0, min(props.pen_stroke_width * 0.5, 8.0))
        else:
            stroke_color = (1.0, 1.0, 1.0, 0.8)
        
        gpu.state.line_width_set(stroke_width_preview)
        shader.uniform_float("color", stroke_color)
        batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": preview_points})
        batch.draw(shader)
    
    # Reset line width for handles (they should be thin regardless of stroke width)
    gpu.state.line_width_set(1.0)
    # Draw anchor points and handles
    for i, pt in enumerate(pen_ops.pen_points):
        ax, ay = pt[0], pt[1]
        hi_x, hi_y = pt[2], pt[3]
        ho_x, ho_y = pt[4], pt[5]
        
        # Screen positions
        anchor_screen = img_to_screen(ax, ay)
        hi_screen = img_to_screen(hi_x, hi_y)
        ho_screen = img_to_screen(ho_x, ho_y)
        
        # Draw handle lines
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        shader.uniform_float("color", (0.5, 0.5, 1.0, 0.6))
        
        if (hi_x != ax or hi_y != ay):
            batch = batch_for_shader(shader, 'LINES', {"pos": [anchor_screen, hi_screen]})
            batch.draw(shader)
        if (ho_x != ax or ho_y != ay):
            batch = batch_for_shader(shader, 'LINES', {"pos": [anchor_screen, ho_screen]})
            batch.draw(shader)
        
        # Draw anchor point (square) - green for first, white for others
        size = 6 if i == 0 else 4
        color = (0.0, 1.0, 0.0, 1.0) if i == 0 else (1.0, 1.0, 1.0, 1.0)
        _draw_square(anchor_screen[0], anchor_screen[1], size, color)
        
        # Draw handle points (circles)
        if (hi_x != ax or hi_y != ay):
            _draw_circle(hi_screen[0], hi_screen[1], 3, (0.5, 0.5, 1.0, 1.0), 8)
        if (ho_x != ax or ho_y != ay):
            _draw_circle(ho_screen[0], ho_screen[1], 3, (0.5, 0.5, 1.0, 1.0), 8)
    
    # Draw preview line to cursor (dashed effect)
    if pen_ops.pen_preview_pos and len(pen_ops.pen_points) > 0:
        last_pt = pen_ops.pen_points[-1]
        last_screen = img_to_screen(last_pt[0], last_pt[1])
        cursor_screen = img_to_screen(pen_ops.pen_preview_pos[0], pen_ops.pen_preview_pos[1])
        
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        shader.uniform_float("color", (1.0, 1.0, 1.0, 0.4))
        batch = batch_for_shader(shader, 'LINES', {"pos": [last_screen, cursor_screen]})
        batch.draw(shader)
    
    # Show displacement indicator
    if pen_ops.pen_displacing:
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        shader.uniform_float("color", (1.0, 0.5, 0.0, 0.8))
        # Draw move icon near first point
        if len(pen_ops.pen_points) > 0:
            first_screen = img_to_screen(pen_ops.pen_points[0][0], pen_ops.pen_points[0][1])
            _draw_move_icon(first_screen[0] - 20, first_screen[1] + 20)
    
    gpu.state.line_width_set(1.0)
    gpu.state.blend_set('NONE')


def _draw_move_icon(x, y, size=8):
    """Draw a small move/arrows icon to indicate displacement mode."""
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    shader.uniform_float("color", (1.0, 0.5, 0.0, 0.9))
    
    # Four arrows pointing outward
    coords = [
        (x, y - size), (x, y + size),  # Vertical
        (x - size, y), (x + size, y),  # Horizontal
    ]
    batch = batch_for_shader(shader, 'LINES', {"pos": coords})
    batch.draw(shader)


def _draw_square(x, y, size, color):
    """Draw a small square."""
    half = size / 2
    vertices = [
        (x - half, y - half),
        (x + half, y - half),
        (x + half, y + half),
        (x - half, y + half),
    ]
    indices = [(0, 1, 2), (0, 2, 3)]
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    shader.uniform_float("color", color)
    batch = batch_for_shader(shader, 'TRIS', {"pos": vertices}, indices=indices)
    batch.draw(shader)




# ----------------------------
# WorkSpace Tools
# ----------------------------


class ImagePenTool(WorkSpaceTool):
    bl_space_type = 'IMAGE_EDITOR'
    bl_context_mode = 'PAINT'
    bl_idname = "image_paint.pen_tool"
    bl_label = "Pen Tool"
    bl_description = "Draw bezier paths with stroke and fill"
    bl_icon = os.path.join(os.path.dirname(__file__), "icons", "ops.paint.pen_tool")
    bl_keymap = (
        ("image_paint.pen_tool", {"type": 'LEFTMOUSE', "value": 'PRESS'}, None),
    )

    def draw_settings(context, layout, tool):
        props = context.scene.text_tool_properties
        is_header = not layout.use_property_split
        
        if is_header:
            layout.prop(props, "pen_use_stroke", text="", icon='STROKE')
            if props.pen_use_stroke:
                layout.prop(props, "pen_stroke_color", text="")
                layout.prop(props, "pen_stroke_width", text="Radius")
            layout.prop(props, "pen_use_fill", text="", icon='SNAP_FACE')
            if props.pen_use_fill:
                layout.prop(props, "pen_fill_color", text="")
            layout.popover("IMAGE_EDIT_PT_pen_options_popover", text="Options")
        else:
            layout.use_property_split = True
            layout.use_property_decorate = False
            col = layout.column()
            
            # Stroke settings
            col.prop(props, "pen_use_stroke")
            if props.pen_use_stroke:
                box = col.box()
                box.prop(props, "pen_stroke_color", text="Color")
                box.prop(props, "pen_stroke_width", text="Width")
            
            # Fill settings
            col.separator()
            col.prop(props, "pen_use_fill")
            if props.pen_use_fill:
                box = col.box()
                box.prop(props, "pen_fill_color", text="Color")
            
            # Blend Mode
            col.separator()
            if context.tool_settings.image_paint.brush:
                col.prop(context.tool_settings.image_paint.brush, "blend", text="Blend Mode")
            
            # Anti-aliasing
            col.prop(props, "use_antialiasing")
            
            col.separator()
            col.label(text="Click to add points, drag for curves", icon='INFO')
            col.label(text="Enter/Space to apply, ESC to cancel")


class ImageCloneTool(WorkSpaceTool):
    bl_space_type = 'IMAGE_EDITOR'
    bl_context_mode = 'PAINT'
    bl_idname = "image_paint.clone_tool"
    bl_label = "Clone Tool"
    bl_description = "Clone pixels from one area to another (Ctrl+Click to set source)"
    bl_icon = os.path.join(os.path.dirname(__file__), "icons", "ops.paint.clone_tool")
    bl_keymap = (
        ("image_paint.clone_tool", {"type": 'LEFTMOUSE', "value": 'PRESS'}, None),
        ("image_paint.clone_tool", {"type": 'LEFTMOUSE', "value": 'PRESS', "ctrl": True}, None),
        ("image_paint.clone_tool", {"type": 'LEFTMOUSE', "value": 'PRESS', "shift": True}, None),
        ("image_paint.clone_adjust_size", {"type": 'F', "value": 'PRESS'}, None),
        ("image_paint.clone_adjust_strength", {"type": 'F', "value": 'PRESS', "shift": True}, None),
    )

    def draw_settings(context, layout, tool):
        props = context.scene.text_tool_properties
        is_header = not layout.use_property_split
        
        if is_header:
            layout.prop(props, "clone_brush_size", text="Radius")
            layout.prop(props, "clone_brush_strength", text="Strength")
            if context.tool_settings.image_paint.brush:
                layout.prop(context.tool_settings.image_paint.brush, "blend", text="")
            layout.popover("IMAGE_EDIT_PT_clone_brush_popover", text="Falloff")
            layout.popover("IMAGE_EDIT_PT_clone_options_popover", text="Options")
        else:
            layout.use_property_split = True
            layout.use_property_decorate = False
            col = layout.column()
            col.prop(props, "clone_brush_size")
            col.prop(props, "clone_falloff_preset")
            
            # Show curve mapping for custom falloff
            if props.clone_falloff_preset == 'CUSTOM':
                brush = context.tool_settings.image_paint.brush
                if brush and brush.curve_distance_falloff:
                    col.template_curve_mapping(brush, "curve_distance_falloff", brush=True,
                                               use_negative_slope=True)
            
            col.prop(props, "clone_brush_strength")
            
            # Blend Mode
            col.separator()
            if context.tool_settings.image_paint.brush:
                col.prop(context.tool_settings.image_paint.brush, "blend", text="Blend Mode")
            
            col.separator()
            if utils.clone_source_set:
                col.label(text="Source set - Click to paint", icon='CHECKMARK')
            else:
                col.label(text="Ctrl+Click to set source", icon='INFO')
            
            col.separator()
            col.prop(props, "use_antialiasing")

class ImageCropTool(WorkSpaceTool):
    bl_space_type = 'IMAGE_EDITOR'
    bl_context_mode = 'PAINT'
    bl_idname = "image_paint.crop_tool"
    bl_label = "Crop Tool"
    bl_description = "Crop the image by selecting a rectangular region"
    bl_icon = os.path.join(os.path.dirname(__file__), "icons", "ops.paint.crop_tool")
    bl_keymap = (
        ("image_paint.crop_tool", {"type": 'LEFTMOUSE', "value": 'PRESS'}, None),
        ("image_paint.crop_tool", {"type": 'LEFTMOUSE', "value": 'PRESS', "ctrl": True}, None),
        ("image_paint.crop_tool", {"type": 'LEFTMOUSE', "value": 'PRESS', "shift": True}, None),
    )

    def draw_settings(context, layout, tool):
        props = context.scene.text_tool_properties
        is_header = not layout.use_property_split
        
        if is_header:
            layout.prop(props, "crop_lock_aspect", text="", icon='LOCKED' if props.crop_lock_aspect else 'UNLOCKED')
            layout.popover("IMAGE_EDIT_PT_crop_aspect_popover", text="Aspect")
            layout.popover("IMAGE_EDIT_PT_crop_canvas_popover", text="Canvas")
        else:
            layout.use_property_split = True
            layout.use_property_decorate = False
            col = layout.column()
            
            # Rule of thirds toggle
            col.prop(props, "crop_show_thirds")
            
            col.separator()
            
            # Aspect ratio section
            row = col.row(align=True)
            row.prop(props, "crop_lock_aspect", text="Lock Aspect Ratio")
            
            if props.crop_lock_aspect:
                box = col.box()
                box_col = box.column(align=True)
                box_col.prop(props, "crop_aspect_width", text="X")
                box_col.prop(props, "crop_aspect_height", text="Y")
            
            col.separator()
            
            # Expand canvas option
            col.prop(props, "crop_expand_canvas")
            if props.crop_expand_canvas:
                col.prop(props, "crop_fill_color")
            
            col.separator()
            
            # Resolution option
            col.prop(props, "crop_use_resolution", text="Set Resolution")
            if props.crop_use_resolution:
                box = col.box()
                box_col = box.column(align=True)
                box_col.prop(props, "crop_resolution_x", text="X")
                box_col.prop(props, "crop_resolution_y", text="Y")
            
            col.separator()
            col.label(text="Enter/Space to apply, ESC to cancel")

class GradientTool(WorkSpaceTool):
    bl_space_type = 'VIEW_3D'
    bl_context_mode = 'PAINT_TEXTURE'
    bl_idname = "texture_paint.gradient_tool"
    bl_label = "Gradient Tool"
    bl_description = "Paint gradients on textures using click-drag"
    bl_icon = "ops.paint.weight_gradient"
    bl_keymap = (
        ("paint.gradient_tool", {"type": 'LEFTMOUSE', "value": 'PRESS'}, None),
        ("paint.gradient_tool", {"type": 'LEFTMOUSE', "value": 'PRESS', "ctrl": True}, None),
        ("paint.gradient_tool", {"type": 'LEFTMOUSE', "value": 'PRESS', "shift": True}, None),
    )

    def draw_settings(context, layout, tool):
        props = context.scene.text_tool_properties
        is_header = not layout.use_property_split
        
        if is_header:
            layout.prop(props, "gradient_type", text="")
            if context.tool_settings.image_paint.brush:
                layout.prop(context.tool_settings.image_paint.brush, "blend", text="")
            layout.popover("IMAGE_EDIT_PT_gradient_settings_popover", text="Color Ramp")
        else:
            layout.use_property_split = True
            layout.use_property_decorate = False
            col = layout.column()
            col.prop(props, "gradient_type")
            
            # Color Ramp
            grad_node = utils.get_gradient_node(create_if_missing=False)
            if grad_node:
                col.template_color_ramp(grad_node, "color_ramp", expand=True)
            
            # Blend Mode
            col.separator()
            if context.tool_settings.image_paint.brush:
                col.prop(context.tool_settings.image_paint.brush, "blend", text="Blend Mode")
            
            col.separator()
            col.prop(props, "use_antialiasing")


class ImageGradientTool(WorkSpaceTool):
    bl_space_type = 'IMAGE_EDITOR'
    bl_context_mode = 'PAINT'
    bl_idname = "image_paint.gradient_tool"
    bl_label = "Gradient Tool"
    bl_description = "Paint gradients on images using click-drag"
    bl_icon = "ops.paint.weight_gradient"
    bl_keymap = (
        ("image_paint.gradient_tool", {"type": 'LEFTMOUSE', "value": 'PRESS'}, None),
        ("image_paint.gradient_tool", {"type": 'LEFTMOUSE', "value": 'PRESS', "ctrl": True}, None),
        ("image_paint.gradient_tool", {"type": 'LEFTMOUSE', "value": 'PRESS', "shift": True}, None),
    )

    def draw_settings(context, layout, tool):
        props = context.scene.text_tool_properties
        is_header = not layout.use_property_split
        
        if is_header:
            layout.prop(props, "gradient_type", text="")
            if context.tool_settings.image_paint.brush:
                layout.prop(context.tool_settings.image_paint.brush, "blend", text="")
            layout.popover("IMAGE_EDIT_PT_gradient_settings_popover", text="Color Ramp")
        else:
            layout.use_property_split = True
            layout.use_property_decorate = False
            col = layout.column()
            col.prop(props, "gradient_type")
            
            # Color Ramp
            grad_node = utils.get_gradient_node(create_if_missing=False)
            if grad_node:
                col.template_color_ramp(grad_node, "color_ramp", expand=True)
            
            # Blend Mode
            col.separator()
            if context.tool_settings.image_paint.brush:
                col.prop(context.tool_settings.image_paint.brush, "blend", text="Blend Mode")
            
            col.separator()
            col.prop(props, "use_antialiasing")


class TextTool(WorkSpaceTool):
    bl_space_type = 'VIEW_3D'
    bl_context_mode = 'PAINT_TEXTURE'
    bl_idname = "texture_paint.text_tool_ttf"
    bl_label = "Text Tool"
    bl_description = "Paint text on textures with TTF/OTF font rendering"
    bl_icon = os.path.join(os.path.dirname(__file__), "icons", "ops.paint.text_tool")
    bl_keymap = (
        ("paint.text_tool_ttf", {"type": 'LEFTMOUSE', "value": 'PRESS'}, None),
        ("paint.text_tool_ttf", {"type": 'LEFTMOUSE', "value": 'PRESS', "ctrl": True}, None),
        ("paint.text_tool_ttf", {"type": 'LEFTMOUSE', "value": 'PRESS', "shift": True}, None),
    )

    def draw_settings(context, layout, tool):
        props = context.scene.text_tool_properties
        
        # Check if we're in header (compact) or side panel (full)
        is_header = not layout.use_property_split
        
        if is_header:
            if not props.use_text_block:
                layout.prop(props, "color", text="")
            layout.prop(props, "font_size", text="Size")
            layout.prop(props, "rotation", text="Rotation")
            layout.popover("IMAGE_EDIT_PT_text_input_popover", text="Text Input")
            layout.popover("IMAGE_EDIT_PT_text_style_popover", text="Style")
            layout.popover("IMAGE_EDIT_PT_text_options_popover", text="Options")
        else:
            # Side panel: full vertical layout
            layout.use_property_split = True
            layout.use_property_decorate = False
            col = layout.column()
            
            # Text input with toggle for text block mode
            row = col.row(align=True)
            if not props.use_text_block:
                row.prop(props, "text", text="")
            else:
                row.template_ID(props, "text_block", new="text.new", open="text.open")
            row.prop(props, "use_text_block", text="", icon='TEXT', toggle=True)
            row = col.row(align=True)
            row.template_ID(props, "font_file", open="font.open")
            row.operator("paint.refresh_fonts_ttf", text="", icon='FILE_REFRESH')
            
            # Color/Gradient tabs
            col.separator()
            row = col.row(align=True)
            row.prop(props, "use_gradient", text="Color", toggle=True, invert_checkbox=True)
            row.prop(props, "use_gradient", text="Gradient", toggle=True)
            
            if not props.use_gradient:
                # Solid color mode
                col.prop(props, "color")
            else:
                # Gradient mode
                box = col.box()
                box.prop(props, "gradient_type", text="Type")
                if props.gradient_type == 'LINEAR':
                    box.prop(props, "gradient_rotation", text="Rotation")
                
                # Native Color Ramp
                grad_node = utils.get_gradient_node(create_if_missing=False)
                if grad_node:
                    box.template_color_ramp(grad_node, "color_ramp", expand=True)
                else:
                    box.label(text="Error loading gradient", icon="ERROR")
            
            col.separator()
            col.prop(props, "rotation")
            col.prop(props, "projection_mode")
            col.prop(props, "anchor_horizontal")
            col.prop(props, "anchor_vertical")
            
            # Text alignment and line spacing (only for multi-line text block mode)
            if props.use_text_block:
                col.prop(props, "text_alignment")
                col.prop(props, "line_spacing")
            # Brush Blend Mode
            brush = context.tool_settings.image_paint.brush
            if brush:
                col.prop(brush, "blend", text="Blend Mode")
            
            # Outline section
            col.separator()
            col.prop(props, "use_outline")
            if props.use_outline:
                box = col.box()
                box.prop(props, "outline_color", text="Color")
                box.prop(props, "outline_size", text="Size")
            
            col.separator()
            col.prop(props, "use_antialiasing")


class ImageTextTool(WorkSpaceTool):
    bl_space_type = 'IMAGE_EDITOR'
    bl_context_mode = 'PAINT'
    bl_idname = "image_paint.text_tool_ttf"
    bl_label = "Text Tool"
    bl_description = "Paint text(TTF/OTF) directly on images"
    bl_icon = os.path.join(os.path.dirname(__file__), "icons", "ops.paint.text_tool")
    bl_keymap = (
        ("image_paint.text_tool_ttf", {"type": 'LEFTMOUSE', "value": 'PRESS'}, None),
        ("image_paint.text_tool_ttf", {"type": 'LEFTMOUSE', "value": 'PRESS', "ctrl": True}, None),
        ("image_paint.text_tool_ttf", {"type": 'LEFTMOUSE', "value": 'PRESS', "shift": True}, None),
    )

    def draw_settings(context, layout, tool):
        props = context.scene.text_tool_properties
        
        # Check if we're in header (compact) or side panel (full)
        is_header = not layout.use_property_split
        
        if is_header:
            if not props.use_text_block:
                layout.prop(props, "color", text="")
            layout.prop(props, "font_size", text="Size")
            layout.prop(props, "rotation", text="Rotation")
            layout.popover("IMAGE_EDIT_PT_text_input_popover", text="Text Input")
            layout.popover("IMAGE_EDIT_PT_text_style_popover", text="Style")
            layout.popover("IMAGE_EDIT_PT_text_options_popover", text="Options")
        else:
            # Side panel: full vertical layout
            layout.use_property_split = True
            layout.use_property_decorate = False
            col = layout.column()
            
            # Text input with toggle for text block mode
            row = col.row(align=True)
            if not props.use_text_block:
                row.prop(props, "text", text="")
            else:
                row.template_ID(props, "text_block", new="text.new", open="text.open")
            row.prop(props, "use_text_block", text="", icon='TEXT', toggle=True)
            row = col.row(align=True)
            row.template_ID(props, "font_file", open="font.open")
            row.operator("paint.refresh_fonts_ttf", text="", icon='FILE_REFRESH')
            
            # Color/Gradient tabs
            col.separator()
            row = col.row(align=True)
            row.prop(props, "use_gradient", text="Color", toggle=True, invert_checkbox=True)
            row.prop(props, "use_gradient", text="Gradient", toggle=True)
            
            if not props.use_gradient:
                # Solid color mode
                col.prop(props, "color")
            else:
                # Gradient mode
                box = col.box()
                box.prop(props, "gradient_type", text="Type")
                if props.gradient_type == 'LINEAR':
                    box.prop(props, "gradient_rotation", text="Rotation")
                
                # Native Color Ramp
                grad_node = utils.get_gradient_node(create_if_missing=False)
                if grad_node:
                    box.template_color_ramp(grad_node, "color_ramp", expand=True)
                else:
                    box.label(text="Error loading gradient", icon="ERROR")
            
            col.separator()
            col.prop(props, "rotation")
            col.prop(props, "anchor_horizontal")
            col.prop(props, "anchor_vertical")
            
            # Text alignment and line spacing (only for multi-line text block mode)
            if props.use_text_block:
                col.prop(props, "text_alignment")
                col.prop(props, "line_spacing")
            # Brush Blend Mode
            brush = context.tool_settings.image_paint.brush
            if brush:
                col.prop(brush, "blend", text="Blend Mode")
            
            # Outline section
            col.separator()
            col.prop(props, "use_outline")
            if props.use_outline:
                box = col.box()
                box.prop(props, "outline_color", text="Color")
                box.prop(props, "outline_size", text="Size")
            
            col.separator()
            col.prop(props, "use_antialiasing")

def draw_pen_preview_3d(context):
    """Draw pen tool path preview in 3D Viewport (Screen Space)."""
    from .operators import pen_ops  # Import module directly to get live variable references
    
    if not pen_ops.pen_points and not pen_ops.pen_preview_pos:
        return
    
    # In 3D Viewport POST_PIXEL, coordinates are already screen pixels.
    # No conversion needed.
    
    gpu.state.blend_set('ALPHA')
    gpu.state.line_width_set(2.0)
    
    # Draw bezier curves between points
    if len(pen_ops.pen_points) >= 2:
        preview_points = []
        for i in range(len(pen_ops.pen_points) - 1):
            p0 = pen_ops.pen_points[i]
            p1 = pen_ops.pen_points[i + 1]
            
            x0, y0 = p0[0], p0[1]
            x1, y1 = p0[4], p0[5]
            x2, y2 = p1[2], p1[3]
            x3, y3 = p1[0], p1[1]
            
            for t in range(21):
                t_val = t / 20
                mt = 1 - t_val
                x = mt**3 * x0 + 3 * mt**2 * t_val * x1 + 3 * mt * t_val**2 * x2 + t_val**3 * x3
                y = mt**3 * y0 + 3 * mt**2 * t_val * y1 + 3 * mt * t_val**2 * y2 + t_val**3 * y3
                preview_points.append((x, y))
        
        if len(preview_points) >= 2:
            shader = gpu.shader.from_builtin('UNIFORM_COLOR')
            shader.uniform_float("color", (1.0, 1.0, 0.0, 0.8)) # Yellow for 3D visibility
            batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": preview_points})
            batch.draw(shader)
    
    # Draw anchor points and handles
    for i, pt in enumerate(pen_ops.pen_points):
        ax, ay = pt[0], pt[1]
        hi_x, hi_y = pt[2], pt[3]
        ho_x, ho_y = pt[4], pt[5]
        
        # Draw handle lines
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        shader.uniform_float("color", (0.5, 0.8, 1.0, 0.6))
        
        if (hi_x != ax or hi_y != ay):
            batch = batch_for_shader(shader, 'LINES', {"pos": [(ax, ay), (hi_x, hi_y)]})
            batch.draw(shader)
        if (ho_x != ax or ho_y != ay):
            batch = batch_for_shader(shader, 'LINES', {"pos": [(ax, ay), (ho_x, ho_y)]})
            batch.draw(shader)
        
        # Draw anchor point
        size = 6 if i == 0 else 4
        color = (0.0, 1.0, 0.0, 1.0) if i == 0 else (1.0, 1.0, 1.0, 1.0)
        _draw_square(ax, ay, size, color)
        
        # Draw handle points
        if (hi_x != ax or hi_y != ay):
            _draw_circle(hi_x, hi_y, 3, (0.5, 0.8, 1.0, 1.0), 8)
        if (ho_x != ax or ho_y != ay):
            _draw_circle(ho_x, ho_y, 3, (0.5, 0.8, 1.0, 1.0), 8)
            
    # Draw preview line to cursor
    if pen_ops.pen_preview_pos and len(pen_ops.pen_points) > 0:
        last_pt = pen_ops.pen_points[-1]
        cursor_pos = pen_ops.pen_preview_pos
        
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        shader.uniform_float("color", (1.0, 1.0, 0.0, 0.4))
        batch = batch_for_shader(shader, 'LINES', {"pos": [(last_pt[0], last_pt[1]), cursor_pos]})
        batch.draw(shader)
    
    gpu.state.line_width_set(1.0)
    gpu.state.blend_set('NONE')


class PenTool(WorkSpaceTool):
    bl_space_type = 'VIEW_3D'
    bl_context_mode = 'PAINT_TEXTURE'
    bl_idname = "texture_paint.pen_tool"
    bl_label = "Pen Tool"
    bl_description = "Draw bezier paths on 3D surface"
    bl_icon = os.path.join(os.path.dirname(__file__), "icons", "ops.paint.pen_tool")
    bl_keymap = (
        ("texture_paint.pen_tool", {"type": 'LEFTMOUSE', "value": 'PRESS'}, None),
    )

    def draw_settings(context, layout, tool):
        props = context.scene.text_tool_properties
        is_header = not layout.use_property_split
        
        if is_header:
            layout.prop(props, "pen_use_stroke", text="", icon='STROKE')
            if props.pen_use_stroke:
                layout.prop(props, "pen_stroke_color", text="")
                layout.prop(props, "pen_stroke_width", text="Radius")
            layout.prop(props, "pen_use_fill", text="", icon='SNAP_FACE')
            if props.pen_use_fill:
                layout.prop(props, "pen_fill_color", text="")
            layout.popover("IMAGE_EDIT_PT_pen_options_popover", text="Options")
        else:
            layout.use_property_split = True
            layout.use_property_decorate = False
            col = layout.column()
            
            # Reuse settings from Image Pen Tool
            col.prop(props, "pen_use_stroke")
            if props.pen_use_stroke:
                box = col.box()
                box.prop(props, "pen_stroke_color", text="Color")
                box.prop(props, "pen_stroke_width", text="Width")
            
            col.separator()
            col.prop(props, "pen_use_fill")
            if props.pen_use_fill:
                box = col.box()
                box.prop(props, "pen_fill_color", text="Color")
            
            col.separator()
            if context.tool_settings.image_paint.brush:
                col.prop(context.tool_settings.image_paint.brush, "blend", text="Blend Mode")
            
            col.separator()
            col.prop(props, "use_antialiasing")
            
            col.separator()
            col.label(text="Click to add points, drag for curves", icon='INFO')
            col.label(text="Enter/Space to apply, ESC to cancel")




from . import operators

def draw_box_select_tool_settings(context, layout):
    """Draw box select tool settings in the Tool tab"""
    wm = context.window_manager
    
    if not hasattr(wm, 'image_edit_properties'):
        return
    
    props = wm.image_edit_properties
    
    # Check if we're in header (compact) or side panel (full)
    is_header = not layout.use_property_split
    
    if is_header:
        layout.prop(props, 'selection_mode', text='')
        layout.prop(props, 'invert_mask', text='', icon='MOD_MASK')
        layout.prop(props, 'foreground_color', text='')
        layout.popover("IMAGE_EDIT_PT_select_mode_popover", text="Actions")
    else:
        # Side panel: full vertical layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        col = layout.column()
        
        # Selection Mode buttons
        col.prop(props, 'selection_mode')
        
        col.separator()
        
        # Deselect button and Invert toggle
        row = col.row(align=True)
        row.operator(operators.IMAGE_EDIT_OT_cancel_selection.bl_idname, text='Deselect', icon='X')
        row.prop(props, 'invert_mask', text='Invert', toggle=True, icon='MOD_MASK')
        
        col.separator()
        
        # Color + Fill
        row = col.row(align=True)
        row.prop(props, 'foreground_color', text='')
        row.operator(operators.IMAGE_EDIT_OT_fill_with_fg_color.bl_idname, text='Fill', icon='BRUSH_DATA')
        
        # Cut/Copy to Layer
        row = col.row(align=True)
        row.operator(operators.IMAGE_EDIT_OT_cut_to_layer.bl_idname, text='Cut to Layer', icon='X')
        row.operator(operators.IMAGE_EDIT_OT_copy_to_layer.bl_idname, text='Copy to Layer', icon='COPYDOWN')
        
        # Delete
        row = col.row(align=True)
        row.operator(operators.IMAGE_EDIT_OT_clear.bl_idname, text='Delete', icon='X')

class IMAGE_EDIT_WT_box_select(WorkSpaceTool):
    bl_space_type = 'IMAGE_EDITOR'
    bl_context_mode = 'PAINT'
    bl_idname = "image_edit.box_select_tool"
    bl_label = "Box Select"
    bl_description = "Draw a box to restrict painting to the selected area (Shift: Add, Ctrl: Subtract)"
    bl_icon = os.path.join(os.path.dirname(__file__), "icons", "ops.paint.box_tool")
    bl_widget = None
    bl_keymap = (
        (operators.IMAGE_EDIT_OT_make_selection.bl_idname, {"type": 'LEFTMOUSE', "value": 'PRESS'}, None),
        (operators.IMAGE_EDIT_OT_make_selection.bl_idname, {"type": 'LEFTMOUSE', "value": 'PRESS', "shift": True}, None),
        (operators.IMAGE_EDIT_OT_make_selection.bl_idname, {"type": 'LEFTMOUSE', "value": 'PRESS', "ctrl": True}, None),
    )
    
    def draw_settings(context, layout, tool):
        draw_box_select_tool_settings(context, layout)

class IMAGE_EDIT_WT_ellipse_select(WorkSpaceTool):
    bl_space_type = 'IMAGE_EDITOR'
    bl_context_mode = 'PAINT'
    bl_idname = "image_edit.ellipse_select_tool"
    bl_label = "Ellipse Select"
    bl_description = "Draw an ellipse to restrict painting to the selected area"
    bl_icon = os.path.join(os.path.dirname(__file__), "icons", "ops.paint.ellipse_tool")
    bl_widget = None
    bl_keymap = (
        (operators.IMAGE_EDIT_OT_make_ellipse_selection.bl_idname, {"type": 'LEFTMOUSE', "value": 'PRESS'}, None),
        (operators.IMAGE_EDIT_OT_make_ellipse_selection.bl_idname, {"type": 'LEFTMOUSE', "value": 'PRESS', "shift": True}, None),
        (operators.IMAGE_EDIT_OT_make_ellipse_selection.bl_idname, {"type": 'LEFTMOUSE', "value": 'PRESS', "ctrl": True}, None),
    )
    
    def draw_settings(context, layout, tool):
        draw_box_select_tool_settings(context, layout)

class IMAGE_EDIT_WT_lasso_select(WorkSpaceTool):
    bl_space_type = 'IMAGE_EDITOR'
    bl_context_mode = 'PAINT'
    bl_idname = "image_edit.lasso_select_tool"
    bl_label = "Lasso Select"
    bl_description = "Draw a freehand lasso to restrict painting to the selected area"
    bl_icon = os.path.join(os.path.dirname(__file__), "icons", "ops.paint.lasso_tool")
    bl_widget = None
    bl_keymap = (
        (operators.IMAGE_EDIT_OT_make_lasso_selection.bl_idname, {"type": 'LEFTMOUSE', "value": 'PRESS'}, None),
        (operators.IMAGE_EDIT_OT_make_lasso_selection.bl_idname, {"type": 'LEFTMOUSE', "value": 'PRESS', "shift": True}, None),
        (operators.IMAGE_EDIT_OT_make_lasso_selection.bl_idname, {"type": 'LEFTMOUSE', "value": 'PRESS', "ctrl": True}, None),
    )
    
    def draw_settings(context, layout, tool):
        draw_box_select_tool_settings(context, layout)

def draw_sculpt_tool_settings(context, layout):
    """Draw sculpt tool settings in the Tool tab"""
    wm = context.window_manager
    
    if not hasattr(wm, 'image_edit_properties'):
        return
    
    props = wm.image_edit_properties
    
    # Check if we're in header (compact) or side panel (full)
    is_header = not layout.use_property_split
    
    if is_header:
        layout.prop(props, 'sculpt_mode', text='')
        layout.prop(props, 'sculpt_radius', text='Radius')
        layout.prop(props, 'sculpt_strength', text='Strength')
        layout.popover("IMAGE_EDIT_PT_sculpt_brush_popover", text="Falloff")
    else:
        # Side panel: full vertical layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        col = layout.column()
        
        # Sculpt Mode dropdown
        col.prop(props, 'sculpt_mode')
        
        col.separator()
        
        # Brush settings
        col.prop(props, 'sculpt_radius')
        col.prop(props, 'sculpt_strength')
        
        # Falloff settings
        col.prop(props, 'sculpt_falloff_preset')
        
        # Show curve mapping for custom falloff
        if props.sculpt_falloff_preset == 'CUSTOM':
            brush = context.tool_settings.image_paint.brush
            if brush and brush.curve_distance_falloff:
                col.template_curve_mapping(brush, "curve_distance_falloff", brush=True,
                                            use_negative_slope=True)

class IMAGE_EDIT_WT_sculpt(WorkSpaceTool):
    bl_space_type = 'IMAGE_EDITOR'
    bl_context_mode = 'PAINT'
    bl_idname = "image_edit.sculpt_tool"
    bl_label = "Image Sculpt"
    bl_description = "Sculpt the image with pixel warping (Grab, Pull, Pinch, Twist)"
    bl_icon = os.path.join(os.path.dirname(__file__), "icons", "ops.paint.sculpt_tool")
    bl_widget = None
    bl_keymap = (
        (operators.IMAGE_EDIT_OT_sculpt_image.bl_idname, {"type": 'LEFTMOUSE', "value": 'PRESS'}, None),
    )
    
    def draw_settings(context, layout, tool):
        draw_sculpt_tool_settings(context, layout)

# Edit menu for Image Editor header
class IMAGE_EDIT_MT_edit_menu(bpy.types.Menu):
    bl_idname = "IMAGE_EDIT_MT_edit_menu"
    bl_label = "Edit"

    @classmethod
    def poll(cls, context):
        return (context.area.spaces.active.mode != 'UV' 
                and context.area.spaces.active.image is not None 
                and context.area.spaces.active.image.source != 'VIEWER')

    def draw(self, context):
        layout = self.layout
        
        # Edit section
        layout.operator(operators.IMAGE_EDIT_OT_cut_to_layer.bl_idname, text='Cut to New Layer', icon='PASTEFLIPDOWN')
        layout.operator(operators.IMAGE_EDIT_OT_copy_to_layer.bl_idname, text='Copy to New Layer', icon='COPYDOWN')
        layout.separator()
        layout.operator(operators.IMAGE_EDIT_OT_clear.bl_idname, text='Clear/Delete', icon='X')
        
        layout.separator()
        
        # Import/Export
        layout.operator(operators.IMAGE_EDIT_OT_import_psd.bl_idname, text='Import PSD', icon='IMPORT')
        
        layout.separator()
        
        # Layer submenu
        layout.menu(IMAGE_EDIT_MT_layers_menu.bl_idname, text='Layer', icon='RENDERLAYERS')
        layout.menu(IMAGE_EDIT_MT_transform_layer_menu.bl_idname, text='Transform Layer', icon='ORIENTATION_GIMBAL')

def edit_header_draw(self, context):
    """Add Edit menu to the Image Editor header"""
    if context.area.spaces.active.mode != 'UV' \
            and context.area.spaces.active.image is not None \
            and context.area.spaces.active.image.source != 'VIEWER':
        layout = self.layout
        layout.menu(IMAGE_EDIT_MT_edit_menu.bl_idname)



class IMAGE_EDIT_UL_layer_list(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname):
        img = bpy.data.images.get(item.name)
        if img:
            icon_value = bpy.types.UILayout.icon(img)
            
            row = layout.row(align=True)
            
            # 1. Multi-selection checkbox
            check_icon = 'CHECKBOX_HLT' if item.checked else 'CHECKBOX_DEHLT'
            row.prop(item, 'checked', text='', emboss=False, icon=check_icon)
            
            # 2. Visibility (eye icon)
            hide_icon = 'HIDE_ON' if item.hide else 'HIDE_OFF'
            row.prop(item, 'hide', text='', emboss=False, icon=hide_icon)
            
            # 3. Layer preview thumbnail + 4. Layer name (editable)
            row.prop(item, 'label', text='', emboss=False, icon_value=icon_value)
            
            # 5. Layer lock toggle
            lock_icon = 'LOCKED' if item.locked else 'UNLOCKED'
            row.prop(item, 'locked', text='', emboss=False, icon=lock_icon)

    def filter_items(self, context, data, propname):
        layers = getattr(data, propname)
        helper_funcs = bpy.types.UI_UL_list
        
        flt_flags = []
        flt_neworder = []
        
        if self.filter_name:
            flt_flags = helper_funcs.filter_items_by_name(self.filter_name, self.bitflag_filter_item, layers, "label")
        
        if not flt_flags:
            flt_flags = [self.bitflag_filter_item] * len(layers)
        
        if self.use_filter_sort_alpha:
            flt_neworder = helper_funcs.sort_items_by_name(layers, "label")
        
        return flt_flags, flt_neworder

class IMAGE_EDIT_MT_layers_menu(bpy.types.Menu):
    bl_idname = "IMAGE_EDIT_MT_layers_menu"
    bl_label = "Layers"

    def draw(self, context):
        layout = self.layout
        layout.operator(operators.IMAGE_EDIT_OT_deselect_layer.bl_idname, text='Deselect Layer', icon='X')
        layout.operator(operators.IMAGE_EDIT_OT_move_layer.bl_idname, text='Move', icon='GRAB')
        layout.operator(operators.IMAGE_EDIT_OT_rotate_layer.bl_idname, text='Rotate', icon='FILE_REFRESH')
        layout.operator(operators.IMAGE_EDIT_OT_scale_layer.bl_idname, text='Scale', icon='FULLSCREEN_ENTER')
        layout.operator(operators.IMAGE_EDIT_OT_delete_layer.bl_idname, text='Delete', icon='TRASH')
        layout.operator(operators.IMAGE_EDIT_OT_merge_layers.bl_idname, text='Merge Layers', icon='CHECKMARK')

class IMAGE_EDIT_MT_layer_options_menu(bpy.types.Menu):
    bl_idname = "IMAGE_EDIT_MT_layer_options_menu"
    bl_label = "Layer Options"

    def draw(self, context):
        layout = self.layout
        
        layout.operator(operators.IMAGE_EDIT_OT_export_layers.bl_idname, text='Export Layers', icon='EXPORT')
        layout.operator(operators.IMAGE_EDIT_OT_export_layers_multilayer.bl_idname, text='Export Multi-Layer File', icon='FILE_BLANK')
        layout.separator()
        
        layout.operator(operators.IMAGE_EDIT_OT_duplicate_layer.bl_idname, text='Duplicate', icon='DUPLICATE')
        
        layout.separator()
        
        layout.operator(operators.IMAGE_EDIT_OT_select_all_layers.bl_idname, text='Select All', icon='CHECKBOX_HLT')
        layout.operator(operators.IMAGE_EDIT_OT_deselect_all_layers.bl_idname, text='Deselect All', icon='CHECKBOX_DEHLT')
        layout.operator(operators.IMAGE_EDIT_OT_invert_layer_selection.bl_idname, text='Invert Selection', icon='UV_SYNC_SELECT')
        
        layout.separator()
        
        layout.operator(operators.IMAGE_EDIT_OT_merge_selected_layers.bl_idname, text='Merge Selected', icon='AUTOMERGE_OFF')
        layout.operator(operators.IMAGE_EDIT_OT_delete_selected_layers.bl_idname, text='Delete Selected', icon='TRASH')
        
        layout.separator()
        
        op = layout.operator(operators.IMAGE_EDIT_OT_flip_layer.bl_idname, text='Flip Horizontal', icon='MOD_MIRROR')
        op.is_vertically = False
        op = layout.operator(operators.IMAGE_EDIT_OT_flip_layer.bl_idname, text='Flip Vertical', icon='MOD_MIRROR')
        op.is_vertically = True
        
        layout.separator()
        
        layout.operator(operators.IMAGE_EDIT_OT_show_all_layers.bl_idname, text='Show All', icon='HIDE_OFF')
        layout.operator(operators.IMAGE_EDIT_OT_hide_all_layers.bl_idname, text='Hide All', icon='HIDE_ON')
        
        layout.separator()
        
        layout.operator(operators.IMAGE_EDIT_OT_lock_all_layers.bl_idname, text='Lock All', icon='LOCKED')
        layout.operator(operators.IMAGE_EDIT_OT_unlock_all_layers.bl_idname, text='Unlock All', icon='UNLOCKED')
        
        layout.separator()
        
        layout.operator(operators.IMAGE_EDIT_OT_merge_layers.bl_idname, text='Merge All', icon='AUTOMERGE_OFF')
        layout.operator(operators.IMAGE_EDIT_OT_delete_all_layers.bl_idname, text='Delete All', icon='TRASH')
        
        layout.separator()
        
        layout.operator(operators.IMAGE_EDIT_OT_update_layer_previews.bl_idname, text='Update Previews', icon='FILE_REFRESH')

class IMAGE_EDIT_MT_transform_layer_menu(bpy.types.Menu):
    bl_idname = "IMAGE_EDIT_MT_transform_layer_menu"
    bl_label = "Transform Layer"

    def draw(self, context):
        layout = self.layout
        layout.operator(operators.IMAGE_EDIT_OT_move_layer.bl_idname, text='Move', icon='ORIENTATION_GLOBAL')
        layout.operator(operators.IMAGE_EDIT_OT_rotate_layer_arbitrary.bl_idname, text='Rotate', icon='FILE_REFRESH')
        layout.operator(operators.IMAGE_EDIT_OT_scale_layer.bl_idname, text='Scale', icon='FULLSCREEN_ENTER')
        layout.separator()
        layout.operator(operators.IMAGE_EDIT_OT_apply_layer_transform.bl_idname, text='Apply Transform', icon='CHECKMARK')
        layout.separator()
        op = layout.operator(operators.IMAGE_EDIT_OT_flip_layer.bl_idname, text='Flip Horizontally', icon='MOD_MIRROR')
        op.is_vertically = False
        op = layout.operator(operators.IMAGE_EDIT_OT_flip_layer.bl_idname, text='Flip Vertically', icon='MOD_MIRROR')
        op.is_vertically = True
        op = layout.operator(operators.IMAGE_EDIT_OT_rotate_layer.bl_idname, text="Rotate 90° Left", icon='FILE_REFRESH')
        op.is_left = True
        op = layout.operator(operators.IMAGE_EDIT_OT_rotate_layer.bl_idname, text="Rotate 90° Right", icon='FILE_REFRESH')
        op.is_left = False

class IMAGE_EDIT_PT_layers_panel(bpy.types.Panel):
    bl_label = "Layers"
    bl_space_type = "IMAGE_EDITOR"
    bl_region_type = "UI"
    bl_category = "Layers"

    @classmethod
    def poll(cls, context):
        return context.area.spaces.active.mode != 'UV' and context.area.spaces.active.image != None and context.area.spaces.active.image.source != 'VIEWER'

    def draw(self, context):
        layout = self.layout
        img = context.area.spaces.active.image
        if img:
            img_props = img.image_edit_properties
            layers = img_props.layers
            
            row = layout.row()
            
            row.template_list("IMAGE_EDIT_UL_layer_list", "", img_props, "layers", img_props, "selected_layer_index", rows=4)
            
            col = row.column(align=True)
            col.operator(operators.IMAGE_EDIT_OT_add_image_layer.bl_idname, text='', icon='IMAGE_DATA')
            col.operator(operators.IMAGE_EDIT_OT_new_image_layer.bl_idname, text='', icon='ADD')
            col.separator()

            if layers:
                col.operator(operators.IMAGE_EDIT_OT_delete_layer.bl_idname, text='', icon='REMOVE')
                col.separator()
                col.menu(IMAGE_EDIT_MT_layer_options_menu.bl_idname, text='', icon='DOWNARROW_HLT')
                col.separator()
                op = col.operator(operators.IMAGE_EDIT_OT_change_image_layer_order.bl_idname, text='', icon="TRIA_UP")
                op.up = True
                op = col.operator(operators.IMAGE_EDIT_OT_change_image_layer_order.bl_idname, text='', icon="TRIA_DOWN")
                op.up = False
            
            if layers and img_props.selected_layer_index >= 0 and img_props.selected_layer_index < len(layers):
                selected_layer = layers[img_props.selected_layer_index]
                row = layout.row()
                row.prop(selected_layer, 'opacity', slider=True)
                row = layout.row()
                row.prop(selected_layer, 'blend_mode', text='')
                
                # Edit Layer button
                row = layout.row()
                if img_props.editing_layer:
                    row.operator(operators.IMAGE_EDIT_OT_edit_layer.bl_idname, text='Exit Edit Mode', icon='LOOP_BACK', depress=True)
                else:
                    row.operator(operators.IMAGE_EDIT_OT_edit_layer.bl_idname, text='Edit Layer', icon='GREASEPENCIL')
            
            # Show editing indicator if currently editing a layer
            if img_props.editing_layer:
                box = layout.box()
                box.alert = True
                box.label(text="Editing Layer Mode", icon='INFO')
                box.label(text="Paint directly on this layer")
                box.operator(operators.IMAGE_EDIT_OT_edit_layer.bl_idname, text='Exit Edit Mode', icon='LOOP_BACK')

class IMAGE_EDIT_PT_layer_transform_panel(bpy.types.Panel):
    bl_label = "Transform"
    bl_space_type = "IMAGE_EDITOR"
    bl_region_type = "UI"
    bl_category = "Layers"
    bl_parent_id = "IMAGE_EDIT_PT_layers_panel"

    @classmethod
    def poll(cls, context):
        return context.area.spaces.active.mode != 'UV' and context.area.spaces.active.image != None and context.area.spaces.active.image.source != 'VIEWER'

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        img = context.area.spaces.active.image
        if not img:
            return

        img_props = img.image_edit_properties
        layers = img_props.layers
        
        if layers and 0 <= img_props.selected_layer_index < len(layers):
            layer = layers[img_props.selected_layer_index]
            
            col = layout.column(align=True)
            col.prop(layer, "location", text="Location")
            
            col.separator()
            col = layout.column(align=True)
            row = col.row(align=True)
            # Blender uses degrees for the UI, converting the radian property
            row.prop(layer, "rotation", text="Rotation")
            
            col.separator()
            col = layout.column(align=True)
            col.prop(layer, "scale", text="Scale")
            
            if layer.rotation != 0.0 or layer.scale[0] != 1.0 or layer.scale[1] != 1.0:
                col.separator()
                col.operator(operators.IMAGE_EDIT_OT_apply_layer_transform.bl_idname, text="Apply Transform", icon='CHECKMARK')

class IMAGE_EDIT_PT_layer_align_panel(bpy.types.Panel):
    bl_label = "Align & Distribute"
    bl_space_type = "IMAGE_EDITOR"
    bl_region_type = "UI"
    bl_category = "Layers"
    bl_parent_id = "IMAGE_EDIT_PT_layers_panel"

    @classmethod
    def poll(cls, context):
        return context.area.spaces.active.mode != 'UV' and context.area.spaces.active.image != None and context.area.spaces.active.image.source != 'VIEWER'

    def draw(self, context):
        from . import operators
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        
        img = context.area.spaces.active.image
        if not img:
            return
            
        img_props = img.image_edit_properties
        layers = img_props.layers
        
        selected_count = sum(1 for layer in layers if layer.checked and not layer.locked)
        
        # Anchor
        col = layout.column(align=True)
        col.prop(img_props, "align_relative_to", text="Anchor:")
        
        # Align Section
        layout.separator()
        row = layout.row()
        row.alignment = 'LEFT'
        icon = 'TRIA_DOWN' if img_props.ui_align_open else 'TRIA_RIGHT'
        row.prop(img_props, "ui_align_open", text="Align", icon=icon, emboss=False)
        
        if img_props.ui_align_open:
            col = layout.column(align=True)
            row = col.row(align=True)
            
            # Align Left
            op = row.operator(operators.IMAGE_EDIT_OT_align_layers.bl_idname, text="Left")
            op.align_type = 'LEFT'
            
            # Align Right
            op = row.operator(operators.IMAGE_EDIT_OT_align_layers.bl_idname, text="Right")
            op.align_type = 'RIGHT'
            
            # Align Y Center (Vertical center alignment)
            op = row.operator(operators.IMAGE_EDIT_OT_align_layers.bl_idname, text="Y Center")
            op.align_type = 'CENTER_V'
            
            row = col.row(align=True)
            
            # Align X Center (Horizontal center alignment)
            op = row.operator(operators.IMAGE_EDIT_OT_align_layers.bl_idname, text="X Center")
            op.align_type = 'CENTER_H'
            
            # Align Bottom
            op = row.operator(operators.IMAGE_EDIT_OT_align_layers.bl_idname, text="Bottom")
            op.align_type = 'BOTTOM'
            
            # Align Top
            op = row.operator(operators.IMAGE_EDIT_OT_align_layers.bl_idname, text="Top")
            op.align_type = 'TOP'
            
        # Distribute Section
        layout.separator()
        row = layout.row()
        row.alignment = 'LEFT'
        icon = 'TRIA_DOWN' if img_props.ui_distribute_open else 'TRIA_RIGHT'
        row.prop(img_props, "ui_distribute_open", text="Distribute", icon=icon, emboss=False)
        
        if img_props.ui_distribute_open:
            col = layout.column(align=True)
            col.use_property_split = True
            col.prop(img_props, "distribute_mode", text="Mode:")
            col.use_property_split = False
            
            col.separator()
            col.prop(img_props, "distribute_use_gap")
            if img_props.distribute_use_gap:
                row = col.row(align=True)
                row.prop(img_props, "distribute_gap_x")
                row.prop(img_props, "distribute_gap_y")
                
            col.separator()
            row = col.row(align=True)
            
            # Distribute Horizontal (X)
            op = row.operator(operators.IMAGE_EDIT_OT_distribute_layers.bl_idname, text="X")
            op.distribute_type = 'HORIZONTAL'
            op.use_gap = img_props.distribute_use_gap
            if img_props.distribute_use_gap:
                op.gap_value = img_props.distribute_gap_x
            
            # Distribute Vertical (Y)
            op = row.operator(operators.IMAGE_EDIT_OT_distribute_layers.bl_idname, text="Y")
            op.distribute_type = 'VERTICAL'
            op.use_gap = img_props.distribute_use_gap
            if img_props.distribute_use_gap:
                op.gap_value = img_props.distribute_gap_y
            
        # Arrange Section
        layout.separator()
        row = layout.row()
        row.alignment = 'LEFT'
        icon = 'TRIA_DOWN' if img_props.ui_arrange_open else 'TRIA_RIGHT'
        row.prop(img_props, "ui_arrange_open", text="Arrange", icon=icon, emboss=False)
        
        if img_props.ui_arrange_open:
            box = layout.box()
            col = box.column(align=True)
            
            row = col.row()
            row.alignment = 'LEFT'
            row.label(text="Grid Arrange", icon='MESH_GRID')
            col.separator()
            
            col.label(text="Grid Size:")
            row = col.row(align=True)
            row.prop(img_props, "arrange_grid_x", text="X")
            row.prop(img_props, "arrange_grid_y", text="Y")
            
            col.separator()
            col.label(text="Gap (pixels):")
            row = col.row(align=True)
            row.prop(img_props, "arrange_gap_x", text="X")
            row.prop(img_props, "arrange_gap_y", text="Y")
            
            col.separator()
            total_slots = img_props.arrange_grid_x * img_props.arrange_grid_y
            
            row = col.row()
            row.alignment = 'LEFT'
            row.label(text=f"Total slots: {total_slots}", icon='INFO')
            
            col.separator()
            col.operator(operators.IMAGE_EDIT_OT_arrange_grid.bl_idname, text="Apply Grid")
            
        if selected_count < 2:
            layout.separator()
            layout.label(text="Select 2+ layers to align", icon='INFO')
        elif selected_count < 3:
            layout.separator()
            layout.label(text="Select 3+ layers to distribute", icon='INFO')


# -----------------
# Tool Setting Popovers
# -----------------


class IMAGE_EDIT_PT_pen_stroke_popover(bpy.types.Panel):
    bl_label = "Stroke & Fill"
    bl_idname = "IMAGE_EDIT_PT_pen_stroke_popover"
    bl_space_type = 'TOPBAR'
    bl_region_type = 'HEADER'

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        props = context.scene.text_tool_properties
        col = layout.column()
        col.prop(props, "pen_use_stroke")
        if props.pen_use_stroke:
            box = col.box()
            box.prop(props, "pen_stroke_color", text="Color")
            box.prop(props, "pen_stroke_width", text="Width")
        col.separator()
        col.prop(props, "pen_use_fill")
        if props.pen_use_fill:
            box = col.box()
            box.prop(props, "pen_fill_color", text="Color")

class IMAGE_EDIT_PT_pen_options_popover(bpy.types.Panel):
    bl_label = "Options"
    bl_idname = "IMAGE_EDIT_PT_pen_options_popover"
    bl_space_type = 'TOPBAR'
    bl_region_type = 'HEADER'

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        props = context.scene.text_tool_properties
        col = layout.column()
        if context.tool_settings.image_paint.brush:
            col.prop(context.tool_settings.image_paint.brush, "blend", text="Blend Mode")
        col.prop(props, "use_antialiasing")

class IMAGE_EDIT_PT_clone_brush_popover(bpy.types.Panel):
    bl_label = "Brush"
    bl_idname = "IMAGE_EDIT_PT_clone_brush_popover"
    bl_space_type = 'TOPBAR'
    bl_region_type = 'HEADER'

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        props = context.scene.text_tool_properties
        col = layout.column()
        col.prop(props, "clone_falloff_preset")
        if props.clone_falloff_preset == 'CUSTOM':
            brush = context.tool_settings.image_paint.brush
            if brush and brush.curve_distance_falloff:
                col.template_curve_mapping(brush, "curve_distance_falloff", brush=True, use_negative_slope=True)

class IMAGE_EDIT_PT_clone_options_popover(bpy.types.Panel):
    bl_label = "Options"
    bl_idname = "IMAGE_EDIT_PT_clone_options_popover"
    bl_space_type = 'TOPBAR'
    bl_region_type = 'HEADER'

    def draw(self, context):
        from . import utils
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        props = context.scene.text_tool_properties
        col = layout.column()
        if utils.clone_source_set:
            col.label(text="Source set - Click to paint", icon='CHECKMARK')
        else:
            col.label(text="Ctrl+Click to set source", icon='INFO')
        col.separator()
        col.prop(props, "use_antialiasing")

class IMAGE_EDIT_PT_crop_aspect_popover(bpy.types.Panel):
    bl_label = "Aspect"
    bl_idname = "IMAGE_EDIT_PT_crop_aspect_popover"
    bl_space_type = 'TOPBAR'
    bl_region_type = 'HEADER'

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        props = context.scene.text_tool_properties
        col = layout.column()
        col.prop(props, "crop_show_thirds")
        col.separator()
        row = col.row(align=True)
        row.prop(props, "crop_lock_aspect", text="Lock Aspect")
        if props.crop_lock_aspect:
            box = col.box()
            box_col = box.column(align=True)
            box_col.prop(props, "crop_aspect_width", text="X")
            box_col.prop(props, "crop_aspect_height", text="Y")

class IMAGE_EDIT_PT_crop_canvas_popover(bpy.types.Panel):
    bl_label = "Canvas"
    bl_idname = "IMAGE_EDIT_PT_crop_canvas_popover"
    bl_space_type = 'TOPBAR'
    bl_region_type = 'HEADER'

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        props = context.scene.text_tool_properties
        col = layout.column()
        col.prop(props, "crop_expand_canvas")
        if props.crop_expand_canvas:
            col.prop(props, "crop_fill_color")
        col.separator()
        col.prop(props, "crop_use_resolution", text="Set Resolution")
        if props.crop_use_resolution:
            box = col.box()
            box_col = box.column(align=True)
            box_col.prop(props, "crop_resolution_x", text="X")
            box_col.prop(props, "crop_resolution_y", text="Y")

class IMAGE_EDIT_PT_gradient_settings_popover(bpy.types.Panel):
    bl_label = "Gradient Settings"
    bl_idname = "IMAGE_EDIT_PT_gradient_settings_popover"
    bl_space_type = 'TOPBAR'
    bl_region_type = 'HEADER'

    def draw(self, context):
        from . import utils
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        props = context.scene.text_tool_properties
        col = layout.column()
        grad_node = utils.get_gradient_node(create_if_missing=False)
        if grad_node:
            col.template_color_ramp(grad_node, "color_ramp", expand=True)
        col.separator()
        col.prop(props, "use_antialiasing")

class IMAGE_EDIT_PT_text_input_popover(bpy.types.Panel):
    bl_label = "Text Box"
    bl_idname = "IMAGE_EDIT_PT_text_input_popover"
    bl_space_type = 'TOPBAR'
    bl_region_type = 'HEADER'

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        props = context.scene.text_tool_properties
        col = layout.column()
        row = col.row(align=True)
        if not props.use_text_block:
            row.prop(props, "text", text="")
        else:
            row.template_ID(props, "text_block", new="text.new", open="text.open")
        row.prop(props, "use_text_block", text="", icon='TEXT', toggle=True)
        row = col.row(align=True)
        row.template_ID(props, "font_file", open="font.open")
        row.operator("paint.refresh_fonts_ttf", text="", icon='FILE_REFRESH')

class IMAGE_EDIT_PT_text_style_popover(bpy.types.Panel):
    bl_label = "Style"
    bl_idname = "IMAGE_EDIT_PT_text_style_popover"
    bl_space_type = 'TOPBAR'
    bl_region_type = 'HEADER'

    def draw(self, context):
        from . import utils
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        props = context.scene.text_tool_properties
        col = layout.column()
        row = col.row(align=True)
        row.prop(props, "use_gradient", text="Color", toggle=True, invert_checkbox=True)
        row.prop(props, "use_gradient", text="Gradient", toggle=True)
        if not props.use_gradient:
            col.prop(props, "color")
        else:
            box = col.box()
            box.prop(props, "gradient_type", text="Type")
            if props.gradient_type == 'LINEAR':
                box.prop(props, "gradient_rotation", text="Rotation")
            grad_node = utils.get_gradient_node(create_if_missing=False)
            if grad_node:
                box.template_color_ramp(grad_node, "color_ramp", expand=True)
        col.separator()
        col.prop(props, "use_outline")
        if props.use_outline:
            box = col.box()
            box.prop(props, "outline_color", text="Color")
            box.prop(props, "outline_size", text="Size")
        col.separator()
        if context.tool_settings.image_paint.brush:
            col.prop(context.tool_settings.image_paint.brush, "blend", text="Blend Mode")

class IMAGE_EDIT_PT_text_options_popover(bpy.types.Panel):
    bl_label = "Options"
    bl_idname = "IMAGE_EDIT_PT_text_options_popover"
    bl_space_type = 'TOPBAR'
    bl_region_type = 'HEADER'

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        props = context.scene.text_tool_properties
        col = layout.column()
        # Check projection mode property exists
        if hasattr(props, "projection_mode"):
            col.prop(props, "projection_mode")
        col.prop(props, "anchor_horizontal")
        col.prop(props, "anchor_vertical")
        if props.use_text_block:
            col.prop(props, "text_alignment")
            col.prop(props, "line_spacing")
        col.separator()
        col.prop(props, "use_antialiasing")

class IMAGE_EDIT_PT_select_mode_popover(bpy.types.Panel):
    bl_label = "Selection Mode & Actions"
    bl_idname = "IMAGE_EDIT_PT_select_mode_popover"
    bl_space_type = 'TOPBAR'
    bl_region_type = 'HEADER'

    def draw(self, context):
        from . import operators
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        wm = context.window_manager
        if not hasattr(wm, 'image_edit_properties'): return
        props = wm.image_edit_properties
        col = layout.column()
        row = col.row(align=True)
        row.operator(operators.IMAGE_EDIT_OT_cancel_selection.bl_idname, text='Deselect', icon='X')
        row.operator(operators.IMAGE_EDIT_OT_fill_with_fg_color.bl_idname, text='Fill (FG)', icon='BRUSH_DATA')
        row = col.row(align=True)
        row.operator(operators.IMAGE_EDIT_OT_cut_to_layer.bl_idname, text='Cut to Layer', icon='X')
        row.operator(operators.IMAGE_EDIT_OT_copy_to_layer.bl_idname, text='Copy to Layer', icon='COPYDOWN')
        row = col.row(align=True)
        row.operator(operators.IMAGE_EDIT_OT_clear.bl_idname, text='Delete', icon='X')

class IMAGE_EDIT_PT_sculpt_brush_popover(bpy.types.Panel):
    bl_label = "Sculpt Brush"
    bl_idname = "IMAGE_EDIT_PT_sculpt_brush_popover"
    bl_space_type = 'TOPBAR'
    bl_region_type = 'HEADER'

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        wm = context.window_manager
        if not hasattr(wm, 'image_edit_properties'): return
        props = wm.image_edit_properties
        col = layout.column()
        col.prop(props, 'sculpt_falloff_preset')
        if props.sculpt_falloff_preset == 'CUSTOM':
            brush = context.tool_settings.image_paint.brush
            if brush and brush.curve_distance_falloff:
                col.template_curve_mapping(brush, "curve_distance_falloff", brush=True, use_negative_slope=True)

