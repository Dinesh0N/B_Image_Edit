import math

import bpy
import bmesh
from bpy.types import Operator
from bpy_extras.view3d_utils import (
    region_2d_to_origin_3d,
    region_2d_to_vector_3d,
)
from mathutils.bvhtree import BVHTree
from mathutils import Vector
from .. import utils


# Global pen tool state - these are imported by ui.py for drawing
pen_points = []  # List of (anchor_x, anchor_y, handle_in_x, handle_in_y, handle_out_x, handle_out_y)
pen_preview_pos = None
pen_is_closed = False
pen_is_adjusting_handle = False
pen_handle_snap_active = False
pen_displace_mode = False
pen_displace_offset = (0, 0)
pen_last_applied_points = None

# Internal pen state
pen_is_dragging = False
pen_drag_handle = None
pen_edit_point_idx = None
pen_edit_element = None
pen_realtime_preview = True
pen_show_fill_preview = True
pen_show_stroke_preview = True
pen_displacing = False
pen_displace_start = None




# ----------------------------
# Pen Tool (Image Editor)
# ----------------------------

# Global pen tool state for drawing
pen_points = []  # List of (anchor_x, anchor_y, handle_in_x, handle_in_y, handle_out_x, handle_out_y)
pen_preview_pos = None
pen_is_dragging = False
pen_drag_handle = None  # 'IN' or 'OUT'
pen_edit_point_idx = None  # Index of point being edited (for re-edit mode)
pen_edit_element = None  # 'ANCHOR', 'HANDLE_IN', 'HANDLE_OUT'

# Real-time preview state (Atelier-Paint style)
pen_realtime_preview = True  # Show real-time fill/stroke preview
pen_show_fill_preview = True  # Toggle for filled preview rendering
pen_show_stroke_preview = True  # Toggle for stroke preview rendering

# Spacebar displacement state
pen_displacing = False  # Whether in displacement mode
pen_displace_start = None  # (x, y) mouse position when displacement started
pen_displace_offset = (0, 0)  # Cumulative offset applied to points

# Curve re-edit state (allows editing previously applied path)
pen_last_applied_points = None  # Stored points from last apply for re-editing


class IMAGE_PAINT_OT_pen_tool(Operator):
    bl_idname = "image_paint.pen_tool"
    bl_label = "Pen Tool"
    bl_description = "Draw bezier paths with stroke and fill"
    bl_options = {'REGISTER', 'UNDO'}
    
    _draw_handler = None
    _image = None
    _width = 0
    _height = 0
    _is_closed = False
    _hit_radius = 12  # Pixel radius for hit detection
    
    @classmethod
    def poll(cls, context):
        sima = context.space_data
        return (context.area.type == 'IMAGE_EDITOR' and 
                sima.mode == 'PAINT' and 
                sima.image is not None)
    
    def _hit_test(self, img_x, img_y):
        """Test if click is near any existing point or handle. Returns (index, element_type) or (None, None)."""
        global pen_points
        
        for i, pt in enumerate(pen_points):
            ax, ay = pt[0], pt[1]
            hi_x, hi_y = pt[2], pt[3]
            ho_x, ho_y = pt[4], pt[5]
            
            # Check anchor point
            dist_anchor = ((img_x - ax)**2 + (img_y - ay)**2)**0.5
            if dist_anchor < self._hit_radius:
                return (i, 'ANCHOR')
            
            # Check handle in (if different from anchor)
            if hi_x != ax or hi_y != ay:
                dist_hi = ((img_x - hi_x)**2 + (img_y - hi_y)**2)**0.5
                if dist_hi < self._hit_radius:
                    return (i, 'HANDLE_IN')
            
            # Check handle out (if different from anchor)
            if ho_x != ax or ho_y != ay:
                dist_ho = ((img_x - ho_x)**2 + (img_y - ho_y)**2)**0.5
                if dist_ho < self._hit_radius:
                    return (i, 'HANDLE_OUT')
        
        return (None, None)
    
    def _snap_handle_30(self, anchor_x, anchor_y, handle_x, handle_y):
        """Snap handle to nearest 30° angle from anchor point."""
        dx = handle_x - anchor_x
        dy = handle_y - anchor_y
        distance = math.sqrt(dx * dx + dy * dy)
        if distance < 1:
            return (handle_x, handle_y)
        
        # Get current angle and snap to nearest 30° (pi/6)
        angle = math.atan2(dy, dx)
        snap_angle = round(angle / (math.pi / 6)) * (math.pi / 6)
        
        # Calculate snapped handle position
        snapped_x = anchor_x + distance * math.cos(snap_angle)
        snapped_y = anchor_y + distance * math.sin(snap_angle)
        return (snapped_x, snapped_y)
    
    def modal(self, context, event):
        global pen_points, pen_preview_pos, pen_is_dragging, pen_drag_handle
        global pen_edit_point_idx, pen_edit_element
        global pen_displacing, pen_displace_start
        context.area.tag_redraw()
        
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE'}:
            return {'PASS_THROUGH'}
        
        mx, my = event.mouse_region_x, event.mouse_region_y
        
        # Convert to image coordinates
        region = context.region
        view2d = region.view2d
        uv = view2d.region_to_view(mx, my)
        img_x = uv[0] * self._width
        img_y = uv[1] * self._height
        
        pen_preview_pos = (img_x, img_y)
        
        if event.type == 'MOUSEMOVE':
            # Spacebar displacement mode - move entire path
            if pen_displacing and pen_displace_start is not None:
                dx = img_x - pen_displace_start[0]
                dy = img_y - pen_displace_start[1]
                # Move all points by the offset
                new_points = []
                for pt in pen_points:
                    new_points.append((
                        pt[0] + dx, pt[1] + dy,
                        pt[2] + dx, pt[3] + dy,
                        pt[4] + dx, pt[5] + dy
                    ))
                pen_points = new_points
                pen_displace_start = (img_x, img_y)  # Update start for continuous drag
                return {'RUNNING_MODAL'}
            
            if pen_is_dragging:
                if pen_edit_point_idx is not None and pen_edit_element is not None:
                    # Re-editing existing point/handle
                    pt = pen_points[pen_edit_point_idx]
                    
                    if pen_edit_element == 'ANCHOR':
                        # Move anchor and both handles together
                        dx = img_x - pt[0]
                        dy = img_y - pt[1]
                        pen_points[pen_edit_point_idx] = (
                            img_x, img_y,
                            pt[2] + dx, pt[3] + dy,
                            pt[4] + dx, pt[5] + dy
                        )
                    elif pen_edit_element == 'HANDLE_IN':
                        # Shift: snap to 30° angles
                        hx, hy = img_x, img_y
                        if event.shift:
                            hx, hy = self._snap_handle_30(pt[0], pt[1], img_x, img_y)
                        # Move handle in, optionally mirror handle out
                        if event.alt:
                            # Alt held: move only this handle
                            pen_points[pen_edit_point_idx] = (pt[0], pt[1], hx, hy, pt[4], pt[5])
                        else:
                            # Mirror the handle out
                            dx = hx - pt[0]
                            dy = hy - pt[1]
                            pen_points[pen_edit_point_idx] = (pt[0], pt[1], hx, hy, pt[0] - dx, pt[1] - dy)
                    elif pen_edit_element == 'HANDLE_OUT':
                        # Shift: snap to 30° angles
                        hx, hy = img_x, img_y
                        if event.shift:
                            hx, hy = self._snap_handle_30(pt[0], pt[1], img_x, img_y)
                        # Move handle out, optionally mirror handle in
                        if event.alt:
                            # Alt held: move only this handle
                            pen_points[pen_edit_point_idx] = (pt[0], pt[1], pt[2], pt[3], hx, hy)
                        else:
                            # Mirror the handle in
                            dx = hx - pt[0]
                            dy = hy - pt[1]
                            pen_points[pen_edit_point_idx] = (pt[0], pt[1], pt[0] - dx, pt[1] - dy, hx, hy)
                
                elif pen_drag_handle == 'OUT' and len(pen_points) > 0:
                    # Creating new point - adjust handle with optional snapping
                    last_pt = pen_points[-1]
                    hx, hy = img_x, img_y
                    if event.shift:
                        hx, hy = self._snap_handle_30(last_pt[0], last_pt[1], img_x, img_y)
                    dx = hx - last_pt[0]
                    dy = hy - last_pt[1]
                    pen_points[-1] = (last_pt[0], last_pt[1], last_pt[0] - dx, last_pt[1] - dy, hx, hy)
            
            return {'RUNNING_MODAL'}
        
        elif event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                # First check if clicking near first point to close path
                if len(pen_points) >= 3:
                    first_pt = pen_points[0]
                    dist = ((img_x - first_pt[0])**2 + (img_y - first_pt[1])**2)**0.5
                    if dist < self._hit_radius:
                        # Close the path
                        self._is_closed = True
                        self._apply_path(context)
                        self._cleanup(context)
                        return {'FINISHED'}
                
                # Hit test for re-editing existing points/handles
                hit_idx, hit_elem = self._hit_test(img_x, img_y)
                
                if hit_idx is not None:
                    # Start editing existing point/handle
                    pen_edit_point_idx = hit_idx
                    pen_edit_element = hit_elem
                    pen_is_dragging = True
                    context.area.header_text_set(f"Editing {hit_elem.lower()} of point {hit_idx + 1} | Alt to break handles | Release to confirm")
                    return {'RUNNING_MODAL'}
                
                # Add new point
                pen_points.append((img_x, img_y, img_x, img_y, img_x, img_y))
                pen_is_dragging = True
                pen_drag_handle = 'OUT'
                pen_edit_point_idx = None
                pen_edit_element = None
                context.area.header_text_set(f"Point {len(pen_points)} | Drag to adjust curve | Enter to apply | ESC to cancel")
                return {'RUNNING_MODAL'}
            
            elif event.value == 'RELEASE':
                pen_is_dragging = False
                pen_drag_handle = None
                pen_edit_point_idx = None
                pen_edit_element = None
                return {'RUNNING_MODAL'}
        
        elif event.type == 'BACK_SPACE' and event.value == 'PRESS':
            # Delete last point
            if len(pen_points) > 0:
                pen_points.pop()
                context.area.header_text_set(f"Point deleted | {len(pen_points)} points remaining")
            return {'RUNNING_MODAL'}
        
        # Spacebar: displacement mode (Atelier-Paint style)
        elif event.type == 'SPACE':
            if event.value == 'PRESS' and len(pen_points) > 0:
                pen_displacing = True
                pen_displace_start = (img_x, img_y)
                context.area.header_text_set("Displacing path - move mouse, release SPACE to confirm")
                return {'RUNNING_MODAL'}
            elif event.value == 'RELEASE':
                pen_displacing = False
                pen_displace_start = None
                context.area.header_text_set(f"{len(pen_points)} points | Enter to apply | ESC to cancel | SPACE to move")
                return {'RUNNING_MODAL'}
        
        # R key: reload last applied path for re-editing
        elif event.type == 'R' and event.value == 'PRESS' and pen_last_applied_points is not None:
            pen_points = list(pen_last_applied_points)
            context.area.header_text_set(f"Re-editing last path ({len(pen_points)} points) | Modify and press Enter")
            return {'RUNNING_MODAL'}
        
        elif event.type in {'RET', 'NUMPAD_ENTER'} and event.value == 'PRESS':
            if len(pen_points) >= 2:
                self._apply_path(context)
            self._cleanup(context)
            return {'FINISHED'}
        
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            context.area.header_text_set(None)
            self._cleanup(context)
            return {'CANCELLED'}
        
        return {'RUNNING_MODAL'}
    
    def invoke(self, context, event):
        global pen_points, pen_preview_pos, pen_is_dragging
        global pen_displacing, pen_displace_start
        
        if context.area.type == 'IMAGE_EDITOR':
            pen_points = []
            pen_preview_pos = None
            pen_is_dragging = False
            pen_displacing = False
            pen_displace_start = None
            
            self._image = context.space_data.image
            self._width, self._height = self._image.size
            self._is_closed = False
            
            from .. import ui
            self._draw_handler = bpy.types.SpaceImageEditor.draw_handler_add(
                ui.draw_pen_preview, (context,), 'WINDOW', 'POST_PIXEL')
            
            context.window_manager.modal_handler_add(self)
            context.area.header_text_set("Click to add points | SHIFT=snap 30° | SPACE=move | R=re-edit last | Enter to apply")
            return {'RUNNING_MODAL'}
        
        return {'CANCELLED'}
    
    def _apply_path(self, context):
        """Render the path to the image using numpy (no PIL required)."""
        global pen_points, pen_last_applied_points
        
        if len(pen_points) < 2:
            return
        
        # Store points for re-editing (copy to avoid reference issues)
        pen_last_applied_points = [tuple(pt) for pt in pen_points]
        
        props = context.scene.text_tool_properties
        
        # Save undo state
        utils.ImageUndoStack.get().push_state(self._image)
        
        import numpy as np
        
        # Get current pixels
        num_pixels = self._width * self._height * 4
        pixels = np.zeros(num_pixels, dtype=np.float32)
        self._image.pixels.foreach_get(pixels)
        pixels = pixels.reshape((self._height, self._width, 4))
        
        # Generate bezier curve points (in image coordinates)
        path_points = self._generate_bezier_points()
        
        if len(path_points) < 2:
            return
        
        # Get blend mode and anti-aliasing setting
        brush = context.tool_settings.image_paint.brush
        blend_mode = brush.blend if brush else 'MIX'
        use_aa = props.use_antialiasing
        
        # Draw fill using scanline algorithm
        if props.pen_use_fill and len(path_points) >= 3:
            fill_color = np.array(props.pen_fill_color, dtype=np.float32)
            self._fill_polygon(pixels, path_points, fill_color, blend_mode)
        
        # Draw stroke
        if props.pen_use_stroke:
            stroke_color = np.array(props.pen_stroke_color, dtype=np.float32)
            stroke_width = props.pen_stroke_width
            self._draw_polyline(pixels, path_points, stroke_color, stroke_width, blend_mode, use_aa)
            
            # Close path if needed
            if self._is_closed and len(path_points) >= 2:
                self._draw_line(pixels, path_points[-1], path_points[0], stroke_color, stroke_width, blend_mode, use_aa)
        
        # Set pixels back
        self._image.pixels.foreach_set(pixels.flatten())
        self._image.update()
    
    def _generate_bezier_points(self, segments_per_curve=20):
        """Generate points along the bezier path in image coordinates."""
        global pen_points
        
        points = []
        for i in range(len(pen_points) - 1):
            p0 = pen_points[i]
            p1 = pen_points[i + 1]
            
            # Bezier control points
            x0, y0 = p0[0], p0[1]
            x1, y1 = p0[4], p0[5]  # handle_out of p0
            x2, y2 = p1[2], p1[3]  # handle_in of p1
            x3, y3 = p1[0], p1[1]
            
            for t in range(segments_per_curve + 1):
                t_val = t / segments_per_curve
                # Cubic bezier formula
                mt = 1 - t_val
                x = mt**3 * x0 + 3 * mt**2 * t_val * x1 + 3 * mt * t_val**2 * x2 + t_val**3 * x3
                y = mt**3 * y0 + 3 * mt**2 * t_val * y1 + 3 * mt * t_val**2 * y2 + t_val**3 * y3
                points.append((int(x), int(y)))
        
        return points
    
    def _fill_polygon(self, pixels, points, color, blend_mode='MIX'):
        """Fill a polygon using vectorized scanline algorithm with blend mode support."""
        import numpy as np
        
        if len(points) < 3:
            return
        
        # Convert to numpy arrays
        pts = np.array(points)
        min_y = max(0, int(pts[:, 1].min()))
        max_y = min(self._height - 1, int(pts[:, 1].max()))
        min_x = max(0, int(pts[:, 0].min()))
        max_x = min(self._width - 1, int(pts[:, 0].max()))
        
        if max_y <= min_y or max_x <= min_x:
            return
        
        # Create coordinate grid for the bounding box
        yy, xx = np.mgrid[min_y:max_y+1, min_x:max_x+1]
        
        # Point-in-polygon using ray casting (vectorized)
        n = len(points)
        inside = np.zeros(yy.shape, dtype=bool)
        
        for i in range(n):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % n]
            
            # Check if edge crosses the horizontal ray
            cond1 = ((y1 <= yy) & (yy < y2)) | ((y2 <= yy) & (yy < y1))
            
            if y1 != y2:
                x_intersect = x1 + (yy - y1) * (x2 - x1) / (y2 - y1)
                crosses = cond1 & (xx < x_intersect)
                inside ^= crosses
        
        # Apply fill with blend mode
        alpha = color[3]
        mask_3d = inside[:, :, np.newaxis]
        
        dst = pixels[min_y:max_y+1, min_x:max_x+1]
        src_color = np.array([color[0], color[1], color[2], color[3]])
        
        # Calculate blended color based on blend mode
        blended = self._apply_blend_mode(dst, src_color, blend_mode)
        
        # Apply with alpha where mask is true
        pixels[min_y:max_y+1, min_x:max_x+1] = np.where(
            mask_3d,
            dst * (1 - alpha) + blended * alpha,
            dst
        )
    
    def _draw_polyline(self, pixels, points, color, width, blend_mode='MIX', use_aa=True):
        """Draw a polyline with given width, blend mode, and anti-aliasing support."""
        import numpy as np
        
        if len(points) < 2:
            return
        
        # Get bounding box of all points + stroke width padding
        pts = np.array(points)
        pad = width + 2
        min_x = max(0, int(pts[:, 0].min()) - pad)
        max_x = min(self._width - 1, int(pts[:, 0].max()) + pad)
        min_y = max(0, int(pts[:, 1].min()) - pad)
        max_y = min(self._height - 1, int(pts[:, 1].max()) + pad)
        
        if max_x <= min_x or max_y <= min_y:
            return
        
        # Create stroke accumulation mask for the bounding region
        stroke_height = max_y - min_y + 1
        stroke_width_px = max_x - min_x + 1
        stroke_mask = np.zeros((stroke_height, stroke_width_px), dtype=np.float32)
        
        # Pre-compute brush kernel
        half_w = width // 2 + 1
        by, bx = np.ogrid[-half_w:half_w+1, -half_w:half_w+1]
        dist = np.sqrt(bx**2 + by**2).astype(np.float32)
        
        # Create brush with optional anti-aliasing
        if use_aa:
            # Anti-aliased soft brush with smooth falloff
            brush = np.clip(1.0 - (dist - width/2.0 + 0.5) / 1.5, 0.0, 1.0)
        else:
            # Hard edge brush (no anti-aliasing)
            brush = (dist <= width/2.0).astype(np.float32)
        brush_h, brush_w = brush.shape
        
        # Collect all stroke centerline points using Bresenham
        for i in range(len(points) - 1):
            x1, y1 = int(points[i][0]), int(points[i][1])
            x2, y2 = int(points[i + 1][0]), int(points[i + 1][1])
            
            dx, dy = abs(x2 - x1), abs(y2 - y1)
            sx, sy = (1 if x1 < x2 else -1), (1 if y1 < y2 else -1)
            err = dx - dy
            
            while True:
                # Stamp brush onto stroke_mask (local coords)
                lx, ly = x1 - min_x, y1 - min_y
                
                # Calculate brush bounds clipped to mask
                msy = max(0, ly - half_w)
                mey = min(stroke_height, ly + half_w + 1)
                msx = max(0, lx - half_w)
                mex = min(stroke_width_px, lx + half_w + 1)
                
                # Corresponding brush region
                bsy = msy - (ly - half_w)
                bey = bsy + (mey - msy)
                bsx = msx - (lx - half_w)
                bex = bsx + (mex - msx)
                
                if mey > msy and mex > msx:
                    # Max blend (accumulate)
                    stroke_mask[msy:mey, msx:mex] = np.maximum(
                        stroke_mask[msy:mey, msx:mex],
                        brush[bsy:bey, bsx:bex]
                    )
                
                if x1 == x2 and y1 == y2:
                    break
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x1 += sx
                if e2 < dx:
                    err += dx
                    y1 += sy
        
        # Apply accumulated stroke to pixels with blend mode
        alpha = color[3]
        final_alpha = stroke_mask * alpha
        
        dst = pixels[min_y:max_y+1, min_x:max_x+1]
        src_color = np.array([color[0], color[1], color[2], color[3]])
        
        # Calculate blended color based on blend mode
        blended = self._apply_blend_mode(dst, src_color, blend_mode)
        
        # Apply with accumulated alpha
        for c in range(4):
            pixels[min_y:max_y+1, min_x:max_x+1, c] = (
                dst[:, :, c] * (1 - final_alpha) +
                blended[:, :, c] * final_alpha
            )
    
    def _draw_line(self, pixels, p1, p2, color, width, blend_mode='MIX', use_aa=True):
        """Draw a single line segment."""
        self._draw_polyline(pixels, [p1, p2], color, width, blend_mode, use_aa)
    
    def _apply_blend_mode(self, dst, src_color, blend_mode):
        """Apply blend mode to destination RGB with source RGB. Alpha is kept separate."""
        import numpy as np
        
        # Split RGB and calculate
        d_rgb = dst[..., :3]
        s_rgb = np.ones_like(d_rgb) * src_color[:3]
        
        out_rgb = s_rgb.copy()
        
        if blend_mode == 'MIX':
            out_rgb = s_rgb
        elif blend_mode == 'DARKEN':
            out_rgb = np.minimum(d_rgb, s_rgb)
        elif blend_mode == 'MUL':
            out_rgb = d_rgb * s_rgb
        elif blend_mode == 'LIGHTEN':
            out_rgb = np.maximum(d_rgb, s_rgb)
        elif blend_mode == 'SCREEN':
            out_rgb = 1.0 - (1.0 - d_rgb) * (1.0 - s_rgb)
        elif blend_mode == 'ADD':
            out_rgb = np.clip(d_rgb + s_rgb, 0.0, 1.0)
        elif blend_mode == 'SUB':
            out_rgb = np.clip(d_rgb - s_rgb, 0.0, 1.0)
        elif blend_mode == 'OVERLAY':
            # Hard light if src < 0.5? No, Overlay is:
            # if dst < 0.5: 2 * dst * src
            # else: 1 - 2 * (1 - dst) * (1 - src)
            mask = d_rgb < 0.5
            out_rgb = np.where(mask, 
                               2.0 * d_rgb * s_rgb, 
                               1.0 - 2.0 * (1.0 - d_rgb) * (1.0 - s_rgb))
        elif blend_mode == 'DIFFERENCE':
            out_rgb = np.abs(d_rgb - s_rgb)
        elif blend_mode == 'EXCLUSION':
            out_rgb = d_rgb + s_rgb - 2.0 * d_rgb * s_rgb
        elif blend_mode == 'SOFT_LIGHT':
            # Pegtop formula
            out_rgb = (1.0 - 2.0 * s_rgb) * (d_rgb ** 2) + 2.0 * s_rgb * d_rgb
        elif blend_mode == 'HARD_LIGHT':
            # Overlay with swapped inputs
            mask = s_rgb < 0.5
            out_rgb = np.where(mask,
                               2.0 * s_rgb * d_rgb,
                               1.0 - 2.0 * (1.0 - s_rgb) * (1.0 - d_rgb))
        elif blend_mode == 'LINEAR_LIGHT':
            out_rgb = np.clip(d_rgb + 2.0 * s_rgb - 1.0, 0.0, 1.0)
        elif blend_mode == 'VIVID_LIGHT':
            # Color Burn / Color Dodge split
            out_rgb = np.where(s_rgb < 0.5,
                               1.0 - np.clip((1.0 - d_rgb) / (2.0 * s_rgb + 0.001), 0.0, 1.0),
                               np.clip(d_rgb / (2.0 * (1.0 - s_rgb) + 0.001), 0.0, 1.0))
        elif blend_mode == 'PIN_LIGHT':
            # Lighten / Darken split
            out_rgb = np.where(s_rgb < 0.5,
                               np.minimum(d_rgb, 2.0 * s_rgb),
                               np.maximum(d_rgb, 2.0 * s_rgb - 1.0))
        elif blend_mode == 'DIVIDE':
             out_rgb = np.clip(d_rgb / (s_rgb + 0.001), 0.0, 1.0)
             
        # Re-attach alpha channel (set to 1.0 so alpha compositing logic works correctly)
        # Using 1.0 ensures that 'src_alpha' controls the mix factor, not the color value itself
        alpha = np.ones(dst.shape[:2] + (1,), dtype=np.float32)
        
        return np.dstack((out_rgb, alpha))
    
    def _cleanup(self, context):
        global pen_points, pen_preview_pos, pen_is_dragging, pen_edit_point_idx, pen_edit_element
        global pen_displacing, pen_displace_start
        pen_points = []
        pen_preview_pos = None
        pen_is_dragging = False
        pen_edit_point_idx = None
        pen_edit_element = None
        pen_displacing = False
        pen_displace_start = None
        
        if self._draw_handler:
            bpy.types.SpaceImageEditor.draw_handler_remove(self._draw_handler, 'WINDOW')
            self._draw_handler = None
        context.area.header_text_set(None)
        context.area.tag_redraw()



# ----------------------------
# Pen Tool (3D Viewport)
# ----------------------------

class TEXTURE_PAINT_OT_pen_tool(Operator):
    bl_idname = "texture_paint.pen_tool"
    bl_label = "Pen Tool"
    bl_description = "Draw bezier paths on 3D surface"
    bl_options = {'REGISTER', 'UNDO'}
    
    _draw_handler = None
    _is_closed = False
    
    @classmethod
    def poll(cls, context):
        return (context.area.type == 'VIEW_3D' and 
                context.mode == 'PAINT_TEXTURE')
    
    def modal(self, context, event):
        global pen_points, pen_preview_pos, pen_is_dragging, pen_drag_handle
        context.area.tag_redraw()
        
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'NUMPAD_1', 'NUMPAD_2', 'NUMPAD_3', 'NUMPAD_4', 'NUMPAD_5', 'NUMPAD_6', 'NUMPAD_7', 'NUMPAD_8', 'NUMPAD_9'}:
            return {'PASS_THROUGH'}
        
        mx, my = event.mouse_region_x, event.mouse_region_y
        
        # Screen coordinates for 3D view are just (mx, my)
        # We store them directly
        
        pen_preview_pos = (mx, my)
        
        if event.type == 'MOUSEMOVE':
            if pen_is_dragging and len(pen_points) > 0:
                # Adjust handle of last point
                last_pt = pen_points[-1]
                if pen_drag_handle == 'OUT':
                    # Set handle out
                    pen_points[-1] = (last_pt[0], last_pt[1], last_pt[2], last_pt[3], mx, my)
                    # Mirror handle in
                    dx = mx - last_pt[0]
                    dy = my - last_pt[1]
                    pen_points[-1] = (last_pt[0], last_pt[1], last_pt[0] - dx, last_pt[1] - dy, mx, my)
            return {'RUNNING_MODAL'}
        
        elif event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                # Check if clicking near first point to close path
                if len(pen_points) >= 3:
                    first_pt = pen_points[0]
                    dist = ((mx - first_pt[0])**2 + (my - first_pt[1])**2)**0.5
                    if dist < 10:
                        # Close the path
                        self._is_closed = True
                        self._apply_path(context)
                        self._cleanup(context)
                        return {'FINISHED'}
                
                # Add new point
                pen_points.append((mx, my, mx, my, mx, my))
                pen_is_dragging = True
                pen_drag_handle = 'OUT'
                context.area.header_text_set(f"Point {len(pen_points)} | Drag to adjust curve | Enter to apply | ESC to cancel")
                return {'RUNNING_MODAL'}
            
            elif event.value == 'RELEASE':
                pen_is_dragging = False
                pen_drag_handle = None
                return {'RUNNING_MODAL'}
        
        elif event.type == 'BACK_SPACE' and event.value == 'PRESS':
            # Delete last point
            if len(pen_points) > 0:
                pen_points.pop()
                context.area.header_text_set(f"Point deleted | {len(pen_points)} points remaining")
            return {'RUNNING_MODAL'}
        
        elif event.type in {'RET', 'NUMPAD_ENTER', 'SPACE'} and event.value == 'PRESS':
            if len(pen_points) >= 2:
                self._apply_path(context)
            self._cleanup(context)
            return {'FINISHED'}
        
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            context.area.header_text_set(None)
            self._cleanup(context)
            return {'CANCELLED'}
        
        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        global pen_points, pen_preview_pos, pen_is_dragging
        
        pen_points = []
        pen_preview_pos = None
        pen_is_dragging = False
        self._is_closed = False
        
        # Add draw handler
        from .. import ui
        # We need a 3D specific draw handler because draw_pen_preview assumes image coords and does conversion
        # Use draw_pen_preview but we need to trick it or modify it?
        # Creating a new draw handler is cleaner.
        self._draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            ui.draw_pen_preview_3d, (context,), 'WINDOW', 'POST_PIXEL')
        
        context.window_manager.modal_handler_add(self)
        context.area.header_text_set("Click to add points | Drag to adjust curves | Enter/Space to apply | ESC to cancel")
        return {'RUNNING_MODAL'}

    def _cleanup(self, context):
        global pen_points, pen_preview_pos, pen_is_dragging
        pen_points = []
        pen_preview_pos = None
        pen_is_dragging = False
        
        if self._draw_handler:
            bpy.types.SpaceView3D.draw_handler_remove(self._draw_handler, 'WINDOW')
            self._draw_handler = None
        context.area.header_text_set(None)
        context.area.tag_redraw()

    def _apply_path(self, context):
        """Project the screen path to 3D surface and paint."""
        global pen_points
        if len(pen_points) < 2:
            return
        
        # Get active object and image
        obj = context.active_object
        if not obj:
            return
            
        # Try to find active image from material
        mat = obj.active_material
        if not mat:
            return
            
        nodes = [n for n in mat.node_tree.nodes if n.type == 'TEX_IMAGE' and n.image]
        active_node = nodes[0] if nodes else None
        for n in nodes:
            if n.select:
                active_node = n
                break
        
        if not active_node or not active_node.image:
            self.report({'WARNING'}, "No active image found on material")
            return
            
        image = active_node.image
        width, height = image.size
        
        # Save undo state
        utils.ImageUndoStack.get().push_state(image)
        
        import numpy as np
        
        # Get pixels
        num_pixels = width * height * 4
        pixels = np.zeros(num_pixels, dtype=np.float32)
        image.pixels.foreach_get(pixels)
        pixels = pixels.reshape((height, width, 4))
        
        # Generate dense screen points
        screen_points = self._generate_bezier_points(segments_per_curve=40)
        
        # Project to UV space
        uv_paths = []
        current_path = []
        
        last_uv = None
        
        # Reuse view3d_raycast_uv from TEXTURE_PAINT_OT_text_tool?
        # We can reproduce the logic here for simplicity or call it if we had an instance (we don't)
        
        for pt in screen_points:
            # Create a mock event for raycasting
            class MockEvent:
                def __init__(self, x, y):
                    self.mouse_region_x = x
                    self.mouse_region_y = y
            
            mock_evt = MockEvent(pt[0], pt[1])
            
            # Perform raycast
            hit_loc, face_idx, uv, _, _ = self.view3d_raycast_uv(context, mock_evt, obj)
            
            if uv:
                uv_pixel = (int(uv[0] * width), int(uv[1] * height))
                
                if last_uv:
                    # Check for seam (distance threshold)
                    # In UV space (0-1), a jump of > 0.1 is likely a seam or gap
                    dist = ((uv[0] - last_uv[0])**2 + (uv[1] - last_uv[1])**2)**0.5
                    if dist > 0.1:
                        # Start new path
                        if current_path:
                            uv_paths.append(current_path)
                            current_path = []
                
                current_path.append(uv_pixel)
                last_uv = uv
            else:
                # Gap in projection (off mesh)
                if current_path:
                    uv_paths.append(current_path)
                    current_path = []
                last_uv = None
        
        if current_path:
            uv_paths.append(current_path)
            
        # Draw strokes on image
        props = context.scene.text_tool_properties
        brush = context.tool_settings.image_paint.brush
        blend_mode = brush.blend if brush else 'MIX'
        use_aa = props.use_antialiasing
        
        if props.pen_use_stroke:
            stroke_color = np.array(props.pen_stroke_color, dtype=np.float32)
            stroke_width = props.pen_stroke_width
            
            # Helper to draw polyline (copied from Image Pen Tool)
            # We can use the method from IMAGE_PAINT_OT_pen_tool if we make it static or mixin
            # For now, let's duplicate the _draw_polyline helper or just define it here
            
            for path in uv_paths:
                if len(path) >= 2:
                    self._draw_polyline(pixels, path, stroke_color, stroke_width, width, height, blend_mode, use_aa)
        
        # Apply pixels
        image.pixels.foreach_set(pixels.flatten())
        image.update()

    def _generate_bezier_points(self, segments_per_curve=20):
        global pen_points
        points = []
        for i in range(len(pen_points) - 1):
            p0 = pen_points[i]
            p1 = pen_points[i + 1]
            x0, y0 = p0[0], p0[1]
            x1, y1 = p0[4], p0[5]
            x2, y2 = p1[2], p1[3]
            x3, y3 = p1[0], p1[1]
            for t in range(segments_per_curve + 1):
                t_val = t / segments_per_curve
                mt = 1 - t_val
                x = mt**3 * x0 + 3 * mt**2 * t_val * x1 + 3 * mt * t_val**2 * x2 + t_val**3 * x3
                y = mt**3 * y0 + 3 * mt**2 * t_val * y1 + 3 * mt * t_val**2 * y2 + t_val**3 * y3
                points.append((int(x), int(y)))
        return points

    def view3d_raycast_uv(self, context, event, obj):
        region = context.region
        rv3d = context.region_data
        if not rv3d:
            return None, None, None, None, None

        # Build ray from mouse in world space
        coord = (event.mouse_region_x, event.mouse_region_y)
        view_origin = region_2d_to_origin_3d(region, rv3d, coord)
        view_dir = region_2d_to_vector_3d(region, rv3d, coord).normalized()

        near = view_origin + view_dir * 0.001
        far = view_origin + view_dir * 1e6

        inv = obj.matrix_world.inverted()
        ro_local = inv @ near
        rf_local = inv @ far
        rd_local = (rf_local - ro_local).normalized()

        bm = bmesh.new()
        depsgraph = context.evaluated_depsgraph_get()
        bm.from_object(obj, depsgraph)
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()

        bvh = BVHTree.FromBMesh(bm)
        hit = bvh.ray_cast(ro_local, rd_local)
        
        if not hit or not hit[0]:
            bm.free()
            return None, None, None, None, None

        hit_loc_local, hit_normal_local, face_index, distance = hit
        
        uv_layer = bm.loops.layers.uv.active
        if not uv_layer:
            bm.free()
            return hit_loc_local, face_index, None, None, None

        face = bm.faces[face_index]
        p = hit_loc_local
        
        from mathutils.interpolate import poly_3d_calc
        
        vert_coords = [v.co for v in face.verts]
        loop_uvs = [loop[uv_layer].uv for loop in face.loops]
        
        weights = poly_3d_calc(vert_coords, p)
        
        u_interp = sum(w * uv.x for w, uv in zip(weights, loop_uvs))
        v_interp = sum(w * uv.y for w, uv in zip(weights, loop_uvs))
        best_uv = Vector((u_interp, v_interp))
        result_uv = (best_uv.x, best_uv.y)
        
        bm.free()
        return hit_loc_local, face_index, result_uv, None, None

    def _apply_blend_mode(self, dst, src_color, blend_mode):
        import numpy as np
        d_rgb = dst[..., :3]
        s_rgb = np.ones_like(d_rgb) * src_color[:3]
        out_rgb = s_rgb.copy()
        
        if blend_mode == 'MIX': out_rgb = s_rgb
        elif blend_mode == 'DARKEN': out_rgb = np.minimum(d_rgb, s_rgb)
        elif blend_mode == 'MUL': out_rgb = d_rgb * s_rgb
        elif blend_mode == 'LIGHTEN': out_rgb = np.maximum(d_rgb, s_rgb)
        elif blend_mode == 'SCREEN': out_rgb = 1.0 - (1.0 - d_rgb) * (1.0 - s_rgb)
        elif blend_mode == 'ADD': out_rgb = np.clip(d_rgb + s_rgb, 0.0, 1.0)
        elif blend_mode == 'SUB': out_rgb = np.clip(d_rgb - s_rgb, 0.0, 1.0)
        elif blend_mode == 'OVERLAY':
            mask = d_rgb < 0.5
            out_rgb = np.where(mask, 2.0 * d_rgb * s_rgb, 1.0 - 2.0 * (1.0 - d_rgb) * (1.0 - s_rgb))
        
        alpha = np.ones(dst.shape[:2] + (1,), dtype=np.float32)
        return np.dstack((out_rgb, alpha))

    def _draw_polyline(self, pixels, points, color, width, img_width, img_height, blend_mode='MIX', use_aa=True):
        import numpy as np
        if len(points) < 2: return
        
        pts = np.array(points)
        pad = width + 2
        min_x = max(0, int(pts[:, 0].min()) - pad)
        max_x = min(img_width - 1, int(pts[:, 0].max()) + pad)
        min_y = max(0, int(pts[:, 1].min()) - pad)
        max_y = min(img_height - 1, int(pts[:, 1].max()) + pad)
        
        if max_x <= min_x or max_y <= min_y: return
        
        stroke_height = max_y - min_y + 1
        stroke_width_px = max_x - min_x + 1
        stroke_mask = np.zeros((stroke_height, stroke_width_px), dtype=np.float32)
        
        half_w = width // 2 + 1
        by, bx = np.ogrid[-half_w:half_w+1, -half_w:half_w+1]
        dist = np.sqrt(bx**2 + by**2).astype(np.float32)
        
        if use_aa: brush = np.clip(1.0 - (dist - width/2.0 + 0.5) / 1.5, 0.0, 1.0)
        else: brush = (dist <= width/2.0).astype(np.float32)
        
        for i in range(len(points) - 1):
            x1, y1 = int(points[i][0]), int(points[i][1])
            x2, y2 = int(points[i + 1][0]), int(points[i + 1][1])
            dx, dy = abs(x2 - x1), abs(y2 - y1)
            sx, sy = (1 if x1 < x2 else -1), (1 if y1 < y2 else -1)
            err = dx - dy
            
            while True:
                lx, ly = x1 - min_x, y1 - min_y
                msy = max(0, ly - half_w)
                mey = min(stroke_height, ly + half_w + 1)
                msx = max(0, lx - half_w)
                mex = min(stroke_width_px, lx + half_w + 1)
                
                bsy = msy - (ly - half_w)
                bey = bsy + (mey - msy)
                bsx = msx - (lx - half_w)
                bex = bsx + (mex - msx)
                
                if mey > msy and mex > msx:
                    stroke_mask[msy:mey, msx:mex] = np.maximum(stroke_mask[msy:mey, msx:mex], brush[bsy:bey, bsx:bex])
                
                if x1 == x2 and y1 == y2: break
                e2 = 2 * err
                if e2 > -dy: err -= dy; x1 += sx
                if e2 < dx: err += dx; y1 += sy
        
        alpha = color[3]
        final_alpha = stroke_mask * alpha
        
        dst = pixels[min_y:max_y+1, min_x:max_x+1]
        src_color = np.array([color[0], color[1], color[2], color[3]])
        blended = self._apply_blend_mode(dst, src_color, blend_mode)
        
        for c in range(4):
            pixels[min_y:max_y+1, min_x:max_x+1, c] = dst[:, :, c] * (1 - final_alpha) + blended[:, :, c] * final_alpha
