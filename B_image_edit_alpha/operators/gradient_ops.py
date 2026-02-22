import math
import array

import bpy
import bmesh
from bpy.types import Operator
from bpy_extras.view3d_utils import (
    region_2d_to_origin_3d,
    region_2d_to_vector_3d,
)
from .. import utils
from .. import ui


# ----------------------------
# Gradient Tool Operators
# ----------------------------
class TEXTURE_PAINT_OT_gradient_tool(Operator):
    bl_idname = "paint.gradient_tool"
    bl_label = "Gradient Tool"
    bl_options = {'REGISTER', 'UNDO'}
    
    _draw_handler = None
    _start_pos = None
    _end_pos = None
    _is_dragging = False
    
    # Realtime preview state
    _image = None
    _original_pixels = None  # Store original for restore
    _cached_face_data = None  # Pre-computed face UV/screen data
    _width = 0
    _height = 0
    
    # Throttling for performance
    _last_update_time = 0.0
    _last_end_pos = None
    _update_interval = 0.016  # ~60 FPS max
    _min_pos_change = 3  # Minimum pixel change to trigger update
    
    @classmethod
    def poll(cls, context):
        return (context.mode == 'PAINT_TEXTURE' and
                context.active_object and
                context.active_object.type == 'MESH')
    
    def modal(self, context, event):
        context.area.tag_redraw()
        
        # Allow viewport navigation to pass through
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE'}:
            return {'PASS_THROUGH'}
        
        if event.type == 'MOUSEMOVE':
            if self._is_dragging and self._image:
                import time
                current_time = time.time()
                new_pos = (event.mouse_region_x, event.mouse_region_y)
                
                # Apply 45-degree snapping when Shift is held
                if event.shift and self._start_pos:
                    dx = new_pos[0] - self._start_pos[0]
                    dy = new_pos[1] - self._start_pos[1]
                    distance = math.sqrt(dx * dx + dy * dy)
                    if distance > 0:
                        angle = math.atan2(dy, dx)
                        # Snap to nearest 45 degrees (pi/4)
                        snap_angle = round(angle / (math.pi / 4)) * (math.pi / 4)
                        new_pos = (
                            self._start_pos[0] + int(distance * math.cos(snap_angle)),
                            self._start_pos[1] + int(distance * math.sin(snap_angle))
                        )
                
                # Throttle updates
                time_ok = (current_time - self._last_update_time) >= self._update_interval
                
                # Check if position changed enough
                pos_ok = True
                if self._last_end_pos:
                    dx = abs(new_pos[0] - self._last_end_pos[0])
                    dy = abs(new_pos[1] - self._last_end_pos[1])
                    pos_ok = (dx >= self._min_pos_change or dy >= self._min_pos_change)
                
                if time_ok and pos_ok:
                    self._end_pos = new_pos
                    self._last_end_pos = new_pos
                    self._last_update_time = current_time
                    utils.gradient_preview_start = self._start_pos
                    utils.gradient_preview_end = self._end_pos
                    self._apply_gradient_realtime(context)
            return {'RUNNING_MODAL'}
        
        elif event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                # Start dragging - initialize state
                self._is_dragging = True
                self._start_pos = (event.mouse_region_x, event.mouse_region_y)
                self._end_pos = self._start_pos
                utils.gradient_preview_start = self._start_pos
                utils.gradient_preview_end = self._end_pos
                # Initialize realtime state
                if not self._init_realtime_state(context):
                    self._is_dragging = False
                    self.remove_handler(context)
                    return {'CANCELLED'}
                return {'RUNNING_MODAL'}
            
            elif event.value == 'RELEASE' and self._is_dragging:
                # Final apply
                self._end_pos = (event.mouse_region_x, event.mouse_region_y)
                self._apply_gradient_realtime(context)
                # Save undo state
                if self._image and self._original_pixels:
                    utils.ImageUndoStack.get().push_state_from_array(self._image, self._original_pixels)
                self._cleanup_state()
                utils.gradient_preview_start = None
                utils.gradient_preview_end = None
                self.remove_handler(context)
                self.report({'INFO'}, "Gradient applied.")
                return {'FINISHED'}
        
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            # Cancel - restore original
            if self._image and self._original_pixels:
                self._image.pixels.foreach_set(self._original_pixels)
                utils.force_texture_refresh(context, self._image)
            self._cleanup_state()
            utils.gradient_preview_start = None
            utils.gradient_preview_end = None
            self.remove_handler(context)
            return {'CANCELLED'}
        
        return {'RUNNING_MODAL'}
    
    def _cleanup_state(self):
        self._is_dragging = False
        self._image = None
        self._original_pixels = None
        self._cached_face_data = None
    
    def _init_realtime_state(self, context):
        """Initialize state for realtime preview: cache image, pixels, and face data."""
        from bpy_extras.view3d_utils import location_3d_to_region_2d
        
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            return False
        
        mat = obj.active_material
        if not mat or not mat.use_nodes:
            return False
        
        # Find active image node
        image_node = None
        for node in mat.node_tree.nodes:
            if node.type == 'TEX_IMAGE' and node.select:
                image_node = node
                break
        if not image_node:
            for node in mat.node_tree.nodes:
                if node.type == 'TEX_IMAGE':
                    image_node = node
                    break
        
        if not image_node or not image_node.image:
            return False
        
        self._image = image_node.image
        self._width, self._height = self._image.size
        
        if self._width == 0 or self._height == 0:
            return False
        
        # Store original pixels
        num_pixels = self._width * self._height * 4
        self._original_pixels = array.array('f', [0.0] * num_pixels)
        self._image.pixels.foreach_get(self._original_pixels)
        
        # Pre-compute face UV/screen data
        region = context.region
        rv3d = context.region_data
        mat_world = obj.matrix_world
        
        bm = bmesh.new()
        depsgraph = context.evaluated_depsgraph_get()
        bm.from_object(obj, depsgraph)
        bm.faces.ensure_lookup_table()
        
        uv_layer = bm.loops.layers.uv.active
        if not uv_layer:
            bm.free()
            return False
        
        self._cached_face_data = []
        
        for face in bm.faces:
            loops = list(face.loops)
            n_verts = len(loops)
            if n_verts < 3:
                continue
            
            # Get UV and screen coords for each vertex
            face_data = []
            valid = True
            for loop in loops:
                uv = loop[uv_layer].uv
                world_pos = mat_world @ loop.vert.co
                screen_pos = location_3d_to_region_2d(region, rv3d, world_pos)
                if screen_pos is None:
                    valid = False
                    break
                face_data.append((uv.x, uv.y, screen_pos.x, screen_pos.y))
            
            if valid:
                self._cached_face_data.append(face_data)
        
        bm.free()
        return True
    
    def _apply_gradient_realtime(self, context):
        """Apply gradient using cached data - OPTIMIZED with NumPy per-triangle."""
        if not self._image or not self._original_pixels or not self._cached_face_data:
            return
        
        props = context.scene.text_tool_properties
        grad_node = utils.get_gradient_node()
        if not grad_node:
            return
        
        lut = utils.get_gradient_lut(grad_node)
        lut_len = len(lut)
        if lut_len < 2:
            return
        
        gradient_type = props.gradient_type
        is_linear = (gradient_type == 'LINEAR')
        width, height = self._width, self._height
        
        # Gradient parameters
        sx1, sy1 = self._start_pos
        sx2, sy2 = self._end_pos
        gdx = sx2 - sx1
        gdy = sy2 - sy1
        grad_len_sq = gdx * gdx + gdy * gdy
        if grad_len_sq < 1.0:
            grad_len_sq = 1.0
        grad_len = math.sqrt(grad_len_sq)
        
        # Try NumPy + multithreading for massive speedup
        try:
            import numpy as np
            from concurrent.futures import ThreadPoolExecutor
            import os
            
            # Convert LUT to numpy array
            lut_arr = np.array([(c[0], c[1], c[2], c[3] if len(c) > 3 else 1.0) for c in lut], dtype=np.float32)
            
            # Get original as numpy
            result = np.array(self._original_pixels, dtype=np.float32).reshape(height, width, 4)
            
            # Flatten triangles for batch processing
            all_triangles = []
            for face_data in self._cached_face_data:
                n_verts = len(face_data)
                for tri_idx in range(n_verts - 2):
                    all_triangles.append((face_data[0], face_data[tri_idx + 1], face_data[tri_idx + 2]))
            
            if not all_triangles:
                self._image.pixels.foreach_set(result.flatten())
                utils.force_texture_refresh(context, self._image)
                return
            
            # Number of threads
            num_threads = min(os.cpu_count() or 4, 8)
            batch_size = max(1, len(all_triangles) // num_threads)
            
            # Process triangles in batch
            def process_triangle_batch(triangles, local_result):
                """Process a batch of triangles."""
                for tri in triangles:
                    v0, v1, v2 = tri
                    u0, v0_uv, sx0, sy0 = v0
                    u1, v1_uv, sx_1, sy_1 = v1
                    u2, v2_uv, sx_2, sy_2 = v2
                    
                    uv_min_u = min(u0, u1, u2)
                    uv_max_u = max(u0, u1, u2)
                    uv_min_v = min(v0_uv, v1_uv, v2_uv)
                    uv_max_v = max(v0_uv, v1_uv, v2_uv)
                    
                    px_min = max(0, int(uv_min_u * width))
                    px_max = min(width, int(uv_max_u * width) + 1)
                    py_min = max(0, int(uv_min_v * height))
                    py_max = min(height, int(uv_max_v * height) + 1)
                    
                    if px_max <= px_min or py_max <= py_min:
                        continue
                    
                    denom = (v1_uv - v2_uv) * (u0 - u2) + (u2 - u1) * (v0_uv - v2_uv)
                    if abs(denom) < 0.0001:
                        continue
                    inv_denom = 1.0 / denom
                    
                    # Create coordinate grids
                    py_range = np.arange(py_min, py_max, dtype=np.float32)
                    px_range = np.arange(px_min, px_max, dtype=np.float32)
                    py_grid, px_grid = np.meshgrid(py_range, px_range, indexing='ij')
                    
                    tex_u = (px_grid + 0.5) / width
                    tex_v = (py_grid + 0.5) / height
                    
                    w0 = ((v1_uv - v2_uv) * (tex_u - u2) + (u2 - u1) * (tex_v - v2_uv)) * inv_denom
                    w1 = ((v2_uv - v0_uv) * (tex_u - u2) + (u0 - u2) * (tex_v - v2_uv)) * inv_denom
                    w2 = 1.0 - w0 - w1
                    
                    inside = (w0 >= -0.001) & (w1 >= -0.001) & (w2 >= -0.001)
                    if not np.any(inside):
                        continue
                    
                    sx = w0 * sx0 + w1 * sx_1 + w2 * sx_2
                    sy = w0 * sy0 + w1 * sy_1 + w2 * sy_2
                    
                    if is_linear:
                        t = ((sx - sx1) * gdx + (sy - sy1) * gdy) / grad_len_sq
                    else:
                        t = np.sqrt((sx - sx1)**2 + (sy - sy1)**2) / grad_len
                    
                    t = np.clip(t, 0.0, 1.0)
                    lut_indices = (t * (lut_len - 1)).astype(np.int32)
                    colors = lut_arr[lut_indices]
                    
                    alpha = colors[:, :, 3:4]
                    region_orig = local_result[py_min:py_max, px_min:px_max, :].copy()
                    
                    blended = np.zeros_like(region_orig)
                    blended[:, :, :3] = colors[:, :, :3] * alpha + region_orig[:, :, :3] * (1.0 - alpha)
                    blended[:, :, 3:4] = alpha + region_orig[:, :, 3:4] * (1.0 - alpha)
                    
                    inside_3d = inside[:, :, np.newaxis]
                    local_result[py_min:py_max, px_min:px_max, :] = np.where(inside_3d, blended, region_orig)
            
            # For thread safety, process all triangles serially but with NumPy vectorization
            # Threading for triangle batches can cause race conditions on overlapping regions
            # Instead, we'll use the optimized NumPy code which is already fast
            process_triangle_batch(all_triangles, result)
            
            # Set pixels
            self._image.pixels.foreach_set(result.flatten())
            utils.force_texture_refresh(context, self._image)
            return
            
        except ImportError:
            pass  # Fall back to Python
        
        # Fallback: Python loop (slower)
        lut_cache = [(c[0], c[1], c[2], c[3] if len(c) > 3 else 1.0) for c in lut]
        lut_max_idx = lut_len - 1
        base = array.array('f', self._original_pixels)
        
        for face_data in self._cached_face_data:
            n_verts = len(face_data)
            
            for tri_idx in range(n_verts - 2):
                v0 = face_data[0]
                v1 = face_data[tri_idx + 1]
                v2 = face_data[tri_idx + 2]
                
                u0, v0_uv, sx0, sy0 = v0
                u1, v1_uv, sx_1, sy_1 = v1
                u2, v2_uv, sx_2, sy_2 = v2
                
                uv_min_u = min(u0, u1, u2)
                uv_max_u = max(u0, u1, u2)
                uv_min_v = min(v0_uv, v1_uv, v2_uv)
                uv_max_v = max(v0_uv, v1_uv, v2_uv)
                
                px_min_x = max(0, int(uv_min_u * width))
                px_max_x = min(width, int(uv_max_u * width) + 1)
                px_min_y = max(0, int(uv_min_v * height))
                px_max_y = min(height, int(uv_max_v * height) + 1)
                
                denom = (v1_uv - v2_uv) * (u0 - u2) + (u2 - u1) * (v0_uv - v2_uv)
                if abs(denom) < 0.0001:
                    continue
                inv_denom = 1.0 / denom
                
                for py in range(px_min_y, px_max_y):
                    row_offset = py * width * 4
                    for px in range(px_min_x, px_max_x):
                        tex_u = (px + 0.5) / width
                        tex_v = (py + 0.5) / height
                        
                        w0 = ((v1_uv - v2_uv) * (tex_u - u2) + (u2 - u1) * (tex_v - v2_uv)) * inv_denom
                        w1 = ((v2_uv - v0_uv) * (tex_u - u2) + (u0 - u2) * (tex_v - v2_uv)) * inv_denom
                        w2 = 1.0 - w0 - w1
                        
                        if w0 < -0.001 or w1 < -0.001 or w2 < -0.001:
                            continue
                        
                        sx = w0 * sx0 + w1 * sx_1 + w2 * sx_2
                        sy = w0 * sy0 + w1 * sy_1 + w2 * sy_2
                        
                        if is_linear:
                            t = ((sx - sx1) * gdx + (sy - sy1) * gdy) / grad_len_sq
                        else:
                            t = math.sqrt((sx - sx1)**2 + (sy - sy1)**2) / grad_len
                        
                        if t < 0.0: t = 0.0
                        elif t > 1.0: t = 1.0
                        
                        color = lut_cache[int(t * lut_max_idx)]
                        
                        b_idx = row_offset + px * 4
                        ta = color[3]
                        inv_ta = 1.0 - ta
                        
                        base[b_idx]   = color[0] * ta + base[b_idx] * inv_ta
                        base[b_idx+1] = color[1] * ta + base[b_idx+1] * inv_ta
                        base[b_idx+2] = color[2] * ta + base[b_idx+2] * inv_ta
                        base[b_idx+3] = ta + base[b_idx+3] * inv_ta
        
        self._image.pixels.foreach_set(base)
        utils.force_texture_refresh(context, self._image)
    
    def invoke(self, context, event):
        if context.area.type == 'VIEW_3D':
            args = ()
            self._draw_handler = bpy.types.SpaceView3D.draw_handler_add(
                ui.draw_gradient_preview_3d, args, 'WINDOW', 'POST_PIXEL')
            
            # Start drag immediately on first click
            self._is_dragging = True
            self._start_pos = (event.mouse_region_x, event.mouse_region_y)
            self._end_pos = self._start_pos
            utils.gradient_preview_start = self._start_pos
            utils.gradient_preview_end = self._end_pos
            
            # Initialize realtime state
            if not self._init_realtime_state(context):
                self._is_dragging = False
                self.remove_handler(context)
                return {'CANCELLED'}
            
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            self.report({'WARNING'}, "View3D not found, cannot run operator")
            return {'CANCELLED'}
    
    def remove_handler(self, context):
        if self._draw_handler:
            bpy.types.SpaceView3D.draw_handler_remove(self._draw_handler, 'WINDOW')
            self._draw_handler = None
        context.area.tag_redraw()
    
    def apply_gradient(self, context, event):
        """Apply gradient to the texture based on start/end positions.
        
        OPTIMIZED: Pre-compute screen positions for face corners and interpolate.
        """
        from bpy_extras.view3d_utils import location_3d_to_region_2d
        
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            return False
        
        mat = obj.active_material
        if not mat or not mat.use_nodes:
            self.report({'WARNING'}, "No active material with nodes found")
            return False
        
        # Find active image node
        image_node = None
        for node in mat.node_tree.nodes:
            if node.type == 'TEX_IMAGE' and node.select:
                image_node = node
                break
        if not image_node:
            for node in mat.node_tree.nodes:
                if node.type == 'TEX_IMAGE':
                    image_node = node
                    break
        
        if not image_node or not image_node.image:
            self.report({'WARNING'}, "No active image texture found")
            return False
        
        image = image_node.image
        props = context.scene.text_tool_properties
        width, height = image.size
        
        if width == 0 or height == 0:
            return False
        
        # Get gradient LUT
        grad_node = utils.get_gradient_node()
        if not grad_node:
            self.report({'WARNING'}, "No gradient color ramp found")
            return False
        
        lut = utils.get_gradient_lut(grad_node)
        lut_len = len(lut)
        if lut_len < 2:
            return False
        
        gradient_type = props.gradient_type
        
        # Save undo state
        utils.ImageUndoStack.get().push_state(image)
        
        # Get pixel buffer
        num_pixels = width * height * 4
        base = array.array('f', [0.0] * num_pixels)
        image.pixels.foreach_get(base)
        
        # Get blend mode
        blend_mode = 'MIX'
        if context.tool_settings.image_paint.brush:
            blend_mode = context.tool_settings.image_paint.brush.blend
        
        # Screen-space gradient parameters
        sx1, sy1 = self._start_pos
        sx2, sy2 = self._end_pos
        
        # Gradient vector
        gdx = sx2 - sx1
        gdy = sy2 - sy1
        grad_len_sq = gdx * gdx + gdy * gdy
        if grad_len_sq < 1.0:
            grad_len_sq = 1.0
        grad_len = math.sqrt(grad_len_sq)
        
        region = context.region
        rv3d = context.region_data
        mat_world = obj.matrix_world
        
        # Build BMesh for UV lookup
        bm = bmesh.new()
        depsgraph = context.evaluated_depsgraph_get()
        bm.from_object(obj, depsgraph)
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        
        uv_layer = bm.loops.layers.uv.active
        if not uv_layer:
            bm.free()
            self.report({'WARNING'}, "No active UV layer found")
            return False
        
        # Pre-cache LUT for faster access
        lut_cache = [(c[0], c[1], c[2], c[3] if len(c) > 3 else 1.0) for c in lut]
        
        # OPTIMIZED: Process triangulated faces with corner screen projection
        for face in bm.faces:
            loops = list(face.loops)
            n_verts = len(loops)
            
            # Get UV and screen coords for each vertex
            face_data = []
            valid = True
            for loop in loops:
                uv = loop[uv_layer].uv
                world_pos = mat_world @ loop.vert.co
                screen_pos = location_3d_to_region_2d(region, rv3d, world_pos)
                if screen_pos is None:
                    valid = False
                    break
                face_data.append((uv.x, uv.y, screen_pos.x, screen_pos.y))
            
            if not valid or n_verts < 3:
                continue
            
            # Triangulate the face for processing
            for tri_idx in range(n_verts - 2):
                v0 = face_data[0]
                v1 = face_data[tri_idx + 1]
                v2 = face_data[tri_idx + 2]
                
                u0, v0_uv, sx0, sy0 = v0
                u1, v1_uv, sx_1, sy_1 = v1
                u2, v2_uv, sx_2, sy_2 = v2
                
                # UV bounding box
                uv_min_u = min(u0, u1, u2)
                uv_max_u = max(u0, u1, u2)
                uv_min_v = min(v0_uv, v1_uv, v2_uv)
                uv_max_v = max(v0_uv, v1_uv, v2_uv)
                
                px_min_x = max(0, int(uv_min_u * width))
                px_max_x = min(width, int(uv_max_u * width) + 1)
                px_min_y = max(0, int(uv_min_v * height))
                px_max_y = min(height, int(uv_max_v * height) + 1)
                
                # Barycentric edge vectors
                denom = (v1_uv - v2_uv) * (u0 - u2) + (u2 - u1) * (v0_uv - v2_uv)
                if abs(denom) < 0.0001:
                    continue
                inv_denom = 1.0 / denom
                
                for py in range(px_min_y, px_max_y):
                    for px in range(px_min_x, px_max_x):
                        tex_u = (px + 0.5) / width
                        tex_v = (py + 0.5) / height
                        
                        # Barycentric coordinates
                        w0 = ((v1_uv - v2_uv) * (tex_u - u2) + (u2 - u1) * (tex_v - v2_uv)) * inv_denom
                        w1 = ((v2_uv - v0_uv) * (tex_u - u2) + (u0 - u2) * (tex_v - v2_uv)) * inv_denom
                        w2 = 1.0 - w0 - w1
                        
                        # Outside triangle
                        if w0 < -0.001 or w1 < -0.001 or w2 < -0.001:
                            continue
                        
                        # Interpolate screen position (MUCH faster than projection)
                        sx = w0 * sx0 + w1 * sx_1 + w2 * sx_2
                        sy = w0 * sy0 + w1 * sy_1 + w2 * sy_2
                        
                        # Calculate gradient factor
                        if gradient_type == 'LINEAR':
                            dx = sx - sx1
                            dy = sy - sy1
                            t = (dx * gdx + dy * gdy) / grad_len_sq
                        else:  # RADIAL
                            dx = sx - sx1
                            dy = sy - sy1
                            t = math.sqrt(dx * dx + dy * dy) / grad_len
                        
                        t = max(0.0, min(1.0, t))
                        
                        # Sample LUT
                        color = lut_cache[int(t * (lut_len - 1))]
                        
                        # Apply gradient
                        b_idx = (py * width + px) * 4
                        dr, dg, db, da = base[b_idx], base[b_idx+1], base[b_idx+2], base[b_idx+3]
                        
                        ta = color[3]
                        inv_ta = 1.0 - ta
                        
                        if blend_mode == 'MIX':
                            base[b_idx]   = color[0] * ta + dr * inv_ta
                            base[b_idx+1] = color[1] * ta + dg * inv_ta
                            base[b_idx+2] = color[2] * ta + db * inv_ta
                            base[b_idx+3] = ta + da * inv_ta
                        else:
                            utils.blend_pixel(base, b_idx, color[0], color[1], color[2], ta, blend_mode)
        
        bm.free()
        image.pixels.foreach_set(base)
        utils.force_texture_refresh(context, image)
        return True


class IMAGE_PAINT_OT_gradient_tool(Operator):
    bl_idname = "image_paint.gradient_tool"
    bl_label = "Image Gradient Tool"
    bl_options = {'REGISTER', 'UNDO'}
    
    _draw_handler = None
    _start_pos = None
    _end_pos = None
    _is_dragging = False
    
    # Realtime preview state
    _image = None
    _original_pixels = None
    _width = 0
    _height = 0
    
    # Throttling for performance
    _last_update_time = 0.0
    _last_end_pos = None
    _update_interval = 0.016  # ~60 FPS max
    _min_pos_change = 3  # Minimum pixel change to trigger update
    
    @classmethod
    def poll(cls, context):
        sima = context.space_data
        return (context.area.type == 'IMAGE_EDITOR' and sima.mode == 'PAINT' and sima.image is not None)
    
    def modal(self, context, event):
        context.area.tag_redraw()
        
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE'}:
            return {'PASS_THROUGH'}
        
        if event.type == 'MOUSEMOVE':
            if self._is_dragging and self._image:
                import time
                current_time = time.time()
                new_pos = (event.mouse_region_x, event.mouse_region_y)
                
                # Apply 45-degree snapping when Shift is held
                if event.shift and self._start_pos:
                    dx = new_pos[0] - self._start_pos[0]
                    dy = new_pos[1] - self._start_pos[1]
                    distance = math.sqrt(dx * dx + dy * dy)
                    if distance > 0:
                        angle = math.atan2(dy, dx)
                        # Snap to nearest 45 degrees (pi/4)
                        snap_angle = round(angle / (math.pi / 4)) * (math.pi / 4)
                        new_pos = (
                            self._start_pos[0] + int(distance * math.cos(snap_angle)),
                            self._start_pos[1] + int(distance * math.sin(snap_angle))
                        )
                
                # Throttle updates
                time_ok = (current_time - self._last_update_time) >= self._update_interval
                
                # Check if position changed enough
                pos_ok = True
                if self._last_end_pos:
                    dx = abs(new_pos[0] - self._last_end_pos[0])
                    dy = abs(new_pos[1] - self._last_end_pos[1])
                    pos_ok = (dx >= self._min_pos_change or dy >= self._min_pos_change)
                
                if time_ok and pos_ok:
                    self._end_pos = new_pos
                    self._last_end_pos = new_pos
                    self._last_update_time = current_time
                    utils.gradient_preview_start = self._start_pos
                    utils.gradient_preview_end = self._end_pos
                    self._apply_gradient_realtime(context)
            return {'RUNNING_MODAL'}
        
        elif event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                self._is_dragging = True
                self._start_pos = (event.mouse_region_x, event.mouse_region_y)
                self._end_pos = self._start_pos
                utils.gradient_preview_start = self._start_pos
                utils.gradient_preview_end = self._end_pos
                # Initialize realtime state
                if not self._init_realtime_state(context):
                    self._is_dragging = False
                    self.remove_handler(context)
                    return {'CANCELLED'}
                return {'RUNNING_MODAL'}
            
            elif event.value == 'RELEASE' and self._is_dragging:
                self._end_pos = (event.mouse_region_x, event.mouse_region_y)
                self._apply_gradient_realtime(context)
                # Save undo state
                if self._image and self._original_pixels:
                    utils.ImageUndoStack.get().push_state_from_array(self._image, self._original_pixels)
                self._cleanup_state()
                utils.gradient_preview_start = None
                utils.gradient_preview_end = None
                self.remove_handler(context)
                self.report({'INFO'}, "Gradient applied.")
                return {'FINISHED'}
        
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            # Cancel - restore original
            if self._image and self._original_pixels:
                self._image.pixels.foreach_set(self._original_pixels)
                self._image.update()
            self._cleanup_state()
            utils.gradient_preview_start = None
            utils.gradient_preview_end = None
            self.remove_handler(context)
            return {'CANCELLED'}
        
        return {'RUNNING_MODAL'}
    
    def _cleanup_state(self):
        self._is_dragging = False
        self._image = None
        self._original_pixels = None
    
    def _init_realtime_state(self, context):
        """Initialize state for realtime preview."""
        sima = context.space_data
        self._image = sima.image
        if not self._image:
            return False
        
        self._width, self._height = self._image.size
        if self._width == 0 or self._height == 0:
            return False
        
        # Store original pixels
        num_pixels = self._width * self._height * 4
        self._original_pixels = array.array('f', [0.0] * num_pixels)
        self._image.pixels.foreach_get(self._original_pixels)
        return True
    
    def _apply_gradient_realtime(self, context):
        """Apply gradient - OPTIMIZED with NumPy vectorization."""
        if not self._image or not self._original_pixels:
            return
        
        props = context.scene.text_tool_properties
        grad_node = utils.get_gradient_node()
        if not grad_node:
            return
        
        lut = utils.get_gradient_lut(grad_node)
        lut_len = len(lut)
        if lut_len < 2:
            return
        
        gradient_type = props.gradient_type
        is_linear = (gradient_type == 'LINEAR')
        width, height = self._width, self._height
        
        # Convert screen positions to image coordinates
        region = context.region
        v1 = region.view2d.region_to_view(*self._start_pos)
        v2 = region.view2d.region_to_view(*self._end_pos)
        ix1, iy1 = v1[0] * width, v1[1] * height
        ix2, iy2 = v2[0] * width, v2[1] * height
        
        gdx, gdy = ix2 - ix1, iy2 - iy1
        grad_len_sq = gdx * gdx + gdy * gdy
        if grad_len_sq < 1.0:
            grad_len_sq = 1.0
        grad_len = math.sqrt(grad_len_sq)
        
        # Try NumPy + multithreading for massive speedup
        try:
            import numpy as np
            from concurrent.futures import ThreadPoolExecutor
            import os
            
            # Convert LUT to numpy array
            lut_arr = np.array([(c[0], c[1], c[2], c[3] if len(c) > 3 else 1.0) for c in lut], dtype=np.float32)
            
            # Get original as numpy
            orig = np.array(self._original_pixels, dtype=np.float32).reshape(height, width, 4)
            
            # Number of threads (use available CPUs)
            num_threads = min(os.cpu_count() or 4, 8)  # Cap at 8 threads
            strip_height = max(1, height // num_threads)
            
            # Result array
            final = np.zeros((height, width, 4), dtype=np.float32)
            
            def process_strip(start_y, end_y):
                """Process a horizontal strip of the image."""
                h = end_y - start_y
                w = width
                
                # Create coordinate grids for this strip
                py, px = np.mgrid[start_y:end_y, 0:w]
                dx = px.astype(np.float32) - ix1
                dy = py.astype(np.float32) - iy1
                
                # Calculate gradient factor
                if is_linear:
                    t = (dx * gdx + dy * gdy) / grad_len_sq
                else:
                    t = np.sqrt(dx * dx + dy * dy) / grad_len
                
                # Clamp
                t = np.clip(t, 0.0, 1.0)
                
                # Sample LUT
                lut_indices = (t * (lut_len - 1)).astype(np.int32)
                colors = lut_arr[lut_indices]
                
                # Get original strip
                orig_strip = orig[start_y:end_y, :, :]
                
                # Blend
                alpha = colors[:, :, 3:4]
                result_rgb = colors[:, :, :3] * alpha + orig_strip[:, :, :3] * (1.0 - alpha)
                result_alpha = alpha[:, :, 0] + orig_strip[:, :, 3] * (1.0 - alpha[:, :, 0])
                
                # Write to final
                final[start_y:end_y, :, :3] = result_rgb
                final[start_y:end_y, :, 3] = result_alpha
            
            # Process strips in parallel
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = []
                for i in range(num_threads):
                    start_y = i * strip_height
                    end_y = min((i + 1) * strip_height, height)
                    if i == num_threads - 1:
                        end_y = height  # Last strip gets remainder
                    if start_y < end_y:
                        futures.append(executor.submit(process_strip, start_y, end_y))
                
                # Wait for all to complete
                for f in futures:
                    f.result()
            
            # Set pixels
            self._image.pixels.foreach_set(final.flatten())
            self._image.update()
            return
            
        except ImportError:
            pass  # Fall back to Python
        
        # Fallback: Python loop (slower)
        lut_cache = [(c[0], c[1], c[2], c[3] if len(c) > 3 else 1.0) for c in lut]
        lut_max_idx = lut_len - 1
        base = array.array('f', self._original_pixels)
        
        for py in range(height):
            row_offset = py * width * 4
            dy = py - iy1
            
            for px in range(width):
                dx = px - ix1
                
                if is_linear:
                    t = (dx * gdx + dy * gdy) / grad_len_sq
                else:
                    t = math.sqrt(dx * dx + dy * dy) / grad_len
                
                if t < 0.0: t = 0.0
                elif t > 1.0: t = 1.0
                
                color = lut_cache[int(t * lut_max_idx)]
                
                b_idx = row_offset + px * 4
                ta = color[3]
                inv_ta = 1.0 - ta
                
                base[b_idx]   = color[0] * ta + base[b_idx] * inv_ta
                base[b_idx+1] = color[1] * ta + base[b_idx+1] * inv_ta
                base[b_idx+2] = color[2] * ta + base[b_idx+2] * inv_ta
                base[b_idx+3] = ta + base[b_idx+3] * inv_ta
        
        self._image.pixels.foreach_set(base)
        self._image.update()
    
    def invoke(self, context, event):
        if context.area.type == 'IMAGE_EDITOR':
            args = ()
            self._draw_handler = bpy.types.SpaceImageEditor.draw_handler_add(
                ui.draw_gradient_preview_image, args, 'WINDOW', 'POST_PIXEL')
            
            # Start drag immediately on first click
            self._is_dragging = True
            self._start_pos = (event.mouse_region_x, event.mouse_region_y)
            self._end_pos = self._start_pos
            utils.gradient_preview_start = self._start_pos
            utils.gradient_preview_end = self._end_pos
            
            # Initialize realtime state
            if not self._init_realtime_state(context):
                self._is_dragging = False
                self.remove_handler(context)
                return {'CANCELLED'}
            
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            self.report({'WARNING'}, "Image Editor not found")
            return {'CANCELLED'}
    
    def remove_handler(self, context):
        if self._draw_handler:
            bpy.types.SpaceImageEditor.draw_handler_remove(self._draw_handler, 'WINDOW')
            self._draw_handler = None
        context.area.tag_redraw()
