import math
import array

import bpy
from bpy.types import Operator
from .. import utils
from .. import ui

# NumPy for optimized array operations (optional)
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False





# ----------------------------
# Clone Tool Size/Strength Adjust Operators
# ----------------------------
class IMAGE_PAINT_OT_clone_adjust_size(Operator):
    bl_idname = "image_paint.clone_adjust_size"
    bl_label = "Adjust Clone Brush Size"
    bl_description = "Interactively adjust clone brush size"
    bl_options = {'REGISTER'}
    
    _start_pos = 0
    _start_value = 0
    
    @classmethod
    def poll(cls, context):
        return context.area.type == 'IMAGE_EDITOR'
    
    def modal(self, context, event):
        props = context.scene.text_tool_properties
        mx = event.mouse_region_x
        
        if event.type == 'MOUSEMOVE':
            delta = (mx - self._start_pos) * 0.5
            new_size = int(max(1, min(500, self._start_value + delta)))
            props.clone_brush_size = new_size
            context.area.header_text_set(f"Clone Brush Size: {new_size} | Click to confirm | ESC to cancel")
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}
        
        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            context.area.header_text_set(None)
            return {'FINISHED'}
        
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            props.clone_brush_size = self._start_value
            context.area.header_text_set(None)
            return {'CANCELLED'}
        
        return {'RUNNING_MODAL'}
    
    def invoke(self, context, event):
        props = context.scene.text_tool_properties
        self._start_pos = event.mouse_region_x
        self._start_value = props.clone_brush_size
        context.area.header_text_set(f"Clone Brush Size: {self._start_value} | Move mouse to adjust | Click to confirm")
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}


class IMAGE_PAINT_OT_clone_adjust_strength(Operator):
    bl_idname = "image_paint.clone_adjust_strength"
    bl_label = "Adjust Clone Brush Strength"
    bl_description = "Interactively adjust clone brush strength"
    bl_options = {'REGISTER'}
    
    _start_pos = 0
    _start_value = 0.0
    
    @classmethod
    def poll(cls, context):
        return context.area.type == 'IMAGE_EDITOR'
    
    def modal(self, context, event):
        props = context.scene.text_tool_properties
        mx = event.mouse_region_x
        
        if event.type == 'MOUSEMOVE':
            delta = (mx - self._start_pos) * 0.005
            new_strength = max(0.0, min(1.0, self._start_value + delta))
            props.clone_brush_strength = new_strength
            context.area.header_text_set(f"Clone Brush Strength: {new_strength:.2f} | Click to confirm | ESC to cancel")
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}
        
        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            context.area.header_text_set(None)
            return {'FINISHED'}
        
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            props.clone_brush_strength = self._start_value
            context.area.header_text_set(None)
            return {'CANCELLED'}
        
        return {'RUNNING_MODAL'}
    
    def invoke(self, context, event):
        props = context.scene.text_tool_properties
        self._start_pos = event.mouse_region_x
        self._start_value = props.clone_brush_strength
        context.area.header_text_set(f"Clone Brush Strength: {self._start_value:.2f} | Move mouse to adjust | Click to confirm")
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}


# ----------------------------
# Clone Tool (Image Editor)
# ----------------------------

class IMAGE_PAINT_OT_clone_set_source(Operator):
    bl_idname = "image_paint.clone_set_source"
    bl_label = "Set Clone Source"
    bl_description = "Set the source position for the clone tool"
    bl_options = {'REGISTER'}
    
    @classmethod
    def poll(cls, context):
        sima = context.space_data
        return (context.area.type == 'IMAGE_EDITOR' and 
                sima.mode == 'PAINT' and 
                sima.image is not None)
    
    def invoke(self, context, event):
        mx, my = event.mouse_region_x, event.mouse_region_y
        utils.clone_source_pos = (mx, my)
        utils.clone_source_set = True
        
        region = context.region
        view2d = region.view2d
        
        IMAGE_PAINT_OT_clone_tool._source_uv = view2d.region_to_view(mx, my)
        
        context.area.header_text_set("Source set! Click to paint | Ctrl+Click for new source | ESC to exit")
        context.area.tag_redraw()
        return {'FINISHED'}

class IMAGE_PAINT_OT_clone_tool(Operator):
    bl_idname = "image_paint.clone_tool"
    bl_label = "Image Clone Tool"
    bl_description = "Clone pixels from source to destination (Ctrl+Click to set source)"
    bl_options = {'REGISTER', 'UNDO'}
    
    _draw_handler = None
    _image = None
    _is_painting = False
    _source_offset = (0, 0)  # Offset from cursor to source in image pixels
    _source_uv = (0, 0)  # Source position in UV space
    _last_paint_pos = None
    _original_pixels = None
    _width = 0
    _height = 0
    _pixel_buffer = None  # Reusable numpy buffer to avoid per-frame allocation
    
    # Interactive adjustment state
    _adjust_mode = None  # 'SIZE' or 'STRENGTH'
    _adjust_start_pos = None
    _adjust_start_value = 0.0
    
    @classmethod
    def poll(cls, context):
        sima = context.space_data
        return (context.area.type == 'IMAGE_EDITOR' and 
                sima.mode == 'PAINT' and 
                sima.image is not None)
    
    def modal(self, context, event):
        context.area.tag_redraw()
        
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE'}:
            return {'PASS_THROUGH'}
        
        mx, my = event.mouse_region_x, event.mouse_region_y
        utils.clone_cursor_pos = (mx, my)
        
        # Handle adjustment mode FIRST (F/Shift+F size/strength adjustment)
        if self._adjust_mode:
            props = context.scene.text_tool_properties
            
            if event.type == 'MOUSEMOVE':
                delta = (mx - self._adjust_start_pos) * 0.5
                
                if self._adjust_mode == 'SIZE':
                    new_size = int(max(1, min(500, self._adjust_start_value + delta)))
                    props.clone_brush_size = new_size
                    context.area.header_text_set(f"Size: {new_size} | Click to confirm | ESC to cancel")
                elif self._adjust_mode == 'STRENGTH':
                    new_strength = max(0.0, min(1.0, self._adjust_start_value + delta * 0.01))
                    props.clone_brush_strength = new_strength
                    context.area.header_text_set(f"Strength: {new_strength:.2f} | Click to confirm | ESC to cancel")
                return {'RUNNING_MODAL'}
            
            elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
                # Confirm adjustment
                self._adjust_mode = None
                self._adjust_start_pos = None
                if utils.clone_source_set:
                    context.area.header_text_set("Source set - Click to paint | F for size | Shift+F for strength")
                else:
                    context.area.header_text_set("Ctrl+Click to set source | F for size | Shift+F for strength")
                return {'RUNNING_MODAL'}
            
            elif event.type in {'RIGHTMOUSE', 'ESC'} and event.value == 'PRESS':
                # Cancel adjustment - restore original value
                if self._adjust_mode == 'SIZE':
                    props.clone_brush_size = int(self._adjust_start_value)
                elif self._adjust_mode == 'STRENGTH':
                    props.clone_brush_strength = self._adjust_start_value
                self._adjust_mode = None
                self._adjust_start_pos = None
                if utils.clone_source_set:
                    context.area.header_text_set("Source set - Click to paint")
                else:
                    context.area.header_text_set("Ctrl+Click to set source first")
                return {'RUNNING_MODAL'}
            
            return {'RUNNING_MODAL'}
        
        # Handle F key for size adjustment, Shift+F for strength
        if event.type == 'F' and event.value == 'PRESS':
            props = context.scene.text_tool_properties
            if event.shift:
                # Shift+F: Adjust strength
                self._adjust_mode = 'STRENGTH'
                self._adjust_start_pos = mx
                self._adjust_start_value = props.clone_brush_strength
                context.area.header_text_set(f"Strength: {props.clone_brush_strength:.2f} | Move mouse to adjust | Click to confirm")
            else:
                # F: Adjust size
                self._adjust_mode = 'SIZE'
                self._adjust_start_pos = mx
                self._adjust_start_value = props.clone_brush_size
                context.area.header_text_set(f"Size: {props.clone_brush_size} | Move mouse to adjust | Click to confirm")
            return {'RUNNING_MODAL'}
        
        # Update source crosshair position if painting
        if self._is_painting and utils.clone_source_set:
            region = context.region
            view2d = region.view2d
            uv_cursor = view2d.region_to_view(mx, my)
            source_uv_x = uv_cursor[0] + self._source_offset[0]
            source_uv_y = uv_cursor[1] + self._source_offset[1]
            source_screen = view2d.view_to_region(source_uv_x, source_uv_y, clip=False)
            utils.clone_source_pos = source_screen
        
        if event.type == 'MOUSEMOVE':
            if self._is_painting and utils.clone_source_set:
                self._paint_clone(context, mx, my)
            return {'RUNNING_MODAL'}
        
        elif event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                if event.ctrl:
                    # Set source point
                    utils.clone_source_pos = (mx, my)
                    utils.clone_source_set = True
                    
                    # Store source position in UV space for offset calculation
                    region = context.region
                    view2d = region.view2d
                    self._source_uv = view2d.region_to_view(mx, my)
                    context.area.header_text_set("Source set - Click to paint | F for size | Shift+F for strength")
                    return {'RUNNING_MODAL'}
                else:
                    if utils.clone_source_set:
                        # Start painting
                        self._is_painting = True
                        self._last_paint_pos = None
                        
                        # Calculate offset in UV space
                        region = context.region
                        view2d = region.view2d
                        cursor_uv = view2d.region_to_view(mx, my)
                        self._source_offset = (
                            self._source_uv[0] - cursor_uv[0],
                            self._source_uv[1] - cursor_uv[1]
                        )
                        
                        # Save undo state
                        utils.ImageUndoStack.get().push_state(self._image)
                        
                        # Paint first dab
                        self._paint_clone(context, mx, my)
                        context.area.header_text_set("Painting... Release to finish")
                    else:
                        context.area.header_text_set("Ctrl+Click to set source first | F for size | Shift+F for strength")
                    return {'RUNNING_MODAL'}
            
            elif event.value == 'RELEASE' and self._is_painting:
                self._is_painting = False
                self._last_paint_pos = None
                context.area.header_text_set("Source set - Click to paint | F for size | Shift+F for strength | ESC to exit")
                return {'RUNNING_MODAL'}
        
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            context.area.header_text_set(None)
            self._cleanup(context)
            return {'FINISHED'}
        
        return {'RUNNING_MODAL'}
    
    def invoke(self, context, event):
        if context.area.type == 'IMAGE_EDITOR':
            self._draw_handler = bpy.types.SpaceImageEditor.draw_handler_add(
                ui.draw_clone_preview_image, (), 'WINDOW', 'POST_PIXEL')
            
            self._image = context.space_data.image
            self._width, self._height = self._image.size
            
            # Read pixels once
            num_pixels = self._width * self._height * 4
            self._original_pixels = array.array('f', [0.0] * num_pixels)
            self._image.pixels.foreach_get(self._original_pixels)
            
            mx, my = event.mouse_region_x, event.mouse_region_y
            utils.clone_cursor_pos = (mx, my)
            
            # If invoked with Ctrl held (from keymap), auto-set source
            if event.ctrl:
                utils.clone_source_pos = (mx, my)
                utils.clone_source_set = True
                region = context.region
                view2d = region.view2d
                self._source_uv = view2d.region_to_view(mx, my)
                context.area.header_text_set("Source set! Click to paint | Ctrl+Click for new source | ESC to exit")
            elif utils.clone_source_set:
                context.area.header_text_set("Click to paint | Ctrl+Click to set new source | ESC to exit")
            else:
                context.area.header_text_set("Ctrl+Click to set source")
            
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            self.report({'WARNING'}, "Image Editor not found")
            return {'CANCELLED'}
    
    def _cleanup(self, context):
        if self._draw_handler:
            bpy.types.SpaceImageEditor.draw_handler_remove(self._draw_handler, 'WINDOW')
            self._draw_handler = None
        utils.clone_cursor_pos = None
        self._image = None
        self._original_pixels = None
        self._pixel_buffer = None  # Free buffer memory
        context.area.tag_redraw()
    
    def _paint_clone(self, context, mx, my):
        """Paint cloned pixels at the cursor position - OPTIMIZED with NumPy."""
        if not self._image or not utils.clone_source_set:
            return
        
        props = context.scene.text_tool_properties
        brush_size = props.clone_brush_size
        falloff_preset = props.clone_falloff_preset
        strength = props.clone_brush_strength
        
        region = context.region
        view2d = region.view2d
        
        # Convert cursor to image coordinates
        cursor_uv = view2d.region_to_view(mx, my)
        cursor_px_x = int(cursor_uv[0] * self._width)
        cursor_px_y = int(cursor_uv[1] * self._height)
        
        # Source position in image coordinates
        source_uv = (cursor_uv[0] + self._source_offset[0], cursor_uv[1] + self._source_offset[1])
        source_px_x = int(source_uv[0] * self._width)
        source_px_y = int(source_uv[1] * self._height)
        
        if HAS_NUMPY:
            # Get pixels as numpy array - OPTIMIZED: reuse buffer
            num_pixels = self._width * self._height * 4
            if self._pixel_buffer is None or self._pixel_buffer.size != num_pixels:
                self._pixel_buffer = np.zeros(num_pixels, dtype=np.float32)
            self._image.pixels.foreach_get(self._pixel_buffer)
            pixels = self._pixel_buffer.reshape((self._height, self._width, 4))
            
            # Calculate brush region bounds
            x_min = max(0, cursor_px_x - brush_size)
            x_max = min(self._width, cursor_px_x + brush_size + 1)
            y_min = max(0, cursor_px_y - brush_size)
            y_max = min(self._height, cursor_px_y + brush_size + 1)
            
            # Source region bounds
            src_x_min = source_px_x - cursor_px_x + x_min
            src_x_max = source_px_x - cursor_px_x + x_max
            src_y_min = source_px_y - cursor_px_y + y_min
            src_y_max = source_px_y - cursor_px_y + y_max
            
            # Clip source bounds
            if src_x_min < 0:
                x_min -= src_x_min
                src_x_min = 0
            if src_y_min < 0:
                y_min -= src_y_min
                src_y_min = 0
            if src_x_max > self._width:
                x_max -= (src_x_max - self._width)
                src_x_max = self._width
            if src_y_max > self._height:
                y_max -= (src_y_max - self._height)
                src_y_max = self._height
            
            if x_max <= x_min or y_max <= y_min:
                return
            
            # Create coordinate grids for the brush region
            y_coords = np.arange(y_min, y_max)
            x_coords = np.arange(x_min, x_max)
            yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')
            
            # Calculate distance from cursor center (normalized 0-1)
            dist = np.sqrt((xx - cursor_px_x)**2 + (yy - cursor_px_y)**2)
            
            # Create brush mask
            mask = dist <= brush_size
            
            # Normalize distance to 0-1 range
            t = np.clip(dist / brush_size, 0.0, 1.0) if brush_size > 0 else np.zeros_like(dist)
            
            # Calculate falloff based on preset
            if falloff_preset == 'SMOOTH':
                # Smooth: 3t² - 2t³ (smoothstep)
                falloff = 1.0 - (3.0 * t**2 - 2.0 * t**3)
            elif falloff_preset == 'SMOOTHER':
                # Smoother: 6t⁵ - 15t⁴ + 10t³ (smootherstep)
                falloff = 1.0 - (6.0 * t**5 - 15.0 * t**4 + 10.0 * t**3)
            elif falloff_preset == 'SPHERE':
                # Sphere: sqrt(1 - t²)
                falloff = np.sqrt(np.clip(1.0 - t**2, 0.0, 1.0))
            elif falloff_preset == 'ROOT':
                # Root: 1 - sqrt(t)
                falloff = 1.0 - np.sqrt(t)
            elif falloff_preset == 'SHARP':
                # Sharp: (1 - t)²
                falloff = (1.0 - t)**2
            elif falloff_preset == 'LINEAR':
                # Linear: 1 - t
                falloff = 1.0 - t
            elif falloff_preset == 'CONSTANT':
                # Constant: no falloff
                falloff = np.ones_like(t)
            elif falloff_preset == 'CUSTOM':
                # Custom: sample from brush curve
                brush = context.tool_settings.image_paint.brush
                if brush and brush.curve_distance_falloff:
                    curve = brush.curve_distance_falloff
                    # Sample curve for each distance value (vectorized approach)
                    falloff = np.zeros_like(t)
                    for i in range(t.shape[0]):
                        for j in range(t.shape[1]):
                            falloff[i, j] = curve.evaluate(curve.curves[0], t[i, j])
                else:
                    # Fallback to smooth if no brush curve
                    falloff = 1.0 - (3.0 * t**2 - 2.0 * t**3)
            else:
                falloff = 1.0 - t  # Default to linear
            
            # Apply strength
            falloff = falloff * mask * strength
            falloff = falloff[:, :, np.newaxis]  # Expand for RGBA
            
            # Extract source and destination regions
            dst_region = pixels[y_min:y_max, x_min:x_max]
            src_region = pixels[src_y_min:src_y_max, src_x_min:src_x_max]
            
            # Ensure regions match
            min_h = min(dst_region.shape[0], src_region.shape[0], falloff.shape[0])
            min_w = min(dst_region.shape[1], src_region.shape[1], falloff.shape[1])
            
            if min_h > 0 and min_w > 0:
                # Get blend mode from brush
                brush = context.tool_settings.image_paint.brush
                blend_mode = brush.blend if brush else 'MIX'
                
                dst = dst_region[:min_h, :min_w]
                src = src_region[:min_h, :min_w]
                fall = falloff[:min_h, :min_w]
                
                # Apply blend mode
                if blend_mode == 'MIX':
                    blended = src
                elif blend_mode == 'DARKEN':
                    blended = np.minimum(dst, src)
                elif blend_mode == 'MUL':
                    blended = dst * src
                elif blend_mode == 'LIGHTEN':
                    blended = np.maximum(dst, src)
                elif blend_mode == 'SCREEN':
                    blended = 1.0 - (1.0 - dst) * (1.0 - src)
                elif blend_mode == 'ADD':
                    blended = np.clip(dst + src, 0.0, 1.0)
                elif blend_mode == 'SUB':
                    blended = np.clip(dst - src, 0.0, 1.0)
                elif blend_mode == 'OVERLAY':
                    # Overlay: combination of multiply and screen
                    blended = np.where(dst < 0.5,
                                       2.0 * dst * src,
                                       1.0 - 2.0 * (1.0 - dst) * (1.0 - src))
                elif blend_mode == 'DIFFERENCE':
                    blended = np.abs(dst - src)
                elif blend_mode == 'DIVIDE':
                    blended = np.clip(dst / (src + 0.001), 0.0, 1.0)
                elif blend_mode == 'ERASE_ALPHA':
                    blended = dst.copy()
                    blended[:, :, 3] = dst[:, :, 3] * (1.0 - src[:, :, 3])
                elif blend_mode == 'ADD_ALPHA':
                    blended = dst.copy()
                    blended[:, :, 3] = np.clip(dst[:, :, 3] + src[:, :, 3], 0.0, 1.0)
                else:
                    # Default to mix
                    blended = src
                
                # Apply with falloff
                dst_region[:min_h, :min_w] = dst + (blended - dst) * fall
            
            # Set pixels back
            self._image.pixels.foreach_set(pixels.flatten())
            self._image.update()
        else:
            # Fallback to simple Python (slower when NumPy unavailable)
            num_pixels = self._width * self._height * 4
            pixels = array.array('f', [0.0] * num_pixels)
            self._image.pixels.foreach_get(pixels)
            
            for dy in range(-brush_size, brush_size + 1):
                for dx in range(-brush_size, brush_size + 1):
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist > brush_size:
                        continue
                    
                    falloff = 1.0 - (dist / brush_size) if brush_size > 0 else 1.0
                    falloff = max(0.0, min(1.0, falloff / (1.0 - hardness + 0.001)))
                    
                    dst_x, dst_y = cursor_px_x + dx, cursor_px_y + dy
                    src_x, src_y = source_px_x + dx, source_px_y + dy
                    
                    if not (0 <= dst_x < self._width and 0 <= dst_y < self._height):
                        continue
                    if not (0 <= src_x < self._width and 0 <= src_y < self._height):
                        continue
                    
                    src_idx = (src_y * self._width + src_x) * 4
                    dst_idx = (dst_y * self._width + dst_x) * 4
                    
                    for c in range(4):
                        pixels[dst_idx + c] += (pixels[src_idx + c] - pixels[dst_idx + c]) * falloff
            
            self._image.pixels.foreach_set(pixels)
            self._image.update()
