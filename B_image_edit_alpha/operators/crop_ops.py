import math
import array

import bpy
from bpy.types import Operator
from .. import utils
from .. import ui


# ----------------------------
# Crop Tool (Image Editor)
# ----------------------------
class IMAGE_PAINT_OT_crop_tool(Operator):
    bl_idname = "image_paint.crop_tool"
    bl_label = "Image Crop Tool"
    bl_description = "Crop the image by selecting a rectangular region (Enter/Space to confirm)"
    bl_options = {'REGISTER', 'UNDO'}
    
    _draw_handler = None
    _start_pos = None  # Top-left corner (min x, min y in screen coords after normalization)
    _end_pos = None    # Bottom-right corner
    _is_dragging = False
    _selection_complete = False
    _image = None
    
    # Resize handle state
    _resize_mode = None  # None, 'TL', 'TR', 'BL', 'BR', 'T', 'B', 'L', 'R', 'MOVE'
    _drag_offset = (0, 0)  # For move mode
    HANDLE_SIZE = 12  # Pixels for handle hit detection
    
    @classmethod
    def poll(cls, context):
        sima = context.space_data
        return (context.area.type == 'IMAGE_EDITOR' and 
                sima.mode == 'PAINT' and 
                sima.image is not None)
    
    def _get_normalized_rect(self):
        """Return (x1, y1, x2, y2) with x1 < x2 and y1 < y2."""
        if not self._start_pos or not self._end_pos:
            return None
        x1, y1 = self._start_pos
        x2, y2 = self._end_pos
        return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
    
    def _hit_test_handle(self, mx, my):
        """Check if mouse is over a resize handle. Returns handle name or None."""
        rect = self._get_normalized_rect()
        if not rect:
            return None
        
        x1, y1, x2, y2 = rect
        hs = self.HANDLE_SIZE
        
        # Corner handles (priority)
        if abs(mx - x1) < hs and abs(my - y1) < hs:
            return 'BL'  # Bottom-left
        if abs(mx - x2) < hs and abs(my - y1) < hs:
            return 'BR'  # Bottom-right
        if abs(mx - x1) < hs and abs(my - y2) < hs:
            return 'TL'  # Top-left
        if abs(mx - x2) < hs and abs(my - y2) < hs:
            return 'TR'  # Top-right
        
        # Edge handles
        if abs(mx - x1) < hs and y1 < my < y2:
            return 'L'  # Left edge
        if abs(mx - x2) < hs and y1 < my < y2:
            return 'R'  # Right edge
        if abs(my - y1) < hs and x1 < mx < x2:
            return 'B'  # Bottom edge
        if abs(my - y2) < hs and x1 < mx < x2:
            return 'T'  # Top edge
        
        # Inside rectangle = move
        if x1 < mx < x2 and y1 < my < y2:
            return 'MOVE'
        
        return None
    
    def modal(self, context, event):
        context.area.tag_redraw()
        
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE'}:
            return {'PASS_THROUGH'}
        
        # Confirm crop with Enter or Spacebar
        if event.type in {'RET', 'NUMPAD_ENTER', 'SPACE'} and event.value == 'PRESS':
            if self._selection_complete and self._start_pos and self._end_pos:
                success = self._apply_crop(context)
                self._cleanup(context)
                if success:
                    self.report({'INFO'}, "Image cropped successfully")
                    return {'FINISHED'}
                else:
                    self.report({'WARNING'}, "Crop cancelled - selection too small")
                    return {'CANCELLED'}
            return {'RUNNING_MODAL'}
        
        mx, my = event.mouse_region_x, event.mouse_region_y
        
        if event.type == 'MOUSEMOVE':
            if self._is_dragging:
                props = context.scene.text_tool_properties
                
                if self._resize_mode == 'MOVE':
                    # Move entire selection
                    dx = mx - self._drag_offset[0]
                    dy = my - self._drag_offset[1]
                    rect = self._get_normalized_rect()
                    if rect:
                        x1, y1, x2, y2 = rect
                        w, h = x2 - x1, y2 - y1
                        self._start_pos = (dx, dy)
                        self._end_pos = (dx + w, dy + h)
                elif self._resize_mode in ('TL', 'TR', 'BL', 'BR', 'T', 'B', 'L', 'R'):
                    # Resize from handle
                    rect = self._get_normalized_rect()
                    if rect:
                        x1, y1, x2, y2 = rect
                        
                        if 'L' in self._resize_mode:
                            x1 = mx
                        if 'R' in self._resize_mode:
                            x2 = mx
                        if 'T' in self._resize_mode:
                            y2 = my
                        if 'B' in self._resize_mode:
                            y1 = my
                        
                        # Apply aspect ratio constraint if locked
                        if props.crop_lock_aspect:
                            aspect = props.crop_aspect_width / props.crop_aspect_height
                            w, h = x2 - x1, y2 - y1
                            if abs(w) > 0 and abs(h) > 0:
                                current_aspect = abs(w) / abs(h)
                                if current_aspect > aspect:
                                    # Too wide, adjust width
                                    new_w = abs(h) * aspect
                                    if 'L' in self._resize_mode:
                                        x1 = x2 - new_w
                                    else:
                                        x2 = x1 + new_w
                                else:
                                    # Too tall, adjust height
                                    new_h = abs(w) / aspect
                                    if 'B' in self._resize_mode:
                                        y1 = y2 - new_h
                                    else:
                                        y2 = y1 + new_h
                        
                        self._start_pos = (x1, y1)
                        self._end_pos = (x2, y2)
                else:
                    # Initial drag - new selection
                    new_x, new_y = mx, my
                    
                    # Apply aspect ratio constraint if locked
                    if props.crop_lock_aspect:
                        aspect = props.crop_aspect_width / props.crop_aspect_height
                        sx, sy = self._start_pos
                        w = new_x - sx
                        h = new_y - sy
                        if abs(w) > 0 and abs(h) > 0:
                            current_aspect = abs(w) / abs(h)
                            if current_aspect > aspect:
                                # Too wide, adjust width
                                new_w = abs(h) * aspect * (1 if w > 0 else -1)
                                new_x = sx + new_w
                            else:
                                # Too tall, adjust height
                                new_h = abs(w) / aspect * (1 if h > 0 else -1)
                                new_y = sy + new_h
                    
                    self._end_pos = (new_x, new_y)
                
                utils.crop_preview_start = self._start_pos
                utils.crop_preview_end = self._end_pos
            return {'RUNNING_MODAL'}
        
        elif event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                if self._selection_complete:
                    # Check if clicking on a handle to resize
                    handle = self._hit_test_handle(mx, my)
                    if handle:
                        self._is_dragging = True
                        self._resize_mode = handle
                        if handle == 'MOVE':
                            rect = self._get_normalized_rect()
                            if rect:
                                self._drag_offset = (mx - rect[0], my - rect[1])
                        context.area.header_text_set(f"Resizing: {handle} | Release to confirm adjustment")
                        return {'RUNNING_MODAL'}
                
                # Start new selection
                self._is_dragging = True
                self._selection_complete = False
                self._resize_mode = None
                self._start_pos = (mx, my)
                self._end_pos = self._start_pos
                self._image = context.space_data.image
                utils.crop_preview_start = self._start_pos
                utils.crop_preview_end = self._end_pos
                context.area.header_text_set("Drag to select crop region")
                return {'RUNNING_MODAL'}
            
            elif event.value == 'RELEASE' and self._is_dragging:
                props = context.scene.text_tool_properties
                
                # Check for single-click (minimal movement)
                if self._start_pos:
                    dx = abs(mx - self._start_pos[0])
                    dy = abs(my - self._start_pos[1])
                    is_single_click = (dx < 5 and dy < 5)
                else:
                    is_single_click = False
                
                # If single-click and resolution mode, create fixed-size region
                if is_single_click and props.crop_use_resolution and self._image:
                    # Get image size and view2d for coordinate conversion
                    img_width, img_height = self._image.size
                    view2d = context.region.view2d
                    
                    # Calculate region size in screen space based on resolution
                    # Convert resolution pixels to screen pixels using view2d scale
                    uv_center = view2d.region_to_view(mx, my)
                    
                    # Calculate half-sizes in UV space
                    half_w_uv = (props.crop_resolution_x / img_width) / 2
                    half_h_uv = (props.crop_resolution_y / img_height) / 2
                    
                    # Convert back to screen coordinates
                    uv_x1, uv_y1 = uv_center[0] - half_w_uv, uv_center[1] - half_h_uv
                    uv_x2, uv_y2 = uv_center[0] + half_w_uv, uv_center[1] + half_h_uv
                    
                    screen_x1, screen_y1 = view2d.view_to_region(uv_x1, uv_y1)
                    screen_x2, screen_y2 = view2d.view_to_region(uv_x2, uv_y2)
                    
                    self._start_pos = (screen_x1, screen_y1)
                    self._end_pos = (screen_x2, screen_y2)
                    utils.crop_preview_start = self._start_pos
                    utils.crop_preview_end = self._end_pos
                else:
                    self._end_pos = (mx, my) if not self._resize_mode else self._end_pos
                    utils.crop_preview_end = self._end_pos
                
                self._is_dragging = False
                self._selection_complete = True
                self._resize_mode = None
                context.area.header_text_set("Drag handles to resize | Enter/Space to crop | ESC to cancel")
                return {'RUNNING_MODAL'}
        
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            context.area.header_text_set(None)
            self._cleanup(context)
            return {'CANCELLED'}
        
        return {'RUNNING_MODAL'}
    
    def invoke(self, context, event):
        if context.area.type == 'IMAGE_EDITOR':
            self._draw_handler = bpy.types.SpaceImageEditor.draw_handler_add(
                ui.draw_crop_preview_image, (), 'WINDOW', 'POST_PIXEL')
            
            # Start drag immediately on first click
            self._is_dragging = True
            self._selection_complete = False
            self._resize_mode = None
            self._start_pos = (event.mouse_region_x, event.mouse_region_y)
            self._end_pos = self._start_pos
            self._image = context.space_data.image
            utils.crop_preview_start = self._start_pos
            utils.crop_preview_end = self._end_pos
            context.area.header_text_set("Drag to select crop region")
            
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            self.report({'WARNING'}, "Image Editor not found")
            return {'CANCELLED'}
    
    def _cleanup(self, context):
        """Clean up state and handlers."""
        if self._draw_handler:
            bpy.types.SpaceImageEditor.draw_handler_remove(self._draw_handler, 'WINDOW')
            self._draw_handler = None
        
        utils.crop_preview_start = None
        utils.crop_preview_end = None
        self._is_dragging = False
        self._selection_complete = False
        self._resize_mode = None
        self._image = None
        context.area.tag_redraw()
    
    def _apply_crop(self, context):
        """Apply the crop operation to the image."""
        if not self._image or not self._start_pos or not self._end_pos:
            return False
        
        props = context.scene.text_tool_properties
        
        # Convert screen coordinates to image coordinates
        region = context.region
        view2d = region.view2d
        
        # Get image dimensions
        img_width, img_height = self._image.size
        if img_width == 0 or img_height == 0:
            return False
        
        # Convert screen positions to UV coordinates (0-1 range)
        uv_start = view2d.region_to_view(*self._start_pos)
        uv_end = view2d.region_to_view(*self._end_pos)
        
        # Convert UV to pixel coordinates
        px_x1 = int(uv_start[0] * img_width)
        px_y1 = int(uv_start[1] * img_height)
        px_x2 = int(uv_end[0] * img_width)
        px_y2 = int(uv_end[1] * img_height)
        
        # Ensure proper ordering (min to max)
        crop_x1 = min(px_x1, px_x2)
        crop_y1 = min(px_y1, px_y2)
        crop_x2 = max(px_x1, px_x2)
        crop_y2 = max(px_y1, px_y2)
        
        # If expand canvas is disabled, clamp to image bounds
        if not props.crop_expand_canvas:
            crop_x1 = max(0, crop_x1)
            crop_y1 = max(0, crop_y1)
            crop_x2 = min(img_width, crop_x2)
            crop_y2 = min(img_height, crop_y2)
        
        # Calculate new dimensions
        new_width = crop_x2 - crop_x1
        new_height = crop_y2 - crop_y1
        
        # Minimum crop size check
        if new_width < 2 or new_height < 2:
            return False
        
        # Save undo state before modifying
        utils.ImageUndoStack.get().push_state(self._image)
        
        # Get original pixels
        num_pixels = img_width * img_height * 4
        original_pixels = array.array('f', [0.0] * num_pixels)
        self._image.pixels.foreach_get(original_pixels)
        
        # Create new pixel array, filled with fill color if expanding
        new_num_pixels = new_width * new_height * 4
        fill_r, fill_g, fill_b, fill_a = props.crop_fill_color
        new_pixels = array.array('f', [fill_r, fill_g, fill_b, fill_a] * (new_width * new_height))
        
        # Copy pixels from source to destination
        for y in range(new_height):
            src_y = crop_y1 + y
            # Skip rows outside original image
            if src_y < 0 or src_y >= img_height:
                continue
            
            # Calculate source and destination X ranges
            src_x_start = max(0, crop_x1)
            src_x_end = min(img_width, crop_x2)
            
            # Calculate offset in destination
            dst_x_offset = src_x_start - crop_x1
            copy_width = src_x_end - src_x_start
            
            if copy_width <= 0:
                continue
            
            src_start = (src_y * img_width + src_x_start) * 4
            src_end = src_start + copy_width * 4
            dst_start = (y * new_width + dst_x_offset) * 4
            dst_end = dst_start + copy_width * 4
            new_pixels[dst_start:dst_end] = original_pixels[src_start:src_end]
        
        # Apply resolution scaling if enabled
        if props.crop_use_resolution:
            final_width = props.crop_resolution_x
            final_height = props.crop_resolution_y
        else:
            final_width = new_width
            final_height = new_height
        
        # Resize image and set new pixels
        self._image.scale(new_width, new_height)
        self._image.pixels.foreach_set(new_pixels)
        
        # If resolution is different, scale again
        if props.crop_use_resolution and (final_width != new_width or final_height != new_height):
            self._image.scale(final_width, final_height)
        
        self._image.update()
        
        return True

