import math
import gpu
import blf
from gpu_extras.batch import batch_for_shader

import bpy
from bpy.types import Operator
from .. import utils







# ----------------------------
# Modal Adjust Operators (Font Size / Rotation)
# ----------------------------

def _draw_adjust_preview_3d():
    """Draw text preview during adjust operations in 3D view."""
    context = bpy.context
    if not hasattr(context.scene, "text_tool_properties"):
        return
    
    props = context.scene.text_tool_properties
    if not props.text:
        return
    
    # Draw at center of region
    region = context.region
    x = region.width // 2
    y = region.height // 2
    
    # Draw using blf
    import blf
    font_path = props.font_file.filepath if props.font_file else None
    font_id = utils._get_blf_font_id(font_path)
    
    font_size = max(8, min(props.font_size, 500))
    blf.size(font_id, font_size)
    
    # Get text dimensions
    text_width, text_height = blf.dimensions(font_id, props.text)
    
    # Set color
    if props.use_gradient:
        r, g, b, a = 1.0, 1.0, 1.0, 0.8
    else:
        r, g, b = props.color[0], props.color[1], props.color[2]
        a = props.color[3] if len(props.color) > 3 else 0.8
    blf.color(font_id, r, g, b, a)
    
    # Apply rotation
    if props.rotation != 0.0:
        blf.enable(font_id, blf.ROTATION)
        blf.rotation(font_id, props.rotation)
        
        # Center the rotated text
        cos_r = math.cos(props.rotation)
        sin_r = math.sin(props.rotation)
        offset_x = -text_width / 2
        offset_y = -text_height / 2
        rotated_offset_x = offset_x * cos_r - offset_y * sin_r
        rotated_offset_y = offset_x * sin_r + offset_y * cos_r
        blf.position(font_id, x + rotated_offset_x, y + rotated_offset_y, 0)
    else:
        blf.position(font_id, x - text_width / 2, y - text_height / 2, 0)
    
    blf.draw(font_id, props.text)
    
    if props.rotation != 0.0:
        blf.disable(font_id, blf.ROTATION)


def _draw_adjust_preview_image():
    """Draw text preview during adjust operations in Image Editor."""
    context = bpy.context
    if not hasattr(context.scene, "text_tool_properties"):
        return
    
    props = context.scene.text_tool_properties
    if not props.text:
        return
    
    # Draw at center of region
    region = context.region
    x = region.width // 2
    y = region.height // 2
    
    # Draw using blf
    import blf
    font_path = props.font_file.filepath if props.font_file else None
    font_id = utils._get_blf_font_id(font_path)
    
    # Scale font size based on zoom
    font_size = props.font_size
    try:
        sima = context.space_data
        if sima.type == 'IMAGE_EDITOR' and sima.image:
            i_width, i_height = sima.image.size
            if i_width > 0:
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
    
    font_size = max(8, min(font_size, 500))
    blf.size(font_id, font_size)
    
    # Get text dimensions
    text_width, text_height = blf.dimensions(font_id, props.text)
    
    # Set color
    if props.use_gradient:
        r, g, b, a = 1.0, 1.0, 1.0, 0.8
    else:
        r, g, b = props.color[0], props.color[1], props.color[2]
        a = props.color[3] if len(props.color) > 3 else 0.8
    blf.color(font_id, r, g, b, a)
    
    # Apply rotation
    if props.rotation != 0.0:
        blf.enable(font_id, blf.ROTATION)
        blf.rotation(font_id, props.rotation)
        
        # Center the rotated text
        cos_r = math.cos(props.rotation)
        sin_r = math.sin(props.rotation)
        offset_x = -text_width / 2
        offset_y = -text_height / 2
        rotated_offset_x = offset_x * cos_r - offset_y * sin_r
        rotated_offset_y = offset_x * sin_r + offset_y * cos_r
        blf.position(font_id, x + rotated_offset_x, y + rotated_offset_y, 0)
    else:
        blf.position(font_id, x - text_width / 2, y - text_height / 2, 0)
    
    blf.draw(font_id, props.text)
    
    if props.rotation != 0.0:
        blf.disable(font_id, blf.ROTATION)


class TEXTTOOL_OT_adjust_font_size(Operator):
    """Adjust font size by dragging left/right"""
    bl_idname = "texttool.adjust_font_size"
    bl_label = "Adjust Font Size"
    bl_options = {'REGISTER', 'UNDO', 'GRAB_CURSOR', 'BLOCKING'}

    _initial_size: int = 64
    _initial_mouse_x: int = 0
    _sensitivity: float = 0.5  # pixels per mouse pixel
    _draw_handler = None
    _handler_space = None

    @classmethod
    def poll(cls, context):
        # Only active when our Text Tool is selected
        if not (context.mode in {'PAINT_TEXTURE', 'PAINT'} or 
                (context.area and context.area.type == 'IMAGE_EDITOR')):
            return False
        
        # Check if our tool is active
        ws = context.workspace
        if not ws:
            return False
        
        try:
            if context.area and context.area.type == 'VIEW_3D':
                tool = ws.tools.from_space_view3d_mode('PAINT_TEXTURE')
                return tool.idname == 'texture_paint.text_tool_ttf'
            elif context.area and context.area.type == 'IMAGE_EDITOR':
                tool = ws.tools.from_space_image_mode('PAINT')
                return tool.idname == 'image_paint.text_tool_ttf'
        except (AttributeError, KeyError):
            pass
        return False

    def invoke(self, context, event):
        props = context.scene.text_tool_properties
        self._initial_size = props.font_size
        self._initial_mouse_x = event.mouse_x
        
        # Add draw handler for preview
        if context.area.type == 'VIEW_3D':
            self._draw_handler = bpy.types.SpaceView3D.draw_handler_add(
                _draw_adjust_preview_3d, (), 'WINDOW', 'POST_PIXEL')
            self._handler_space = 'VIEW_3D'
        elif context.area.type == 'IMAGE_EDITOR':
            self._draw_handler = bpy.types.SpaceImageEditor.draw_handler_add(
                _draw_adjust_preview_image, (), 'WINDOW', 'POST_PIXEL')
            self._handler_space = 'IMAGE_EDITOR'
        
        context.window_manager.modal_handler_add(self)
        context.area.header_text_set(f"Font Size: {props.font_size}  |  Drag Left/Right  |  LMB: Confirm  |  RMB/Esc: Cancel")
        return {'RUNNING_MODAL'}

    def _remove_handler(self):
        if self._draw_handler:
            if self._handler_space == 'VIEW_3D':
                bpy.types.SpaceView3D.draw_handler_remove(self._draw_handler, 'WINDOW')
            elif self._handler_space == 'IMAGE_EDITOR':
                bpy.types.SpaceImageEditor.draw_handler_remove(self._draw_handler, 'WINDOW')
            self._draw_handler = None

    def modal(self, context, event):
        props = context.scene.text_tool_properties
        
        if event.type == 'MOUSEMOVE':
            delta = event.mouse_x - self._initial_mouse_x
            new_size = int(self._initial_size + delta * self._sensitivity)
            new_size = max(8, min(512, new_size))  # Clamp to valid range
            props.font_size = new_size
            context.area.header_text_set(f"Font Size: {props.font_size}  |  Drag Left/Right  |  LMB: Confirm  |  RMB/Esc: Cancel")
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}
        
        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            self._remove_handler()
            context.area.header_text_set(None)
            context.area.tag_redraw()
            self.report({'INFO'}, f"Font Size: {props.font_size}")
            return {'FINISHED'}
        
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            props.font_size = self._initial_size
            self._remove_handler()
            context.area.header_text_set(None)
            context.area.tag_redraw()
            self.report({'INFO'}, "Font size change cancelled")
            return {'CANCELLED'}
        
        return {'RUNNING_MODAL'}


class TEXTTOOL_OT_adjust_rotation(Operator):
    """Adjust text rotation by dragging left/right"""
    bl_idname = "texttool.adjust_rotation"
    bl_label = "Adjust Rotation"
    bl_options = {'REGISTER', 'UNDO', 'GRAB_CURSOR', 'BLOCKING'}

    _initial_rotation: float = 0.0
    _initial_mouse_x: int = 0
    _sensitivity: float = 0.01  # radians per mouse pixel
    _draw_handler = None
    _handler_space = None

    @classmethod
    def poll(cls, context):
        # Only active when our Text Tool is selected
        if not (context.mode in {'PAINT_TEXTURE', 'PAINT'} or 
                (context.area and context.area.type == 'IMAGE_EDITOR')):
            return False
        
        # Check if our tool is active
        ws = context.workspace
        if not ws:
            return False
        
        try:
            if context.area and context.area.type == 'VIEW_3D':
                tool = ws.tools.from_space_view3d_mode('PAINT_TEXTURE')
                return tool.idname == 'texture_paint.text_tool_ttf'
            elif context.area and context.area.type == 'IMAGE_EDITOR':
                tool = ws.tools.from_space_image_mode('PAINT')
                return tool.idname == 'image_paint.text_tool_ttf'
        except (AttributeError, KeyError):
            pass
        return False

    def invoke(self, context, event):
        props = context.scene.text_tool_properties
        self._initial_rotation = props.rotation
        self._initial_mouse_x = event.mouse_x
        
        # Add draw handler for preview
        if context.area.type == 'VIEW_3D':
            self._draw_handler = bpy.types.SpaceView3D.draw_handler_add(
                _draw_adjust_preview_3d, (), 'WINDOW', 'POST_PIXEL')
            self._handler_space = 'VIEW_3D'
        elif context.area.type == 'IMAGE_EDITOR':
            self._draw_handler = bpy.types.SpaceImageEditor.draw_handler_add(
                _draw_adjust_preview_image, (), 'WINDOW', 'POST_PIXEL')
            self._handler_space = 'IMAGE_EDITOR'
        
        context.window_manager.modal_handler_add(self)
        context.area.header_text_set(f"Rotation: {math.degrees(props.rotation):.1f}°  |  Drag Left/Right  |  LMB: Confirm  |  RMB/Esc: Cancel")
        return {'RUNNING_MODAL'}

    def _remove_handler(self):
        if self._draw_handler:
            if self._handler_space == 'VIEW_3D':
                bpy.types.SpaceView3D.draw_handler_remove(self._draw_handler, 'WINDOW')
            elif self._handler_space == 'IMAGE_EDITOR':
                bpy.types.SpaceImageEditor.draw_handler_remove(self._draw_handler, 'WINDOW')
            self._draw_handler = None

    def modal(self, context, event):
        props = context.scene.text_tool_properties
        
        if event.type == 'MOUSEMOVE':
            delta = event.mouse_x - self._initial_mouse_x
            new_rotation = self._initial_rotation + delta * self._sensitivity
            # Wrap to 0-360 degrees (0 to 2*pi radians)
            new_rotation = new_rotation % (2 * math.pi)
            if new_rotation < 0:
                new_rotation += 2 * math.pi
            props.rotation = new_rotation
            context.area.header_text_set(f"Rotation: {math.degrees(props.rotation):.1f}°  |  Drag Left/Right  |  LMB: Confirm  |  RMB/Esc: Cancel")
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}
        
        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            self._remove_handler()
            context.area.header_text_set(None)
            context.area.tag_redraw()
            self.report({'INFO'}, f"Rotation: {math.degrees(props.rotation):.1f}°")
            return {'FINISHED'}
        
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            props.rotation = self._initial_rotation
            self._remove_handler()
            context.area.header_text_set(None)
            context.area.tag_redraw()
            self.report({'INFO'}, "Rotation change cancelled")
            return {'CANCELLED'}
        
        return {'RUNNING_MODAL'}

