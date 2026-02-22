import math
import array

import bpy
from bpy.types import Operator
from .. import utils
from .misc_ops import _get_active_image_for_undo




# ============================================================
# Layer Operatorsnow only this 
# ============================================================

import numpy as np

class IMAGE_EDIT_OT_make_selection(bpy.types.Operator):
    """Make a selection on the image (Shift: Add, Ctrl: Subtract)"""
    bl_idname = "image_edit.make_selection"
    bl_label = "Make Selection"
    bl_options = {'REGISTER'}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lmb = False
        self.mode = 'SET'  # 'SET', 'ADD', 'SUBTRACT'

    def modal(self, context, event):
        area_session = utils.layer_get_area_session(context)
        context.area.tag_redraw()
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        width, height = img.size[0], img.size[1]

        if event.type == 'MOUSEMOVE':
            if self.lmb:
                region_pos = [event.mouse_region_x, event.mouse_region_y]
                if area_session.selection_region:
                    area_session.selection_region[1] = region_pos
        elif event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                self.lmb = True
                region_pos = [event.mouse_region_x, event.mouse_region_y]
                area_session.selection_region = [region_pos, region_pos]
            elif event.value == 'RELEASE':
                self.lmb = False
                if area_session.selection_region:
                    area_session.selecting = False
                    # Pass mode to convert_selection
                    utils.layer_convert_selection(context, mode=self.mode)
                    # Clear the temporary region data after converting
                    area_session.selection_region = None
                    utils.layer_resume_paint_mask(context)
                    utils.layer_apply_selection_as_paint_mask(context)
                    img_props = img.image_edit_properties
                    img_props.selected_layer_index = -1
                    return {'FINISHED'}
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            # Cancel only the current drag, not all selections
            area_session.selection_region = None
            area_session.selecting = False
            utils.layer_resume_paint_mask(context)
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        area_session = utils.layer_get_area_session(context)
        if context.area.type != 'IMAGE_EDITOR':
            return {'CANCELLED'}
        
        # Determine selection mode: modifier keys override UI selection
        if event.shift:
            self.mode = 'ADD'
        elif event.ctrl:
            self.mode = 'SUBTRACT'
        else:
            # Use UI-selected mode when no modifier keys
            wm = context.window_manager
            if hasattr(wm, 'image_edit_properties'):
                self.mode = wm.image_edit_properties.selection_mode
            else:
                self.mode = 'SET'
        
        # Clear previous selections only in SET mode
        if self.mode == 'SET':
            area_session.clear_selections()
        
        area_session.selection_region = None
        area_session.selecting = True
        self.lmb = True
        region_pos = [event.mouse_region_x, event.mouse_region_y]
        area_session.selection_region = [region_pos, region_pos[:]]
        # Pause timer during interaction
        utils.layer_pause_paint_mask(context)
        # Force cache update at start of selection
        utils.layer_apply_selection_as_paint_mask(context, force_image_update=True)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

class IMAGE_EDIT_OT_cancel_selection(bpy.types.Operator):
    """Cancel the selection"""
    bl_idname = "image_edit.cancel_selection"
    bl_label = "Cancel Selection"

    def execute(self, context):
        area_session = utils.layer_get_area_session(context)
        # Check if there's anything to cancel (any selection type or negation)
        has_selections = (area_session.selections or area_session.ellipses or 
                         area_session.lassos or area_session._neg_rects or
                         area_session._neg_ellipses or area_session._neg_lassos or
                         area_session.selection_region or area_session.ellipse_region or
                         area_session.lasso_points)
        if not has_selections:
            return {'CANCELLED'}
        utils.layer_cancel_selection(context)
        context.area.tag_redraw()
        return {'FINISHED'}

class IMAGE_EDIT_OT_undo_selection(bpy.types.Operator):
    """Undo the last selection change"""
    bl_idname = "image_edit.undo_selection"
    bl_label = "Undo"

    def execute(self, context):
        # 1. Try Selection Undo
        area_session = utils.layer_get_area_session(context)
        if area_session.undo_selection():
            utils.layer_clear_paint_mask(context)
            # Re-apply paint mask if there are selections
            if area_session.selections or area_session.ellipses or area_session.lassos:
                utils.layer_apply_selection_as_paint_mask(context)
            context.area.tag_redraw()
            return {'FINISHED'}
            
        # 2. Try Image Pixel Undo (Custom Stack)
        image = _get_active_image_for_undo(context)
        if image and utils.ImageUndoStack.get().undo(image):
            utils.force_texture_refresh(context, image)
            self.report({'INFO'}, "Paint undone")
            return {'FINISHED'}
            
        # 3. Try Native Blender Undo
        try:
            bpy.ops.ed.undo()
            return {'FINISHED'}
        except Exception:
            return {'CANCELLED'}

class IMAGE_EDIT_OT_redo_selection(bpy.types.Operator):
    """Redo the last undone selection change"""
    bl_idname = "image_edit.redo_selection"
    bl_label = "Redo"

    def execute(self, context):
        # 1. Try Selection Redo
        area_session = utils.layer_get_area_session(context)
        if area_session.redo_selection():
            utils.layer_clear_paint_mask(context)
            # Re-apply paint mask if there are selections
            if area_session.selections or area_session.ellipses or area_session.lassos:
                utils.layer_apply_selection_as_paint_mask(context)
            context.area.tag_redraw()
            return {'FINISHED'}
            
        # 2. Try Image Pixel Redo (Custom Stack)
        image = _get_active_image_for_undo(context)
        if image and utils.ImageUndoStack.get().redo(image):
            utils.force_texture_refresh(context, image)
            self.report({'INFO'}, "Paint redone")
            return {'FINISHED'}
            
        # 3. Try Native Blender Redo
        try:
            bpy.ops.ed.redo()
            return {'FINISHED'}
        except Exception:
            return {'CANCELLED'}


class IMAGE_EDIT_OT_make_ellipse_selection(bpy.types.Operator):
    """Make an ellipse selection on the image (Shift: Add, Ctrl: Subtract)"""
    bl_idname = "image_edit.make_ellipse_selection"
    bl_label = "Make Ellipse Selection"
    bl_options = {'REGISTER'}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lmb = False
        self.mode = 'SET'

    def modal(self, context, event):
        area_session = utils.layer_get_area_session(context)
        context.area.tag_redraw()
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}

        if event.type == 'MOUSEMOVE':
            if self.lmb:
                region_pos = [event.mouse_region_x, event.mouse_region_y]
                if area_session.ellipse_region:
                    area_session.ellipse_region[1] = region_pos
        elif event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                self.lmb = True
                region_pos = [event.mouse_region_x, event.mouse_region_y]
                area_session.ellipse_region = [region_pos, region_pos[:]]
            elif event.value == 'RELEASE':
                self.lmb = False
                if area_session.ellipse_region:
                    area_session.selecting = False
                    utils.layer_convert_ellipse_selection(context, mode=self.mode)
                    # Clear the temporary region data after converting
                    area_session.ellipse_region = None
                    utils.layer_resume_paint_mask(context)
                    utils.layer_apply_selection_as_paint_mask(context)
                    img_props = img.image_edit_properties
                    img_props.selected_layer_index = -1
                    return {'FINISHED'}
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            area_session.ellipse_region = None
            area_session.selecting = False
            utils.layer_resume_paint_mask(context)
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        area_session = utils.layer_get_area_session(context)
        if context.area.type != 'IMAGE_EDITOR':
            return {'CANCELLED'}
        
        # Determine selection mode: modifier keys override UI selection
        if event.shift:
            self.mode = 'ADD'
        elif event.ctrl:
            self.mode = 'SUBTRACT'
        else:
            wm = context.window_manager
            if hasattr(wm, 'image_edit_properties'):
                self.mode = wm.image_edit_properties.selection_mode
            else:
                self.mode = 'SET'
        
        # Clear previous selections only in SET mode
        if self.mode == 'SET':
            area_session.clear_selections()
        
        area_session.ellipse_region = None
        area_session.selecting = True
        self.lmb = True
        region_pos = [event.mouse_region_x, event.mouse_region_y]
        area_session.ellipse_region = [region_pos, region_pos[:]]
        # Pause timer during interaction
        utils.layer_pause_paint_mask(context)
        # Force cache update at start of selection
        utils.layer_apply_selection_as_paint_mask(context, force_image_update=True)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

class IMAGE_EDIT_OT_make_lasso_selection(bpy.types.Operator):
    """Make a lasso selection on the image (Shift: Add, Ctrl: Subtract)"""
    bl_idname = "image_edit.make_lasso_selection"
    bl_label = "Make Lasso Selection"
    bl_options = {'REGISTER'}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lmb = False
        self.mode = 'SET'
        self.min_dist = 3  # Minimum distance between points for performance

    def modal(self, context, event):
        area_session = utils.layer_get_area_session(context)
        context.area.tag_redraw()
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}

        if event.type == 'MOUSEMOVE':
            if self.lmb and area_session.lasso_points:
                x, y = event.mouse_region_x, event.mouse_region_y
                # Add point if far enough from last point (performance optimization)
                last = area_session.lasso_points[-1]
                dist = ((x - last[0])**2 + (y - last[1])**2)**0.5
                if dist >= self.min_dist:
                    area_session.lasso_points.append([x, y])
        elif event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                self.lmb = True
                x, y = event.mouse_region_x, event.mouse_region_y
                area_session.lasso_points = [[x, y]]
            elif event.value == 'RELEASE':
                self.lmb = False
                if area_session.lasso_points and len(area_session.lasso_points) >= 3:
                    area_session.selecting = False
                    utils.layer_convert_lasso_selection(context, mode=self.mode)
                    utils.layer_resume_paint_mask(context)
                    utils.layer_apply_selection_as_paint_mask(context)
                    img_props = img.image_edit_properties
                    img_props.selected_layer_index = -1
                    area_session.lasso_points = None
                    return {'FINISHED'}
                else:
                    area_session.lasso_points = None
                    return {'CANCELLED'}
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            area_session.lasso_points = None
            area_session.selecting = False
            utils.layer_resume_paint_mask(context)
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        area_session = utils.layer_get_area_session(context)
        if context.area.type != 'IMAGE_EDITOR':
            return {'CANCELLED'}
        
        # Determine selection mode
        if event.shift:
            self.mode = 'ADD'
        elif event.ctrl:
            self.mode = 'SUBTRACT'
        else:
            wm = context.window_manager
            if hasattr(wm, 'image_edit_properties'):
                self.mode = wm.image_edit_properties.selection_mode
            else:
                self.mode = 'SET'
        
        # Clear previous selections only in SET mode
        if self.mode == 'SET':
            area_session.clear_selections()
        
        area_session.lasso_points = None
        area_session.selecting = True
        self.lmb = True
        x, y = event.mouse_region_x, event.mouse_region_y
        area_session.lasso_points = [[x, y]]
        # Pause timer during interaction
        utils.layer_pause_paint_mask(context)
        # Force cache update at start of selection
        utils.layer_apply_selection_as_paint_mask(context, force_image_update=True)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

class IMAGE_EDIT_OT_swap_colors(bpy.types.Operator):
    """Swap foreground and background color"""
    bl_idname = "image_edit.swap_colors"
    bl_label = "Swap Colors"

    def execute(self, context):
        wm = context.window_manager
        props = wm.image_edit_properties
        props.foreground_color, props.background_color = props.background_color[:], props.foreground_color[:]
        return {'FINISHED'}

class IMAGE_EDIT_OT_fill_with_fg_color(bpy.types.Operator):
    """Fill the image with foreground color"""
    bl_idname = "image_edit.fill_with_fg_color"
    bl_label = "Fill with FG Color"

    def execute(self, context):
        wm = context.window_manager
        props = wm.image_edit_properties
        color = props.foreground_color[:] + (1.0,)
        img = utils.layer_get_target_image(context)
        if not img:
            return {'CANCELLED'}
        pixels = utils.layer_read_pixels_from_image(img)
        selection = utils.layer_get_target_selection(context)
        if selection:
            pixels[selection[0][1]:selection[1][1], selection[0][0]:selection[1][0]] = color
        elif selection == []:
            return {'CANCELLED'}
        else:
            pixels[:] = color
        utils.ImageUndoStack.get().push_state(img)
        utils.layer_write_pixels_to_image(img, pixels)
        utils.layer_refresh_image(context)
        return {'FINISHED'}

class IMAGE_EDIT_OT_fill_with_bg_color(bpy.types.Operator):
    """Fill the image with background color"""
    bl_idname = "image_edit.fill_with_bg_color"
    bl_label = "Fill with BG Color"

    def execute(self, context):
        wm = context.window_manager
        props = wm.image_edit_properties
        color = props.background_color[:] + (1.0,)
        img = utils.layer_get_target_image(context)
        if not img:
            return {'CANCELLED'}
        pixels = utils.layer_read_pixels_from_image(img)
        selection = utils.layer_get_target_selection(context)
        if selection:
            pixels[selection[0][1]:selection[1][1], selection[0][0]:selection[1][0]] = color
        elif selection == []:
            return {'CANCELLED'}
        else:
            pixels[:] = color
        utils.ImageUndoStack.get().push_state(img)
        utils.layer_write_pixels_to_image(img, pixels)
        utils.layer_refresh_image(context)
        return {'FINISHED'}

class IMAGE_EDIT_OT_clear(bpy.types.Operator):
    """Clear the image"""
    bl_idname = "image_edit.clear"
    bl_label = "Clear"

    def execute(self, context):
        img = utils.layer_get_target_image(context)
        if not img:
            return {'CANCELLED'}
        pixels = utils.layer_read_pixels_from_image(img)
        selection = utils.layer_get_target_selection(context)
        if selection:
            pixels[selection[0][1]:selection[1][1], selection[0][0]:selection[1][0]] = (0, 0, 0, 0)
        elif selection == []:
            return {'CANCELLED'}
        else:
            pixels[:] = (0, 0, 0, 0)
        utils.ImageUndoStack.get().push_state(img)
        utils.layer_write_pixels_to_image(img, pixels)
        utils.layer_refresh_image(context)
        return {'FINISHED'}

class IMAGE_EDIT_OT_cut(bpy.types.Operator):
    """Cut the image"""
    bl_idname = "image_edit.cut"
    bl_label = "Cut"

    def execute(self, context):
        session = utils.layer_get_session()
        img = utils.layer_get_target_image(context)
        if not img:
            return {'CANCELLED'}
        width, height = img.size
        pixels = utils.layer_read_pixels_from_image(img)
        selection = utils.layer_get_target_selection(context)
        if selection:
            target_pixels = pixels[selection[0][1]:selection[1][1], selection[0][0]:selection[1][0]]
        elif selection == []:
            return {'CANCELLED'}
        else:
            target_pixels = pixels
        session.copied_image_pixels = target_pixels.copy()
        session.copied_image_settings = {'is_float': img.is_float, 'colorspace_name': img.colorspace_settings.name}
        layer = utils.layer_get_active_layer(context)
        if layer:
            session.copied_layer_settings = {'rotation': layer.rotation, 'scale': layer.scale, 'custom_data': layer.custom_data}
        else:
            session.copied_layer_settings = None
        utils.ImageUndoStack.get().push_state(img)
        if selection:
            pixels[selection[0][1]:selection[1][1], selection[0][0]:selection[1][0]] = (0, 0, 0, 0)
        else:
            pixels[:] = (0, 0, 0, 0)
        utils.layer_write_pixels_to_image(img, pixels)
        utils.layer_refresh_image(context)
        self.report({'INFO'}, 'Cut selected image.')
        return {'FINISHED'}

class IMAGE_EDIT_OT_copy(bpy.types.Operator):
    """Copy the image"""
    bl_idname = "image_edit.copy"
    bl_label = "Copy"

    def execute(self, context):
        session = utils.layer_get_session()
        img = utils.layer_get_target_image(context)
        if not img:
            return {'CANCELLED'}
        width, height = img.size
        pixels = utils.layer_read_pixels_from_image(img)
        selection = utils.layer_get_target_selection(context)
        if selection:
            target_pixels = pixels[selection[0][1]:selection[1][1], selection[0][0]:selection[1][0]]
        elif selection == []:
            return {'CANCELLED'}
        else:
            target_pixels = pixels
        session.copied_image_pixels = target_pixels
        session.copied_image_settings = {'is_float': img.is_float, 'colorspace_name': img.colorspace_settings.name}
        layer = utils.layer_get_active_layer(context)
        if layer:
            session.copied_layer_settings = {'rotation': layer.rotation, 'scale': layer.scale, 'custom_data': layer.custom_data}
        else:
            session.copied_layer_settings = None
        self.report({'INFO'}, 'Copied selected image.')
        return {'FINISHED'}

class IMAGE_EDIT_OT_paste(bpy.types.Operator):
    """Paste the image"""
    bl_idname = "image_edit.paste"
    bl_label = "Paste"

    def execute(self, context):
        session = utils.layer_get_session()
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        target_pixels = session.copied_image_pixels
        if target_pixels is None:
            return {'CANCELLED'}
        utils.layer_create_layer(img, target_pixels, session.copied_image_settings, session.copied_layer_settings)
        utils.layer_cancel_selection(context)
        utils.layer_refresh_image(context)
        return {'FINISHED'}

class IMAGE_EDIT_OT_cut_to_layer(bpy.types.Operator):
    """Cut selection and paste as new layer"""
    bl_idname = "image_edit.cut_to_layer"
    bl_label = "Cut Selection to New Layer"

    def execute(self, context):
        img = utils.layer_get_target_image(context)
        base_img = context.area.spaces.active.image
        if not img or not base_img:
            return {'CANCELLED'}
        
        width, height = img.size
        pixels = utils.layer_read_pixels_from_image(img)
        
        # Get combined boolean mask of current selection
        mask = utils.layer_get_combined_selection_mask(context, width, height)
        
        target_pixels = None
        
        if mask is not None:
            # Find bounding box of selected pixels
            rows, cols = np.where(mask)
            if len(rows) == 0:
                 return {'CANCELLED'}
                 
            y1, y2 = min(rows), max(rows) + 1
            x1, x2 = min(cols), max(cols) + 1
            
            # Extract pixels within bounding box
            target_pixels = pixels[y1:y2, x1:x2].copy()
            
            # mask is (height, width), slice it to match target_pixels
            mask_slice = mask[y1:y2, x1:x2]
            
            # Set non-selected pixels to transparent in the copy
            # target_pixels is (h, w, 4), mask_slice is (h, w)
            # broadcasting requires shape compatibility
            target_pixels[~mask_slice] = (0, 0, 0, 0)
        else:
            # No selection - duplicate entire image
            target_pixels = pixels.copy()
            # For Cut operation on full image, we clear the entire image later
        
        img_settings = {'is_float': img.is_float, 'colorspace_name': img.colorspace_settings.name}
        layer = utils.layer_get_active_layer(context)
        if layer:
            layer_settings = {'rotation': layer.rotation, 'scale': layer.scale, 'custom_data': layer.custom_data}
        else:
            layer_settings = None
        
        utils.ImageUndoStack.get().push_state(img)
        
        if mask is not None:
            # Clear SELECTED pixels in original image
            pixels[mask] = (0, 0, 0, 0)
        else:
            # Clear ALL pixels
            pixels[:] = (0, 0, 0, 0)
            
        utils.layer_write_pixels_to_image(img, pixels)
        
        utils.layer_create_layer(base_img, target_pixels, img_settings, layer_settings)
        utils.layer_cancel_selection(context)
        utils.layer_refresh_image(context)
        self.report({'INFO'}, 'Cut selection to new layer.')
        return {'FINISHED'}

class IMAGE_EDIT_OT_copy_to_layer(bpy.types.Operator):
    """Copy selection and paste as new layer"""
    bl_idname = "image_edit.copy_to_layer"
    bl_label = "Copy Selection to New Layer"

    def execute(self, context):
        img = utils.layer_get_target_image(context)
        base_img = context.area.spaces.active.image
        if not img or not base_img:
            return {'CANCELLED'}
        
        width, height = img.size
        pixels = utils.layer_read_pixels_from_image(img)
        
        # Get combined boolean mask of current selection
        mask = utils.layer_get_combined_selection_mask(context, width, height)
        
        target_pixels = None
        
        if mask is not None:
            # Find bounding box of selected pixels mask is (height, width)
            # np.where returns (row_indices, col_indices) -> (y, x)
            rows, cols = np.where(mask)
            if len(rows) == 0:
                 return {'CANCELLED'}

            y1, y2 = min(rows), max(rows) + 1
            x1, x2 = min(cols), max(cols) + 1
            
            # Extract pixels within bounding box
            target_pixels = pixels[y1:y2, x1:x2].copy()
            
            # mask is (height, width), slice it to match target_pixels
            mask_slice = mask[y1:y2, x1:x2]
            
            # Set non-selected pixels to transparent in the copy
            target_pixels[~mask_slice] = (0, 0, 0, 0)
        else:
            # No selection - duplicate entire image
            target_pixels = pixels.copy()
        
        img_settings = {'is_float': img.is_float, 'colorspace_name': img.colorspace_settings.name}
        layer = utils.layer_get_active_layer(context)
        if layer:
            layer_settings = {'rotation': layer.rotation, 'scale': layer.scale, 'custom_data': layer.custom_data}
        else:
            layer_settings = None
        
        utils.layer_create_layer(base_img, target_pixels, img_settings, layer_settings)
        utils.layer_cancel_selection(context)
        utils.layer_refresh_image(context)
        self.report({'INFO'}, 'Copied selection to new layer.')
        return {'FINISHED'}

