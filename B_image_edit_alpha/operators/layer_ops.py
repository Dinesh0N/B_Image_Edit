import math
import array
import numpy as np

import bpy
from bpy.types import Operator
from .. import utils


class IMAGE_EDIT_OT_add_image_layer(bpy.types.Operator):
    """Add image file(s) as new layer(s)"""
    bl_idname = "image_edit.add_image_layer"
    bl_label = "Add Image as Layer"
    
    filepath: bpy.props.StringProperty(subtype='FILE_PATH')
    directory: bpy.props.StringProperty(subtype='DIR_PATH')
    files: bpy.props.CollectionProperty(type=bpy.types.OperatorFileListElement)
    filter_glob: bpy.props.StringProperty(default="*.png;*.jpg;*.jpeg;*.bmp;*.tga;*.tiff;*.exr;*.hdr", options={'HIDDEN'})

    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        import os
        img = context.area.spaces.active.image
        if not img:
            self.report({'ERROR'}, "No active image")
            return {'CANCELLED'}
        
        if self.files:
            filepaths = [os.path.join(self.directory, f.name) for f in self.files if f.name]
        else:
            filepaths = [self.filepath]
        
        added_count = 0
        for filepath in filepaths:
            if not filepath or not os.path.isfile(filepath):
                continue
            
            try:
                layer_source = bpy.data.images.load(filepath)
            except Exception:
                self.report({'WARNING'}, f"Could not load: {os.path.basename(filepath)}")
                continue
            
            original_filename = os.path.splitext(os.path.basename(filepath))[0]
            target_pixels = utils.layer_read_pixels_from_image(layer_source)
            
            img_settings = {
                'is_float': layer_source.is_float,
                'colorspace_name': layer_source.colorspace_settings.name
            }
            
            utils.layer_create_layer(img, target_pixels, img_settings, None, custom_label=original_filename)
            bpy.data.images.remove(layer_source)
            added_count += 1
        
        if added_count == 0:
            self.report({'ERROR'}, "No images were added")
            return {'CANCELLED'}
        
        utils.layer_cancel_selection(context)
        utils.layer_refresh_image(context)
        
        self.report({'INFO'}, f'{added_count} image(s) added as layers.')
        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

class IMAGE_EDIT_OT_new_image_layer(bpy.types.Operator):
    """Create a new blank image as a layer"""
    bl_idname = "image_edit.new_image_layer"
    bl_label = "New Image Layer"
    
    layer_name: bpy.props.StringProperty(name='Name', default='New Layer')
    width: bpy.props.IntProperty(name='Width', default=512, min=1, max=16384)
    height: bpy.props.IntProperty(name='Height', default=512, min=1, max=16384)
    color: bpy.props.FloatVectorProperty(name='Color', subtype='COLOR', size=4, min=0, max=1, default=(1.0, 1.0, 1.0, 0.0))
    use_base_size: bpy.props.BoolProperty(name='Use Base Image Size', default=True)

    bl_options = {'REGISTER', 'UNDO'}
    def invoke(self, context, event):
        img = context.area.spaces.active.image
        if img:
            self.width = img.size[0]
            self.height = img.size[1]
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            self.report({'ERROR'}, "No active image")
            return {'CANCELLED'}
        
        if self.use_base_size:
            layer_width = img.size[0]
            layer_height = img.size[1]
        else:
            layer_width = self.width
            layer_height = self.height
        
        pixels = np.full((layer_height, layer_width, 4), self.color, dtype=np.float32)
        
        img_settings = {
            'is_float': img.is_float,
            'colorspace_name': img.colorspace_settings.name
        }
        
        layer_label = self.layer_name if self.layer_name else "New Layer"
        utils.layer_create_layer(img, pixels, img_settings, None, custom_label=layer_label)
        
        utils.layer_cancel_selection(context)
        utils.layer_refresh_image(context)
        
        return {'FINISHED'}

    def draw(self, context):
        layout = self.layout
        layout.prop(self, 'layer_name')
        layout.prop(self, 'use_base_size')
        if not self.use_base_size:
            layout.prop(self, 'width')
            layout.prop(self, 'height')
        layout.prop(self, 'color')

class IMAGE_EDIT_OT_crop(bpy.types.Operator):
    """Crop the image to the boundary of the selection"""
    bl_idname = "image_edit.crop"
    bl_label = "Crop"

    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        pixels = utils.layer_read_pixels_from_image(img)
        selection = utils.layer_get_selection(context)
        if selection:
            target_pixels = pixels[selection[0][1]:selection[1][1], selection[0][0]:selection[1][0]]
        else:
            target_pixels = pixels
        target_width, target_height = target_pixels.shape[1], target_pixels.shape[0]
        img.scale(target_width, target_height)
        utils.layer_write_pixels_to_image(img, target_pixels)
        if selection:
            img_props = img.image_edit_properties
            layers = img_props.layers
            for layer in reversed(layers):
                layer_pos = layer.location
                layer_pos[0] -= selection[0][0]
                layer_pos[1] -= selection[0][1]
        utils.layer_cancel_selection(context)
        utils.layer_refresh_image(context)
        return {'FINISHED'}

class IMAGE_EDIT_OT_deselect_layer(bpy.types.Operator):
    bl_idname = "image_edit.deselect_layer"
    bl_label = "Deselect Layer"

    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        img_props = img.image_edit_properties
        img_props.selected_layer_index = -1
        utils.layer_refresh_image(context)
        return {'FINISHED'}

class IMAGE_EDIT_OT_move_layer(bpy.types.Operator):
    """Move the layer"""
    bl_idname = "image_edit.move_layer"
    bl_label = "Move Layer"

    bl_options = {'REGISTER', 'UNDO'}
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_input_position = [0, 0]
        self.start_layer_location = [0, 0]

    def modal(self, context, event):
        area_session = utils.layer_get_area_session(context)
        context.area.tag_redraw()
        img = context.area.spaces.active.image
        width, height = img.size[0], img.size[1]
        layer = utils.layer_get_active_layer(context)
        if not layer:
            return {'CANCELLED'}
        if event.type == 'MOUSEMOVE':
            region_pos = [event.mouse_region_x, event.mouse_region_y]
            view_x, view_y = context.region.view2d.region_to_view(*region_pos)
            target_x = width * view_x
            target_y = height * view_y
            layer.location[0] = self.start_layer_location[0] + target_x - self.start_input_position[0]
            layer.location[1] = self.start_layer_location[1] + target_y - self.start_input_position[1]
        elif event.type == 'LEFTMOUSE':
            utils.layer_rebuild_image_layers_nodes(img)
            area_session.layer_moving = False
            area_session.prevent_layer_update_event = False
            return {'FINISHED'}
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            layer.location = self.start_layer_location
            area_session.layer_moving = False
            area_session.prevent_layer_update_event = False
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        area_session = utils.layer_get_area_session(context)
        img = context.area.spaces.active.image
        width, height = img.size[0], img.size[1]
        layer = utils.layer_get_active_layer(context)
        if not layer:
            return {'CANCELLED'}
        if layer.locked:
            self.report({'WARNING'}, 'Layer is locked.')
            return {'CANCELLED'}
        region_pos = [event.mouse_region_x, event.mouse_region_y]
        view_x, view_y = context.region.view2d.region_to_view(*region_pos)
        self.start_input_position = [width * view_x, height * view_y]
        self.start_layer_location = layer.location[:]
        area_session.layer_moving = True
        area_session.prevent_layer_update_event = True
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

class IMAGE_EDIT_OT_delete_layer(bpy.types.Operator):
    bl_idname = "image_edit.delete_layer"
    bl_label = "Delete Layer"

    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        img_props = img.image_edit_properties
        layers = img_props.layers
        selected_layer_index = img_props.selected_layer_index
        if selected_layer_index == -1 or selected_layer_index >= len(layers):
            return {'CANCELLED'}
        layer = layers[selected_layer_index]
        if layer.locked:
            self.report({'WARNING'}, 'Layer is locked.')
            return {'CANCELLED'}
        layer_img = bpy.data.images.get(layer.name, None)
        if layer_img:
            bpy.data.images.remove(layer_img)
        layers.remove(selected_layer_index)
        selected_layer_index = min(max(selected_layer_index, 0), len(layers) - 1)
        img_props.selected_layer_index = selected_layer_index
        utils.layer_rebuild_image_layers_nodes(img)
        return {'FINISHED'}

class IMAGE_EDIT_OT_edit_layer(bpy.types.Operator):
    """Toggle layer edit mode - paint directly on the selected layer"""
    bl_idname = "image_edit.edit_layer"
    bl_label = "Edit Layer"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        if context.area.type != 'IMAGE_EDITOR':
            return False
        img = context.area.spaces.active.image
        if not img:
            return False
        img_props = img.image_edit_properties
        # Allow if currently editing (to exit) or if a layer is selected (to enter)
        if img_props.editing_layer:
            return True
        if img_props.selected_layer_index >= 0 and img_props.selected_layer_index < len(img_props.layers):
            return True
        return False

    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        
        img_props = img.image_edit_properties
        
        # Toggle mode
        if img_props.editing_layer:
            # Exit edit mode
            if utils.layer_exit_edit_mode(context):
                self.report({'INFO'}, 'Exited layer edit mode')
                return {'FINISHED'}
            else:
                self.report({'WARNING'}, 'Failed to exit layer edit mode')
                return {'CANCELLED'}
        else:
            # Enter edit mode
            layer = utils.layer_get_active_layer(context)
            if layer and layer.locked:
                self.report({'WARNING'}, 'Layer is locked')
                return {'CANCELLED'}
            
            if utils.layer_enter_edit_mode(context):
                self.report({'INFO'}, 'Editing layer - paint directly on layer image')
                return {'FINISHED'}
            else:
                self.report({'WARNING'}, 'No layer selected')
                return {'CANCELLED'}

class IMAGE_EDIT_OT_duplicate_layer(bpy.types.Operator):

    """Duplicate the selected layer"""
    bl_idname = "image_edit.duplicate_layer"
    bl_label = "Duplicate Layer"

    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        layer = utils.layer_get_active_layer(context)
        if not layer:
            return {'CANCELLED'}
        
        layer_img = bpy.data.images.get(layer.name, None)
        if not layer_img:
            return {'CANCELLED'}
        
        pixels = utils.layer_read_pixels_from_image(layer_img)
        img_settings = {'is_float': layer_img.is_float, 'colorspace_name': layer_img.colorspace_settings.name}
        layer_settings = {'rotation': layer.rotation, 'scale': list(layer.scale), 'custom_data': layer.custom_data}
        
        utils.layer_create_layer(img, pixels, img_settings, layer_settings, custom_label=layer.label + " Copy")
        utils.layer_refresh_image(context)
        self.report({'INFO'}, 'Layer duplicated.')
        return {'FINISHED'}

class IMAGE_EDIT_OT_lock_all_layers(bpy.types.Operator):
    """Lock all layers"""
    bl_idname = "image_edit.lock_all_layers"
    bl_label = "Lock All Layers"

    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        img_props = img.image_edit_properties
        for layer in img_props.layers:
            layer.locked = True
        self.report({'INFO'}, 'All layers locked.')
        return {'FINISHED'}

class IMAGE_EDIT_OT_unlock_all_layers(bpy.types.Operator):
    """Unlock all layers"""
    bl_idname = "image_edit.unlock_all_layers"
    bl_label = "Unlock All Layers"

    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        img_props = img.image_edit_properties
        for layer in img_props.layers:
            layer.locked = False
        self.report({'INFO'}, 'All layers unlocked.')
        return {'FINISHED'}

class IMAGE_EDIT_OT_hide_all_layers(bpy.types.Operator):
    """Hide all layers"""
    bl_idname = "image_edit.hide_all_layers"
    bl_label = "Hide All Layers"

    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        img_props = img.image_edit_properties
        for layer in img_props.layers:
            layer.hide = True
        utils.layer_refresh_image(context)
        self.report({'INFO'}, 'All layers hidden.')
        return {'FINISHED'}

class IMAGE_EDIT_OT_show_all_layers(bpy.types.Operator):
    """Show all layers"""
    bl_idname = "image_edit.show_all_layers"
    bl_label = "Show All Layers"

    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        img_props = img.image_edit_properties
        for layer in img_props.layers:
            layer.hide = False
        utils.layer_refresh_image(context)
        self.report({'INFO'}, 'All layers shown.')
        return {'FINISHED'}

class IMAGE_EDIT_OT_delete_all_layers(bpy.types.Operator):
    """Delete all layers"""
    bl_idname = "image_edit.delete_all_layers"
    bl_label = "Delete All Layers"

    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        img_props = img.image_edit_properties
        layers = img_props.layers
        
        for layer in layers:
            layer_img = bpy.data.images.get(layer.name, None)
            if layer_img:
                bpy.data.images.remove(layer_img)
        
        layers.clear()
        img_props.selected_layer_index = -1
        utils.layer_rebuild_image_layers_nodes(img)
        utils.layer_refresh_image(context)
        self.report({'INFO'}, 'All layers deleted.')
        return {'FINISHED'}

class IMAGE_EDIT_OT_update_layer_previews(bpy.types.Operator):
    """Update all layer preview thumbnails"""
    bl_idname = "image_edit.update_layer_previews"
    bl_label = "Update Layer Previews"

    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        img_props = img.image_edit_properties
        
        for layer in img_props.layers:
            layer_img = bpy.data.images.get(layer.name, None)
            if layer_img:
                layer_img.update()
                if layer_img.preview:
                    layer_img.preview.reload()
        
        img.update()
        if img.preview:
            img.preview.reload()
        
        context.area.tag_redraw()
        self.report({'INFO'}, 'Layer previews updated.')
        return {'FINISHED'}

class IMAGE_EDIT_OT_select_all_layers(bpy.types.Operator):
    """Select all layers"""
    bl_idname = "image_edit.select_all_layers"
    bl_label = "Select All Layers"

    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        img_props = img.image_edit_properties
        for layer in img_props.layers:
            layer.checked = True
        self.report({'INFO'}, 'All layers selected.')
        return {'FINISHED'}

class IMAGE_EDIT_OT_deselect_all_layers(bpy.types.Operator):
    """Deselect all layers"""
    bl_idname = "image_edit.deselect_all_layers"
    bl_label = "Deselect All Layers"

    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        img_props = img.image_edit_properties
        for layer in img_props.layers:
            layer.checked = False
        self.report({'INFO'}, 'All layers deselected.')
        return {'FINISHED'}

class IMAGE_EDIT_OT_invert_layer_selection(bpy.types.Operator):
    """Invert layer selection"""
    bl_idname = "image_edit.invert_layer_selection"
    bl_label = "Invert Layer Selection"

    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        img_props = img.image_edit_properties
        for layer in img_props.layers:
            layer.checked = not layer.checked
        self.report({'INFO'}, 'Layer selection inverted.')
        return {'FINISHED'}

class IMAGE_EDIT_OT_delete_selected_layers(bpy.types.Operator):
    """Delete all selected (checked) layers"""
    bl_idname = "image_edit.delete_selected_layers"
    bl_label = "Delete Selected Layers"

    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        img_props = img.image_edit_properties
        layers = img_props.layers
        
        indices_to_remove = []
        for i, layer in enumerate(layers):
            if layer.checked and not layer.locked:
                indices_to_remove.append(i)
        
        if not indices_to_remove:
            self.report({'WARNING'}, 'No unlocked layers selected.')
            return {'CANCELLED'}
        
        for i in reversed(indices_to_remove):
            layer = layers[i]
            layer_img = bpy.data.images.get(layer.name, None)
            if layer_img:
                bpy.data.images.remove(layer_img)
            layers.remove(i)
        
        if len(layers) > 0:
            img_props.selected_layer_index = min(img_props.selected_layer_index, len(layers) - 1)
        else:
            img_props.selected_layer_index = -1
        
        utils.layer_rebuild_image_layers_nodes(img)
        utils.layer_refresh_image(context)
        self.report({'INFO'}, f'{len(indices_to_remove)} layers deleted.')
        return {'FINISHED'}

class IMAGE_EDIT_OT_merge_selected_layers(bpy.types.Operator):
    """Merge all selected (checked) layers"""
    bl_idname = "image_edit.merge_selected_layers"
    bl_label = "Merge Selected Layers"

    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        width, height = img.size
        img_props = img.image_edit_properties
        layers = img_props.layers
        
        checked_layers = [(i, layer) for i, layer in enumerate(layers) if layer.checked]
        
        if len(checked_layers) < 2:
            self.report({'WARNING'}, 'Select at least 2 layers to merge.')
            return {'CANCELLED'}
        
        pixels = utils.layer_read_pixels_from_image(img)
        
        merged_count = 0
        indices_to_remove = []
        
        for i, layer in reversed(checked_layers):
            layer_img = bpy.data.images.get(layer.name, None)
            if not layer_img:
                continue
            layer_width, layer_height = layer_img.size[0], layer_img.size[1]
            layer_pos = layer.location
            layer_x1, layer_y1 = layer_pos[0], height - layer_height - layer_pos[1]
            
            if layer.rotation == 0 and layer.scale[0] == 1.0 and layer.scale[1] == 1.0:
                layer_pixels = utils.layer_read_pixels_from_image(layer_img)
            else:
                layer_pixels, new_layer_width, new_layer_height = utils.layer_apply_layer_transform(layer_img, layer.rotation, layer.scale)
                layer_x1 = int(layer_x1 - (new_layer_width - layer_width) / 2.0)
                layer_y1 = int(layer_y1 - (new_layer_height - layer_height) / 2.0)
                layer_width = new_layer_width
                layer_height = new_layer_height
            
            layer_x2 = layer_x1 + layer_width
            layer_y2 = layer_y1 + layer_height
            target_x1 = max(min(layer_x1, width), 0)
            target_y1 = max(min(layer_y1, height), 0)
            target_x2 = max(min(layer_x2, width), 0)
            target_y2 = max(min(layer_y2, height), 0)
            
            if layer_x1 == layer_x2 or layer_y1 == layer_y2:
                continue
            
            src_x1 = target_x1 - layer_x1
            src_y1 = target_y1 - layer_y1
            src_x2 = layer_width - (layer_x2 - target_x2)
            src_y2 = layer_height - (layer_y2 - target_y2)
            
            target_range = pixels[target_y1:target_y2, target_x1:target_x2]
            target_color_chan = target_range[:, :, :3]
            target_alpha_chan = target_range[:, :, 3:4]
            layer_range = layer_pixels[src_y1:src_y2, src_x1:src_x2]
            layer_color_chan = layer_range[:, :, :3]
            layer_alpha_chan = layer_range[:, :, 3:4]
            temp_alpha_chan = target_alpha_chan * (1.0 - layer_alpha_chan) + layer_alpha_chan
            temp_alpha_chan_safe = np.where(temp_alpha_chan == 0, 1.0, temp_alpha_chan)
            pixels[target_y1:target_y2, target_x1:target_x2, :3] = (target_color_chan * target_alpha_chan * (1.0 - layer_alpha_chan) + layer_color_chan * layer_alpha_chan) / temp_alpha_chan_safe
            pixels[target_y1:target_y2, target_x1:target_x2, 3:4] = temp_alpha_chan
            
            bpy.data.images.remove(layer_img)
            indices_to_remove.append(i)
            merged_count += 1
        
        for i in sorted(indices_to_remove, reverse=True):
            layers.remove(i)
        
        utils.ImageUndoStack.get().push_state(img)
        utils.layer_write_pixels_to_image(img, pixels)
        
        if len(layers) > 0:
            img_props.selected_layer_index = min(img_props.selected_layer_index, len(layers) - 1)
        else:
            img_props.selected_layer_index = -1
        
        utils.layer_rebuild_image_layers_nodes(img)
        utils.layer_refresh_image(context)
        self.report({'INFO'}, f'{merged_count} layers merged.')
        return {'FINISHED'}

class IMAGE_EDIT_OT_change_image_layer_order(bpy.types.Operator):
    bl_idname = "image_edit.change_image_layer_order"
    bl_label = "Change Image Layer Order"
    up: bpy.props.BoolProperty()

    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        img_props = img.image_edit_properties
        layers = img_props.layers
        selected_layer_index = img_props.selected_layer_index
        if selected_layer_index == -1 or selected_layer_index >= len(layers):
            return {'CANCELLED'}
        if (self.up and selected_layer_index == 0) or (not self.up and selected_layer_index >= len(layers) - 1):
            return {'CANCELLED'}
        new_layer_index = selected_layer_index + (-1 if self.up else 1)
        layers.move(selected_layer_index, new_layer_index)
        img_props.selected_layer_index = new_layer_index
        utils.layer_rebuild_image_layers_nodes(img)
        return {'FINISHED'}

class IMAGE_EDIT_OT_merge_layers(bpy.types.Operator):
    """Merge all layers"""
    bl_idname = "image_edit.merge_layers"
    bl_label = "Merge Layers"

    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        width, height = img.size
        pixels = utils.layer_read_pixels_from_image(img)
        img_props = img.image_edit_properties
        layers = img_props.layers
        for layer in reversed(layers):
            layer_img = bpy.data.images.get(layer.name, None)
            if not layer_img:
                continue
            layer_width, layer_height = layer_img.size[0], layer_img.size[1]
            layer_pos = layer.location
            layer_x1, layer_y1 = layer_pos[0], height - layer_height - layer_pos[1]
            if layer.rotation == 0 and layer.scale[0] == 1.0 and layer.scale[1] == 1.0:
                layer_pixels = utils.layer_read_pixels_from_image(layer_img)
            else:
                layer_pixels, new_layer_width, new_layer_height = utils.layer_apply_layer_transform(layer_img, layer.rotation, layer.scale)
                layer_x1 = int(layer_x1 - (new_layer_width - layer_width) / 2.0)
                layer_y1 = int(layer_y1 - (new_layer_height - layer_height) / 2.0)
                layer_width = new_layer_width
                layer_height = new_layer_height
            layer_x2 = layer_x1 + layer_width
            layer_y2 = layer_y1 + layer_height
            target_x1 = max(min(layer_x1, width), 0)
            target_y1 = max(min(layer_y1, height), 0)
            target_x2 = max(min(layer_x2, width), 0)
            target_y2 = max(min(layer_y2, height), 0)
            if layer_x1 == layer_x2 or layer_y1 == layer_y2:
                continue
            src_x1 = target_x1 - layer_x1
            src_y1 = target_y1 - layer_y1
            src_x2 = layer_width - (layer_x2 - target_x2)
            src_y2 = layer_height - (layer_y2 - target_y2)
            target_range = pixels[target_y1:target_y2, target_x1:target_x2]
            target_color_chan = target_range[:, :, :3]
            target_alpha_chan = target_range[:, :, 3:4]
            layer_range = layer_pixels[src_y1:src_y2, src_x1:src_x2]
            layer_color_chan = layer_range[:, :, :3]
            layer_alpha_chan = layer_range[:, :, 3:4]
            temp_alpha_chan = target_alpha_chan * (1.0 - layer_alpha_chan) + layer_alpha_chan
            temp_alpha_chan_safe = np.where(temp_alpha_chan == 0, 1.0, temp_alpha_chan)
            pixels[target_y1:target_y2, target_x1:target_x2, :3] = (target_color_chan * target_alpha_chan * (1.0 - layer_alpha_chan) + layer_color_chan * layer_alpha_chan) / temp_alpha_chan_safe
            pixels[target_y1:target_y2, target_x1:target_x2, 3:4] = temp_alpha_chan
            bpy.data.images.remove(layer_img)
        utils.ImageUndoStack.get().push_state(img)
        utils.layer_write_pixels_to_image(img, pixels)
        layers.clear()
        utils.layer_rebuild_image_layers_nodes(img)
        utils.layer_refresh_image(context)
        return {'FINISHED'}

class IMAGE_EDIT_OT_flip_layer(bpy.types.Operator):
    """Flip the layer"""
    bl_idname = "image_edit.flip_layer"
    bl_label = "Flip Layer"
    is_vertically: bpy.props.BoolProperty(name="Vertically", default=False)

    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        layer = utils.layer_get_active_layer(context)
        if not layer:
            return {'CANCELLED'}
        if layer.locked:
            self.report({'WARNING'}, 'Layer is locked.')
            return {'CANCELLED'}
        if self.is_vertically:
            layer.scale[1] *= -1.0
        else:
            layer.scale[0] *= -1.0
        utils.layer_refresh_image(context)
        return {'FINISHED'}

class IMAGE_EDIT_OT_rotate_layer(bpy.types.Operator):
    """Rotate the layer"""
    bl_idname = "image_edit.rotate_layer"
    bl_label = "Rotate Layer"
    is_left: bpy.props.BoolProperty(name="Left", default=False)

    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        layer = utils.layer_get_active_layer(context)
        if not layer:
            return {'CANCELLED'}
        if layer.locked:
            self.report({'WARNING'}, 'Layer is locked.')
            return {'CANCELLED'}
        layer.rotation += math.pi / 2.0 if self.is_left else -math.pi / 2.0
        utils.layer_refresh_image(context)
        return {'FINISHED'}

class IMAGE_EDIT_OT_rotate_layer_arbitrary(bpy.types.Operator):
    """Rotate the image by a specified angle"""
    bl_idname = "image_edit.rotate_layer_arbitrary"
    bl_label = "Rotate Layer Arbitrary"

    bl_options = {'REGISTER', 'UNDO'}
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_input_position = [0, 0]
        self.start_layer_angle = 0

    def modal(self, context, event):
        area_session = utils.layer_get_area_session(context)
        context.area.tag_redraw()
        img = context.area.spaces.active.image
        width, height = img.size[0], img.size[1]
        layer = utils.layer_get_active_layer(context)
        if not layer:
            return {'CANCELLED'}
        layer_width, layer_height = 1, 1
        layer_img = bpy.data.images.get(layer.name, None)
        if layer_img:
            layer_width, layer_height = layer_img.size[0], layer_img.size[1]
        if event.type == 'MOUSEMOVE':
            center_x = width / 2.0 + layer.location[0]
            center_y = height / 2.0 + layer.location[1]
            region_pos = [event.mouse_region_x, event.mouse_region_y]
            view_x, view_y = context.region.view2d.region_to_view(*region_pos)
            target_x = width * view_x
            target_y = height * view_y
            angle1 = math.atan2(self.start_input_position[1] - center_y, self.start_input_position[0] - center_x)
            angle2 = math.atan2(target_y - center_y, target_x - center_x)
            layer.rotation = self.start_layer_angle + angle2 - angle1
        elif event.type == 'LEFTMOUSE':
            utils.layer_rebuild_image_layers_nodes(img)
            area_session.layer_rotating = False
            area_session.prevent_layer_update_event = False
            return {'FINISHED'}
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            layer.rotation = self.start_layer_angle
            area_session.layer_rotating = False
            area_session.prevent_layer_update_event = False
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        area_session = utils.layer_get_area_session(context)
        img = context.area.spaces.active.image
        width, height = img.size[0], img.size[1]
        layer = utils.layer_get_active_layer(context)
        if not layer:
            return {'CANCELLED'}
        if layer.locked:
            self.report({'WARNING'}, 'Layer is locked.')
            return {'CANCELLED'}
        region_pos = [event.mouse_region_x, event.mouse_region_y]
        view_x, view_y = context.region.view2d.region_to_view(*region_pos)
        self.start_input_position = [width * view_x, height * view_y]
        self.start_layer_angle = layer.rotation
        area_session.layer_rotating = True
        area_session.prevent_layer_update_event = True
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

class IMAGE_EDIT_OT_scale_layer(bpy.types.Operator):
    """Scale the layer"""
    bl_idname = "image_edit.scale_layer"
    bl_label = "Scale Layer"

    bl_options = {'REGISTER', 'UNDO'}
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_input_position = [0, 0]
        self.start_layer_scale_x = 1.0
        self.start_layer_scale_y = 1.0

    def modal(self, context, event):
        area_session = utils.layer_get_area_session(context)
        context.area.tag_redraw()
        img = context.area.spaces.active.image
        width, height = img.size[0], img.size[1]
        layer = utils.layer_get_active_layer(context)
        if not layer:
            return {'CANCELLED'}
        layer_width, layer_height = 1, 1
        layer_img = bpy.data.images.get(layer.name, None)
        if layer_img:
            layer_width, layer_height = layer_img.size[0], layer_img.size[1]
        if event.type == 'MOUSEMOVE':
            center_x = width / 2.0 + layer.location[0]
            center_y = height / 2.0 + layer.location[1]
            region_pos = [event.mouse_region_x, event.mouse_region_y]
            view_x, view_y = context.region.view2d.region_to_view(*region_pos)
            target_x = width * view_x
            target_y = height * view_y
            dist1 = math.hypot(self.start_input_position[0] - center_x, self.start_input_position[1] - center_y)
            dist2 = math.hypot(target_x - center_x, target_y - center_y)
            layer.scale[0] = self.start_layer_scale_x * dist2 / dist1
            layer.scale[1] = self.start_layer_scale_y * dist2 / dist1
        elif event.type == 'LEFTMOUSE':
            utils.layer_rebuild_image_layers_nodes(img)
            area_session.layer_scaling = False
            area_session.prevent_layer_update_event = False
            return {'FINISHED'}
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            layer.scale[0] = self.start_layer_scale_x
            layer.scale[1] = self.start_layer_scale_y
            area_session.layer_scaling = False
            area_session.prevent_layer_update_event = False
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        area_session = utils.layer_get_area_session(context)
        img = context.area.spaces.active.image
        width, height = img.size[0], img.size[1]
        layer = utils.layer_get_active_layer(context)
        if not layer:
            return {'CANCELLED'}
        if layer.locked:
            self.report({'WARNING'}, 'Layer is locked.')
            return {'CANCELLED'}
        region_pos = [event.mouse_region_x, event.mouse_region_y]
        view_x, view_y = context.region.view2d.region_to_view(*region_pos)
        self.start_input_position = [width * view_x, height * view_y]
        self.start_layer_scale_x = layer.scale[0]
        self.start_layer_scale_y = layer.scale[1]
        area_session.layer_scaling = True
        area_session.prevent_layer_update_event = True
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}


class IMAGE_EDIT_OT_sculpt_image(bpy.types.Operator):
    """Sculpt the image with brush-based pixel warping"""
    bl_idname = "image_edit.sculpt_image"
    bl_label = "Image Sculpt"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return (context.area.type == 'IMAGE_EDITOR' and 
                context.area.spaces.active.image is not None)

    def modal(self, context, event):
        import numpy as np
        
        if event.type == 'MOUSEMOVE':
            if self.last_pos is not None:
                region_pos = (event.mouse_region_x, event.mouse_region_y)
                view_x, view_y = context.region.view2d.region_to_view(*region_pos)
                curr_x = int(view_x * self.width)
                curr_y = int(view_y * self.height)
                
                wm = context.window_manager
                props = wm.image_edit_properties
                
                dx = curr_x - self.last_pos[0]
                dy = curr_y - self.last_pos[1]
                
                if dx != 0 or dy != 0:
                    # Apply to working buffer
                    self._apply_to_buffer(curr_x, curr_y, dx, dy, 
                                         props.sculpt_mode, props.sculpt_radius, props.sculpt_strength)
                    self.modified = True
                    self.frame_count += 1
                    
                    # Real-time feedback: update image every few frames for performance
                    if self.frame_count % 3 == 0:
                        self.img.pixels.foreach_set(self.working.ravel())
                        self.img.update()
                        context.area.tag_redraw()
                
                self.last_pos = [curr_x, curr_y]
            
            return {'RUNNING_MODAL'}
        
        elif event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
            # Final write on release
            if self.modified:
                # Push undo state BEFORE finalizing (save the original pixels efficiently)
                utils.ImageUndoStack.get().push_state_from_numpy(self.img, self.original_pixels)
                
                # Now apply the final result
                self.img.pixels.foreach_set(self.working.ravel())
                self.img.update()
            context.area.tag_redraw()
            return {'FINISHED'}
        
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            # Cancel - restore original
            if self.modified:
                self.img.pixels.foreach_set(self.original_pixels)
                self.img.update()
            context.area.tag_redraw()
            return {'CANCELLED'}
        
        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        import numpy as np
        
        self.img = context.area.spaces.active.image
        if not self.img:
            return {'CANCELLED'}
        
        self.width, self.height = self.img.size[0], self.img.size[1]
        self.modified = False
        self.frame_count = 0
        
        # Read pixels once - use foreach_get for speed
        pixel_count = self.width * self.height * 4
        self.original_pixels = np.empty(pixel_count, dtype=np.float32)
        self.img.pixels.foreach_get(self.original_pixels)
        
        # Working copy reshaped for manipulation
        self.working = self.original_pixels.reshape((self.height, self.width, 4)).copy()
        
        region_pos = [event.mouse_region_x, event.mouse_region_y]
        view_x, view_y = context.region.view2d.region_to_view(*region_pos)
        self.last_pos = [int(view_x * self.width), int(view_y * self.height)]
        
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def _apply_to_buffer(self, cx, cy, dx, dy, mode, radius, strength):
        """MLS Rigid deformation - O(k) per pixel, highly optimized."""
        import numpy as np
        import bpy
        
        # Get falloff preset from properties
        props = bpy.context.window_manager.image_edit_properties
        falloff_preset = props.sculpt_falloff_preset
        
        # Bounding box
        x1 = max(0, cx - radius)
        y1 = max(0, cy - radius)
        x2 = min(self.width, cx + radius)
        y2 = min(self.height, cy + radius)
        
        if x1 >= x2 or y1 >= y2:
            return
        
        # Coordinate grids (float32 for speed)
        py = np.arange(y1, y2, dtype=np.float32)
        px = np.arange(x1, x2, dtype=np.float32)
        gx, gy = np.meshgrid(px, py)
        
        # Distance from brush center
        rel_x = gx - cx
        rel_y = gy - cy
        dist_sq = rel_x * rel_x + rel_y * rel_y
        radius_sq = float(radius * radius)
        
        # Mask for valid region
        mask = dist_sq < radius_sq
        if not np.any(mask):
            return
        
        # Calculate normalized distance t (0 at center, 1 at edge)
        dist = np.sqrt(dist_sq)
        t = np.clip(dist / radius, 0.0, 1.0)
        
        # Calculate falloff based on preset (vectorized for performance)
        if falloff_preset == 'SMOOTH':
            # Hermite smoothstep: 3t² - 2t³
            weights = (1.0 - (3.0 * t**2 - 2.0 * t**3)) * strength
        elif falloff_preset == 'SMOOTHER':
            # Perlin smootherstep: 6t⁵ - 15t⁴ + 10t³
            weights = (1.0 - (6.0 * t**5 - 15.0 * t**4 + 10.0 * t**3)) * strength
        elif falloff_preset == 'SPHERE':
            # Spherical: sqrt(1 - t²)
            weights = np.sqrt(np.clip(1.0 - t**2, 0.0, 1.0)) * strength
        elif falloff_preset == 'ROOT':
            # Root: 1 - sqrt(t)
            weights = (1.0 - np.sqrt(t)) * strength
        elif falloff_preset == 'SHARP':
            # Sharp: (1 - t)²
            weights = ((1.0 - t)**2) * strength
        elif falloff_preset == 'LINEAR':
            # Linear: 1 - t
            weights = (1.0 - t) * strength
        elif falloff_preset == 'CONSTANT':
            # Constant: no falloff (hard edge)
            weights = np.where(mask, strength, 0.0)
        elif falloff_preset == 'CUSTOM':
            # Use brush curve - evaluate per-pixel (slower but accurate)
            brush = bpy.context.tool_settings.image_paint.brush
            if brush and brush.curve_distance_falloff:
                curve = brush.curve_distance_falloff
                weights = np.zeros_like(t)
                # Vectorized curve evaluation using pre-sampled LUT for performance
                lut_size = 256
                lut = np.array([curve.evaluate(curve.curves[0], i / (lut_size - 1)) 
                               for i in range(lut_size)], dtype=np.float32)
                t_indices = np.clip((t * (lut_size - 1)).astype(np.int32), 0, lut_size - 1)
                weights = lut[t_indices] * strength
            else:
                # Fallback to smooth
                weights = (1.0 - (3.0 * t**2 - 2.0 * t**3)) * strength
        else:
            # Default to smooth
            weights = (1.0 - (3.0 * t**2 - 2.0 * t**3)) * strength
        
        # Apply mask
        weights = np.where(mask, weights, 0.0)
        
        # Control point: brush center
        p_x, p_y = float(cx), float(cy)
        
        # MLS Rigid transformation based on mode
        if mode == 'GRAB':
            # Weighted translation
            offset_x = dx * weights
            offset_y = dy * weights
            src_x = gx - offset_x
            src_y = gy - offset_y
            
        elif mode == 'PINCH':
            # Scale toward center
            scale = 1.0 + weights * 0.5
            src_x = p_x + rel_x * scale
            src_y = p_y + rel_y * scale
        elif mode == 'DRIP':
            # Realistic paint drip effect - teardrop shape
            # Distance from vertical center line (for tapering)
            dist_from_center = np.abs(rel_x)
            
            # Taper factor: 1 at center, 0 at edges (creates narrow drip shape)
            taper = np.maximum(0, 1 - dist_from_center / (radius * 0.3))
            taper = taper ** 0.5  # Softer taper
            
            # Drip amount: stronger in center, combines with gaussian weight
            drip_amount = weights * taper * radius * strength
            
            # Create elongation: pixels stretch downward from where brush is
            # Sample from above to create the stretching effect
            src_x = gx
            src_y = gy - drip_amount
            
            # Add bulge at bottom: pixels near brush center get extra stretch
            center_boost = np.exp(-dist_from_center**2 / (radius * 0.15)**2)
            src_y = src_y - center_boost * weights * radius * 0.2
            
        elif mode == 'WAVE':
            # Ripple/wave distortion - concentric rings from center
            dist = np.sqrt(dist_sq) + 0.001  # avoid division by zero
            
            # Wave parameters
            frequency = 0.3  # waves per pixel
            amplitude = weights * radius * 0.15
            
            # Radial wave displacement
            wave = np.sin(dist * frequency * 2 * np.pi) * amplitude
            
            # Displace perpendicular to radius (creates ripple effect)
            norm_x = rel_x / dist
            norm_y = rel_y / dist
            src_x = gx + norm_x * wave
            src_y = gy + norm_y * wave
            
        elif mode == 'JITTER':
            # Turbulence/noise displacement
            # Use hash-based pseudo-random for deterministic jitter
            hash_x = np.sin(gx * 12.9898 + gy * 78.233) * 43758.5453
            hash_y = np.sin(gx * 78.233 + gy * 12.9898) * 43758.5453
            noise_x = (hash_x - np.floor(hash_x)) * 2 - 1  # -1 to 1
            noise_y = (hash_y - np.floor(hash_y)) * 2 - 1
            
            # Apply weighted displacement
            jitter_amount = weights * radius * 0.2
            src_x = gx + noise_x * jitter_amount
            src_y = gy + noise_y * jitter_amount
            
        elif mode == 'HAZE':
            # Heat haze - vertical refractive shimmer
            # Use position-based sine for shimmer pattern
            phase = gy * 0.3 + gx * 0.1
            shimmer_x = np.sin(phase) * weights * radius * 0.1
            shimmer_y = np.sin(phase * 1.3 + 1.0) * weights * radius * 0.05
            
            src_x = gx + shimmer_x
            src_y = gy + shimmer_y
            
        elif mode == 'ERODE':
            # Edge breaking - displace based on local contrast/gradient
            # Sample luminance gradient using neighbors
            patch = self.working[y1:y2, x1:x2]
            lum = 0.299 * patch[:,:,0] + 0.587 * patch[:,:,1] + 0.114 * patch[:,:,2]
            
            # Compute gradient using Sobel-like kernel
            grad_x = np.zeros_like(lum)
            grad_y = np.zeros_like(lum)
            if lum.shape[0] > 2 and lum.shape[1] > 2:
                grad_x[1:-1, 1:-1] = lum[1:-1, 2:] - lum[1:-1, :-2]
                grad_y[1:-1, 1:-1] = lum[2:, 1:-1] - lum[:-2, 1:-1]
            
            # Displace along gradient (erodes edges)
            erode_strength = weights * radius * 0.3
            src_x = gx + grad_x * erode_strength
            src_y = gy + grad_y * erode_strength
            
        elif mode == 'CREASE':
            # Sharp linear deformation - crease along drag direction
            # Project onto drag direction for sharp line effect
            drag_len = np.sqrt(dx*dx + dy*dy) + 0.001
            drag_nx = dx / drag_len
            drag_ny = dy / drag_len
            
            # Distance along drag direction
            proj = rel_x * drag_nx + rel_y * drag_ny
            
            # Sharp crease using tanh for step-like transition
            crease = np.tanh(proj * 0.2) * weights * radius * 0.2
            
            # Displace perpendicular to drag
            src_x = gx - drag_ny * crease
            src_y = gy + drag_nx * crease
            
        elif mode == 'BRISTLE':
            # Directional streaks and striations
            drag_len = np.sqrt(dx*dx + dy*dy) + 0.001
            drag_nx, drag_ny = dx / drag_len, dy / drag_len
            # Parallel striations 
            striation = np.sin((rel_x * drag_ny - rel_y * drag_nx) * 0.5) * 0.5 + 0.5
            offset_x = dx * weights * striation * 0.3
            offset_y = dy * weights * striation * 0.3
            src_x = gx - offset_x
            src_y = gy - offset_y
            
        elif mode == 'DRYPULL':
            # Broken dry-brush skipped pixels
            skip = ((gx.astype(np.int32) + gy.astype(np.int32)) % 3 != 0).astype(np.float32)
            src_x = gx - dx * weights * skip * 0.3
            src_y = gy - dy * weights * skip * 0.3
            
        elif mode == 'BLOOM':
            # Soft expanding overlap like petals
            dist = np.sqrt(dist_sq) + 0.001
            expand = weights * radius * 0.2
            norm_x, norm_y = rel_x / dist, rel_y / dist
            src_x = gx - norm_x * expand
            src_y = gy - norm_y * expand
            
        elif mode == 'INFLATE':
            # Organic uneven bulging with noise
            dist = np.sqrt(dist_sq) + 0.001
            noise = np.sin(gx * 0.1 + gy * 0.13) * np.cos(gx * 0.07 - gy * 0.11)
            bulge = weights * radius * 0.15 * (1 + noise * 0.5)
            norm_x, norm_y = rel_x / dist, rel_y / dist
            src_x = gx - norm_x * bulge
            src_y = gy - norm_y * bulge
            
        elif mode == 'LIQUIFY':
            # Fluid-like warp deformation
            dist = np.sqrt(dist_sq) + 0.001
            fluid = np.sin(dist * 0.1) * weights * radius * 0.15
            norm_x, norm_y = rel_x / dist, rel_y / dist
            src_x = gx - dx * weights * 0.3 + norm_y * fluid
            src_y = gy - dy * weights * 0.3 - norm_x * fluid
            
        elif mode == 'SPIRAL':
            # Swirling vortex distortion
            dist = np.sqrt(dist_sq) + 0.001
            spiral_angle = weights * 2.0  # Strong spiral
            cos_s = np.cos(spiral_angle)
            sin_s = np.sin(spiral_angle)
            src_x = p_x + rel_x * cos_s - rel_y * sin_s
            src_y = p_y + rel_x * sin_s + rel_y * cos_s
            
        elif mode == 'STRETCH':
            # Directional elongation
            drag_len = np.sqrt(dx*dx + dy*dy) + 0.001
            drag_nx, drag_ny = dx / drag_len, dy / drag_len
            # Project onto drag direction
            proj = rel_x * drag_nx + rel_y * drag_ny
            stretch = weights * proj * 0.3
            src_x = gx - drag_nx * stretch
            src_y = gy - drag_ny * stretch
            
        elif mode == 'PIXELATE':
            # Pixelated mosaic effect - snap to grid
            grid_size = max(2, int(radius * 0.1 * strength) + 1)
            grid_x = (gx // grid_size) * grid_size + grid_size / 2
            grid_y = (gy // grid_size) * grid_size + grid_size / 2
            blend = weights
            src_x = gx * (1 - blend) + grid_x * blend
            src_y = gy * (1 - blend) + grid_y * blend
            
        elif mode == 'GLITCH':
            # Digital scan line displacement
            line_height = 3
            line_idx = (gy / line_height).astype(np.int32)
            offset = np.sin(line_idx * 3.7) * weights * radius * 0.3
            src_x = gx + offset
            src_y = gy
            
        else:
            return
        
        # Clamp source coordinates
        src_x = np.clip(src_x, 0, self.width - 1.001)
        src_y = np.clip(src_y, 0, self.height - 1.001)
        
        # Bilinear interpolation for quality
        x0 = src_x.astype(np.int32)
        y0 = src_y.astype(np.int32)
        x1i = np.minimum(x0 + 1, self.width - 1)
        y1i = np.minimum(y0 + 1, self.height - 1)
        
        fx = (src_x - x0)[:, :, np.newaxis]
        fy = (src_y - y0)[:, :, np.newaxis]
        
        # Sample 4 corners
        p00 = self.working[y0, x0]
        p10 = self.working[y0, x1i]
        p01 = self.working[y1i, x0]
        p11 = self.working[y1i, x1i]
        
        # Bilinear interpolate
        sampled = (p00 * (1 - fx) * (1 - fy) +
                   p10 * fx * (1 - fy) +
                   p01 * (1 - fx) * fy +
                   p11 * fx * fy)
        
        # Apply with mask
        patch = self.working[y1:y2, x1:x2]
        mask_3d = mask[:, :, np.newaxis]
        patch[:] = np.where(mask_3d, sampled, patch)


# ============================================================
# Lattice Deformation Tool
# ============================================================

class IMAGE_EDIT_OT_lattice_deform(bpy.types.Operator):
    """Deform image using lattice grid with perspective or mesh mode"""
    bl_idname = "image_edit.lattice_deform"
    bl_label = "Lattice Deform"
    bl_options = {'REGISTER', 'UNDO'}
    
    _draw_handler = None
    _image = None
    _original_pixels = None
    _working_pixels = None
    _width = 0
    _height = 0
    
    # Control points: grid [row][col] = [x, y]
    _control_points = []
    _original_grid = []
    
    # Interaction state
    _active_point = None
    _is_dragging = False
    _initialized = False
    
    HANDLE_SIZE = 10
    
    @classmethod
    def poll(cls, context):
        sima = context.space_data
        return (context.area.type == 'IMAGE_EDITOR' and 
                sima.mode == 'PAINT' and 
                sima.image is not None)
    
    def _init_grid(self, context):
        """Initialize control point grid based on resolution."""
        props = context.window_manager.image_edit_properties
        mode = props.lattice_mode
        
        if mode == 'PERSPECTIVE':
            res_u, res_v = 2, 2
        else:
            res_u = props.lattice_resolution_u
            res_v = props.lattice_resolution_v
        
        self._control_points = []
        self._original_grid = []
        
        for j in range(res_v):
            row, orig_row = [], []
            for i in range(res_u):
                x = (i / (res_u - 1)) * (self._width - 1) if res_u > 1 else self._width / 2
                y = (j / (res_v - 1)) * (self._height - 1) if res_v > 1 else self._height / 2
                row.append([x, y])
                orig_row.append([x, y])
            self._control_points.append(row)
            self._original_grid.append(orig_row)
    
    def _screen_to_image(self, context, mx, my):
        """Convert screen to image pixel coordinates."""
        view2d = context.region.view2d
        img_x, img_y = view2d.region_to_view(mx, my)
        return img_x * self._width, img_y * self._height
    
    def _image_to_screen(self, context, ix, iy):
        """Convert image to screen coordinates."""
        nx = ix / self._width if self._width > 0 else 0
        ny = iy / self._height if self._height > 0 else 0
        return context.region.view2d.view_to_region(nx, ny)
    
    def _hit_test(self, context, mx, my):
        """Check if mouse is over a control point."""
        ix, iy = self._screen_to_image(context, mx, my)
        hs = self.HANDLE_SIZE * 2
        for j, row in enumerate(self._control_points):
            for i, pt in enumerate(row):
                if abs(ix - pt[0]) < hs and abs(iy - pt[1]) < hs:
                    return (j, i)
        return None
    
    def _find_homography(self, src, dst):
        """Compute 3x3 homography using DLT algorithm."""
        import numpy as np
        n = src.shape[0]
        A = np.zeros((2*n, 9), dtype=np.float64)
        for i in range(n):
            x, y = src[i]
            xp, yp = dst[i]
            A[2*i] = [-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp]
            A[2*i+1] = [0, 0, 0, -x, -y, -1, x*yp, y*yp, yp]
        _, _, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape(3, 3)
        return H / H[2, 2]
    
    def _apply_perspective_warp(self):
        """Apply 4-point perspective transformation."""
        import numpy as np
        
        src = np.array([self._original_grid[0][0], self._original_grid[0][-1],
                        self._original_grid[-1][-1], self._original_grid[-1][0]], dtype=np.float32)
        dst = np.array([self._control_points[0][0], self._control_points[0][-1],
                        self._control_points[-1][-1], self._control_points[-1][0]], dtype=np.float32)
        
        H = self._find_homography(src, dst)
        H_inv = np.linalg.inv(H)
        
        y, x = np.meshgrid(np.arange(self._height, dtype=np.float32),
                           np.arange(self._width, dtype=np.float32), indexing='ij')
        coords = np.stack([x, y, np.ones_like(x)], axis=-1)
        src_coords = np.einsum('ij,...j->...i', H_inv, coords)
        
        w = src_coords[..., 2:3]
        w = np.where(np.abs(w) < 1e-10, 1e-10, w)
        src_x = src_coords[..., 0] / w[..., 0]
        src_y = src_coords[..., 1] / w[..., 0]
        
        self._bilinear_sample(src_x, src_y)
    
    def _apply_mesh_warp(self):
        """Apply mesh deformation with bilinear cell interpolation."""
        import numpy as np
        
        rows, cols = len(self._control_points), len(self._control_points[0])
        if rows < 2 or cols < 2:
            return
        
        y, x = np.meshgrid(np.arange(self._height, dtype=np.float32),
                           np.arange(self._width, dtype=np.float32), indexing='ij')
        src_x, src_y = np.copy(x), np.copy(y)
        
        for j in range(rows - 1):
            for i in range(cols - 1):
                o_tl, o_tr = self._original_grid[j][i], self._original_grid[j][i+1]
                o_bl, o_br = self._original_grid[j+1][i], self._original_grid[j+1][i+1]
                d_tl, d_tr = self._control_points[j][i], self._control_points[j][i+1]
                d_bl, d_br = self._control_points[j+1][i], self._control_points[j+1][i+1]
                
                mask = (x >= o_tl[0]) & (x < o_tr[0]) & (y >= o_tl[1]) & (y < o_bl[1])
                if not np.any(mask):
                    continue
                
                cw, ch = o_tr[0] - o_tl[0], o_bl[1] - o_tl[1]
                if cw < 1 or ch < 1:
                    continue
                
                u = (x[mask] - o_tl[0]) / cw
                v = (y[mask] - o_tl[1]) / ch
                
                src_x[mask] = (1-u)*(1-v)*d_tl[0] + u*(1-v)*d_tr[0] + (1-u)*v*d_bl[0] + u*v*d_br[0]
                src_y[mask] = (1-u)*(1-v)*d_tl[1] + u*(1-v)*d_tr[1] + (1-u)*v*d_bl[1] + u*v*d_br[1]
        
        self._bilinear_sample(src_x, src_y)
    
    def _bilinear_sample(self, src_x, src_y):
        """Sample original pixels with bilinear interpolation."""
        import numpy as np
        src_x = np.clip(src_x, 0, self._width - 1.001)
        src_y = np.clip(src_y, 0, self._height - 1.001)
        
        x0, y0 = src_x.astype(np.int32), src_y.astype(np.int32)
        x1, y1 = np.minimum(x0 + 1, self._width - 1), np.minimum(y0 + 1, self._height - 1)
        fx, fy = (src_x - x0)[:,:,np.newaxis], (src_y - y0)[:,:,np.newaxis]
        
        self._working_pixels[:] = (self._original_pixels[y0, x0] * (1-fx) * (1-fy) +
                                    self._original_pixels[y0, x1] * fx * (1-fy) +
                                    self._original_pixels[y1, x0] * (1-fx) * fy +
                                    self._original_pixels[y1, x1] * fx * fy)
    
    def _update_preview(self, context):
        """Apply deformation and update image preview."""
        props = context.window_manager.image_edit_properties
        if props.lattice_mode == 'PERSPECTIVE':
            self._apply_perspective_warp()
        else:
            self._apply_mesh_warp()
        self._image.pixels.foreach_set(self._working_pixels.ravel())
        self._image.update()
    
    def modal(self, context, event):
        context.area.tag_redraw()
        
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE'}:
            return {'PASS_THROUGH'}
        
        if event.type in {'RET', 'NUMPAD_ENTER', 'SPACE'} and event.value == 'PRESS':
            utils.ImageUndoStack.get().push_state_from_numpy(self._image, self._original_pixels)
            self._image.pixels.foreach_set(self._working_pixels.ravel())
            self._image.update()
            self._cleanup(context)
            self.report({'INFO'}, "Lattice deformation applied")
            return {'FINISHED'}
        
        if event.type == 'ESC' and event.value == 'PRESS':
            self._image.pixels.foreach_set(self._original_pixels.ravel())
            self._image.update()
            self._cleanup(context)
            return {'CANCELLED'}
        
        mx, my = event.mouse_region_x, event.mouse_region_y
        
        if event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                hit = self._hit_test(context, mx, my)
                if hit:
                    self._active_point = hit
                    self._is_dragging = True
            elif event.value == 'RELEASE':
                self._is_dragging = False
                self._active_point = None
        
        elif event.type == 'MOUSEMOVE' and self._is_dragging and self._active_point:
            row, col = self._active_point
            ix, iy = self._screen_to_image(context, mx, my)
            ix = max(0, min(self._width - 1, ix))
            iy = max(0, min(self._height - 1, iy))
            self._control_points[row][col] = [ix, iy]
            self._update_preview(context)
        
        return {'RUNNING_MODAL'}
    
    def invoke(self, context, event):
        import numpy as np
        
        self._image = context.space_data.image
        self._width, self._height = self._image.size
        
        if self._width < 2 or self._height < 2:
            self.report({'ERROR'}, "Image too small")
            return {'CANCELLED'}
        
        pixels = np.zeros(self._width * self._height * 4, dtype=np.float32)
        self._image.pixels.foreach_get(pixels)
        self._original_pixels = pixels.reshape((self._height, self._width, 4))
        self._working_pixels = np.copy(self._original_pixels)
        
        self._init_grid(context)
        self._initialized = True
        
        self._draw_handler = context.space_data.draw_handler_add(
            draw_lattice_overlay, (self, context), 'WINDOW', 'POST_PIXEL')
        
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}
    
    def _cleanup(self, context):
        if self._draw_handler:
            context.space_data.draw_handler_remove(self._draw_handler, 'WINDOW')
            self._draw_handler = None
        self._initialized = False


def draw_lattice_overlay(op, context):
    """Draw lattice grid and control points."""
    import gpu
    from gpu_extras.batch import batch_for_shader
    
    if not op._initialized or not op._control_points:
        return
    
    gpu.state.blend_set('ALPHA')
    gpu.state.line_width_set(1.5)
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    
    rows = len(op._control_points)
    cols = len(op._control_points[0]) if rows else 0
    
    # Grid lines
    lines = []
    for j in range(rows):
        for i in range(cols - 1):
            s1 = op._image_to_screen(context, *op._control_points[j][i])
            s2 = op._image_to_screen(context, *op._control_points[j][i+1])
            lines.extend([s1, s2])
    for j in range(rows - 1):
        for i in range(cols):
            s1 = op._image_to_screen(context, *op._control_points[j][i])
            s2 = op._image_to_screen(context, *op._control_points[j+1][i])
            lines.extend([s1, s2])
    
    if lines:
        shader.uniform_float("color", (0.4, 0.7, 1.0, 0.7))
        batch_for_shader(shader, 'LINES', {"pos": lines}).draw(shader)
    
    # Control points
    pts = [op._image_to_screen(context, *op._control_points[j][i]) 
           for j in range(rows) for i in range(cols)]
    if pts:
        gpu.state.point_size_set(10.0)
        shader.uniform_float("color", (0.2, 0.5, 0.9, 1.0))
        batch_for_shader(shader, 'POINTS', {"pos": pts}).draw(shader)
        gpu.state.point_size_set(6.0)
        shader.uniform_float("color", (1.0, 1.0, 1.0, 1.0))
        batch_for_shader(shader, 'POINTS', {"pos": pts}).draw(shader)
    
    gpu.state.blend_set('NONE')
    gpu.state.line_width_set(1.0)
    gpu.state.point_size_set(1.0)
class IMAGE_EDIT_OT_add_image_layer(bpy.types.Operator):
    """Add image file(s) as new layer(s)"""
    bl_idname = "image_edit.add_image_layer"
    bl_label = "Add Image as Layer"
    
    filepath: bpy.props.StringProperty(subtype='FILE_PATH')
    directory: bpy.props.StringProperty(subtype='DIR_PATH')
    files: bpy.props.CollectionProperty(type=bpy.types.OperatorFileListElement)
    filter_glob: bpy.props.StringProperty(default="*.png;*.jpg;*.jpeg;*.bmp;*.tga;*.tiff;*.exr;*.hdr", options={'HIDDEN'})

    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        import os
        img = context.area.spaces.active.image
        if not img:
            self.report({'ERROR'}, "No active image")
            return {'CANCELLED'}
        
        if self.files:
            filepaths = [os.path.join(self.directory, f.name) for f in self.files if f.name]
        else:
            filepaths = [self.filepath]
        
        added_count = 0
        for filepath in filepaths:
            if not filepath or not os.path.isfile(filepath):
                continue
            
            try:
                layer_source = bpy.data.images.load(filepath)
            except Exception:
                self.report({'WARNING'}, f"Could not load: {os.path.basename(filepath)}")
                continue
            
            original_filename = os.path.splitext(os.path.basename(filepath))[0]
            target_pixels = utils.layer_read_pixels_from_image(layer_source)
            
            img_settings = {
                'is_float': layer_source.is_float,
                'colorspace_name': layer_source.colorspace_settings.name
            }
            
            utils.layer_create_layer(img, target_pixels, img_settings, None, custom_label=original_filename)
            bpy.data.images.remove(layer_source)
            added_count += 1
        
        if added_count == 0:
            self.report({'ERROR'}, "No images were added")
            return {'CANCELLED'}
        
        utils.layer_cancel_selection(context)
        utils.layer_refresh_image(context)
        
        self.report({'INFO'}, f'{added_count} image(s) added as layers.')
        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

class IMAGE_EDIT_OT_new_image_layer(bpy.types.Operator):
    """Create a new blank image as a layer"""
    bl_idname = "image_edit.new_image_layer"
    bl_label = "New Image Layer"
    
    layer_name: bpy.props.StringProperty(name='Name', default='New Layer')
    width: bpy.props.IntProperty(name='Width', default=512, min=1, max=16384)
    height: bpy.props.IntProperty(name='Height', default=512, min=1, max=16384)
    color: bpy.props.FloatVectorProperty(name='Color', subtype='COLOR', size=4, min=0, max=1, default=(1.0, 1.0, 1.0, 0.0))
    use_base_size: bpy.props.BoolProperty(name='Use Base Image Size', default=True)

    bl_options = {'REGISTER', 'UNDO'}
    def invoke(self, context, event):
        img = context.area.spaces.active.image
        if img:
            self.width = img.size[0]
            self.height = img.size[1]
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            self.report({'ERROR'}, "No active image")
            return {'CANCELLED'}
        
        if self.use_base_size:
            layer_width = img.size[0]
            layer_height = img.size[1]
        else:
            layer_width = self.width
            layer_height = self.height
        
        pixels = np.full((layer_height, layer_width, 4), self.color, dtype=np.float32)
        
        img_settings = {
            'is_float': img.is_float,
            'colorspace_name': img.colorspace_settings.name
        }
        
        layer_label = self.layer_name if self.layer_name else "New Layer"
        utils.layer_create_layer(img, pixels, img_settings, None, custom_label=layer_label)
        
        utils.layer_cancel_selection(context)
        utils.layer_refresh_image(context)
        
        return {'FINISHED'}

    def draw(self, context):
        layout = self.layout
        layout.prop(self, 'layer_name')
        layout.prop(self, 'use_base_size')
        if not self.use_base_size:
            layout.prop(self, 'width')
            layout.prop(self, 'height')
        layout.prop(self, 'color')

class IMAGE_EDIT_OT_crop(bpy.types.Operator):
    """Crop the image to the boundary of the selection"""
    bl_idname = "image_edit.crop"
    bl_label = "Crop"

    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        pixels = utils.layer_read_pixels_from_image(img)
        selection = utils.layer_get_selection(context)
        if selection:
            target_pixels = pixels[selection[0][1]:selection[1][1], selection[0][0]:selection[1][0]]
        else:
            target_pixels = pixels
        target_width, target_height = target_pixels.shape[1], target_pixels.shape[0]
        img.scale(target_width, target_height)
        utils.layer_write_pixels_to_image(img, target_pixels)
        if selection:
            img_props = img.image_edit_properties
            layers = img_props.layers
            for layer in reversed(layers):
                layer_pos = layer.location
                layer_pos[0] -= selection[0][0]
                layer_pos[1] -= selection[0][1]
        utils.layer_cancel_selection(context)
        utils.layer_refresh_image(context)
        return {'FINISHED'}

class IMAGE_EDIT_OT_deselect_layer(bpy.types.Operator):
    bl_idname = "image_edit.deselect_layer"
    bl_label = "Deselect Layer"

    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        img_props = img.image_edit_properties
        img_props.selected_layer_index = -1
        utils.layer_refresh_image(context)
        return {'FINISHED'}

class IMAGE_EDIT_OT_move_layer(bpy.types.Operator):
    """Move the layer"""
    bl_idname = "image_edit.move_layer"
    bl_label = "Move Layer"

    # Numeric key type mapping
    _NUM_KEYS = {
        'ZERO': '0', 'ONE': '1', 'TWO': '2', 'THREE': '3', 'FOUR': '4',
        'FIVE': '5', 'SIX': '6', 'SEVEN': '7', 'EIGHT': '8', 'NINE': '9',
        'NUMPAD_0': '0', 'NUMPAD_1': '1', 'NUMPAD_2': '2', 'NUMPAD_3': '3',
        'NUMPAD_4': '4', 'NUMPAD_5': '5', 'NUMPAD_6': '6', 'NUMPAD_7': '7',
        'NUMPAD_8': '8', 'NUMPAD_9': '9',
        'PERIOD': '.', 'NUMPAD_PERIOD': '.',
        'MINUS': '-', 'NUMPAD_MINUS': '-',
    }

    bl_options = {'REGISTER', 'UNDO'}
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_input_position = [0, 0]
        self.start_layer_location = [0, 0]
        self.axis_constraint = None  # None, 'X', or 'Y'
        self.numeric_input = ''  # String buffer for typed numeric value

    def _apply_numeric(self, layer):
        """Apply numeric input value to the layer location."""
        try:
            value = float(self.numeric_input) if self.numeric_input else 0
        except ValueError:
            value = 0
        val = float(value)
        if self.axis_constraint == 'X':
            layer.location[0] = self.start_layer_location[0] + val
            layer.location[1] = self.start_layer_location[1]
        elif self.axis_constraint == 'Y':
            layer.location[0] = self.start_layer_location[0]
            layer.location[1] = self.start_layer_location[1] + val
        else:
            # No axis = apply to both
            layer.location[0] = self.start_layer_location[0] + val
            layer.location[1] = self.start_layer_location[1] + val

    def _update_header(self, context):
        axis = f" {self.axis_constraint}" if self.axis_constraint else ""
        if self.numeric_input:
            context.area.header_text_set(f"Move{axis}: {self.numeric_input}  (Enter to confirm)")
        else:
            context.area.header_text_set(f"Move{axis}: mouse  (X/Y axis, type number)")

    def modal(self, context, event):
        area_session = utils.layer_get_area_session(context)
        context.area.tag_redraw()
        img = context.area.spaces.active.image
        width, height = img.size[0], img.size[1]
        layer = utils.layer_get_active_layer(context)
        if not layer:
            return {'CANCELLED'}

        # Numeric input handling
        if event.type in self._NUM_KEYS and event.value == 'PRESS':
            char = self._NUM_KEYS[event.type]
            if char == '-':
                if self.numeric_input.startswith('-'):
                    self.numeric_input = self.numeric_input[1:]
                else:
                    self.numeric_input = '-' + self.numeric_input
            elif char == '.':
                if '.' not in self.numeric_input:
                    self.numeric_input += char
            else:
                self.numeric_input += char
            self._apply_numeric(layer)
            self._update_header(context)
            return {'RUNNING_MODAL'}

        if event.type == 'BACK_SPACE' and event.value == 'PRESS':
            if self.numeric_input:
                self.numeric_input = self.numeric_input[:-1]
                self._apply_numeric(layer)
                self._update_header(context)
            return {'RUNNING_MODAL'}

        if event.type == 'MOUSEMOVE' and not self.numeric_input:
            region_pos = [event.mouse_region_x, event.mouse_region_y]
            view_x, view_y = context.region.view2d.region_to_view(*region_pos)
            target_x = width * view_x
            target_y = height * view_y
            delta_x = target_x - self.start_input_position[0]
            delta_y = target_y - self.start_input_position[1]
            if self.axis_constraint == 'X':
                layer.location[0] = self.start_layer_location[0] + delta_x
                layer.location[1] = self.start_layer_location[1]
            elif self.axis_constraint == 'Y':
                layer.location[0] = self.start_layer_location[0]
                layer.location[1] = self.start_layer_location[1] + delta_y
            else:
                layer.location[0] = self.start_layer_location[0] + delta_x
                layer.location[1] = self.start_layer_location[1] + delta_y
        elif event.type == 'X' and event.value == 'PRESS':
            self.axis_constraint = 'X' if self.axis_constraint != 'X' else None
            if self.numeric_input:
                self._apply_numeric(layer)
            self._update_header(context)
        elif event.type == 'Y' and event.value == 'PRESS':
            self.axis_constraint = 'Y' if self.axis_constraint != 'Y' else None
            if self.numeric_input:
                self._apply_numeric(layer)
            self._update_header(context)
        elif event.type == 'S' and event.value == 'PRESS' and not self.numeric_input:
            layer.location = self.start_layer_location
            area_session.layer_moving = False
            area_session.prevent_layer_update_event = False
            context.area.header_text_set(None)
            bpy.ops.image_edit.scale_layer('INVOKE_DEFAULT')
            return {'FINISHED'}
        elif event.type == 'R' and event.value == 'PRESS' and not self.numeric_input:
            layer.location = self.start_layer_location
            area_session.layer_moving = False
            area_session.prevent_layer_update_event = False
            context.area.header_text_set(None)
            bpy.ops.image_edit.rotate_layer_arbitrary('INVOKE_DEFAULT')
            return {'FINISHED'}
        elif event.type in {'LEFTMOUSE', 'RET', 'NUMPAD_ENTER'}:
            utils.layer_rebuild_image_layers_nodes(img)
            area_session.layer_moving = False
            area_session.prevent_layer_update_event = False
            context.area.header_text_set(None)
            return {'FINISHED'}
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            layer.location = self.start_layer_location
            area_session.layer_moving = False
            area_session.prevent_layer_update_event = False
            context.area.header_text_set(None)
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        area_session = utils.layer_get_area_session(context)
        img = context.area.spaces.active.image
        width, height = img.size[0], img.size[1]
        layer = utils.layer_get_active_layer(context)
        if not layer:
            return {'CANCELLED'}
        if layer.locked:
            self.report({'WARNING'}, 'Layer is locked.')
            return {'CANCELLED'}
        region_pos = [event.mouse_region_x, event.mouse_region_y]
        view_x, view_y = context.region.view2d.region_to_view(*region_pos)
        self.start_input_position = [width * view_x, height * view_y]
        self.start_layer_location = layer.location[:]
        self.axis_constraint = None
        self.numeric_input = ''
        area_session.layer_moving = True
        area_session.prevent_layer_update_event = True
        self._update_header(context)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

class IMAGE_EDIT_OT_delete_layer(bpy.types.Operator):
    bl_idname = "image_edit.delete_layer"
    bl_label = "Delete Layer"

    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        img_props = img.image_edit_properties
        layers = img_props.layers
        selected_layer_index = img_props.selected_layer_index
        if selected_layer_index == -1 or selected_layer_index >= len(layers):
            return {'CANCELLED'}
        layer = layers[selected_layer_index]
        if layer.locked:
            self.report({'WARNING'}, 'Layer is locked.')
            return {'CANCELLED'}
        layer_img = bpy.data.images.get(layer.name, None)
        if layer_img:
            bpy.data.images.remove(layer_img)
        layers.remove(selected_layer_index)
        selected_layer_index = min(max(selected_layer_index, 0), len(layers) - 1)
        img_props.selected_layer_index = selected_layer_index
        utils.layer_rebuild_image_layers_nodes(img)
        return {'FINISHED'}

class IMAGE_EDIT_OT_edit_layer(bpy.types.Operator):
    """Toggle layer edit mode - paint directly on the selected layer"""
    bl_idname = "image_edit.edit_layer"
    bl_label = "Edit Layer"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        if context.area.type != 'IMAGE_EDITOR':
            return False
        img = context.area.spaces.active.image
        if not img:
            return False
        img_props = img.image_edit_properties
        # Allow if currently editing (to exit) or if a layer is selected (to enter)
        if img_props.editing_layer:
            return True
        if img_props.selected_layer_index >= 0 and img_props.selected_layer_index < len(img_props.layers):
            return True
        return False

    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        
        img_props = img.image_edit_properties
        
        # Toggle mode
        if img_props.editing_layer:
            # Exit edit mode
            if utils.layer_exit_edit_mode(context):
                self.report({'INFO'}, 'Exited layer edit mode')
                return {'FINISHED'}
            else:
                self.report({'WARNING'}, 'Failed to exit layer edit mode')
                return {'CANCELLED'}
        else:
            # Enter edit mode
            layer = utils.layer_get_active_layer(context)
            if layer and layer.locked:
                self.report({'WARNING'}, 'Layer is locked')
                return {'CANCELLED'}
            
            if utils.layer_enter_edit_mode(context):
                self.report({'INFO'}, 'Editing layer - paint directly on layer image')
                return {'FINISHED'}
            else:
                self.report({'WARNING'}, 'No layer selected')
                return {'CANCELLED'}

class IMAGE_EDIT_OT_duplicate_layer(bpy.types.Operator):

    """Duplicate the selected layer"""
    bl_idname = "image_edit.duplicate_layer"
    bl_label = "Duplicate Layer"

    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        layer = utils.layer_get_active_layer(context)
        if not layer:
            return {'CANCELLED'}
        
        layer_img = bpy.data.images.get(layer.name, None)
        if not layer_img:
            return {'CANCELLED'}
        
        pixels = utils.layer_read_pixels_from_image(layer_img)
        img_settings = {'is_float': layer_img.is_float, 'colorspace_name': layer_img.colorspace_settings.name}
        layer_settings = {'rotation': layer.rotation, 'scale': list(layer.scale), 'custom_data': layer.custom_data}
        
        utils.layer_create_layer(img, pixels, img_settings, layer_settings, custom_label=layer.label + " Copy")
        utils.layer_refresh_image(context)
        self.report({'INFO'}, 'Layer duplicated.')
        return {'FINISHED'}

class IMAGE_EDIT_OT_lock_all_layers(bpy.types.Operator):
    """Lock all layers"""
    bl_idname = "image_edit.lock_all_layers"
    bl_label = "Lock All Layers"

    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        img_props = img.image_edit_properties
        for layer in img_props.layers:
            layer.locked = True
        self.report({'INFO'}, 'All layers locked.')
        return {'FINISHED'}

class IMAGE_EDIT_OT_unlock_all_layers(bpy.types.Operator):
    """Unlock all layers"""
    bl_idname = "image_edit.unlock_all_layers"
    bl_label = "Unlock All Layers"

    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        img_props = img.image_edit_properties
        for layer in img_props.layers:
            layer.locked = False
        self.report({'INFO'}, 'All layers unlocked.')
        return {'FINISHED'}

class IMAGE_EDIT_OT_hide_all_layers(bpy.types.Operator):
    """Hide all layers"""
    bl_idname = "image_edit.hide_all_layers"
    bl_label = "Hide All Layers"

    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        img_props = img.image_edit_properties
        for layer in img_props.layers:
            layer.hide = True
        utils.layer_refresh_image(context)
        self.report({'INFO'}, 'All layers hidden.')
        return {'FINISHED'}

class IMAGE_EDIT_OT_show_all_layers(bpy.types.Operator):
    """Show all layers"""
    bl_idname = "image_edit.show_all_layers"
    bl_label = "Show All Layers"

    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        img_props = img.image_edit_properties
        for layer in img_props.layers:
            layer.hide = False
        utils.layer_refresh_image(context)
        self.report({'INFO'}, 'All layers shown.')
        return {'FINISHED'}

class IMAGE_EDIT_OT_delete_all_layers(bpy.types.Operator):
    """Delete all layers"""
    bl_idname = "image_edit.delete_all_layers"
    bl_label = "Delete All Layers"

    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        img_props = img.image_edit_properties
        layers = img_props.layers
        
        for layer in layers:
            layer_img = bpy.data.images.get(layer.name, None)
            if layer_img:
                bpy.data.images.remove(layer_img)
        
        layers.clear()
        img_props.selected_layer_index = -1
        utils.layer_rebuild_image_layers_nodes(img)
        utils.layer_refresh_image(context)
        self.report({'INFO'}, 'All layers deleted.')
        return {'FINISHED'}

class IMAGE_EDIT_OT_update_layer_previews(bpy.types.Operator):
    """Update all layer preview thumbnails"""
    bl_idname = "image_edit.update_layer_previews"
    bl_label = "Update Layer Previews"

    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        img_props = img.image_edit_properties
        
        for layer in img_props.layers:
            layer_img = bpy.data.images.get(layer.name, None)
            if layer_img:
                layer_img.update()
                if layer_img.preview:
                    layer_img.preview.reload()
        
        img.update()
        if img.preview:
            img.preview.reload()
        
        context.area.tag_redraw()
        self.report({'INFO'}, 'Layer previews updated.')
        return {'FINISHED'}

class IMAGE_EDIT_OT_select_all_layers(bpy.types.Operator):
    """Select all layers"""
    bl_idname = "image_edit.select_all_layers"
    bl_label = "Select All Layers"

    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        img_props = img.image_edit_properties
        for layer in img_props.layers:
            layer.checked = True
        self.report({'INFO'}, 'All layers selected.')
        return {'FINISHED'}

class IMAGE_EDIT_OT_deselect_all_layers(bpy.types.Operator):
    """Deselect all layers"""
    bl_idname = "image_edit.deselect_all_layers"
    bl_label = "Deselect All Layers"

    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        img_props = img.image_edit_properties
        for layer in img_props.layers:
            layer.checked = False
        self.report({'INFO'}, 'All layers deselected.')
        return {'FINISHED'}

class IMAGE_EDIT_OT_invert_layer_selection(bpy.types.Operator):
    """Invert layer selection"""
    bl_idname = "image_edit.invert_layer_selection"
    bl_label = "Invert Layer Selection"

    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        img_props = img.image_edit_properties
        for layer in img_props.layers:
            layer.checked = not layer.checked
        self.report({'INFO'}, 'Layer selection inverted.')
        return {'FINISHED'}

class IMAGE_EDIT_OT_delete_selected_layers(bpy.types.Operator):
    """Delete all selected (checked) layers"""
    bl_idname = "image_edit.delete_selected_layers"
    bl_label = "Delete Selected Layers"

    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        img_props = img.image_edit_properties
        layers = img_props.layers
        
        indices_to_remove = []
        for i, layer in enumerate(layers):
            if layer.checked and not layer.locked:
                indices_to_remove.append(i)
        
        if not indices_to_remove:
            self.report({'WARNING'}, 'No unlocked layers selected.')
            return {'CANCELLED'}
        
        for i in reversed(indices_to_remove):
            layer = layers[i]
            layer_img = bpy.data.images.get(layer.name, None)
            if layer_img:
                bpy.data.images.remove(layer_img)
            layers.remove(i)
        
        if len(layers) > 0:
            img_props.selected_layer_index = min(img_props.selected_layer_index, len(layers) - 1)
        else:
            img_props.selected_layer_index = -1
        
        utils.layer_rebuild_image_layers_nodes(img)
        utils.layer_refresh_image(context)
        self.report({'INFO'}, f'{len(indices_to_remove)} layers deleted.')
        return {'FINISHED'}

class IMAGE_EDIT_OT_merge_selected_layers(bpy.types.Operator):
    """Merge all selected (checked) layers"""
    bl_idname = "image_edit.merge_selected_layers"
    bl_label = "Merge Selected Layers"

    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        width, height = img.size
        img_props = img.image_edit_properties
        layers = img_props.layers
        
        checked_layers = [(i, layer) for i, layer in enumerate(layers) if layer.checked]
        
        if len(checked_layers) < 2:
            self.report({'WARNING'}, 'Select at least 2 layers to merge.')
            return {'CANCELLED'}
        
        pixels = utils.layer_read_pixels_from_image(img)
        
        merged_count = 0
        indices_to_remove = []
        
        for i, layer in reversed(checked_layers):
            layer_img = bpy.data.images.get(layer.name, None)
            if not layer_img:
                continue
            layer_width, layer_height = layer_img.size[0], layer_img.size[1]
            layer_pos = layer.location
            layer_x1, layer_y1 = layer_pos[0], height - layer_height - layer_pos[1]
            
            if layer.rotation == 0 and layer.scale[0] == 1.0 and layer.scale[1] == 1.0:
                layer_pixels = utils.layer_read_pixels_from_image(layer_img)
            else:
                layer_pixels, new_layer_width, new_layer_height = utils.layer_apply_layer_transform(layer_img, layer.rotation, layer.scale)
                layer_x1 = int(layer_x1 - (new_layer_width - layer_width) / 2.0)
                layer_y1 = int(layer_y1 - (new_layer_height - layer_height) / 2.0)
                layer_width = new_layer_width
                layer_height = new_layer_height
            
            layer_x2 = layer_x1 + layer_width
            layer_y2 = layer_y1 + layer_height
            target_x1 = max(min(layer_x1, width), 0)
            target_y1 = max(min(layer_y1, height), 0)
            target_x2 = max(min(layer_x2, width), 0)
            target_y2 = max(min(layer_y2, height), 0)
            
            if layer_x1 == layer_x2 or layer_y1 == layer_y2:
                continue
            
            src_x1 = target_x1 - layer_x1
            src_y1 = target_y1 - layer_y1
            src_x2 = layer_width - (layer_x2 - target_x2)
            src_y2 = layer_height - (layer_y2 - target_y2)
            
            target_range = pixels[target_y1:target_y2, target_x1:target_x2]
            target_color_chan = target_range[:, :, :3]
            target_alpha_chan = target_range[:, :, 3:4]
            layer_range = layer_pixels[src_y1:src_y2, src_x1:src_x2]
            layer_color_chan = layer_range[:, :, :3]
            layer_alpha_chan = layer_range[:, :, 3:4]
            temp_alpha_chan = target_alpha_chan * (1.0 - layer_alpha_chan) + layer_alpha_chan
            temp_alpha_chan_safe = np.where(temp_alpha_chan == 0, 1.0, temp_alpha_chan)
            pixels[target_y1:target_y2, target_x1:target_x2, :3] = (target_color_chan * target_alpha_chan * (1.0 - layer_alpha_chan) + layer_color_chan * layer_alpha_chan) / temp_alpha_chan_safe
            pixels[target_y1:target_y2, target_x1:target_x2, 3:4] = temp_alpha_chan
            
            bpy.data.images.remove(layer_img)
            indices_to_remove.append(i)
            merged_count += 1
        
        for i in sorted(indices_to_remove, reverse=True):
            layers.remove(i)
        
        utils.ImageUndoStack.get().push_state(img)
        utils.layer_write_pixels_to_image(img, pixels)
        
        if len(layers) > 0:
            img_props.selected_layer_index = min(img_props.selected_layer_index, len(layers) - 1)
        else:
            img_props.selected_layer_index = -1
        
        utils.layer_rebuild_image_layers_nodes(img)
        utils.layer_refresh_image(context)
        self.report({'INFO'}, f'{merged_count} layers merged.')
        return {'FINISHED'}

class IMAGE_EDIT_OT_change_image_layer_order(bpy.types.Operator):
    bl_idname = "image_edit.change_image_layer_order"
    bl_label = "Change Image Layer Order"
    up: bpy.props.BoolProperty()

    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        img_props = img.image_edit_properties
        layers = img_props.layers
        selected_layer_index = img_props.selected_layer_index
        if selected_layer_index == -1 or selected_layer_index >= len(layers):
            return {'CANCELLED'}
        if (self.up and selected_layer_index == 0) or (not self.up and selected_layer_index >= len(layers) - 1):
            return {'CANCELLED'}
        new_layer_index = selected_layer_index + (-1 if self.up else 1)
        layers.move(selected_layer_index, new_layer_index)
        img_props.selected_layer_index = new_layer_index
        utils.layer_rebuild_image_layers_nodes(img)
        return {'FINISHED'}

class IMAGE_EDIT_OT_merge_layers(bpy.types.Operator):
    """Merge all layers"""
    bl_idname = "image_edit.merge_layers"
    bl_label = "Merge Layers"

    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        width, height = img.size
        pixels = utils.layer_read_pixels_from_image(img)
        img_props = img.image_edit_properties
        layers = img_props.layers
        for layer in reversed(layers):
            layer_img = bpy.data.images.get(layer.name, None)
            if not layer_img:
                continue
            layer_width, layer_height = layer_img.size[0], layer_img.size[1]
            layer_pos = layer.location
            layer_x1, layer_y1 = layer_pos[0], height - layer_height - layer_pos[1]
            if layer.rotation == 0 and layer.scale[0] == 1.0 and layer.scale[1] == 1.0:
                layer_pixels = utils.layer_read_pixels_from_image(layer_img)
            else:
                layer_pixels, new_layer_width, new_layer_height = utils.layer_apply_layer_transform(layer_img, layer.rotation, layer.scale)
                layer_x1 = int(layer_x1 - (new_layer_width - layer_width) / 2.0)
                layer_y1 = int(layer_y1 - (new_layer_height - layer_height) / 2.0)
                layer_width = new_layer_width
                layer_height = new_layer_height
            layer_x2 = layer_x1 + layer_width
            layer_y2 = layer_y1 + layer_height
            target_x1 = max(min(layer_x1, width), 0)
            target_y1 = max(min(layer_y1, height), 0)
            target_x2 = max(min(layer_x2, width), 0)
            target_y2 = max(min(layer_y2, height), 0)
            if layer_x1 == layer_x2 or layer_y1 == layer_y2:
                continue
            src_x1 = target_x1 - layer_x1
            src_y1 = target_y1 - layer_y1
            src_x2 = layer_width - (layer_x2 - target_x2)
            src_y2 = layer_height - (layer_y2 - target_y2)
            target_range = pixels[target_y1:target_y2, target_x1:target_x2]
            target_color_chan = target_range[:, :, :3]
            target_alpha_chan = target_range[:, :, 3:4]
            layer_range = layer_pixels[src_y1:src_y2, src_x1:src_x2]
            layer_color_chan = layer_range[:, :, :3]
            layer_alpha_chan = layer_range[:, :, 3:4]
            temp_alpha_chan = target_alpha_chan * (1.0 - layer_alpha_chan) + layer_alpha_chan
            temp_alpha_chan_safe = np.where(temp_alpha_chan == 0, 1.0, temp_alpha_chan)
            pixels[target_y1:target_y2, target_x1:target_x2, :3] = (target_color_chan * target_alpha_chan * (1.0 - layer_alpha_chan) + layer_color_chan * layer_alpha_chan) / temp_alpha_chan_safe
            pixels[target_y1:target_y2, target_x1:target_x2, 3:4] = temp_alpha_chan
            bpy.data.images.remove(layer_img)
        utils.ImageUndoStack.get().push_state(img)
        utils.layer_write_pixels_to_image(img, pixels)
        layers.clear()
        utils.layer_rebuild_image_layers_nodes(img)
        utils.layer_refresh_image(context)
        return {'FINISHED'}

class IMAGE_EDIT_OT_flip_layer(bpy.types.Operator):
    """Flip the layer"""
    bl_idname = "image_edit.flip_layer"
    bl_label = "Flip Layer"
    is_vertically: bpy.props.BoolProperty(name="Vertically", default=False)

    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        layer = utils.layer_get_active_layer(context)
        if not layer:
            return {'CANCELLED'}
        if layer.locked:
            self.report({'WARNING'}, 'Layer is locked.')
            return {'CANCELLED'}
        if self.is_vertically:
            layer.scale[1] *= -1.0
        else:
            layer.scale[0] *= -1.0
        utils.layer_refresh_image(context)
        return {'FINISHED'}

class IMAGE_EDIT_OT_rotate_layer(bpy.types.Operator):
    """Rotate the layer"""
    bl_idname = "image_edit.rotate_layer"
    bl_label = "Rotate Layer"
    is_left: bpy.props.BoolProperty(name="Left", default=False)

    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        img = context.area.spaces.active.image
        if not img:
            return {'CANCELLED'}
        layer = utils.layer_get_active_layer(context)
        if not layer:
            return {'CANCELLED'}
        if layer.locked:
            self.report({'WARNING'}, 'Layer is locked.')
            return {'CANCELLED'}
        layer.rotation += math.pi / 2.0 if self.is_left else -math.pi / 2.0
        utils.layer_refresh_image(context)
        return {'FINISHED'}

class IMAGE_EDIT_OT_rotate_layer_arbitrary(bpy.types.Operator):
    """Rotate the image by a specified angle"""
    bl_idname = "image_edit.rotate_layer_arbitrary"
    bl_label = "Rotate Layer Arbitrary"

    # Numeric key type mapping
    _NUM_KEYS = {
        'ZERO': '0', 'ONE': '1', 'TWO': '2', 'THREE': '3', 'FOUR': '4',
        'FIVE': '5', 'SIX': '6', 'SEVEN': '7', 'EIGHT': '8', 'NINE': '9',
        'NUMPAD_0': '0', 'NUMPAD_1': '1', 'NUMPAD_2': '2', 'NUMPAD_3': '3',
        'NUMPAD_4': '4', 'NUMPAD_5': '5', 'NUMPAD_6': '6', 'NUMPAD_7': '7',
        'NUMPAD_8': '8', 'NUMPAD_9': '9',
        'PERIOD': '.', 'NUMPAD_PERIOD': '.',
        'MINUS': '-', 'NUMPAD_MINUS': '-',
    }

    bl_options = {'REGISTER', 'UNDO'}
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_input_position = [0, 0]
        self.start_layer_angle = 0
        self.numeric_input = ''

    def _apply_numeric(self, layer):
        """Apply numeric input as degrees of rotation."""
        try:
            degrees = float(self.numeric_input) if self.numeric_input else 0
        except ValueError:
            degrees = 0
        layer.rotation = self.start_layer_angle + math.radians(degrees)

    def _update_header(self, context):
        if self.numeric_input:
            context.area.header_text_set(f"Rotate: {self.numeric_input}°  (Enter to confirm)")
        else:
            context.area.header_text_set(f"Rotate: mouse  (type degrees)")

    def modal(self, context, event):
        area_session = utils.layer_get_area_session(context)
        context.area.tag_redraw()
        img = context.area.spaces.active.image
        width, height = img.size[0], img.size[1]
        layer = utils.layer_get_active_layer(context)
        if not layer:
            return {'CANCELLED'}
        layer_width, layer_height = 1, 1
        layer_img = bpy.data.images.get(layer.name, None)
        if layer_img:
            layer_width, layer_height = layer_img.size[0], layer_img.size[1]

        # Numeric input handling
        if event.type in self._NUM_KEYS and event.value == 'PRESS':
            char = self._NUM_KEYS[event.type]
            if char == '-':
                if self.numeric_input.startswith('-'):
                    self.numeric_input = self.numeric_input[1:]
                else:
                    self.numeric_input = '-' + self.numeric_input
            elif char == '.':
                if '.' not in self.numeric_input:
                    self.numeric_input += char
            else:
                self.numeric_input += char
            self._apply_numeric(layer)
            self._update_header(context)
            return {'RUNNING_MODAL'}

        if event.type == 'BACK_SPACE' and event.value == 'PRESS':
            if self.numeric_input:
                self.numeric_input = self.numeric_input[:-1]
                self._apply_numeric(layer)
                self._update_header(context)
            return {'RUNNING_MODAL'}

        if event.type == 'MOUSEMOVE' and not self.numeric_input:
            center_x = width / 2.0 + layer.location[0]
            center_y = height / 2.0 + layer.location[1]
            region_pos = [event.mouse_region_x, event.mouse_region_y]
            view_x, view_y = context.region.view2d.region_to_view(*region_pos)
            target_x = width * view_x
            target_y = height * view_y
            angle1 = math.atan2(self.start_input_position[1] - center_y, self.start_input_position[0] - center_x)
            angle2 = math.atan2(target_y - center_y, target_x - center_x)
            layer.rotation = self.start_layer_angle + angle2 - angle1
        elif event.type == 'G' and event.value == 'PRESS' and not self.numeric_input:
            layer.rotation = self.start_layer_angle
            area_session.layer_rotating = False
            area_session.prevent_layer_update_event = False
            context.area.header_text_set(None)
            bpy.ops.image_edit.move_layer('INVOKE_DEFAULT')
            return {'FINISHED'}
        elif event.type == 'S' and event.value == 'PRESS' and not self.numeric_input:
            layer.rotation = self.start_layer_angle
            area_session.layer_rotating = False
            area_session.prevent_layer_update_event = False
            context.area.header_text_set(None)
            bpy.ops.image_edit.scale_layer('INVOKE_DEFAULT')
            return {'FINISHED'}
        elif event.type in {'LEFTMOUSE', 'RET', 'NUMPAD_ENTER'}:
            utils.layer_rebuild_image_layers_nodes(img)
            area_session.layer_rotating = False
            area_session.prevent_layer_update_event = False
            context.area.header_text_set(None)
            return {'FINISHED'}
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            layer.rotation = self.start_layer_angle
            area_session.layer_rotating = False
            area_session.prevent_layer_update_event = False
            context.area.header_text_set(None)
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        area_session = utils.layer_get_area_session(context)
        img = context.area.spaces.active.image
        width, height = img.size[0], img.size[1]
        layer = utils.layer_get_active_layer(context)
        if not layer:
            return {'CANCELLED'}
        if layer.locked:
            self.report({'WARNING'}, 'Layer is locked.')
            return {'CANCELLED'}
        region_pos = [event.mouse_region_x, event.mouse_region_y]
        view_x, view_y = context.region.view2d.region_to_view(*region_pos)
        self.start_input_position = [width * view_x, height * view_y]
        self.start_layer_angle = layer.rotation
        self.numeric_input = ''
        area_session.layer_rotating = True
        area_session.prevent_layer_update_event = True
        self._update_header(context)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

class IMAGE_EDIT_OT_scale_layer(bpy.types.Operator):
    """Scale the layer"""
    bl_idname = "image_edit.scale_layer"
    bl_label = "Scale Layer"

    # Numeric key type mapping
    _NUM_KEYS = {
        'ZERO': '0', 'ONE': '1', 'TWO': '2', 'THREE': '3', 'FOUR': '4',
        'FIVE': '5', 'SIX': '6', 'SEVEN': '7', 'EIGHT': '8', 'NINE': '9',
        'NUMPAD_0': '0', 'NUMPAD_1': '1', 'NUMPAD_2': '2', 'NUMPAD_3': '3',
        'NUMPAD_4': '4', 'NUMPAD_5': '5', 'NUMPAD_6': '6', 'NUMPAD_7': '7',
        'NUMPAD_8': '8', 'NUMPAD_9': '9',
        'PERIOD': '.', 'NUMPAD_PERIOD': '.',
        'MINUS': '-', 'NUMPAD_MINUS': '-',
    }

    bl_options = {'REGISTER', 'UNDO'}
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_input_position = [0, 0]
        self.start_layer_scale_x = 1.0
        self.start_layer_scale_y = 1.0
        self.axis_constraint = None  # None, 'X', or 'Y'
        self.numeric_input = ''

    def _apply_numeric(self, layer):
        """Apply numeric input as scale factor."""
        try:
            factor = float(self.numeric_input) if self.numeric_input else 1.0
        except ValueError:
            factor = 1.0
        if self.axis_constraint == 'X':
            layer.scale[0] = self.start_layer_scale_x * factor
            layer.scale[1] = self.start_layer_scale_y
        elif self.axis_constraint == 'Y':
            layer.scale[0] = self.start_layer_scale_x
            layer.scale[1] = self.start_layer_scale_y * factor
        else:
            layer.scale[0] = self.start_layer_scale_x * factor
            layer.scale[1] = self.start_layer_scale_y * factor

    def _update_header(self, context):
        axis = f" {self.axis_constraint}" if self.axis_constraint else ""
        if self.numeric_input:
            context.area.header_text_set(f"Scale{axis}: {self.numeric_input}x  (Enter to confirm)")
        else:
            context.area.header_text_set(f"Scale{axis}: mouse  (X/Y axis, type factor)")

    def modal(self, context, event):
        area_session = utils.layer_get_area_session(context)
        context.area.tag_redraw()
        img = context.area.spaces.active.image
        width, height = img.size[0], img.size[1]
        layer = utils.layer_get_active_layer(context)
        if not layer:
            return {'CANCELLED'}
        layer_width, layer_height = 1, 1
        layer_img = bpy.data.images.get(layer.name, None)
        if layer_img:
            layer_width, layer_height = layer_img.size[0], layer_img.size[1]

        # Numeric input handling
        if event.type in self._NUM_KEYS and event.value == 'PRESS':
            char = self._NUM_KEYS[event.type]
            if char == '-':
                if self.numeric_input.startswith('-'):
                    self.numeric_input = self.numeric_input[1:]
                else:
                    self.numeric_input = '-' + self.numeric_input
            elif char == '.':
                if '.' not in self.numeric_input:
                    self.numeric_input += char
            else:
                self.numeric_input += char
            self._apply_numeric(layer)
            self._update_header(context)
            return {'RUNNING_MODAL'}

        if event.type == 'BACK_SPACE' and event.value == 'PRESS':
            if self.numeric_input:
                self.numeric_input = self.numeric_input[:-1]
                self._apply_numeric(layer)
                self._update_header(context)
            return {'RUNNING_MODAL'}

        if event.type == 'MOUSEMOVE' and not self.numeric_input:
            center_x = width / 2.0 + layer.location[0]
            center_y = height / 2.0 + layer.location[1]
            region_pos = [event.mouse_region_x, event.mouse_region_y]
            view_x, view_y = context.region.view2d.region_to_view(*region_pos)
            target_x = width * view_x
            target_y = height * view_y
            dist1 = math.hypot(self.start_input_position[0] - center_x, self.start_input_position[1] - center_y)
            dist2 = math.hypot(target_x - center_x, target_y - center_y)
            scale_factor = dist2 / dist1 if dist1 > 0 else 1.0
            if self.axis_constraint == 'X':
                layer.scale[0] = self.start_layer_scale_x * scale_factor
                layer.scale[1] = self.start_layer_scale_y
            elif self.axis_constraint == 'Y':
                layer.scale[0] = self.start_layer_scale_x
                layer.scale[1] = self.start_layer_scale_y * scale_factor
            else:
                layer.scale[0] = self.start_layer_scale_x * scale_factor
                layer.scale[1] = self.start_layer_scale_y * scale_factor
        elif event.type == 'X' and event.value == 'PRESS':
            self.axis_constraint = 'X' if self.axis_constraint != 'X' else None
            if self.numeric_input:
                self._apply_numeric(layer)
            self._update_header(context)
        elif event.type == 'Y' and event.value == 'PRESS':
            self.axis_constraint = 'Y' if self.axis_constraint != 'Y' else None
            if self.numeric_input:
                self._apply_numeric(layer)
            self._update_header(context)
        elif event.type == 'G' and event.value == 'PRESS' and not self.numeric_input:
            layer.scale[0] = self.start_layer_scale_x
            layer.scale[1] = self.start_layer_scale_y
            area_session.layer_scaling = False
            area_session.prevent_layer_update_event = False
            context.area.header_text_set(None)
            bpy.ops.image_edit.move_layer('INVOKE_DEFAULT')
            return {'FINISHED'}
        elif event.type == 'R' and event.value == 'PRESS' and not self.numeric_input:
            layer.scale[0] = self.start_layer_scale_x
            layer.scale[1] = self.start_layer_scale_y
            area_session.layer_scaling = False
            area_session.prevent_layer_update_event = False
            context.area.header_text_set(None)
            bpy.ops.image_edit.rotate_layer_arbitrary('INVOKE_DEFAULT')
            return {'FINISHED'}
        elif event.type in {'LEFTMOUSE', 'RET', 'NUMPAD_ENTER'}:
            utils.layer_rebuild_image_layers_nodes(img)
            area_session.layer_scaling = False
            area_session.prevent_layer_update_event = False
            context.area.header_text_set(None)
            return {'FINISHED'}
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            layer.scale[0] = self.start_layer_scale_x
            layer.scale[1] = self.start_layer_scale_y
            area_session.layer_scaling = False
            area_session.prevent_layer_update_event = False
            context.area.header_text_set(None)
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        area_session = utils.layer_get_area_session(context)
        img = context.area.spaces.active.image
        width, height = img.size[0], img.size[1]
        layer = utils.layer_get_active_layer(context)
        if not layer:
            return {'CANCELLED'}
        if layer.locked:
            self.report({'WARNING'}, 'Layer is locked.')
            return {'CANCELLED'}
        region_pos = [event.mouse_region_x, event.mouse_region_y]
        view_x, view_y = context.region.view2d.region_to_view(*region_pos)
        self.start_input_position = [width * view_x, height * view_y]
        self.start_layer_scale_x = layer.scale[0]
        self.start_layer_scale_y = layer.scale[1]
        self.axis_constraint = None
        self.numeric_input = ''
        area_session.layer_scaling = True
        area_session.prevent_layer_update_event = True
        self._update_header(context)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}


class IMAGE_EDIT_OT_sculpt_image(bpy.types.Operator):
    """Sculpt the image with brush-based pixel warping"""
    bl_idname = "image_edit.sculpt_image"
    bl_label = "Image Sculpt"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return (context.area.type == 'IMAGE_EDITOR' and 
                context.area.spaces.active.image is not None)

    def modal(self, context, event):
        import numpy as np
        
        if event.type == 'MOUSEMOVE':
            if self.last_pos is not None:
                region_pos = (event.mouse_region_x, event.mouse_region_y)
                view_x, view_y = context.region.view2d.region_to_view(*region_pos)
                curr_x = int(view_x * self.width)
                curr_y = int(view_y * self.height)
                
                wm = context.window_manager
                props = wm.image_edit_properties
                
                dx = curr_x - self.last_pos[0]
                dy = curr_y - self.last_pos[1]
                
                if dx != 0 or dy != 0:
                    # Apply to working buffer
                    self._apply_to_buffer(curr_x, curr_y, dx, dy, 
                                         props.sculpt_mode, props.sculpt_radius, props.sculpt_strength)
                    self.modified = True
                    self.frame_count += 1
                    
                    # Real-time feedback: update image every few frames for performance
                    if self.frame_count % 3 == 0:
                        self.img.pixels.foreach_set(self.working.ravel())
                        self.img.update()
                        context.area.tag_redraw()
                
                self.last_pos = [curr_x, curr_y]
            
            return {'RUNNING_MODAL'}
        
        elif event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
            # Final write on release
            if self.modified:
                # Push undo state BEFORE finalizing (save the original pixels efficiently)
                utils.ImageUndoStack.get().push_state_from_numpy(self.img, self.original_pixels)
                
                # Now apply the final result
                self.img.pixels.foreach_set(self.working.ravel())
                self.img.update()
            context.area.tag_redraw()
            return {'FINISHED'}
        
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            # Cancel - restore original
            if self.modified:
                self.img.pixels.foreach_set(self.original_pixels)
                self.img.update()
            context.area.tag_redraw()
            return {'CANCELLED'}
        
        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        import numpy as np
        
        self.img = context.area.spaces.active.image
        if not self.img:
            return {'CANCELLED'}
        
        self.width, self.height = self.img.size[0], self.img.size[1]
        self.modified = False
        self.frame_count = 0
        
        # Read pixels once - use foreach_get for speed
        pixel_count = self.width * self.height * 4
        self.original_pixels = np.empty(pixel_count, dtype=np.float32)
        self.img.pixels.foreach_get(self.original_pixels)
        
        # Working copy reshaped for manipulation
        self.working = self.original_pixels.reshape((self.height, self.width, 4)).copy()
        
        region_pos = [event.mouse_region_x, event.mouse_region_y]
        view_x, view_y = context.region.view2d.region_to_view(*region_pos)
        self.last_pos = [int(view_x * self.width), int(view_y * self.height)]
        
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def _apply_to_buffer(self, cx, cy, dx, dy, mode, radius, strength):
        """MLS Rigid deformation - O(k) per pixel, highly optimized."""
        import numpy as np
        import bpy
        
        # Get falloff preset from properties
        props = bpy.context.window_manager.image_edit_properties
        falloff_preset = props.sculpt_falloff_preset
        
        # Bounding box
        x1 = max(0, cx - radius)
        y1 = max(0, cy - radius)
        x2 = min(self.width, cx + radius)
        y2 = min(self.height, cy + radius)
        
        if x1 >= x2 or y1 >= y2:
            return
        
        # Coordinate grids (float32 for speed)
        py = np.arange(y1, y2, dtype=np.float32)
        px = np.arange(x1, x2, dtype=np.float32)
        gx, gy = np.meshgrid(px, py)
        
        # Distance from brush center
        rel_x = gx - cx
        rel_y = gy - cy
        dist_sq = rel_x * rel_x + rel_y * rel_y
        radius_sq = float(radius * radius)
        
        # Mask for valid region
        mask = dist_sq < radius_sq
        if not np.any(mask):
            return
        
        # Calculate normalized distance t (0 at center, 1 at edge)
        dist = np.sqrt(dist_sq)
        t = np.clip(dist / radius, 0.0, 1.0)
        
        # Calculate falloff based on preset (vectorized for performance)
        if falloff_preset == 'SMOOTH':
            # Hermite smoothstep: 3t² - 2t³
            weights = (1.0 - (3.0 * t**2 - 2.0 * t**3)) * strength
        elif falloff_preset == 'SMOOTHER':
            # Perlin smootherstep: 6t⁵ - 15t⁴ + 10t³
            weights = (1.0 - (6.0 * t**5 - 15.0 * t**4 + 10.0 * t**3)) * strength
        elif falloff_preset == 'SPHERE':
            # Spherical: sqrt(1 - t²)
            weights = np.sqrt(np.clip(1.0 - t**2, 0.0, 1.0)) * strength
        elif falloff_preset == 'ROOT':
            # Root: 1 - sqrt(t)
            weights = (1.0 - np.sqrt(t)) * strength
        elif falloff_preset == 'SHARP':
            # Sharp: (1 - t)²
            weights = ((1.0 - t)**2) * strength
        elif falloff_preset == 'LINEAR':
            # Linear: 1 - t
            weights = (1.0 - t) * strength
        elif falloff_preset == 'CONSTANT':
            # Constant: no falloff (hard edge)
            weights = np.where(mask, strength, 0.0)
        elif falloff_preset == 'CUSTOM':
            # Use brush curve - evaluate per-pixel (slower but accurate)
            brush = bpy.context.tool_settings.image_paint.brush
            if brush and brush.curve_distance_falloff:
                curve = brush.curve_distance_falloff
                weights = np.zeros_like(t)
                # Vectorized curve evaluation using pre-sampled LUT for performance
                lut_size = 256
                lut = np.array([curve.evaluate(curve.curves[0], i / (lut_size - 1)) 
                               for i in range(lut_size)], dtype=np.float32)
                t_indices = np.clip((t * (lut_size - 1)).astype(np.int32), 0, lut_size - 1)
                weights = lut[t_indices] * strength
            else:
                # Fallback to smooth
                weights = (1.0 - (3.0 * t**2 - 2.0 * t**3)) * strength
        else:
            # Default to smooth
            weights = (1.0 - (3.0 * t**2 - 2.0 * t**3)) * strength
        
        # Apply mask
        weights = np.where(mask, weights, 0.0)
        
        # Control point: brush center
        p_x, p_y = float(cx), float(cy)
        
        # MLS Rigid transformation based on mode
        if mode == 'GRAB':
            # Weighted translation
            offset_x = dx * weights
            offset_y = dy * weights
            src_x = gx - offset_x
            src_y = gy - offset_y
            
        elif mode == 'PINCH':
            # Scale toward center
            scale = 1.0 + weights * 0.5
            src_x = p_x + rel_x * scale
            src_y = p_y + rel_y * scale
        elif mode == 'DRIP':
            # Realistic paint drip effect - teardrop shape
            # Distance from vertical center line (for tapering)
            dist_from_center = np.abs(rel_x)
            
            # Taper factor: 1 at center, 0 at edges (creates narrow drip shape)
            taper = np.maximum(0, 1 - dist_from_center / (radius * 0.3))
            taper = taper ** 0.5  # Softer taper
            
            # Drip amount: stronger in center, combines with gaussian weight
            drip_amount = weights * taper * radius * strength
            
            # Create elongation: pixels stretch downward from where brush is
            # Sample from above to create the stretching effect
            src_x = gx
            src_y = gy - drip_amount
            
            # Add bulge at bottom: pixels near brush center get extra stretch
            center_boost = np.exp(-dist_from_center**2 / (radius * 0.15)**2)
            src_y = src_y - center_boost * weights * radius * 0.2
            
        elif mode == 'WAVE':
            # Ripple/wave distortion - concentric rings from center
            dist = np.sqrt(dist_sq) + 0.001  # avoid division by zero
            
            # Wave parameters
            frequency = 0.3  # waves per pixel
            amplitude = weights * radius * 0.15
            
            # Radial wave displacement
            wave = np.sin(dist * frequency * 2 * np.pi) * amplitude
            
            # Displace perpendicular to radius (creates ripple effect)
            norm_x = rel_x / dist
            norm_y = rel_y / dist
            src_x = gx + norm_x * wave
            src_y = gy + norm_y * wave
            
        elif mode == 'JITTER':
            # Turbulence/noise displacement
            # Use hash-based pseudo-random for deterministic jitter
            hash_x = np.sin(gx * 12.9898 + gy * 78.233) * 43758.5453
            hash_y = np.sin(gx * 78.233 + gy * 12.9898) * 43758.5453
            noise_x = (hash_x - np.floor(hash_x)) * 2 - 1  # -1 to 1
            noise_y = (hash_y - np.floor(hash_y)) * 2 - 1
            
            # Apply weighted displacement
            jitter_amount = weights * radius * 0.2
            src_x = gx + noise_x * jitter_amount
            src_y = gy + noise_y * jitter_amount
            
        elif mode == 'HAZE':
            # Heat haze - vertical refractive shimmer
            # Use position-based sine for shimmer pattern
            phase = gy * 0.3 + gx * 0.1
            shimmer_x = np.sin(phase) * weights * radius * 0.1
            shimmer_y = np.sin(phase * 1.3 + 1.0) * weights * radius * 0.05
            
            src_x = gx + shimmer_x
            src_y = gy + shimmer_y
            
        elif mode == 'ERODE':
            # Edge breaking - displace based on local contrast/gradient
            # Sample luminance gradient using neighbors
            patch = self.working[y1:y2, x1:x2]
            lum = 0.299 * patch[:,:,0] + 0.587 * patch[:,:,1] + 0.114 * patch[:,:,2]
            
            # Compute gradient using Sobel-like kernel
            grad_x = np.zeros_like(lum)
            grad_y = np.zeros_like(lum)
            if lum.shape[0] > 2 and lum.shape[1] > 2:
                grad_x[1:-1, 1:-1] = lum[1:-1, 2:] - lum[1:-1, :-2]
                grad_y[1:-1, 1:-1] = lum[2:, 1:-1] - lum[:-2, 1:-1]
            
            # Displace along gradient (erodes edges)
            erode_strength = weights * radius * 0.3
            src_x = gx + grad_x * erode_strength
            src_y = gy + grad_y * erode_strength
            
        elif mode == 'CREASE':
            # Sharp linear deformation - crease along drag direction
            # Project onto drag direction for sharp line effect
            drag_len = np.sqrt(dx*dx + dy*dy) + 0.001
            drag_nx = dx / drag_len
            drag_ny = dy / drag_len
            
            # Distance along drag direction
            proj = rel_x * drag_nx + rel_y * drag_ny
            
            # Sharp crease using tanh for step-like transition
            crease = np.tanh(proj * 0.2) * weights * radius * 0.2
            
            # Displace perpendicular to drag
            src_x = gx - drag_ny * crease
            src_y = gy + drag_nx * crease
            
        elif mode == 'BRISTLE':
            # Directional streaks and striations
            drag_len = np.sqrt(dx*dx + dy*dy) + 0.001
            drag_nx, drag_ny = dx / drag_len, dy / drag_len
            # Parallel striations 
            striation = np.sin((rel_x * drag_ny - rel_y * drag_nx) * 0.5) * 0.5 + 0.5
            offset_x = dx * weights * striation * 0.3
            offset_y = dy * weights * striation * 0.3
            src_x = gx - offset_x
            src_y = gy - offset_y
            
        elif mode == 'DRYPULL':
            # Broken dry-brush skipped pixels
            skip = ((gx.astype(np.int32) + gy.astype(np.int32)) % 3 != 0).astype(np.float32)
            src_x = gx - dx * weights * skip * 0.3
            src_y = gy - dy * weights * skip * 0.3
            
        elif mode == 'BLOOM':
            # Soft expanding overlap like petals
            dist = np.sqrt(dist_sq) + 0.001
            expand = weights * radius * 0.2
            norm_x, norm_y = rel_x / dist, rel_y / dist
            src_x = gx - norm_x * expand
            src_y = gy - norm_y * expand
            
        elif mode == 'INFLATE':
            # Organic uneven bulging with noise
            dist = np.sqrt(dist_sq) + 0.001
            noise = np.sin(gx * 0.1 + gy * 0.13) * np.cos(gx * 0.07 - gy * 0.11)
            bulge = weights * radius * 0.15 * (1 + noise * 0.5)
            norm_x, norm_y = rel_x / dist, rel_y / dist
            src_x = gx - norm_x * bulge
            src_y = gy - norm_y * bulge
            
        elif mode == 'LIQUIFY':
            # Fluid-like warp deformation
            dist = np.sqrt(dist_sq) + 0.001
            fluid = np.sin(dist * 0.1) * weights * radius * 0.15
            norm_x, norm_y = rel_x / dist, rel_y / dist
            src_x = gx - dx * weights * 0.3 + norm_y * fluid
            src_y = gy - dy * weights * 0.3 - norm_x * fluid
            
        elif mode == 'SPIRAL':
            # Swirling vortex distortion
            dist = np.sqrt(dist_sq) + 0.001
            spiral_angle = weights * 2.0  # Strong spiral
            cos_s = np.cos(spiral_angle)
            sin_s = np.sin(spiral_angle)
            src_x = p_x + rel_x * cos_s - rel_y * sin_s
            src_y = p_y + rel_x * sin_s + rel_y * cos_s
            
        elif mode == 'STRETCH':
            # Directional elongation
            drag_len = np.sqrt(dx*dx + dy*dy) + 0.001
            drag_nx, drag_ny = dx / drag_len, dy / drag_len
            # Project onto drag direction
            proj = rel_x * drag_nx + rel_y * drag_ny
            stretch = weights * proj * 0.3
            src_x = gx - drag_nx * stretch
            src_y = gy - drag_ny * stretch
            
        elif mode == 'PIXELATE':
            # Pixelated mosaic effect - snap to grid
            grid_size = max(2, int(radius * 0.1 * strength) + 1)
            grid_x = (gx // grid_size) * grid_size + grid_size / 2
            grid_y = (gy // grid_size) * grid_size + grid_size / 2
            blend = weights
            src_x = gx * (1 - blend) + grid_x * blend
            src_y = gy * (1 - blend) + grid_y * blend
            
        elif mode == 'GLITCH':
            # Digital scan line displacement
            line_height = 3
            line_idx = (gy / line_height).astype(np.int32)
            offset = np.sin(line_idx * 3.7) * weights * radius * 0.3
            src_x = gx + offset
            src_y = gy
            
        else:
            return
        
        # Clamp source coordinates
        src_x = np.clip(src_x, 0, self.width - 1.001)
        src_y = np.clip(src_y, 0, self.height - 1.001)
        
        # Bilinear interpolation for quality
        x0 = src_x.astype(np.int32)
        y0 = src_y.astype(np.int32)
        x1i = np.minimum(x0 + 1, self.width - 1)
        y1i = np.minimum(y0 + 1, self.height - 1)
        
        fx = (src_x - x0)[:, :, np.newaxis]
        fy = (src_y - y0)[:, :, np.newaxis]
        
        # Sample 4 corners
        p00 = self.working[y0, x0]
        p10 = self.working[y0, x1i]
        p01 = self.working[y1i, x0]
        p11 = self.working[y1i, x1i]
        
        # Bilinear interpolate
        sampled = (p00 * (1 - fx) * (1 - fy) +
                   p10 * fx * (1 - fy) +
                   p01 * (1 - fx) * fy +
                   p11 * fx * fy)
        
        # Apply with mask
        patch = self.working[y1:y2, x1:x2]
        mask_3d = mask[:, :, np.newaxis]
        patch[:] = np.where(mask_3d, sampled, patch)


# ============================================================
# Lattice Deformation Tool
# ============================================================

class IMAGE_EDIT_OT_lattice_deform(bpy.types.Operator):
    """Deform image using lattice grid with perspective or mesh mode"""
    bl_idname = "image_edit.lattice_deform"
    bl_label = "Lattice Deform"
    bl_options = {'REGISTER', 'UNDO'}
    
    _draw_handler = None
    _image = None
    _original_pixels = None
    _working_pixels = None
    _width = 0
    _height = 0
    
    # Control points: grid [row][col] = [x, y]
    _control_points = []
    _original_grid = []
    
    # Interaction state
    _active_point = None
    _is_dragging = False
    _initialized = False
    
    HANDLE_SIZE = 10
    
    @classmethod
    def poll(cls, context):
        sima = context.space_data
        return (context.area.type == 'IMAGE_EDITOR' and 
                sima.mode == 'PAINT' and 
                sima.image is not None)
    
    def _init_grid(self, context):
        """Initialize control point grid based on resolution."""
        props = context.window_manager.image_edit_properties
        mode = props.lattice_mode
        
        if mode == 'PERSPECTIVE':
            res_u, res_v = 2, 2
        else:
            res_u = props.lattice_resolution_u
            res_v = props.lattice_resolution_v
        
        self._control_points = []
        self._original_grid = []
        
        for j in range(res_v):
            row, orig_row = [], []
            for i in range(res_u):
                x = (i / (res_u - 1)) * (self._width - 1) if res_u > 1 else self._width / 2
                y = (j / (res_v - 1)) * (self._height - 1) if res_v > 1 else self._height / 2
                row.append([x, y])
                orig_row.append([x, y])
            self._control_points.append(row)
            self._original_grid.append(orig_row)
    
    def _screen_to_image(self, context, mx, my):
        """Convert screen to image pixel coordinates."""
        view2d = context.region.view2d
        img_x, img_y = view2d.region_to_view(mx, my)
        return img_x * self._width, img_y * self._height
    
    def _image_to_screen(self, context, ix, iy):
        """Convert image to screen coordinates."""
        nx = ix / self._width if self._width > 0 else 0
        ny = iy / self._height if self._height > 0 else 0
        return context.region.view2d.view_to_region(nx, ny)
    
    def _hit_test(self, context, mx, my):
        """Check if mouse is over a control point."""
        ix, iy = self._screen_to_image(context, mx, my)
        hs = self.HANDLE_SIZE * 2
        for j, row in enumerate(self._control_points):
            for i, pt in enumerate(row):
                if abs(ix - pt[0]) < hs and abs(iy - pt[1]) < hs:
                    return (j, i)
        return None
    
    def _find_homography(self, src, dst):
        """Compute 3x3 homography using DLT algorithm."""
        import numpy as np
        n = src.shape[0]
        A = np.zeros((2*n, 9), dtype=np.float64)
        for i in range(n):
            x, y = src[i]
            xp, yp = dst[i]
            A[2*i] = [-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp]
            A[2*i+1] = [0, 0, 0, -x, -y, -1, x*yp, y*yp, yp]
        _, _, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape(3, 3)
        return H / H[2, 2]
    
    def _apply_perspective_warp(self):
        """Apply 4-point perspective transformation."""
        import numpy as np
        
        src = np.array([self._original_grid[0][0], self._original_grid[0][-1],
                        self._original_grid[-1][-1], self._original_grid[-1][0]], dtype=np.float32)
        dst = np.array([self._control_points[0][0], self._control_points[0][-1],
                        self._control_points[-1][-1], self._control_points[-1][0]], dtype=np.float32)
        
        H = self._find_homography(src, dst)
        H_inv = np.linalg.inv(H)
        
        y, x = np.meshgrid(np.arange(self._height, dtype=np.float32),
                           np.arange(self._width, dtype=np.float32), indexing='ij')
        coords = np.stack([x, y, np.ones_like(x)], axis=-1)
        src_coords = np.einsum('ij,...j->...i', H_inv, coords)
        
        w = src_coords[..., 2:3]
        w = np.where(np.abs(w) < 1e-10, 1e-10, w)
        src_x = src_coords[..., 0] / w[..., 0]
        src_y = src_coords[..., 1] / w[..., 0]
        
        self._bilinear_sample(src_x, src_y)
    
    def _apply_mesh_warp(self):
        """Apply mesh deformation with bilinear cell interpolation."""
        import numpy as np
        
        rows, cols = len(self._control_points), len(self._control_points[0])
        if rows < 2 or cols < 2:
            return
        
        y, x = np.meshgrid(np.arange(self._height, dtype=np.float32),
                           np.arange(self._width, dtype=np.float32), indexing='ij')
        src_x, src_y = np.copy(x), np.copy(y)
        
        for j in range(rows - 1):
            for i in range(cols - 1):
                o_tl, o_tr = self._original_grid[j][i], self._original_grid[j][i+1]
                o_bl, o_br = self._original_grid[j+1][i], self._original_grid[j+1][i+1]
                d_tl, d_tr = self._control_points[j][i], self._control_points[j][i+1]
                d_bl, d_br = self._control_points[j+1][i], self._control_points[j+1][i+1]
                
                mask = (x >= o_tl[0]) & (x < o_tr[0]) & (y >= o_tl[1]) & (y < o_bl[1])
                if not np.any(mask):
                    continue
                
                cw, ch = o_tr[0] - o_tl[0], o_bl[1] - o_tl[1]
                if cw < 1 or ch < 1:
                    continue
                
                u = (x[mask] - o_tl[0]) / cw
                v = (y[mask] - o_tl[1]) / ch
                
                src_x[mask] = (1-u)*(1-v)*d_tl[0] + u*(1-v)*d_tr[0] + (1-u)*v*d_bl[0] + u*v*d_br[0]
                src_y[mask] = (1-u)*(1-v)*d_tl[1] + u*(1-v)*d_tr[1] + (1-u)*v*d_bl[1] + u*v*d_br[1]
        
        self._bilinear_sample(src_x, src_y)
    
    def _bilinear_sample(self, src_x, src_y):
        """Sample original pixels with bilinear interpolation."""
        import numpy as np
        src_x = np.clip(src_x, 0, self._width - 1.001)
        src_y = np.clip(src_y, 0, self._height - 1.001)
        
        x0, y0 = src_x.astype(np.int32), src_y.astype(np.int32)
        x1, y1 = np.minimum(x0 + 1, self._width - 1), np.minimum(y0 + 1, self._height - 1)
        fx, fy = (src_x - x0)[:,:,np.newaxis], (src_y - y0)[:,:,np.newaxis]
        
        self._working_pixels[:] = (self._original_pixels[y0, x0] * (1-fx) * (1-fy) +
                                    self._original_pixels[y0, x1] * fx * (1-fy) +
                                    self._original_pixels[y1, x0] * (1-fx) * fy +
                                    self._original_pixels[y1, x1] * fx * fy)
    
    def _update_preview(self, context):
        """Apply deformation and update image preview."""
        props = context.window_manager.image_edit_properties
        if props.lattice_mode == 'PERSPECTIVE':
            self._apply_perspective_warp()
        else:
            self._apply_mesh_warp()
        self._image.pixels.foreach_set(self._working_pixels.ravel())
        self._image.update()
    
    def modal(self, context, event):
        context.area.tag_redraw()
        
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE'}:
            return {'PASS_THROUGH'}
        
        if event.type in {'RET', 'NUMPAD_ENTER', 'SPACE'} and event.value == 'PRESS':
            utils.ImageUndoStack.get().push_state_from_numpy(self._image, self._original_pixels)
            self._image.pixels.foreach_set(self._working_pixels.ravel())
            self._image.update()
            self._cleanup(context)
            self.report({'INFO'}, "Lattice deformation applied")
            return {'FINISHED'}
        
        if event.type == 'ESC' and event.value == 'PRESS':
            self._image.pixels.foreach_set(self._original_pixels.ravel())
            self._image.update()
            self._cleanup(context)
            return {'CANCELLED'}
        
        mx, my = event.mouse_region_x, event.mouse_region_y
        
        if event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                hit = self._hit_test(context, mx, my)
                if hit:
                    self._active_point = hit
                    self._is_dragging = True
            elif event.value == 'RELEASE':
                self._is_dragging = False
                self._active_point = None
        
        elif event.type == 'MOUSEMOVE' and self._is_dragging and self._active_point:
            row, col = self._active_point
            ix, iy = self._screen_to_image(context, mx, my)
            ix = max(0, min(self._width - 1, ix))
            iy = max(0, min(self._height - 1, iy))
            self._control_points[row][col] = [ix, iy]
            self._update_preview(context)
        
        return {'RUNNING_MODAL'}
    
    def invoke(self, context, event):
        import numpy as np
        
        self._image = context.space_data.image
        self._width, self._height = self._image.size
        
        if self._width < 2 or self._height < 2:
            self.report({'ERROR'}, "Image too small")
            return {'CANCELLED'}
        
        pixels = np.zeros(self._width * self._height * 4, dtype=np.float32)
        self._image.pixels.foreach_get(pixels)
        self._original_pixels = pixels.reshape((self._height, self._width, 4))
        self._working_pixels = np.copy(self._original_pixels)
        
        self._init_grid(context)
        self._initialized = True
        
        self._draw_handler = context.space_data.draw_handler_add(
            draw_lattice_overlay, (self, context), 'WINDOW', 'POST_PIXEL')
        
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}
    
    def _cleanup(self, context):
        if self._draw_handler:
            context.space_data.draw_handler_remove(self._draw_handler, 'WINDOW')
            self._draw_handler = None
        self._initialized = False


def draw_lattice_overlay(op, context):
    """Draw lattice grid and control points."""
    import gpu
    from gpu_extras.batch import batch_for_shader
    
    if not op._initialized or not op._control_points:
        return
    
    gpu.state.blend_set('ALPHA')
    gpu.state.line_width_set(1.5)
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    
    rows = len(op._control_points)
    cols = len(op._control_points[0]) if rows else 0
    
    # Grid lines
    lines = []
    for j in range(rows):
        for i in range(cols - 1):
            s1 = op._image_to_screen(context, *op._control_points[j][i])
            s2 = op._image_to_screen(context, *op._control_points[j][i+1])
            lines.extend([s1, s2])
    for j in range(rows - 1):
        for i in range(cols):
            s1 = op._image_to_screen(context, *op._control_points[j][i])
            s2 = op._image_to_screen(context, *op._control_points[j+1][i])
            lines.extend([s1, s2])
    
    if lines:
        shader.uniform_float("color", (0.4, 0.7, 1.0, 0.7))
        batch_for_shader(shader, 'LINES', {"pos": lines}).draw(shader)
    
    # Control points
    pts = [op._image_to_screen(context, *op._control_points[j][i]) 
           for j in range(rows) for i in range(cols)]
    if pts:
        gpu.state.point_size_set(10.0)
        shader.uniform_float("color", (0.2, 0.5, 0.9, 1.0))
        batch_for_shader(shader, 'POINTS', {"pos": pts}).draw(shader)
        gpu.state.point_size_set(6.0)
        shader.uniform_float("color", (1.0, 1.0, 1.0, 1.0))
        batch_for_shader(shader, 'POINTS', {"pos": pts}).draw(shader)
    
    gpu.state.blend_set('NONE')
    gpu.state.line_width_set(1.0)
    gpu.state.point_size_set(1.0)


def _apply_transform_cpu(pixels, orig_w, orig_h, rotation, scale):
    """Apply rotation and scale to layer pixels using CPU (NumPy).

    Args:
        pixels: (H, W, 4) float32 array (bottom-to-top, Blender convention)
        orig_w, orig_h: Original layer dimensions
        rotation: Rotation angle in radians
        scale: (sx, sy) scale factors

    Returns:
        (transformed_pixels, new_width, new_height)
    """
    import numpy as np
    import math

    sx, sy = float(scale[0]), float(scale[1])
    rot = float(rotation)
    cos_r = math.cos(rot)
    sin_r = math.sin(rot)

    # Compute bounding box of transformed corners
    corners = np.array([
        [0, 0], [orig_w, 0], [0, orig_h], [orig_w, orig_h]
    ], dtype=np.float64)
    # Center origin
    cx, cy = orig_w / 2.0, orig_h / 2.0
    corners -= [cx, cy]
    # Apply scale then rotation
    scaled = corners * [sx, sy]
    rotated = np.column_stack([
        scaled[:, 0] * cos_r - scaled[:, 1] * sin_r,
        scaled[:, 0] * sin_r + scaled[:, 1] * cos_r
    ])

    min_xy = rotated.min(axis=0)
    max_xy = rotated.max(axis=0)
    new_w = int(math.ceil(max_xy[0] - min_xy[0]))
    new_h = int(math.ceil(max_xy[1] - min_xy[1]))
    new_w = max(1, new_w)
    new_h = max(1, new_h)

    # Build inverse transform: for each output pixel, find source pixel
    # Output center
    ncx, ncy = new_w / 2.0, new_h / 2.0

    # Create output coordinate grid
    oy, ox = np.mgrid[0:new_h, 0:new_w].astype(np.float64)
    ox -= ncx
    oy -= ncy

    # Inverse rotation
    inv_cos = cos_r   # cos(-r) = cos(r)
    inv_sin = -sin_r   # sin(-r) = -sin(r)
    rx = ox * inv_cos - oy * inv_sin
    ry = ox * inv_sin + oy * inv_cos

    # Inverse scale
    if abs(sx) > 1e-6:
        rx /= sx
    if abs(sy) > 1e-6:
        ry /= sy

    # Shift back to source pixel coords
    rx += cx
    ry += cy

    # Bilinear interpolation
    x0 = np.floor(rx).astype(np.int32)
    y0 = np.floor(ry).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    fx = (rx - x0).astype(np.float32)
    fy = (ry - y0).astype(np.float32)

    # Clamp
    x0c = np.clip(x0, 0, orig_w - 1)
    x1c = np.clip(x1, 0, orig_w - 1)
    y0c = np.clip(y0, 0, orig_h - 1)
    y1c = np.clip(y1, 0, orig_h - 1)

    # Mask for out-of-bounds (make transparent)
    valid = (x0 >= 0) & (x1 < orig_w) & (y0 >= 0) & (y1 < orig_h)

    fx = fx[:, :, np.newaxis]
    fy = fy[:, :, np.newaxis]

    result = (pixels[y0c, x0c] * (1 - fx) * (1 - fy) +
              pixels[y1c, x0c] * (1 - fx) * fy +
              pixels[y0c, x1c] * fx * (1 - fy) +
              pixels[y1c, x1c] * fx * fy)

    # Zero out-of-bounds pixels
    result[~valid] = 0.0

    return result, new_w, new_h


class IMAGE_EDIT_OT_export_layers(bpy.types.Operator):
    """Export all layers as a single image"""
    bl_idname = "image_edit.export_layers"
    bl_label = "Export Layers"
    
    filepath: bpy.props.StringProperty(subtype='FILE_PATH')
    filename: bpy.props.StringProperty(default="files.png")
    directory: bpy.props.StringProperty(subtype='DIR_PATH')

    # Enhanced options
    only_selected: bpy.props.BoolProperty(name="Selected Only", default=False, description="Export only selected layers")
    layout_mode: bpy.props.EnumProperty(
        name="Layout",
        items=[
            ('GRID', "Grid", "Pack layers into a grid"),
            ('LOCATION', "Original Location", "Preserve layer positions (Composition)")
        ],
        default='GRID'
    )
    apply_transforms: bpy.props.BoolProperty(name="Apply Transforms", default=True, description="Apply rotation and scale to layers")
    include_background: bpy.props.BoolProperty(name="Include Background", default=False, description="Include the canvas background image in the export")
    resolution_percentage: bpy.props.IntProperty(name="Resolution %", default=100, min=1, max=1000, description="Scale the final output")
    
    # Grid options
    padding: bpy.props.IntProperty(name="Padding", default=10, min=0)
    columns: bpy.props.IntProperty(name="Columns", default=0, min=0, description="0 for auto (Grid only)")

    bl_options = {'REGISTER', 'UNDO'}
    def invoke(self, context, event):
        img = context.area.spaces.active.image
        if img:
            self.filename = img.name + "_layers.png"
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
    
    def draw(self, context):
        layout = self.layout
        layout.prop(self, "only_selected")
        layout.prop(self, "layout_mode")
        layout.prop(self, "apply_transforms")
        layout.prop(self, "include_background")
        layout.prop(self, "resolution_percentage")
        
        if self.layout_mode == 'GRID':
            box = layout.box()
            box.label(text="Grid Settings")
            box.prop(self, "padding")
            box.prop(self, "columns")
        elif self.layout_mode == 'LOCATION':
            box = layout.box()
            box.label(text="Location Settings")
            box.prop(self, "padding", text="Margin")

    def execute(self, context):
        import os
        img = context.area.spaces.active.image
        if not img:
            self.report({'ERROR'}, "No active image")
            return {'CANCELLED'}
        
        img_props = img.image_edit_properties
        layers = img_props.layers
        
        if not layers:
            self.report({'WARNING'}, "No layers to export")
            return {'CANCELLED'}
            
        # Collect layer data
        layers_data = []

        canvas_w, canvas_h = img.size[0], img.size[1]
        
        for layer in layers:
            # Filter selected
            if self.only_selected and not layer.checked:
                continue
                
            layer_img = bpy.data.images.get(layer.name, None)
            if not layer_img:
                continue
            
            orig_w, orig_h = layer_img.size[0], layer_img.size[1]
            pixels = utils.layer_read_pixels_from_image(layer_img)
            width, height = orig_w, orig_h

            has_transform = (layer.rotation != 0
                             or abs(layer.scale[0] - 1.0) > 1e-4
                             or abs(layer.scale[1] - 1.0) > 1e-4)

            # Apply transforms if requested
            if self.apply_transforms and has_transform:
                pixels, width, height = _apply_transform_cpu(
                    pixels, orig_w, orig_h, layer.rotation, layer.scale)
                
            layers_data.append({
                'name': layer.name,
                'width': width,
                'height': height,
                'orig_w': orig_w,
                'orig_h': orig_h,
                'pixels': pixels, # (h, w, 4)
                'layer_obj': layer,
                'orig_x': layer.location[0],
                'orig_y': layer.location[1]
            })

        # Append background image at the end (bottom of stack)
        if self.include_background:
            bg_pixels = utils.layer_read_pixels_from_image(img)
            layers_data.append({
                'name': img.name,
                'width': canvas_w,
                'height': canvas_h,
                'orig_w': canvas_w,
                'orig_h': canvas_h,
                'pixels': bg_pixels,
                'layer_obj': None,
                'orig_x': 0,
                'orig_y': 0,
            })

        if not layers_data:
            self.report({'WARNING'}, "No layers to export (check selection)")
            return {'CANCELLED'}

        # Compute Layout
        final_w, final_h = 0, 0
        
        if self.layout_mode == 'GRID':
            count = len(layers_data)
            cols = self.columns
            if cols <= 0:
                cols = math.ceil(math.sqrt(count))
            rows = math.ceil(count / cols)
            
            col_widths = [0] * cols
            row_heights = [0] * rows
            
            for i, data in enumerate(layers_data):
                c = i % cols
                r = i // cols
                col_widths[c] = max(col_widths[c], data['width'])
                row_heights[r] = max(row_heights[r], data['height'])
                
            final_w = sum(col_widths) + self.padding * (cols + 1)
            final_h = sum(row_heights) + self.padding * (rows + 1)
            
            # Calculate positions for Grid
            current_y = final_h - self.padding
            for r in range(rows):
                row_h = row_heights[r]
                current_y -= row_h
                current_x = self.padding
                for c in range(cols):
                    idx = r * cols + c
                    if idx >= count:
                        continue
                    data = layers_data[idx]
                    col_w = col_widths[c]
                    
                    off_x = (col_w - data['width']) // 2
                    off_y = (row_h - data['height']) // 2
                    
                    data['x'] = current_x + off_x
                    data['y'] = current_y + off_y
                    
                    current_x += col_w + self.padding
                current_y -= self.padding
                
        elif self.layout_mode == 'LOCATION':
            # Composition mode
            # Find bounds
            min_x, min_y = float('inf'), float('inf')
            max_x, max_y = float('-inf'), float('-inf')
            
            vals_x = []
            vals_y = []
            
            for data in layers_data:
                # Use center relative to canvas to get bottom-left corner of layer contents
                cx = canvas_w / 2.0 + data['orig_x']
                cy = canvas_h / 2.0 + data['orig_y'] 
                
                # Bottom-left coordinate of the layer's content
                x = cx - data['width'] / 2.0
                y = cy - data['height'] / 2.0
                
                data['comp_x'] = x
                data['comp_y'] = y
                
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x + data['width'])
                max_y = max(max_y, y + data['height'])
            
            # Add padding
            content_w = max_x - min_x
            content_h = max_y - min_y
            
            final_w = int(content_w + self.padding * 2)
            final_h = int(content_h + self.padding * 2)
            
            # Relocate to new image space in standard Bottom-Up logic
            for data in layers_data:
                data['x'] = int(data['comp_x'] - min_x + self.padding)
                data['y'] = int(data['comp_y'] - min_y + self.padding)


        # Create output buffer
        out_pixels = np.zeros((final_h, final_w, 4), dtype=np.float32)
        
        # Fill buffer
        for data in layers_data:
            x0 = int(data['x'])
            y0 = int(data['y'])
            x1 = x0 + data['width']
            y1 = y0 + data['height']
            
            # Clipping check
            if x0 >= final_w or y0 >= final_h or x1 <= 0 or y1 <= 0:
                continue
                
            # Crop to bounds
            px = data['pixels']
            
            sx0, sy0 = 0, 0
            sx1, sy1 = data['width'], data['height']
            
            if x0 < 0:
                sx0 = -x0
                x0 = 0
            if y0 < 0:
                sy0 = -y0
                y0 = 0
            if x1 > final_w:
                sx1 -= (x1 - final_w)
                x1 = final_w
            if y1 > final_h:
                sy1 -= (y1 - final_h)
                y1 = final_h
                
            if sx1 <= sx0 or sy1 <= sy0:
                continue
                
            # Copy with alpha blending? 
            # If masking, we should Blend?
            # Or just Overwrite? 
            # In Grid mode -> Overwrite (no overlap).
            # In Location mode -> Overwrite (layers order).
            
            # We iterate layers in order (Top to Bottom in list).
            # We should draw Bottom layers first for correct composition.
            # `layers` list in Blender is usually Top-to-Bottom.
            # So we should iterate reversed(layers_data) for Painter's Algorithm.
            pass
            
        # Draw in reverse order (Bottom-First)
        for data in reversed(layers_data):
            x0 = int(data['x'])
            y0 = int(data['y'])
            x1 = x0 + data['width']
            y1 = y0 + data['height']
            
            # Clipping (re-calc)
            sx0, sy0 = 0, 0
            sx1, sy1 = data['width'], data['height']
            
            if x0 < 0: sx0 = -x0; x0 = 0
            if y0 < 0: sy0 = -y0; y0 = 0
            if x1 > final_w: sx1 -= (x1 - final_w); x1 = final_w
            if y1 > final_h: sy1 -= (y1 - final_h); y1 = final_h
                
            if sx1 <= sx0 or sy1 <= sy0: continue
            
            # Read source slice
            src = data['pixels'][sy0:sy1, sx0:sx1]
            
            # Determine blend mode and opacity from layer properties
            layer_obj = data['layer_obj']
            blend_mode = layer_obj.blend_mode if layer_obj else 'MIX'
            opacity = layer_obj.opacity if layer_obj else 1.0
            
            # Composite with blend mode and opacity
            utils.composite_layer_numpy(
                out_pixels[y0:y1, x0:x1], src, blend_mode, opacity
            )

        # Resolution Scaling
        if self.resolution_percentage != 100:
            import cv2
            # Not sure if cv2 is available in Blender python. 
            # Safest is to rely on Blender's image scaling or simple NN/Bilinear if manual.
            # Using Blender's scale method on the image object is best.
            pass
            
        # Create output image
        output_name = self.filename if self.filename else "Exported Layers"
        internal_name = os.path.splitext(os.path.basename(output_name))[0]
        
        new_img = bpy.data.images.new(internal_name, width=final_w, height=final_h, alpha=True, float_buffer=img.is_float)
        new_img.colorspace_settings.name = img.colorspace_settings.name
        utils.layer_write_pixels_to_image(new_img, out_pixels)
        
        if self.resolution_percentage != 100:
            new_w = int(final_w * self.resolution_percentage / 100.0)
            new_h = int(final_h * self.resolution_percentage / 100.0)
            new_img.scale(new_w, new_h)
        
        # Save
        if self.filepath:
            new_img.filepath_raw = self.filepath
            new_img.file_format = 'PNG'
            new_img.save()
            self.report({'INFO'}, f"Saved to {self.filepath}")
        else:
             self.report({'INFO'}, f"Created image {internal_name}")

        return {'FINISHED'}


class IMAGE_EDIT_OT_export_layers_multilayer(bpy.types.Operator):
    """Export layers as a multi-layer PSD file"""
    bl_idname = "image_edit.export_layers_multilayer"
    bl_label = "Export Multi-Layer File"

    filepath: bpy.props.StringProperty(subtype='FILE_PATH')
    filename: bpy.props.StringProperty(default="export.psd")
    directory: bpy.props.StringProperty(subtype='DIR_PATH')

    export_format: bpy.props.EnumProperty(
        name="Format",
        items=[
            ('PSD', "PSD", "Adobe Photoshop (.psd)"),
        ],
        default='PSD'
    )
    only_selected: bpy.props.BoolProperty(
        name="Selected Only", default=False,
        description="Export only checked layers"
    )
    apply_transforms: bpy.props.BoolProperty(
        name="Apply Transforms", default=True,
        description="Bake rotation and scale into layer pixels"
    )
    include_background: bpy.props.BoolProperty(
        name="Include Background", default=False,
        description="Include the canvas background image as a layer"
    )

    _EXT_MAP = {'PSD': '.psd'}

    bl_options = {'REGISTER', 'UNDO'}
    def invoke(self, context, event):
        img = context.area.spaces.active.image
        if img:
            ext = self._EXT_MAP.get(self.export_format, '.psd')
            self.filename = img.name + "_layers" + ext
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def check(self, context):
        import os
        # Auto-update extension when format changes
        ext = self._EXT_MAP.get(self.export_format, '.psd')
        base = os.path.splitext(self.filename)[0]
        new_name = base + ext
        if new_name != self.filename:
            self.filename = new_name
            if self.filepath:
                dir_part = os.path.dirname(self.filepath)
                self.filepath = os.path.join(dir_part, new_name)
            return True
        return False

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "export_format")
        layout.prop(self, "only_selected")
        layout.prop(self, "apply_transforms")
        layout.prop(self, "include_background")

    def execute(self, context):
        import os
        img = context.area.spaces.active.image
        if not img:
            self.report({'ERROR'}, "No active image")
            return {'CANCELLED'}

        img_props = img.image_edit_properties
        layers = img_props.layers

        if not layers:
            self.report({'WARNING'}, "No layers to export")
            return {'CANCELLED'}

        # Collect layer data
        layers_data = []
        canvas_w, canvas_h = img.size[0], img.size[1]
        import numpy as np

        for layer in layers:
            if self.only_selected and not layer.checked:
                continue

            layer_img = bpy.data.images.get(layer.name, None)
            if not layer_img:
                continue

            orig_w, orig_h = layer_img.size[0], layer_img.size[1]
            pixels = utils.layer_read_pixels_from_image(layer_img)
            width, height = orig_w, orig_h

            has_transform = (layer.rotation != 0
                             or abs(layer.scale[0] - 1.0) > 1e-4
                             or abs(layer.scale[1] - 1.0) > 1e-4)

            if self.apply_transforms and has_transform:
                pixels, width, height = self._apply_transform_cpu(
                    pixels, orig_w, orig_h, layer.rotation, layer.scale)

            # Blender pixels are bottom-to-top; PSD expects top-to-bottom
            pixels = np.flipud(pixels)

            # layer.location is (x, y_from_top)
            lx = int(layer.location[0])
            ly = int(layer.location[1])

            # Adjust position for center-pivot transform expansion
            if self.apply_transforms and has_transform:
                lx += (orig_w - width) // 2
                ly += (orig_h - height) // 2

            layers_data.append({
                'name': layer.label if layer.label else layer.name,
                'pixels': pixels,
                'x': lx,
                'y': ly,
                'width': width,
                'height': height,
                'opacity': layer.opacity,
                'blend_mode': layer.blend_mode,
                'visible': not layer.hide,
            })

        # Append background image at the end (bottom of layer stack)
        if self.include_background:
            bg_pixels = utils.layer_read_pixels_from_image(img)
            bg_pixels = np.flipud(bg_pixels)
            layers_data.append({
                'name': 'Background',
                'pixels': bg_pixels,
                'x': 0,
                'y': 0,
                'width': canvas_w,
                'height': canvas_h,
                'opacity': 1.0,
                'blend_mode': 'MIX',
                'visible': True,
            })

        if not layers_data:
            self.report({'WARNING'}, "No layers to export (check selection)")
            return {'CANCELLED'}

        # Ensure filepath has proper extension
        ext = self._EXT_MAP.get(self.export_format, '.psd')
        filepath = self.filepath
        if not filepath.lower().endswith(ext):
            filepath = os.path.splitext(filepath)[0] + ext

        try:
            if self.export_format == 'PSD':
                from ..utils.psd_writer import write_psd
                write_psd(filepath, layers_data, canvas_w, canvas_h)

        except Exception as e:
            self.report({'ERROR'}, f"Export failed: {e}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}

        self.report({'INFO'}, f"Exported {len(layers_data)} layers to {filepath}")
        return {'FINISHED'}

    @staticmethod
    def _apply_transform_cpu(pixels, orig_w, orig_h, rotation, scale):
        return _apply_transform_cpu(pixels, orig_w, orig_h, rotation, scale)


class IMAGE_EDIT_OT_import_psd(bpy.types.Operator):
    """Import a PSD file as layers"""
    bl_idname = "image_edit.import_psd"
    bl_label = "Import PSD"

    filepath: bpy.props.StringProperty(subtype='FILE_PATH')
    filter_glob: bpy.props.StringProperty(default="*.psd", options={'HIDDEN'})

    clear_existing: bpy.props.BoolProperty(
        name="Clear Existing Layers",
        description="Remove existing layers before importing",
        default=True
    )

    bl_options = {'REGISTER', 'UNDO'}
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "clear_existing")

    def execute(self, context):
        import os
        import traceback

        img = context.area.spaces.active.image
        if not img:
            self.report({'ERROR'}, "No active image")
            return {'CANCELLED'}

        if not self.filepath or not os.path.isfile(self.filepath):
            self.report({'ERROR'}, "Invalid file path")
            return {'CANCELLED'}

        try:
            from ..utils.psd_reader import read_psd
            psd = read_psd(self.filepath)
        except ValueError as e:
            self.report({'ERROR'}, f"PSD read error: {e}")
            return {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, f"Failed to read PSD: {e}")
            traceback.print_exc()
            return {'CANCELLED'}

        if not psd.layers:
            self.report({'WARNING'}, "No layers found in PSD file")
            return {'CANCELLED'}

        # Clear existing layers if requested
        if self.clear_existing:
            img_props = img.image_edit_properties
            for layer in img_props.layers:
                layer_img = bpy.data.images.get(layer.name, None)
                if layer_img:
                    bpy.data.images.remove(layer_img)
            img_props.layers.clear()
            img_props.selected_layer_index = -1

        # Resize base image to PSD canvas size
        canvas_w, canvas_h = psd.width, psd.height
        if img.size[0] != canvas_w or img.size[1] != canvas_h:
            img.scale(canvas_w, canvas_h)

        # PSD layers are stored bottom-to-top in psd_reader output.
        # We iterate in order (bottom first) and add each layer.
        # layer_create_layer inserts at top (index 0), so we process
        # bottom-to-top which results in correct final order.
        imported_count = 0

        for psd_layer in psd.layers:
            if psd_layer.pixels is None or psd_layer.width <= 0 or psd_layer.height <= 0:
                continue

            # Flip pixels: PSD is top-to-bottom, Blender is bottom-to-top
            pixels = np.flipud(psd_layer.pixels)

            img_settings = {
                'is_float': img.is_float,
                'colorspace_name': img.colorspace_settings.name
            }

            # Create the layer (inserts at top of stack)
            utils.layer_create_layer(img, pixels, img_settings, None,
                                     custom_label=psd_layer.name)

            # Get the newly created layer (it's at index 0 after layer_create_layer)
            img_props = img.image_edit_properties
            new_layer = img_props.layers[0]

            # Set position: PSD uses (left, top) from top-left origin
            # Blender layer.location is center offset relative to the canvas center
            layer_center_x = psd_layer.left + psd_layer.width / 2.0
            layer_center_y = canvas_h - (psd_layer.top + psd_layer.height / 2.0)
            new_layer.location = [layer_center_x - canvas_w / 2.0, layer_center_y - canvas_h / 2.0]

            # Set properties
            new_layer.opacity = psd_layer.opacity
            new_layer.blend_mode = psd_layer.blend_mode
            new_layer.hide = not psd_layer.visible

            imported_count += 1

        utils.layer_cancel_selection(context)
        utils.layer_refresh_image(context)

        self.report({'INFO'}, f"Imported {imported_count} layers from PSD")
        return {'FINISHED'}

class IMAGE_EDIT_OT_apply_layer_transform(bpy.types.Operator):
    """Apply the transform (Rotation and Scale) to the layer pixels"""
    bl_idname = "image_edit.apply_layer_transform"
    bl_label = "Apply Transform"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        if context.area.type != 'IMAGE_EDITOR':
            return False
        img = context.area.spaces.active.image
        if not img:
            return False
        from .. import utils
        layer = utils.layer_get_active_layer(context)
        if not layer:
            return False
        return (layer.rotation != 0.0 or layer.scale[0] != 1.0 or layer.scale[1] != 1.0)

    def execute(self, context):
        from .. import utils
        img = context.area.spaces.active.image
        layer = utils.layer_get_active_layer(context)
        
        if layer.locked:
            self.report({'WARNING'}, 'Layer is locked.')
            return {'CANCELLED'}
            
        layer_img = bpy.data.images.get(layer.name, None)
        if not layer_img:
            return {'CANCELLED'}

        layer_pixels, new_layer_width, new_layer_height = utils.layer_apply_layer_transform(layer_img, layer.rotation, layer.scale)
        
        utils.ImageUndoStack.get().push_state(img)
        
        layer_img.scale(new_layer_width, new_layer_height)
        utils.layer_write_pixels_to_image(layer_img, layer_pixels)
        
        layer.rotation = 0.0
        layer.scale[0] = 1.0
        layer.scale[1] = 1.0
        
        utils.layer_rebuild_image_layers_nodes(img)
        utils.layer_refresh_image(context)
        
        self.report({'INFO'}, 'Layer transform applied.')
        return {'FINISHED'}


class IMAGE_EDIT_OT_align_layers(bpy.types.Operator):
    """Align selected layers"""
    bl_idname = "image_edit.align_layers"
    bl_label = "Align Layers"
    bl_options = {'REGISTER', 'UNDO'}

    align_type: bpy.props.EnumProperty(
        name="Align",
        items=[
            ('LEFT', "Left", "Align Left"),
            ('CENTER_H', "Center Horizontal", "Align Center Horizontal"),
            ('RIGHT', "Right", "Align Right"),
            ('TOP', "Top", "Align Top"),
            ('CENTER_V', "Center Vertical", "Align Center Vertical"),
            ('BOTTOM', "Bottom", "Align Bottom"),
        ],
        default='LEFT'
    )

    @classmethod
    def poll(cls, context):
        if context.area.type != 'IMAGE_EDITOR':
            return False
        img = context.area.spaces.active.image
        if not img:
            return False
        return True

    def execute(self, context):
        from .. import utils
        img = context.area.spaces.active.image
        
        selected_layers = [lay for lay in img.image_edit_properties.layers if lay.checked and not lay.locked]
        
        if len(selected_layers) < 2:
            self.report({'WARNING'}, "Needs at least 2 checked and unlocked layers.")
            return {'CANCELLED'}
            
        utils.ImageUndoStack.get().push_state(img)
            
        if img.image_edit_properties.align_relative_to == 'CANVAS':
            min_x = 0
            min_y = 0
            max_x = img.size[0]
            max_y = img.size[1]
        else:
            # Get bounding box of all selected layers
            min_x = float('inf')
            min_y = float('inf')
            max_x = float('-inf')
            max_y = float('-inf')
            
            for lay in selected_layers:
                layer_img = bpy.data.images.get(lay.name, None)
                if not layer_img:
                    continue
                    
                orig_w, orig_h = layer_img.size[0], layer_img.size[1]
                scaled_w = orig_w * lay.scale[0]
                scaled_h = orig_h * lay.scale[1]
                
                # location is center relative to page center
                lx_center = lay.location[0] + img.size[0] / 2.0
                ly_center = lay.location[1] + img.size[1] / 2.0
                
                l_min_x = lx_center - scaled_w / 2.0
                l_max_x = lx_center + scaled_w / 2.0
                l_min_y = ly_center - scaled_h / 2.0
                l_max_y = ly_center + scaled_h / 2.0
                
                min_x = min(min_x, l_min_x)
                min_y = min(min_y, l_min_y)
                max_x = max(max_x, l_max_x)
                max_y = max(max_y, l_max_y)
            
        # Align
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0
        
        for i, lay in enumerate(selected_layers):
            layer_img = bpy.data.images.get(lay.name, None)
            if not layer_img:
                continue
            
            orig_w, orig_h = layer_img.size[0], layer_img.size[1]
            scaled_w = orig_w * lay.scale[0]
            scaled_h = orig_h * lay.scale[1]
            
            if img.image_edit_properties.align_relative_to == 'PAGE':
                # Convert the page bounds back to relative offsets
                rel_min_x = -img.size[0] / 2.0
                rel_max_x = img.size[0] / 2.0
                rel_min_y = -img.size[1] / 2.0
                rel_max_y = img.size[1] / 2.0
                rel_center_x = 0
                rel_center_y = 0
                
                if self.align_type == 'LEFT':
                    lay.location[0] = rel_min_x + scaled_w / 2.0
                elif self.align_type == 'RIGHT':
                    lay.location[0] = rel_max_x - scaled_w / 2.0
                elif self.align_type == 'CENTER_H':
                    lay.location[0] = rel_center_x
                elif self.align_type == 'BOTTOM':
                    lay.location[1] = rel_min_y + scaled_h / 2.0
                elif self.align_type == 'TOP':
                    lay.location[1] = rel_max_y - scaled_h / 2.0
                elif self.align_type == 'CENTER_V':
                    lay.location[1] = rel_center_y
            else:
                if self.align_type == 'LEFT':
                    lay.location[0] = min_x + scaled_w / 2.0 - img.size[0] / 2.0
                elif self.align_type == 'RIGHT':
                    lay.location[0] = max_x - scaled_w / 2.0 - img.size[0] / 2.0
                elif self.align_type == 'CENTER_H':
                    lay.location[0] = center_x - img.size[0] / 2.0
                elif self.align_type == 'BOTTOM':
                    lay.location[1] = min_y + scaled_h / 2.0 - img.size[1] / 2.0
                elif self.align_type == 'TOP':
                    lay.location[1] = max_y - scaled_h / 2.0 - img.size[1] / 2.0
                elif self.align_type == 'CENTER_V':
                    lay.location[1] = center_y - img.size[1] / 2.0
                
        utils.layer_rebuild_image_layers_nodes(img)
        utils.layer_refresh_image(context)
        return {'FINISHED'}


class IMAGE_EDIT_OT_distribute_layers(bpy.types.Operator):
    """Distribute selected layers"""
    bl_idname = "image_edit.distribute_layers"
    bl_label = "Distribute Layers"
    bl_options = {'REGISTER', 'UNDO'}

    distribute_mode: bpy.props.EnumProperty(
        name="Mode",
        items=[
            ('CENTERS', "Centers", "Distribute layers by their centers"),
            ('GAPS', "Even Gaps", "Distribute layers with even gaps between them"),
        ],
        default='CENTERS'
    )
    
    distribute_type: bpy.props.EnumProperty(
        name="Distribute Axis",
        items=[
            ('HORIZONTAL', "Horizontal", "Distribute Horizontally"),
            ('VERTICAL', "Vertical", "Distribute Vertically"),
        ],
        default='HORIZONTAL'
    )

    use_gap: bpy.props.BoolProperty(
        name="Use Gap",
        default=False
    )
    
    gap_value: bpy.props.FloatProperty(
        name="Gap Value",
        default=50.0
    )

    @classmethod
    def poll(cls, context):
        if context.area.type != 'IMAGE_EDITOR':
            return False
        img = context.area.spaces.active.image
        if not img:
            return False
        return True

    def execute(self, context):
        from .. import utils
        img = context.area.spaces.active.image
        
        selected_layers = [lay for lay in img.image_edit_properties.layers if lay.checked and not lay.locked]
        
        if len(selected_layers) < 3 and not (len(selected_layers) >= 2 and self.use_gap):
            self.report({'WARNING'}, "Needs at least 3 layers (or 2 if using specific gap).")
            return {'CANCELLED'}
            
        utils.ImageUndoStack.get().push_state(img)
            
        is_horiz = self.distribute_type == 'HORIZONTAL'
        axis_idx = 0 if is_horiz else 1
        
        # Sort layers by center position on the axis
        selected_layers.sort(key=lambda l: l.location[axis_idx])
        
        # We need the extents of each layer
        layer_extents = []
        for lay in selected_layers:
            layer_img = bpy.data.images.get(lay.name, None)
            if not layer_img:
                layer_extents.append(0)
                continue
            extent = (layer_img.size[axis_idx] * lay.scale[axis_idx])
            layer_extents.append(extent)
            
        if self.distribute_mode == 'CENTERS' and not self.use_gap:
            min_val = selected_layers[0].location[axis_idx]
            max_val = selected_layers[-1].location[axis_idx]
            step = (max_val - min_val) / (len(selected_layers) - 1)
            for i, lay in enumerate(selected_layers):
                lay.location[axis_idx] = min_val + step * i
                
        elif self.distribute_mode == 'CENTERS' and self.use_gap:
            current_pos = selected_layers[0].location[axis_idx]
            for i in range(1, len(selected_layers)):
                lay = selected_layers[i]
                current_pos += self.gap_value
                lay.location[axis_idx] = current_pos
                
        elif self.distribute_mode == 'GAPS':
            first_lay = selected_layers[0]
            first_edge = first_lay.location[axis_idx] + layer_extents[0]/2.0
            
            if self.use_gap:
                gap_size = self.gap_value
            else:
                last_lay = selected_layers[-1]
                last_edge = last_lay.location[axis_idx] - layer_extents[-1]/2.0
                
                total_gap_space = last_edge - first_edge
                for i in range(1, len(selected_layers)-1):
                    total_gap_space -= layer_extents[i]
                gap_size = total_gap_space / (len(selected_layers) - 1)
            
            # Now we use gap_size directly starting from edge of first item
            current_pos = first_edge
            # Note range: if we are using an explicit gap, we want to move ALL remaining layers
            # including the last one, to exact locations. If not using explicit gap, moving the
            # last one won't strictly affect anything since it's theoretically at the end bounds.
            loop_end = len(selected_layers) if self.use_gap else len(selected_layers) - 1
            for i in range(1, loop_end):
                lay = selected_layers[i]
                extent = layer_extents[i]
                lay.location[axis_idx] = current_pos + gap_size + extent/2.0
                current_pos += gap_size + extent
                
        utils.layer_rebuild_image_layers_nodes(img)
        utils.layer_refresh_image(context)
        return {'FINISHED'}


class IMAGE_EDIT_OT_arrange_grid(bpy.types.Operator):
    """Arrange selected layers into a grid"""
    bl_idname = "image_edit.arrange_grid"
    bl_label = "Arrange Grid"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        if context.area.type != 'IMAGE_EDITOR':
            return False
        img = context.area.spaces.active.image
        if not img:
            return False
        return True

    def execute(self, context):
        from .. import utils
        img = context.area.spaces.active.image
        
        selected_layers = [lay for lay in img.image_edit_properties.layers if lay.checked and not lay.locked]
        
        if len(selected_layers) < 2:
            self.report({'WARNING'}, "Needs at least 2 checked and unlocked layers.")
            return {'CANCELLED'}
            
        utils.ImageUndoStack.get().push_state(img)
        
        img_props = img.image_edit_properties
        cols = img_props.arrange_grid_x
        rows = img_props.arrange_grid_y
        gap_x = img_props.arrange_gap_x
        gap_y = img_props.arrange_gap_y
        
        # Organize layers into grid
        grid = []
        for r in range(rows):
            grid.append([])
            for c in range(cols):
                idx = r * cols + c
                if idx < len(selected_layers):
                    grid[r].append(selected_layers[idx])
                else:
                    grid[r].append(None)
                    
        # Calculate individual extents
        extents = {}
        for lay in selected_layers:
            layer_img = bpy.data.images.get(lay.name, None)
            if layer_img:
                w = layer_img.size[0] * lay.scale[0]
                h = layer_img.size[1] * lay.scale[1]
                extents[lay] = (w, h)
            else:
                extents[lay] = (0, 0)
                
        # Calculate row dimensions and overall grid dimensions to center it
        row_widths = []
        row_max_heights = []
        
        for r in range(rows):
            r_width = 0
            r_height = 0
            items_in_row = 0
            for c in range(cols):
                lay = grid[r][c]
                if lay:
                    w, h = extents[lay]
                    r_width += w
                    r_height = max(r_height, h)
                    items_in_row += 1
            if items_in_row > 0:
                r_width += gap_x * (items_in_row - 1)
            row_widths.append(r_width)
            row_max_heights.append(r_height)
            
        total_w = max(row_widths) if row_widths else 0
        total_h = sum(row_max_heights) + gap_y * (len([h for h in row_max_heights if h > 0]) - 1)
        
        # Start top-left
        start_x = -total_w / 2.0
        start_y = total_h / 2.0
        
        current_y = start_y
        for r in range(rows):
            r_height = row_max_heights[r]
            if r_height == 0:
                continue # Empty row
                
            current_x = start_x
            for c in range(cols):
                lay = grid[r][c]
                if lay:
                    w, h = extents[lay]
                    # Center the item vertically within the row's max height
                    lay.location[0] = current_x + w / 2.0
                    lay.location[1] = current_y - r_height / 2.0
                    
                    current_x += w + gap_x
                    
            current_y -= (r_height + gap_y)
            
        utils.layer_rebuild_image_layers_nodes(img)
        utils.layer_refresh_image(context)
        return {'FINISHED'}

