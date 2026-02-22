import bpy
from bpy.types import Operator
from .. import utils


class TEXTURE_PAINT_OT_refresh_fonts(Operator):
    bl_idname = "paint.refresh_fonts_ttf"
    bl_label = "Refresh Fonts"
    bl_description = "Rescan font directories and reload fonts"
    bl_options = {'REGISTER'}

    def execute(self, context):
        utils.reset_font_cache()
        utils.load_custom_fonts_to_blender()
        self.report({'INFO'}, "Font cache refreshed")
        return {'FINISHED'}




# ----------------------------
# Undo/Redo Operators
# ----------------------------
def _get_active_image_for_undo(context):
    """Get the active image for undo/redo based on current context."""
    # Check if we're in image editor
    if context.area and context.area.type == 'IMAGE_EDITOR':
        if context.space_data and context.space_data.image:
            return context.space_data.image
    
    # Check if we're in 3D viewport texture paint
    if context.mode == 'PAINT_TEXTURE':
        obj = context.active_object
        if obj and obj.type == 'MESH' and obj.active_material:
            mat = obj.active_material
            if mat.use_nodes:
                for node in mat.node_tree.nodes:
                    if node.type == 'TEX_IMAGE' and node.select:
                        return node.image
                for node in mat.node_tree.nodes:
                    if node.type == 'TEX_IMAGE' and node.image:
                        return node.image
    return None





class TEXTTOOL_OT_undo(Operator):
    bl_idname = "texttool.undo"
    bl_label = "Undo Text Paint"
    bl_description = "Undo the last text paint operation"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        image = _get_active_image_for_undo(context)
        return utils.ImageUndoStack.get().can_undo(image)

    def execute(self, context):
        image = _get_active_image_for_undo(context)
        if image and utils.ImageUndoStack.get().undo(image):
            utils.force_texture_refresh(context, image)
            self.report({'INFO'}, "Text paint undone")
            return {'FINISHED'}
        self.report({'WARNING'}, "Nothing to undo")
        return {'CANCELLED'}


class TEXTTOOL_OT_redo(Operator):
    bl_idname = "texttool.redo"
    bl_label = "Redo Text Paint"
    bl_description = "Redo the last undone text paint operation"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        image = _get_active_image_for_undo(context)
        return utils.ImageUndoStack.get().can_redo(image)

    def execute(self, context):
        image = _get_active_image_for_undo(context)
        if image and utils.ImageUndoStack.get().redo(image):
            utils.force_texture_refresh(context, image)
            self.report({'INFO'}, "Text paint redone")
            return {'FINISHED'}
        self.report({'WARNING'}, "Nothing to redo")
        return {'CANCELLED'}

# ----------------------------
# Texture Randomizer Operators
# ----------------------------

class TEXTTOOL_OT_add_texture(Operator):
    bl_idname = "texttool.add_texture"
    bl_label = "Add Texture"
    bl_description = "Add a texture to the randomization list"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.text_tool_properties
        props.texture_list.add()
        props.texture_index = len(props.texture_list) - 1
        return {'FINISHED'}

class TEXTTOOL_OT_remove_texture(Operator):
    bl_idname = "texttool.remove_texture"
    bl_label = "Remove Texture"
    bl_description = "Remove the selected texture from the list"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        props = context.scene.text_tool_properties
        return props.texture_list and len(props.texture_list) > 0

    def execute(self, context):
        props = context.scene.text_tool_properties
        index = props.texture_index
        props.texture_list.remove(index)
        props.texture_index = min(max(0, index - 1), len(props.texture_list) - 1)
        return {'FINISHED'}

class TEXTTOOL_OT_move_texture(Operator):
    bl_idname = "texttool.move_texture"
    bl_label = "Move Texture"
    bl_description = "Move the selected texture up or down in the list"
    bl_options = {'REGISTER', 'UNDO'}

    direction: bpy.props.StringProperty()

    @classmethod
    def poll(cls, context):
        props = context.scene.text_tool_properties
        return props.texture_list and len(props.texture_list) > 1

    def execute(self, context):
        props = context.scene.text_tool_properties
        index = props.texture_index
        new_index = index - 1 if self.direction == 'UP' else index + 1
        
        if 0 <= new_index < len(props.texture_list):
            props.texture_list.move(index, new_index)
            props.texture_index = new_index
            return {'FINISHED'}
        return {'CANCELLED'}



