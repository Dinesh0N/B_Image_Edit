import bpy
from bpy.types import AddonPreferences, PropertyGroup, Operator
from bpy.props import StringProperty, CollectionProperty, IntProperty


class FontPathItem(PropertyGroup):
    """A single custom font directory path."""
    path: StringProperty(
        name="Path",
        description="Path to a custom font directory",
        subtype='DIR_PATH',
        default=""
    )


class TEXTTOOL_OT_add_font_path(Operator):
    """Add a new custom font directory"""
    bl_idname = "texttool.add_font_path"
    bl_label = "Add Font Path"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        prefs = context.preferences.addons[__package__].preferences
        prefs.custom_font_paths.add()
        # Reset font cache to include new path
        from . import utils
        utils.reset_font_cache()
        return {'FINISHED'}


class TEXTTOOL_OT_remove_font_path(Operator):
    """Remove a custom font directory"""
    bl_idname = "texttool.remove_font_path"
    bl_label = "Remove Font Path"
    bl_options = {'REGISTER', 'UNDO'}

    index: IntProperty()

    def execute(self, context):
        prefs = context.preferences.addons[__package__].preferences
        if 0 <= self.index < len(prefs.custom_font_paths):
            prefs.custom_font_paths.remove(self.index)
            # Reset font cache
            from . import utils
            utils.reset_font_cache()
        return {'FINISHED'}


class TextToolPreferences(AddonPreferences):
    bl_idname = __package__

    custom_font_paths: CollectionProperty(
        type=FontPathItem,
        name="Custom Font Paths",
        description="Additional directories to scan for fonts"
    )
    
    active_font_path_index: IntProperty(
        name="Active Font Path Index",
        default=0
    )

    def draw(self, context):
        layout = self.layout
        col = layout.column()
        col.label(text="Uses Blender's built-in font rendering (blf)", icon='INFO')
        col.separator()
        
        # Custom font paths section
        box = layout.box()
        box.label(text="Custom Font Directories:", icon='FONT_DATA')
        
        for i, item in enumerate(self.custom_font_paths):
            row = box.row(align=True)
            row.prop(item, "path", text="")
            op = row.operator("texttool.remove_font_path", text="", icon='X')
            op.index = i
        
        box.operator("texttool.add_font_path", text="Add Font Directory", icon='ADD')
        
        col.separator()
        col.operator("paint.refresh_fonts_ttf", text="Refresh Fonts", icon='FILE_REFRESH')


# Classes to register
preference_classes = (
    FontPathItem,
    TEXTTOOL_OT_add_font_path,
    TEXTTOOL_OT_remove_font_path,
    TextToolPreferences,
)
