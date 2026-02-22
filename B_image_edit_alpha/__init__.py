bl_info = {
    "name": "B Image Editor",
    "author": "Dinesh007",
    "version": (1, 0, 0),
    "blender": (5, 0, 0),
    "location": "Texture Paint Mode > Toolbar",
    "description": "Text, Gradient, Crop, Clone, Layers and Selection tools.",
    "category": "Paint",
}

import bpy
from . import properties
from . import preferences
from . import ui
from . import operators
from . import sync
from . import utils
from . import api

classes = (
    # Main addon property groups
    properties.TextTexProperties,
    *preferences.preference_classes,
    
    # Main addon operators
    operators.TEXTURE_PAINT_OT_input_text,
    operators.TEXTURE_PAINT_OT_refresh_fonts,
    operators.TEXTURE_PAINT_OT_text_tool,
    operators.IMAGE_PAINT_OT_text_tool,
    operators.TEXTURE_PAINT_OT_gradient_tool,
    operators.IMAGE_PAINT_OT_gradient_tool,
    operators.IMAGE_PAINT_OT_crop_tool,
    operators.IMAGE_PAINT_OT_clone_set_source,
    operators.IMAGE_PAINT_OT_clone_tool,
    operators.IMAGE_PAINT_OT_clone_adjust_size,
    operators.IMAGE_PAINT_OT_clone_adjust_strength,
    operators.IMAGE_PAINT_OT_pen_tool,
    operators.TEXTTOOL_OT_undo,
    operators.TEXTTOOL_OT_redo,
    operators.TEXTTOOL_OT_adjust_font_size,
    operators.TEXTTOOL_OT_adjust_rotation,
    operators.TEXTTOOL_OT_add_texture,
    operators.TEXTTOOL_OT_remove_texture,
    operators.TEXTTOOL_OT_move_texture,
    operators.TEXTURE_PAINT_OT_pen_tool,
    
    # Layer property groups (merged from im_edit)
    properties.IMAGE_EDIT_WindowPropertyGroup,
    properties.IMAGE_EDIT_LayerPropertyGroup,
    properties.IMAGE_EDIT_ImagePropertyGroup,
    
    # Layer operators (merged from im_edit)
    operators.IMAGE_EDIT_OT_make_selection,
    operators.IMAGE_EDIT_OT_make_ellipse_selection,
    operators.IMAGE_EDIT_OT_make_lasso_selection,
    operators.IMAGE_EDIT_OT_cancel_selection,
    operators.IMAGE_EDIT_OT_undo_selection,
    operators.IMAGE_EDIT_OT_redo_selection,
    operators.IMAGE_EDIT_OT_swap_colors,
    operators.IMAGE_EDIT_OT_fill_with_fg_color,
    operators.IMAGE_EDIT_OT_fill_with_bg_color,
    operators.IMAGE_EDIT_OT_clear,
    operators.IMAGE_EDIT_OT_cut,
    operators.IMAGE_EDIT_OT_copy,
    operators.IMAGE_EDIT_OT_paste,
    operators.IMAGE_EDIT_OT_cut_to_layer,
    operators.IMAGE_EDIT_OT_copy_to_layer,
    operators.IMAGE_EDIT_OT_add_image_layer,
    operators.IMAGE_EDIT_OT_new_image_layer,
    operators.IMAGE_EDIT_OT_crop,
    operators.IMAGE_EDIT_OT_deselect_layer,
    operators.IMAGE_EDIT_OT_move_layer,
    operators.IMAGE_EDIT_OT_delete_layer,
    operators.IMAGE_EDIT_OT_edit_layer,
    operators.IMAGE_EDIT_OT_duplicate_layer,
    operators.IMAGE_EDIT_OT_lock_all_layers,
    operators.IMAGE_EDIT_OT_unlock_all_layers,
    operators.IMAGE_EDIT_OT_hide_all_layers,
    operators.IMAGE_EDIT_OT_show_all_layers,
    operators.IMAGE_EDIT_OT_delete_all_layers,
    operators.IMAGE_EDIT_OT_update_layer_previews,
    operators.IMAGE_EDIT_OT_select_all_layers,
    operators.IMAGE_EDIT_OT_deselect_all_layers,
    operators.IMAGE_EDIT_OT_invert_layer_selection,
    operators.IMAGE_EDIT_OT_delete_selected_layers,
    operators.IMAGE_EDIT_OT_merge_selected_layers,
    operators.IMAGE_EDIT_OT_change_image_layer_order,
    operators.IMAGE_EDIT_OT_merge_layers,
    operators.IMAGE_EDIT_OT_flip_layer,
    operators.IMAGE_EDIT_OT_rotate_layer,
    operators.IMAGE_EDIT_OT_rotate_layer_arbitrary,
    operators.IMAGE_EDIT_OT_scale_layer,
    operators.IMAGE_EDIT_OT_sculpt_image,
    operators.IMAGE_EDIT_OT_export_layers,
    operators.IMAGE_EDIT_OT_export_layers_multilayer,
    operators.IMAGE_EDIT_OT_import_psd,
    operators.IMAGE_EDIT_OT_apply_layer_transform,
    operators.IMAGE_EDIT_OT_align_layers,
    operators.IMAGE_EDIT_OT_distribute_layers,
    operators.IMAGE_EDIT_OT_arrange_grid,
    
    # Layer UI (merged from im_edit)
    ui.IMAGE_EDIT_MT_edit_menu,
    ui.IMAGE_EDIT_UL_layer_list,
    ui.IMAGE_EDIT_MT_layers_menu,
    ui.IMAGE_EDIT_MT_layer_options_menu,
    ui.IMAGE_EDIT_MT_transform_layer_menu,
    ui.IMAGE_EDIT_PT_layers_panel,
    ui.IMAGE_EDIT_PT_layer_transform_panel,
    ui.IMAGE_EDIT_PT_layer_align_panel,
    ui.IMAGE_EDIT_PT_pen_stroke_popover,
    ui.IMAGE_EDIT_PT_pen_options_popover,
    ui.IMAGE_EDIT_PT_clone_brush_popover,
    ui.IMAGE_EDIT_PT_clone_options_popover,
    ui.IMAGE_EDIT_PT_crop_aspect_popover,
    ui.IMAGE_EDIT_PT_crop_canvas_popover,
    ui.IMAGE_EDIT_PT_gradient_settings_popover,
    ui.IMAGE_EDIT_PT_text_input_popover,
    ui.IMAGE_EDIT_PT_text_style_popover,
    ui.IMAGE_EDIT_PT_text_options_popover,
    ui.IMAGE_EDIT_PT_select_mode_popover,
    ui.IMAGE_EDIT_PT_sculpt_brush_popover,
)

_addon_keymaps = []
_layer_draw_handler_images = None
_layer_draw_handler_ui = None

def register():
    global _layer_draw_handler_images, _layer_draw_handler_ui
    
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.text_tool_properties = bpy.props.PointerProperty(type=properties.TextTexProperties)
    
    # Register workspace tools (main addon)
    bpy.utils.register_tool(ui.TextTool, separator=True, group=False)
    bpy.utils.register_tool(ui.GradientTool, separator=False, group=False)
    bpy.utils.register_tool(ui.ImageTextTool, separator=True, group=False)
    bpy.utils.register_tool(ui.ImageGradientTool, separator=False, group=False)
    bpy.utils.register_tool(ui.ImageCropTool, separator=False, group=False)
    bpy.utils.register_tool(ui.ImageCloneTool, separator=False, group=False)
    bpy.utils.register_tool(ui.ImagePenTool, separator=False, group=False)
    
    # Register Selection tools as a group (like in the toolbar)
    bpy.utils.register_tool(ui.IMAGE_EDIT_WT_box_select, after={"builtin.select_box"}, separator=True, group=True)
    bpy.utils.register_tool(ui.IMAGE_EDIT_WT_ellipse_select, after={ui.IMAGE_EDIT_WT_box_select.bl_idname})
    bpy.utils.register_tool(ui.IMAGE_EDIT_WT_lasso_select, after={ui.IMAGE_EDIT_WT_box_select.bl_idname})
    # Sculpt tool - register separately, not in the selection group
    bpy.utils.register_tool(ui.IMAGE_EDIT_WT_sculpt, separator=True, group=False)
    
    if not bpy.app.timers.is_registered(sync.sync_tool_timer):
        bpy.app.timers.register(sync.sync_tool_timer)
    if not bpy.app.timers.is_registered(sync.layer_name_sync_timer):
        bpy.app.timers.register(sync.layer_name_sync_timer)
    
    # Register keymaps for undo/redo
    wm = bpy.context.window_manager
    if wm.keyconfigs.addon:
        # Texture Paint mode keymap
        km = wm.keyconfigs.addon.keymaps.new(name='Texture Paint', space_type='EMPTY')
        kmi = km.keymap_items.new('texttool.undo', 'Z', 'PRESS', ctrl=True)
        _addon_keymaps.append((km, kmi))
        kmi = km.keymap_items.new('texttool.redo', 'Z', 'PRESS', ctrl=True, shift=True)
        _addon_keymaps.append((km, kmi))
        kmi = km.keymap_items.new('texttool.adjust_font_size', 'F', 'PRESS')
        _addon_keymaps.append((km, kmi))
        kmi = km.keymap_items.new('texttool.adjust_rotation', 'F', 'PRESS', ctrl=True)
        _addon_keymaps.append((km, kmi))
        
        # Image Paint mode keymap
        km = wm.keyconfigs.addon.keymaps.new(name='Image Paint', space_type='EMPTY')
        kmi = km.keymap_items.new('texttool.undo', 'Z', 'PRESS', ctrl=True)
        _addon_keymaps.append((km, kmi))
        kmi = km.keymap_items.new('texttool.redo', 'Z', 'PRESS', ctrl=True, shift=True)
        _addon_keymaps.append((km, kmi))
        kmi = km.keymap_items.new('texttool.adjust_font_size', 'F', 'PRESS')
        _addon_keymaps.append((km, kmi))
        kmi = km.keymap_items.new('texttool.adjust_rotation', 'F', 'PRESS', ctrl=True)
        _addon_keymaps.append((km, kmi))
        
        # Layer keymaps (merged from im_edit)
        km = wm.keyconfigs.addon.keymaps.new(name='Image Generic', space_type='IMAGE_EDITOR')
        kmi = km.keymap_items.new(operators.IMAGE_EDIT_OT_make_selection.bl_idname, 'B', 'PRESS')
        _addon_keymaps.append((km, kmi))
        kmi = km.keymap_items.new(operators.IMAGE_EDIT_OT_cancel_selection.bl_idname, 'A', 'PRESS', alt=True)
        _addon_keymaps.append((km, kmi))
        kmi = km.keymap_items.new(operators.IMAGE_EDIT_OT_cut.bl_idname, 'X', 'PRESS', ctrl=True)
        _addon_keymaps.append((km, kmi))
        kmi = km.keymap_items.new(operators.IMAGE_EDIT_OT_copy.bl_idname, 'C', 'PRESS', ctrl=True)
        _addon_keymaps.append((km, kmi))
        kmi = km.keymap_items.new(operators.IMAGE_EDIT_OT_paste.bl_idname, 'V', 'PRESS', ctrl=True)
        _addon_keymaps.append((km, kmi))
        kmi = km.keymap_items.new(operators.IMAGE_EDIT_OT_fill_with_fg_color.bl_idname, 'DEL', 'PRESS', ctrl=True)
        _addon_keymaps.append((km, kmi))
        kmi = km.keymap_items.new(operators.IMAGE_EDIT_OT_clear.bl_idname, 'DEL', 'PRESS')
        _addon_keymaps.append((km, kmi))
        kmi = km.keymap_items.new(operators.IMAGE_EDIT_OT_move_layer.bl_idname, 'G', 'PRESS')
        _addon_keymaps.append((km, kmi))
        kmi = km.keymap_items.new(operators.IMAGE_EDIT_OT_rotate_layer_arbitrary.bl_idname, 'R', 'PRESS')
        _addon_keymaps.append((km, kmi))
        kmi = km.keymap_items.new(operators.IMAGE_EDIT_OT_scale_layer.bl_idname, 'S', 'PRESS')
        _addon_keymaps.append((km, kmi))
        # Selection undo/redo
        kmi = km.keymap_items.new(operators.IMAGE_EDIT_OT_undo_selection.bl_idname, 'Z', 'PRESS', ctrl=True)
        _addon_keymaps.append((km, kmi))
        kmi = km.keymap_items.new(operators.IMAGE_EDIT_OT_redo_selection.bl_idname, 'Z', 'PRESS', ctrl=True, shift=True)
        _addon_keymaps.append((km, kmi))
    
    # Register API version
    api.API.VERSION = (1, 0, 0)
    
    # Add Edit menu to Image Editor menu bar (next to View, Image)
    bpy.types.IMAGE_MT_editor_menus.append(ui.edit_header_draw)
    
    # Add save handler (merged from im_edit)
    bpy.app.handlers.save_pre.append(utils.layer_save_pre_handler)
    
    # Add draw handlers
    # POST_VIEW for images (behind UI)
    _layer_draw_handler_images = bpy.types.SpaceImageEditor.draw_handler_add(
        utils.layer_draw_images, (), 'WINDOW', 'POST_VIEW')
    
    # POST_PIXEL for UI overlays (top of everything)
    _layer_draw_handler_ui = bpy.types.SpaceImageEditor.draw_handler_add(
        utils.layer_draw_ui, (), 'WINDOW', 'POST_PIXEL')
    
    # Register layer properties
    wm_type = bpy.types.WindowManager
    wm_type.image_edit_api = api.API
    wm_type.image_edit_properties = bpy.props.PointerProperty(type=properties.IMAGE_EDIT_WindowPropertyGroup)
    bpy.types.Image.image_edit_properties = bpy.props.PointerProperty(type=properties.IMAGE_EDIT_ImagePropertyGroup)

def unregister():
    global _layer_draw_handler_images, _layer_draw_handler_ui
    
    # Unregister keymaps
    for km, kmi in _addon_keymaps:
        km.keymap_items.remove(kmi)
    _addon_keymaps.clear()
    
    if bpy.app.timers.is_registered(sync.sync_tool_timer):
        try:
            bpy.app.timers.unregister(sync.sync_tool_timer)
        except ValueError:
            pass
    if bpy.app.timers.is_registered(sync.layer_name_sync_timer):
        try:
            bpy.app.timers.unregister(sync.layer_name_sync_timer)
        except ValueError:
            pass

    # Unregister tools
    bpy.utils.unregister_tool(ui.IMAGE_EDIT_WT_sculpt)
    bpy.utils.unregister_tool(ui.IMAGE_EDIT_WT_lasso_select)
    bpy.utils.unregister_tool(ui.IMAGE_EDIT_WT_ellipse_select)
    bpy.utils.unregister_tool(ui.IMAGE_EDIT_WT_box_select)
    bpy.utils.unregister_tool(ui.ImagePenTool)
    bpy.utils.unregister_tool(ui.ImageCloneTool)
    bpy.utils.unregister_tool(ui.ImageCropTool)
    bpy.utils.unregister_tool(ui.ImageGradientTool)
    bpy.utils.unregister_tool(ui.ImageTextTool)
    bpy.utils.unregister_tool(ui.GradientTool)
    bpy.utils.unregister_tool(ui.TextTool)
    del bpy.types.Scene.text_tool_properties
    
    # Cleanup layer scene
    # Cleanup layer scene and resources
    utils.layer_cleanup_scene()
    utils.layer_free_resources()
    
    # Remove Edit menu from menu bar
    bpy.types.IMAGE_MT_editor_menus.remove(ui.edit_header_draw)
    
    # Remove save handler
    bpy.app.handlers.save_pre.remove(utils.layer_save_pre_handler)
    
    # Remove draw handlers
    if _layer_draw_handler_images:
        bpy.types.SpaceImageEditor.draw_handler_remove(_layer_draw_handler_images, 'WINDOW')
        _layer_draw_handler_images = None
        
    if _layer_draw_handler_ui:
        bpy.types.SpaceImageEditor.draw_handler_remove(_layer_draw_handler_ui, 'WINDOW')
        _layer_draw_handler_ui = None
    
    # Remove properties
    wm_type = bpy.types.WindowManager
    del wm_type.image_edit_api
    del wm_type.image_edit_properties
    del bpy.types.Image.image_edit_properties
    
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
        
    utils.TextPreviewCache.get().cleanup()
    utils.ImageUndoStack.get().clear()
