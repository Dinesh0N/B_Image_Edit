# Operators package - re-exports all operators for backward compatibility
# This allows `from . import operators` to work unchanged

from .text_ops import (
    get_text_content,
    TEXTURE_PAINT_OT_text_tool,
    IMAGE_PAINT_OT_text_tool,
    TEXTURE_PAINT_OT_input_text,
)

from .gradient_ops import (
    TEXTURE_PAINT_OT_gradient_tool,
    IMAGE_PAINT_OT_gradient_tool,
)

from .pen_ops import (
    pen_points,
    pen_preview_pos,
    pen_is_closed,
    pen_is_adjusting_handle,
    pen_handle_snap_active,
    pen_displace_mode,
    pen_displace_offset,
    pen_last_applied_points,
    pen_displacing,
    pen_realtime_preview,
    IMAGE_PAINT_OT_pen_tool,
    TEXTURE_PAINT_OT_pen_tool,
)

from .clone_ops import (
    IMAGE_PAINT_OT_clone_adjust_size,
    IMAGE_PAINT_OT_clone_adjust_strength,
    IMAGE_PAINT_OT_clone_set_source,
    IMAGE_PAINT_OT_clone_tool,
)

from .crop_ops import (
    IMAGE_PAINT_OT_crop_tool,
)

from .adjust_ops import (
    TEXTTOOL_OT_adjust_font_size,
    TEXTTOOL_OT_adjust_rotation,
)

from .selection_ops import (
    IMAGE_EDIT_OT_make_selection,
    IMAGE_EDIT_OT_make_ellipse_selection,
    IMAGE_EDIT_OT_make_lasso_selection,
    IMAGE_EDIT_OT_cancel_selection,
    IMAGE_EDIT_OT_undo_selection,
    IMAGE_EDIT_OT_redo_selection,
    IMAGE_EDIT_OT_swap_colors,
    IMAGE_EDIT_OT_fill_with_fg_color,
    IMAGE_EDIT_OT_fill_with_bg_color,
    IMAGE_EDIT_OT_clear,
    IMAGE_EDIT_OT_cut,
    IMAGE_EDIT_OT_copy,
    IMAGE_EDIT_OT_paste,
    IMAGE_EDIT_OT_cut_to_layer,
    IMAGE_EDIT_OT_copy_to_layer,
)

from .layer_ops import (
    IMAGE_EDIT_OT_add_image_layer,
    IMAGE_EDIT_OT_new_image_layer,
    IMAGE_EDIT_OT_crop,
    IMAGE_EDIT_OT_deselect_layer,
    IMAGE_EDIT_OT_move_layer,
    IMAGE_EDIT_OT_delete_layer,
    IMAGE_EDIT_OT_edit_layer,
    IMAGE_EDIT_OT_duplicate_layer,
    IMAGE_EDIT_OT_lock_all_layers,
    IMAGE_EDIT_OT_unlock_all_layers,
    IMAGE_EDIT_OT_hide_all_layers,
    IMAGE_EDIT_OT_show_all_layers,
    IMAGE_EDIT_OT_delete_all_layers,
    IMAGE_EDIT_OT_update_layer_previews,
    IMAGE_EDIT_OT_select_all_layers,
    IMAGE_EDIT_OT_deselect_all_layers,
    IMAGE_EDIT_OT_invert_layer_selection,
    IMAGE_EDIT_OT_delete_selected_layers,
    IMAGE_EDIT_OT_merge_selected_layers,
    IMAGE_EDIT_OT_change_image_layer_order,
    IMAGE_EDIT_OT_merge_layers,
    IMAGE_EDIT_OT_flip_layer,
    IMAGE_EDIT_OT_rotate_layer,
    IMAGE_EDIT_OT_rotate_layer_arbitrary,
    IMAGE_EDIT_OT_scale_layer,
    IMAGE_EDIT_OT_sculpt_image,
    IMAGE_EDIT_OT_export_layers,
    IMAGE_EDIT_OT_export_layers_multilayer,
    IMAGE_EDIT_OT_import_psd,
    IMAGE_EDIT_OT_apply_layer_transform,
    IMAGE_EDIT_OT_align_layers,
    IMAGE_EDIT_OT_distribute_layers,
    IMAGE_EDIT_OT_arrange_grid,
)

from .misc_ops import (
    TEXTURE_PAINT_OT_refresh_fonts,
    TEXTTOOL_OT_undo,
    TEXTTOOL_OT_redo,
    TEXTTOOL_OT_add_texture,
    TEXTTOOL_OT_remove_texture,
    TEXTTOOL_OT_move_texture,
)

__all__ = [
    # Text ops
    'get_text_content',
    'TEXTURE_PAINT_OT_text_tool',
    'IMAGE_PAINT_OT_text_tool',
    'TEXTURE_PAINT_OT_input_text',
    # Gradient ops
    'TEXTURE_PAINT_OT_gradient_tool',
    'IMAGE_PAINT_OT_gradient_tool',
    # Pen ops
    'pen_points',
    'pen_preview_pos',
    'pen_is_closed',
    'pen_is_adjusting_handle',
    'pen_handle_snap_active',
    'pen_displace_mode',
    'pen_displace_offset',
    'pen_last_applied_points',
    'pen_displacing',
    'pen_realtime_preview',
    'IMAGE_PAINT_OT_pen_tool',
    'TEXTURE_PAINT_OT_pen_tool',
    # Clone ops
    'IMAGE_PAINT_OT_clone_adjust_size',
    'IMAGE_PAINT_OT_clone_adjust_strength',
    'IMAGE_PAINT_OT_clone_set_source',
    'IMAGE_PAINT_OT_clone_tool',
    # Crop ops
    'IMAGE_PAINT_OT_crop_tool',
    # Adjust ops
    'TEXTTOOL_OT_adjust_font_size',
    'TEXTTOOL_OT_adjust_rotation',
    # Selection ops
    'IMAGE_EDIT_OT_make_selection',
    'IMAGE_EDIT_OT_make_ellipse_selection',
    'IMAGE_EDIT_OT_make_lasso_selection',
    'IMAGE_EDIT_OT_cancel_selection',
    'IMAGE_EDIT_OT_undo_selection',
    'IMAGE_EDIT_OT_redo_selection',
    'IMAGE_EDIT_OT_swap_colors',
    'IMAGE_EDIT_OT_fill_with_fg_color',
    'IMAGE_EDIT_OT_fill_with_bg_color',
    'IMAGE_EDIT_OT_clear',
    'IMAGE_EDIT_OT_cut',
    'IMAGE_EDIT_OT_copy',
    'IMAGE_EDIT_OT_paste',
    'IMAGE_EDIT_OT_cut_to_layer',
    'IMAGE_EDIT_OT_copy_to_layer',
    # Layer ops
    'IMAGE_EDIT_OT_add_image_layer',
    'IMAGE_EDIT_OT_new_image_layer',
    'IMAGE_EDIT_OT_crop',
    'IMAGE_EDIT_OT_deselect_layer',
    'IMAGE_EDIT_OT_move_layer',
    'IMAGE_EDIT_OT_delete_layer',
    'IMAGE_EDIT_OT_edit_layer',
    'IMAGE_EDIT_OT_duplicate_layer',
    'IMAGE_EDIT_OT_lock_all_layers',
    'IMAGE_EDIT_OT_unlock_all_layers',
    'IMAGE_EDIT_OT_hide_all_layers',
    'IMAGE_EDIT_OT_show_all_layers',
    'IMAGE_EDIT_OT_delete_all_layers',
    'IMAGE_EDIT_OT_update_layer_previews',
    'IMAGE_EDIT_OT_select_all_layers',
    'IMAGE_EDIT_OT_deselect_all_layers',
    'IMAGE_EDIT_OT_invert_layer_selection',
    'IMAGE_EDIT_OT_delete_selected_layers',
    'IMAGE_EDIT_OT_merge_selected_layers',
    'IMAGE_EDIT_OT_change_image_layer_order',
    'IMAGE_EDIT_OT_merge_layers',
    'IMAGE_EDIT_OT_flip_layer',
    'IMAGE_EDIT_OT_rotate_layer',
    'IMAGE_EDIT_OT_rotate_layer_arbitrary',
    'IMAGE_EDIT_OT_scale_layer',
    'IMAGE_EDIT_OT_sculpt_image',
    'IMAGE_EDIT_OT_export_layers',
    'IMAGE_EDIT_OT_export_layers_multilayer',
    'IMAGE_EDIT_OT_import_psd',
    'IMAGE_EDIT_OT_apply_layer_transform',
    'IMAGE_EDIT_OT_align_layers',
    'IMAGE_EDIT_OT_distribute_layers',
    'IMAGE_EDIT_OT_arrange_grid',
    # Misc ops
    'TEXTURE_PAINT_OT_refresh_fonts',
    'TEXTTOOL_OT_undo',
    'TEXTTOOL_OT_redo',
    'TEXTTOOL_OT_add_texture',
    'TEXTTOOL_OT_remove_texture',
    'TEXTTOOL_OT_move_texture',
]
