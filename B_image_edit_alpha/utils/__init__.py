# Utils package for B_image_edit addon
# Re-exports all utilities for backward compatibility

# Font utilities
from .fonts import (
    get_custom_font_dirs,
    load_custom_fonts_to_blender,
    _get_blf_font_id,
    reset_font_cache,
    FontManager,
)

# Gradient utilities
from .gradient import (
    get_gradient_node,
    get_gradient_lut,
)

# Shared state and utilities
from .shared import (
    cursor_pos,
    show_cursor,
    cursor_pixel_scale,
    gradient_preview_start,
    gradient_preview_end,
    crop_preview_start,
    crop_preview_end,
    clone_source_pos,
    clone_cursor_pos,
    clone_source_set,
    blend_pixel,
    blend_pixels_numpy,
    composite_layer_numpy,
    force_texture_refresh,
)

# Undo/Redo stack
from .undo import ImageUndoStack

# Text preview cache
from .text_preview import TextPreviewCache

# Layer system
from .layer import (
    LayerAreaSession,
    LayerSession,
    layer_get_session,
    layer_get_area_session,
    layer_draw_images,
    layer_draw_ui,
    math_utils,
    layer_get_active_layer,
    layer_get_target_image,
    layer_enter_edit_mode,
    layer_exit_edit_mode,
    layer_is_editing,
    layer_convert_selection,
    layer_convert_ellipse_selection,
    layer_convert_lasso_selection,
    layer_crop_selection,
    layer_cancel_selection,
    layer_build_selection_mask,
    layer_get_combined_selection_mask,
    layer_apply_selection_as_paint_mask,
    layer_clear_paint_mask,
    layer_pause_paint_mask,
    layer_resume_paint_mask,
    layer_get_selection,
    layer_get_target_selection,
    layer_refresh_image,
    layer_apply_layer_transform,
    layer_create_layer,
    layer_rebuild_image_layers_nodes,
    layer_cleanup_scene,
    layer_save_pre_handler,
    layer_read_pixels_from_image,
    layer_write_pixels_to_image,
    layer_free_resources,
)

# Try to import colorspace conversion if available
try:
    from .layer import layer_convert_colorspace
except ImportError:
    pass
