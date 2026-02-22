import bpy

# ID of our custom tools
TOOL_ID_3D = "texture_paint.text_tool_ttf"
TOOL_ID_IMAGE = "image_paint.text_tool_ttf"

# Cache last known state to detect changes
# Format: {workspace_name: {'3d': tool_id, 'image': tool_id}}
_last_state = {}

def get_tool_id(context, space_type, mode):
    """Get the active tool ID for a specific space and mode."""
    # We can't easily get the tool for a specific *space* instance directly without iterating areas,
    # but tools are actually stored per workspace + context mode.
    # bpy.context.workspace.tools.from_space_view3d_mode(mode).idname
    ws = context.workspace
    if not ws:
        return None
        
    try:
        if space_type == 'VIEW_3D':
            return ws.tools.from_space_view3d_mode(mode).idname
        elif space_type == 'IMAGE_EDITOR':
            return ws.tools.from_space_image_mode(mode).idname
    except (AttributeError, KeyError):
        return None

def set_tool_id(context, space_type, mode, tool_id):
    """Set the active tool ID using context override to ensure correct area targeting."""
    ws = context.workspace
    if not ws:
        return
        
    found_area = None
    target_window = None
    target_screen = None
    
    # search for the first area matching the space_type in the active window (or any window)
    for window in context.window_manager.windows:
        screen = window.screen
        for area in screen.areas:
            if area.type == space_type:
                found_area = area
                target_window = window
                target_screen = screen
                break
        if found_area:
            break
            
    if found_area:
        try:
            # Context override is crucial for tool_set_by_id to know which editor instance/mode to affect
            with context.temp_override(window=target_window, screen=target_screen, area=found_area):
                bpy.ops.wm.tool_set_by_id(name=tool_id, space_type=space_type)
        except Exception as e:
            print(f"[TextTex] Sync Error: {e}")
            pass
    else:
        # Fallback if no area found (mostly for completeness, though likely won't work well)
        try:
            bpy.ops.wm.tool_set_by_id(name=tool_id, space_type=space_type)
        except Exception:
            pass

def sync_tool_timer():
    """Timer callback to sync tools."""
    context = bpy.context
    ws = context.workspace
    if not ws:
        return 0.1

    ws_name = ws.name
    
    # Initialize state for this workspace if needed
    if ws_name not in _last_state:
        _last_state[ws_name] = {
            '3d': get_tool_id(context, 'VIEW_3D', 'PAINT_TEXTURE'),
            'image': get_tool_id(context, 'IMAGE_EDITOR', 'PAINT')
        }
        return 0.1

    # Get current state
    current_3d = get_tool_id(context, 'VIEW_3D', 'PAINT_TEXTURE')
    current_image = get_tool_id(context, 'IMAGE_EDITOR', 'PAINT')
    
    last_3d = _last_state[ws_name]['3d']
    last_image = _last_state[ws_name]['image']
    
    # Logic: If 3D changed TO our tool, sync Image.
    if current_3d != last_3d:
        # 3D tool changed
        if current_3d == TOOL_ID_3D:
            # User selected our tool in 3D View
            if current_image != TOOL_ID_IMAGE:
                # Sync Image Editor
                set_tool_id(context, 'IMAGE_EDITOR', 'PAINT', TOOL_ID_IMAGE)
                current_image = TOOL_ID_IMAGE # Update current because we just changed it

    # Logic: If Image changed TO our tool, sync 3D.
    if current_image != last_image:
        # Image tool changed
        if current_image == TOOL_ID_IMAGE:
            # User selected our tool in Image Editor
            if current_3d != TOOL_ID_3D:
                # Sync 3D View
                set_tool_id(context, 'VIEW_3D', 'PAINT_TEXTURE', TOOL_ID_3D)
                current_3d = TOOL_ID_3D # Update current because we just changed it

    # Update last state
    _last_state[ws_name]['3d'] = current_3d
    _last_state[ws_name]['image'] = current_image
    
    return 0.1

# ============================================================
# Layer ↔ Image Name Sync
# ============================================================

# Track image names by their pointer (stable identity across renames)
# Format: {image_pointer: last_known_name}
_image_name_tracker = {}

def layer_name_sync_timer():
    """Timer that detects image datablock renames and syncs layer names.
    
    When a user renames an image in the datablock dropdown, this timer
    detects the change and updates the layer's .name and .label accordingly.
    """
    import os
    from . import properties as _props
    
    global _image_name_tracker
    
    try:
        # Build a set of all image names currently used by layers
        layer_img_names = set()
        for img in bpy.data.images:
            if not hasattr(img, 'image_edit_properties'):
                continue
            for layer in img.image_edit_properties.layers:
                layer_img_names.add(layer.name)
        
        # Check all tracked images for renames
        updated = {}
        for img in bpy.data.images:
            ptr = img.as_pointer()
            current_name = img.name
            
            if ptr in _image_name_tracker:
                old_name = _image_name_tracker[ptr]
                if old_name != current_name and old_name in layer_img_names:
                    # This image was renamed! Find the layer(s) referencing old_name
                    if not _props._syncing_layer_name:
                        _props._syncing_layer_name = True
                        try:
                            for base_img in bpy.data.images:
                                if not hasattr(base_img, 'image_edit_properties'):
                                    continue
                                for layer in base_img.image_edit_properties.layers:
                                    if layer.name == old_name:
                                        layer.name = current_name
                                        # Extract new label from the renamed image name
                                        base_prefix = os.path.splitext(base_img.name)[0] + '_'
                                        if current_name.startswith(base_prefix):
                                            layer.label = current_name[len(base_prefix):]
                                        else:
                                            layer.label = current_name
                        finally:
                            _props._syncing_layer_name = False
            
            updated[ptr] = current_name
        
        _image_name_tracker = updated
    except Exception as e:
        print(f"[B_image_edit] Layer name sync error: {e}")
    
    return 0.5
