# Layer system for B_image_edit addon

import bpy
import blf
import gpu
import copy
import numpy as np
from gpu_extras.batch import batch_for_shader
from . import math_utils
from ..ui_renderer import UIRenderer

# Blend mode string to integer mapping (module-level constant)
BLEND_MODE_MAP = {
    'MIX': 0, 'DARKEN': 1, 'MULTIPLY': 2, 'COLOR_BURN': 3,
    'LIGHTEN': 4, 'SCREEN': 5, 'COLOR_DODGE': 6, 'ADD': 7,
    'OVERLAY': 8, 'SOFT_LIGHT': 9, 'LINEAR_LIGHT': 10,
    'DIFFERENCE': 11, 'EXCLUSION': 12, 'SUBTRACT': 13, 'DIVIDE': 14,
    'HUE': 15, 'SATURATION': 16, 'COLOR': 17, 'VALUE': 18
}



class LayerAreaSession:
    """Session state for a single Image Editor area."""
    def __init__(self):
        self._selections = []  # List of [[x1, y1], [x2, y2]] rectangles
        self._ellipses = []    # List of [[x1, y1], [x2, y2]] ellipses (bounding boxes)
        self._lassos = []      # List of [[x1,y1], [x2,y2], ...] polygon point lists
        # Negation shapes (for subtract mode - affects ALL selection types)
        self._neg_rects = []     # Rectangles to subtract
        self._neg_ellipses = []  # Ellipses to subtract
        self._neg_lassos = []    # Lassos to subtract
        self.selection_region = None
        self.ellipse_region = None  # Current drag region for ellipse tool
        self.lasso_points = None    # Current lasso points being drawn
        self.selecting = False
        self.selection_mode = 'SET'  # 'SET', 'ADD', 'SUBTRACT'
        self.selection_mask = None  # Numpy boolean mask for paint restriction
        self.layer_moving = False
        self.layer_rotating = False
        self.layer_scaling = False
        self.prevent_layer_update_event = False
        self.prev_image = None
        self.prev_image_width = 0
        self.prev_image_height = 0
        # Selection undo/redo history
        self._selection_history = []  # Stack of previous selection states
        self._selection_redo = []     # Stack of redo states
        self._max_history = 50        # Max undo history size
    
    def _get_selection_state(self):
        """Get current selection state as a dictionary."""
        return {
            'selections': copy.deepcopy(self._selections),
            'ellipses': copy.deepcopy(self._ellipses),
            'lassos': copy.deepcopy(self._lassos),
            'neg_rects': copy.deepcopy(self._neg_rects),
            'neg_ellipses': copy.deepcopy(self._neg_ellipses),
            'neg_lassos': copy.deepcopy(self._neg_lassos),
        }
    
    def _set_selection_state(self, state):
        """Restore selection state from a dictionary."""
        self._selections = copy.deepcopy(state.get('selections', []))
        self._ellipses = copy.deepcopy(state.get('ellipses', []))
        self._lassos = copy.deepcopy(state.get('lassos', []))
        self._neg_rects = copy.deepcopy(state.get('neg_rects', []))
        self._neg_ellipses = copy.deepcopy(state.get('neg_ellipses', []))
        self._neg_lassos = copy.deepcopy(state.get('neg_lassos', []))
        self.selection_mask = None
    
    def push_undo(self):
        """Push current selection state to undo history."""
        state = self._get_selection_state()
        self._selection_history.append(state)
        # Limit history size
        if len(self._selection_history) > self._max_history:
            self._selection_history.pop(0)
        # Clear redo stack when new action is performed
        self._selection_redo.clear()
    
    def undo_selection(self):
        """Undo last selection change. Returns True if successful."""
        if not self._selection_history:
            return False
        # Save current state to redo stack
        current = self._get_selection_state()
        self._selection_redo.append(current)
        # Restore previous state
        prev_state = self._selection_history.pop()
        self._set_selection_state(prev_state)
        return True
    
    def redo_selection(self):
        """Redo last undone selection change. Returns True if successful."""
        if not self._selection_redo:
            return False
        # Save current state to undo stack
        current = self._get_selection_state()
        self._selection_history.append(current)
        # Restore redo state
        redo_state = self._selection_redo.pop()
        self._set_selection_state(redo_state)
        return True
    
    @property
    def selection(self):
        """Get bounding box of all selections (backward compatibility)."""
        if not self._selections:
            return None
        # Compute bounding box of all selection rectangles
        min_x = min(s[0][0] for s in self._selections)
        min_y = min(s[0][1] for s in self._selections)
        max_x = max(s[1][0] for s in self._selections)
        max_y = max(s[1][1] for s in self._selections)
        return [[min_x, min_y], [max_x, max_y]]
    
    @selection.setter
    def selection(self, value):
        """Set selection (backward compatibility - replaces all selections)."""
        if value is None:
            self._selections = []
            self.selection_mask = None
        else:
            self._selections = [value]
    
    @property
    def selections(self):
        """Get list of all selection rectangles."""
        return self._selections
    
    def add_selection(self, rect):
        """Add a rectangle to selections (extend mode)."""
        self._selections.append(rect)
        self.selection_mask = None  # Invalidate mask
    
    def subtract_selection(self, rect):
        """Subtract a rectangle from selections.
        
        This computes the geometric difference between existing selections
        and the subtracted rectangle, potentially splitting rectangles.
        """
        if not self._selections:
            return
        
        sub_x1, sub_y1 = rect[0]
        sub_x2, sub_y2 = rect[1]
        
        new_selections = []
        for sel in self._selections:
            sel_x1, sel_y1 = sel[0]
            sel_x2, sel_y2 = sel[1]
            
            # Check if rectangles overlap
            if sub_x1 >= sel_x2 or sub_x2 <= sel_x1 or sub_y1 >= sel_y2 or sub_y2 <= sel_y1:
                # No overlap, keep original
                new_selections.append(sel)
            else:
                # Overlap - split into up to 4 rectangles
                # Top portion (above subtracted area)
                if sub_y1 > sel_y1:
                    new_selections.append([[sel_x1, sel_y1], [sel_x2, sub_y1]])
                # Bottom portion (below subtracted area)
                if sub_y2 < sel_y2:
                    new_selections.append([[sel_x1, sub_y2], [sel_x2, sel_y2]])
                # Left portion (within vertical overlap)
                top = max(sel_y1, sub_y1)
                bottom = min(sel_y2, sub_y2)
                if sub_x1 > sel_x1 and bottom > top:
                    new_selections.append([[sel_x1, top], [sub_x1, bottom]])
                # Right portion (within vertical overlap)
                if sub_x2 < sel_x2 and bottom > top:
                    new_selections.append([[sub_x2, top], [sel_x2, bottom]])
        
        self._selections = new_selections
        self.selection_mask = None
    
    @property
    def ellipses(self):
        """Get list of all ellipse selections (bounding boxes)."""
        return self._ellipses

    def add_ellipse(self, bbox):
        """Add an ellipse to selections (extend mode)."""
        self._ellipses.append(bbox)
        self.selection_mask = None

    @property
    def lassos(self):
        """Get list of all lasso selections (point lists)."""
        return self._lassos

    def add_lasso(self, points):
        """Add a lasso polygon to selections (extend mode)."""
        self._lassos.append(points)
        self.selection_mask = None

    def clear_selections(self):
        """Clear all selections."""
        self._selections = []
        self._ellipses = []
        self._lassos = []
        self._neg_rects = []
        self._neg_ellipses = []
        self._neg_lassos = []
        self.selection_mask = None

class LayerSession:
    """Global layer session state."""
    def __init__(self):
        self.icons = None
        self.keymaps = []
        self.ui_renderer = None
        self.copied_image_pixels = None
        self.copied_image_settings = None
        self.copied_layer_settings = None
        self.areas = {}

# Global layer session
_layer_session = LayerSession()

def layer_get_session():
    """Get the global layer session."""
    global _layer_session
    return _layer_session

def layer_get_area_session(context):
    """Get or create the area session for an Image Editor area."""
    global _layer_session
    area_session = _layer_session.areas.get(context.area, None)
    if not area_session:
        area_session = LayerAreaSession()
        _layer_session.areas[context.area] = area_session
    return area_session

def layer_draw_images():
    """Draw handler for layer images (POST_VIEW)."""
    global _layer_session
    context = bpy.context
    
    # Only draw if we have an active image
    try:
        if not context.area or not context.area.spaces.active:
            return
        img = context.area.spaces.active.image
    except AttributeError:
        # Context might not be fully ready
        return

    if not img:
        return

    width, height = img.size[0], img.size[1]
    
    # Initialize renderer if needed
    if not _layer_session.ui_renderer:
        _layer_session.ui_renderer = UIRenderer()
        
    img_props = img.image_edit_properties
    try:
        layers = img_props.layers
    except AttributeError:
        return
        
    # Draw layers
    # Note: Using POST_VIEW means we draw in Image Space coordinates
    # (0,0 is bottom-left of the image).
    # layer.location stored as (x, y_from_top) needs conversion.
    
    gpu.matrix.push()
    try:
        # Scale Pixel Space -> Normalized Space
        # POST_VIEW in Image Editor uses 0..1 coordinates for the image bounds
        gpu.matrix.scale((1.0 / width, 1.0 / height))
        
        for layer in reversed(layers):
            if layer.hide:
                continue
                
            layer_img = bpy.data.images.get(layer.name, None)
            if layer_img:
                layer_width, layer_height = layer_img.size[0], layer_img.size[1]
                layer_pos = layer.location
                
                # Convert from Canvas-Center coordinates to Bottom-Left for GL drawing
                x = layer_pos[0] + width / 2.0 - layer_width / 2.0
                y = layer_pos[1] + height / 2.0 - layer_height / 2.0
                
                blend_mode_int = BLEND_MODE_MAP.get(layer.blend_mode, 0)
                
                _layer_session.ui_renderer.render_image(
                    layer_img, 
                    (x, y), 
                    (layer_width, layer_height), 
                    layer.rotation, 
                    layer.scale, 
                    layer.opacity, 
                    blend_mode_int
                )
    finally:
        gpu.matrix.pop()

def layer_draw_ui():
    """Draw handler for layer UI overlays (POST_PIXEL)."""
    global _layer_session
    context = bpy.context
    area_session = layer_get_area_session(context)
    info_text = None
    width, height = 0, 0
    
    try:
        if not context.area or not context.area.spaces.active:
            return
        img = context.area.spaces.active.image
    except AttributeError:
        return
        
    if img:
        width, height = img.size[0], img.size[1]

    if img and (area_session.selections or area_session.selection_region or area_session.ellipses or area_session.ellipse_region or area_session.lassos or area_session.lasso_points):
        if not _layer_session.ui_renderer:
            _layer_session.ui_renderer = UIRenderer()
        
        # Draw current drag rectangle if selecting
        if area_session.selection_region and area_session.selecting:
            region_pos1, region_pos2 = area_session.selection_region
            region_size = [region_pos2[0] - region_pos1[0], region_pos2[1] - region_pos1[1]]
            _layer_session.ui_renderer.render_selection_frame(region_pos1, region_size)
        
        # Draw current drag ellipse if selecting
        if area_session.ellipse_region and area_session.selecting:
            region_pos1, region_pos2 = area_session.ellipse_region
            # Normalize for width/height
            x = min(region_pos1[0], region_pos2[0])
            y = min(region_pos1[1], region_pos2[1])
            w = abs(region_pos2[0] - region_pos1[0])
            h = abs(region_pos2[1] - region_pos1[1])
            _layer_session.ui_renderer.render_ellipse_selection((x, y), (w, h))
        
        # Draw current lasso points if selecting
        if area_session.lasso_points and area_session.selecting:
            _layer_session.ui_renderer.render_lasso_preview(area_session.lasso_points)
        
        # Draw ALL committed selections as unified merged contour (Krita-style)
        screen_rects = []
        screen_ellipses = []
        screen_lassos = []
        
        if area_session.selections:
            for selection in area_session.selections:
                view_x1 = selection[0][0] / width
                view_y1 = selection[0][1] / height
                view_x2 = selection[1][0] / width
                view_y2 = selection[1][1] / height
                region_pos1 = context.region.view2d.view_to_region(view_x1, view_y1, clip=False)
                region_pos2 = context.region.view2d.view_to_region(view_x2, view_y2, clip=False)
                screen_rects.append((region_pos1[0], region_pos1[1], region_pos2[0], region_pos2[1]))

        if area_session.ellipses:
            for selection in area_session.ellipses:
                view_x1 = selection[0][0] / width
                view_y1 = selection[0][1] / height
                view_x2 = selection[1][0] / width
                view_y2 = selection[1][1] / height
                region_pos1 = context.region.view2d.view_to_region(view_x1, view_y1, clip=False)
                region_pos2 = context.region.view2d.view_to_region(view_x2, view_y2, clip=False)
                
                x1 = min(region_pos1[0], region_pos2[0])
                y1 = min(region_pos1[1], region_pos2[1])
                x2 = max(region_pos1[0], region_pos2[0])
                y2 = max(region_pos1[1], region_pos2[1])
                screen_ellipses.append((x1, y1, x2, y2))
        
        if area_session.lassos:
            for lasso in area_session.lassos:
                screen_pts = []
                for pt in lasso:
                    vx = pt[0] / width
                    vy = pt[1] / height
                    sx, sy = context.region.view2d.view_to_region(vx, vy, clip=False)
                    screen_pts.append((sx, sy))
                if len(screen_pts) >= 3:
                    screen_lassos.append(screen_pts)
        
        # Collect negation shapes for subtraction
        neg_screen_ellipses = []
        neg_screen_lassos = []
        
        if area_session._neg_ellipses:
            for sel in area_session._neg_ellipses:
                view_x1 = sel[0][0] / width
                view_y1 = sel[0][1] / height
                view_x2 = sel[1][0] / width
                view_y2 = sel[1][1] / height
                region_pos1 = context.region.view2d.view_to_region(view_x1, view_y1, clip=False)
                region_pos2 = context.region.view2d.view_to_region(view_x2, view_y2, clip=False)
                
                x1 = min(region_pos1[0], region_pos2[0])
                y1 = min(region_pos1[1], region_pos2[1])
                x2 = max(region_pos1[0], region_pos2[0])
                y2 = max(region_pos1[1], region_pos2[1])
                neg_screen_ellipses.append((x1, y1, x2, y2))
        
        neg_screen_rects = []
        if area_session._neg_rects:
            for sel in area_session._neg_rects:
                view_x1 = sel[0][0] / width
                view_y1 = sel[0][1] / height
                view_x2 = sel[1][0] / width
                view_y2 = sel[1][1] / height
                region_pos1 = context.region.view2d.view_to_region(view_x1, view_y1, clip=False)
                region_pos2 = context.region.view2d.view_to_region(view_x2, view_y2, clip=False)
                neg_screen_rects.append((region_pos1[0], region_pos1[1], region_pos2[0], region_pos2[1]))
        
        if area_session._neg_lassos:
            for lasso in area_session._neg_lassos:
                screen_pts = []
                for pt in lasso:
                    vx = pt[0] / width
                    vy = pt[1] / height
                    sx, sy = context.region.view2d.view_to_region(vx, vy, clip=False)
                    screen_pts.append((sx, sy))
                if len(screen_pts) >= 3:
                    neg_screen_lassos.append(screen_pts)
        
        if screen_rects or screen_ellipses or screen_lassos or neg_screen_rects or neg_screen_ellipses or neg_screen_lassos:
            _layer_session.ui_renderer.render_merged_all(screen_rects, screen_ellipses, screen_lassos, neg_screen_rects, neg_screen_ellipses, neg_screen_lassos)

    if img:
        if not _layer_session.ui_renderer:
            _layer_session.ui_renderer = UIRenderer()
        img_props = img.image_edit_properties
        selected_layer_index = img_props.selected_layer_index
        layers = img_props.layers

        # Draw selection frame for active layer
        if 0 <= selected_layer_index < len(layers):
            layer = layers[selected_layer_index]
            layer_img = bpy.data.images.get(layer.name, None)
            
            if layer_img:
                layer_width, layer_height = layer_img.size[0], layer_img.size[1]
                layer_pos = layer.location
                
                # Calculate bottom-left (pos1) and top-right (pos2) using canvas-center coordinates
                layer_center_x = layer_pos[0] + width / 2.0
                layer_center_y = layer_pos[1] + height / 2.0
                
                layer_pos1 = [layer_center_x - layer_width / 2.0, layer_center_y - layer_height / 2.0]
                layer_pos2 = [layer_center_x + layer_width / 2.0, layer_center_y + layer_height / 2.0]
                
                layer_view_x1 = layer_pos1[0] / width
                layer_view_y1 = layer_pos1[1] / height
                
                layer_view_x2 = layer_pos2[0] / width
                layer_view_y2 = layer_pos2[1] / height
                
                layer_region_pos1 = context.region.view2d.view_to_region(layer_view_x1, layer_view_y1, clip=False)
                layer_region_pos2 = context.region.view2d.view_to_region(layer_view_x2, layer_view_y2, clip=False)
                
                layer_region_size = [layer_region_pos2[0] - layer_region_pos1[0], layer_region_pos2[1] - layer_region_pos1[1]]
                
                _layer_session.ui_renderer.render_selection_frame(layer_region_pos1, layer_region_size, layer.rotation, layer.scale)

    if img:
        if area_session.selection or area_session.selection_region:
            if area_session.prev_image:
                if img != area_session.prev_image:
                    layer_cancel_selection(context)
                elif width != area_session.prev_image_width or height != area_session.prev_image_height:
                    layer_crop_selection(context)

    area_session.prev_image = img
    area_session.prev_image_width = width
    area_session.prev_image_height = height

    if area_session.layer_moving or area_session.layer_rotating or area_session.layer_scaling:
        info_text = "LMB: Perform   RMB: Cancel"

    area_width = context.area.width
    if info_text:
        ui_scale = context.preferences.system.ui_scale
        _layer_session.ui_renderer.render_info_box((0, 0), (area_width, 20 * ui_scale))
        blf.position(0, 8 * ui_scale, 6 * ui_scale, 0)
        blf.size(0, 11 * ui_scale) if bpy.app.version >= (3, 6) else blf.size(0, 11 * ui_scale, 72)
        blf.color(0, 0.7, 0.7, 0.7, 1.0)
        blf.draw(0, info_text)

def layer_get_active_layer(context):
    """Get the currently active layer."""
    img = context.area.spaces.active.image
    if not img:
        return None
    img_props = img.image_edit_properties
    layers = img_props.layers
    selected_layer_index = img_props.selected_layer_index
    if selected_layer_index == -1 or selected_layer_index >= len(layers):
        return None
    return layers[selected_layer_index]

def layer_get_target_image(context):
    """Get the target image (layer image or base image)."""
    layer = layer_get_active_layer(context)
    if layer:
        return bpy.data.images.get(layer.name, None)
    else:
        return context.area.spaces.active.image

def layer_enter_edit_mode(context):
    """Enter layer edit mode - swap to layer image for painting."""
    img = context.area.spaces.active.image
    if not img:
        return False
    
    img_props = img.image_edit_properties
    layers = img_props.layers
    selected_layer_index = img_props.selected_layer_index
    
    if selected_layer_index == -1 or selected_layer_index >= len(layers):
        return False
    
    layer = layers[selected_layer_index]
    layer_img = bpy.data.images.get(layer.name, None)
    if not layer_img:
        return False
    
    # Store base image info on the layer image so we can get back
    layer_img_props = layer_img.image_edit_properties
    layer_img_props.base_image_name = img.name
    layer_img_props.editing_layer = True
    
    # Swap to layer image
    context.area.spaces.active.image = layer_img
    return True

def layer_exit_edit_mode(context):
    """Exit layer edit mode - swap back to base image."""
    img = context.area.spaces.active.image
    if not img:
        return False
    
    img_props = img.image_edit_properties
    
    if not img_props.editing_layer:
        return False
    
    base_img_name = img_props.base_image_name
    base_img = bpy.data.images.get(base_img_name, None)
    if not base_img:
        return False
    
    # Clear editing state
    img_props.editing_layer = False
    img_props.base_image_name = ''
    
    # Update the layer image
    img.update()
    if img.preview:
        img.preview.reload()
    
    # Invalidate GPU texture cache so the draw handler picks up painted pixels
    if _layer_session.ui_renderer:
        _layer_session.ui_renderer.update_texture(img)
    
    # Swap back to base image
    context.area.spaces.active.image = base_img
    
    # Trigger layer refresh
    layer_rebuild_image_layers_nodes(base_img)
    
    return True

def layer_is_editing(context):
    """Check if currently in layer edit mode."""
    img = context.area.spaces.active.image
    if not img:
        return False
    return img.image_edit_properties.editing_layer

def layer_convert_selection(context, mode='SET'):
    """Convert screen-space selection to image-space.
    
    Args:
        context: Blender context
        mode: 'SET' (replace), 'ADD' (extend), or 'SUBTRACT'
    """
    area_session = layer_get_area_session(context)
    img = context.area.spaces.active.image
    if not img:
        return
    width, height = img.size[0], img.size[1]
    selection_region = area_session.selection_region
    if not selection_region:
        return
    
    # Convert region coordinates to image coordinates
    x1, y1 = context.region.view2d.region_to_view(*selection_region[0])
    x2, y2 = context.region.view2d.region_to_view(*selection_region[1])
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = round(x1 * width)
    y1 = round(y1 * height)
    x2 = round(x2 * width)
    y2 = round(y2 * height)
    
    # Don't clamp here - allow out-of-bounds selections
    # Clamping happens in paint mask and other operations that need it
    
    # Ensure minimum size
    if x2 - x1 <= 0:
        x2 = x1 + 1
    if y2 - y1 <= 0:
        y2 = y1 + 1
    
    new_rect = [[x1, y1], [x2, y2]]
    
    # Push current state to undo history before modifying
    area_session.push_undo()
    
    if mode == 'SET':
        # Replace all selections with new one (clear both boxes, ellipses, lassos, and negations)
        area_session._selections = [new_rect]
        area_session._ellipses = []
        area_session._lassos = []
        area_session._neg_rects = []
        area_session._neg_ellipses = []
        area_session._neg_lassos = []
    elif mode == 'ADD':
        # Add to existing selections
        area_session.add_selection(new_rect)
    elif mode == 'SUBTRACT':
        # Add to negation list (subtracts from ALL selection types)
        area_session._neg_rects.append(new_rect)
        area_session.selection_mask = None
    
    area_session.selection_mode = mode

def layer_convert_ellipse_selection(context, mode='SET'):
    """Convert screen-space ellipse selection to image-space."""
    area_session = layer_get_area_session(context)
    img = context.area.spaces.active.image
    if not img:
        return
    width, height = img.size[0], img.size[1]
    ellipse_region = area_session.ellipse_region
    if not ellipse_region:
        return
    
    # Convert region coordinates to image coordinates
    x1, y1 = context.region.view2d.region_to_view(*ellipse_region[0])
    x2, y2 = context.region.view2d.region_to_view(*ellipse_region[1])
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = round(x1 * width)
    y1 = round(y1 * height)
    x2 = round(x2 * width)
    y2 = round(y2 * height)
    
    # Ensure minimum size
    if x2 - x1 <= 0:
        x2 = x1 + 1
    if y2 - y1 <= 0:
        y2 = y1 + 1
    
    new_bbox = [[x1, y1], [x2, y2]]
    
    # Push current state to undo history before modifying
    area_session.push_undo()
    
    if mode == 'SET':
        area_session._selections = []
        area_session._ellipses = [new_bbox]
        area_session._lassos = []
        area_session._neg_rects = []
        area_session._neg_ellipses = []
        area_session._neg_lassos = []
    elif mode == 'ADD':
        area_session.add_ellipse(new_bbox)
    elif mode == 'SUBTRACT':
        # Add to negation list for subtract
        area_session._neg_ellipses.append(new_bbox)
        area_session.selection_mask = None

    area_session.selection_mode = mode

def layer_convert_lasso_selection(context, mode='SET'):
    """Convert screen-space lasso selection to image-space."""
    area_session = layer_get_area_session(context)
    img = context.area.spaces.active.image
    if not img:
        return
    width, height = img.size[0], img.size[1]
    lasso_points = area_session.lasso_points
    if not lasso_points or len(lasso_points) < 3:
        return
    
    # Convert region coordinates to image coordinates
    image_points = []
    for point in lasso_points:
        vx, vy = context.region.view2d.region_to_view(point[0], point[1])
        ix = round(vx * width)
        iy = round(vy * height)
        image_points.append([ix, iy])
    
    # Push current state to undo history before modifying
    area_session.push_undo()
    
    if mode == 'SET':
        area_session._selections = []
        area_session._ellipses = []
        area_session._lassos = [image_points]
        area_session._neg_rects = []
        area_session._neg_ellipses = []
        area_session._neg_lassos = []
    elif mode == 'ADD':
        area_session.add_lasso(image_points)
    elif mode == 'SUBTRACT':
        # Add to negation list for subtract
        area_session._neg_lassos.append(image_points)
        area_session.selection_mask = None

    area_session.selection_mode = mode

def layer_crop_selection(context):
    """Clamp all selections to image bounds."""
    area_session = layer_get_area_session(context)
    img = context.area.spaces.active.image
    if not img:
        return
    width, height = img.size[0], img.size[1]
    if not area_session._selections:
        return
    
    # Clamp all selection rectangles
    cropped = []
    for sel in area_session._selections:
        [x1, y1], [x2, y2] = sel
        x1 = max(min(x1, width), 0)
        y1 = max(min(y1, height), 0)
        x2 = max(min(x2, width), 0)
        y2 = max(min(y2, height), 0)
        if x2 - x1 <= 0:
            if x2 < width:
                x2 = x2 + 1
            else:
                x1 = x1 - 1
        if y2 - y1 <= 0:
            if y2 < height:
                y2 = y2 + 1
            else:
                y1 = y1 - 1
        cropped.append([[x1, y1], [x2, y2]])
    area_session._selections = cropped

def layer_cancel_selection(context):
    """Cancel the current selection."""
    area_session = layer_get_area_session(context)
    # Push to undo history so clear can be undone
    area_session.push_undo()
    area_session.clear_selections()
    area_session.selection_region = None
    area_session.ellipse_region = None
    area_session.lasso_points = None
    area_session.selecting = False
    layer_clear_paint_mask(context)

# Paint mask data storage
_layer_paint_mask_data = {
    'enabled': False,
    'image_name': None,
    'selections': [],  # List of valid selection rectangles
    'full_cached': None,  # Cached original image
    'img_size': None,
    'timer': None
}

def layer_build_selection_mask(width, height, selections, subtract_rect=None):
    """Build numpy boolean mask from selection rectangles.
    
    Args:
        width, height: Image dimensions
        selections: List of [[x1, y1], [x2, y2]] rectangles
        subtract_rect: Optional rectangle to subtract from mask
    
    Returns:
        Numpy boolean array (height, width) - True = inside selection
    """
    # Start with all False (nothing selected)
    mask = np.zeros((height, width), dtype=bool)
    
    # Add all selection rectangles
    for sel in selections:
        [[x1, y1], [x2, y2]] = sel
        # Clamp to image bounds
        x1 = max(0, min(x1, width))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height))
        y2 = max(0, min(y2, height))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = True
    
    # Subtract rectangle if provided
    if subtract_rect:
        [[x1, y1], [x2, y2]] = subtract_rect
        x1 = max(0, min(x1, width))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height))
        y2 = max(0, min(y2, height))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = False
    
    return mask

def _compute_gap_rectangles(width, height, selections):
    """Precompute rectangles that need to be restored (gaps between selections).
    
    Returns list of (y1, y2, x1, x2) tuples for efficient slicing.
    """
    if not selections:
        return [(0, height, 0, width)]  # Entire image
    
    # Clamp selections to image bounds
    clamped = []
    for sel in selections:
        x1 = max(0, min(sel[0][0], width))
        y1 = max(0, min(sel[0][1], height))
        x2 = max(0, min(sel[1][0], width))
        y2 = max(0, min(sel[1][1], height))
        if x2 > x1 and y2 > y1:
            clamped.append([[x1, y1], [x2, y2]])
    
    if not clamped:
        return [(0, height, 0, width)]
    
    selections = clamped
    
    # Compute bounding box
    min_x = min(s[0][0] for s in selections)
    min_y = min(s[0][1] for s in selections)
    max_x = max(s[1][0] for s in selections)
    max_y = max(s[1][1] for s in selections)
    
    gap_rects = []
    
    # Add outer edges (outside bounding box)
    if min_y > 0:
        gap_rects.append((0, min_y, 0, width))  # Top
    if max_y < height:
        gap_rects.append((max_y, height, 0, width))  # Bottom
    if min_x > 0:
        gap_rects.append((min_y, max_y, 0, min_x))  # Left
    if max_x < width:
        gap_rects.append((min_y, max_y, max_x, width))  # Right
    
    # Find ALL interior gaps between selection pairs (works for any count)
    for i, sel1 in enumerate(selections):
        for sel2 in selections[i+1:]:
            # Horizontal gap: selections side by side
            y_overlap_start = max(sel1[0][1], sel2[0][1])
            y_overlap_end = min(sel1[1][1], sel2[1][1])
            if y_overlap_start < y_overlap_end:
                if sel1[1][0] < sel2[0][0]:
                    gap_rects.append((y_overlap_start, y_overlap_end, sel1[1][0], sel2[0][0]))
                elif sel2[1][0] < sel1[0][0]:
                    gap_rects.append((y_overlap_start, y_overlap_end, sel2[1][0], sel1[0][0]))
            
            # Vertical gap: selections stacked
            x_overlap_start = max(sel1[0][0], sel2[0][0])
            x_overlap_end = min(sel1[1][0], sel2[1][0])
            if x_overlap_start < x_overlap_end:
                if sel1[1][1] < sel2[0][1]:
                    gap_rects.append((sel1[1][1], sel2[0][1], x_overlap_start, x_overlap_end))
                elif sel2[1][1] < sel1[0][1]:
                    gap_rects.append((sel2[1][1], sel1[0][1], x_overlap_start, x_overlap_end))
    
    return gap_rects

def layer_apply_selection_as_paint_mask(context, force_image_update=False):
    """Create paint mask from multi-selection regions."""
    global _layer_paint_mask_data
    
    area_session = layer_get_area_session(context)
    selections = area_session.selections
    ellipses = area_session.ellipses
    lassos = area_session.lassos
    neg_rects = area_session._neg_rects
    neg_ellipses = area_session._neg_ellipses
    neg_lassos = area_session._neg_lassos
    
    if not selections and not ellipses and not lassos:
        return
    
    img = context.area.spaces.active.image
    if not img:
        return
    
    width, height = img.size
    
    # OPTIMIZATION: Check if we can reuse existing cache
    reuse_cache = False
    
    # If image is clean (not dirty), we can trust the cache if it exists, even if force_update is requested
    # because the pixels definitely haven't changed since the last clean state or load.
    can_skip_force = not img.is_dirty and force_image_update
    
    if not force_image_update or can_skip_force:
        if _layer_paint_mask_data.get('image_name') == img.name:
            cached = _layer_paint_mask_data.get('full_cached')
            if cached is not None:
                # Check if size matches
                if cached.shape[0] == height and cached.shape[1] == width:
                    reuse_cache = True
    
    if reuse_cache:
        pixels = _layer_paint_mask_data['full_cached']
    else:
        # Cache the full image (this is the guaranteed correct approach)
        pixels = layer_read_pixels_from_image(img)
    
    # Clamp and validate selections
    valid_selections = []
    if selections:
        for sel in selections:
            x1 = max(0, min(sel[0][0], width))
            y1 = max(0, min(sel[0][1], height))
            x2 = max(0, min(sel[1][0], width))
            y2 = max(0, min(sel[1][1], height))
            if x2 > x1 and y2 > y1:
                valid_selections.append([[x1, y1], [x2, y2]])
    
    valid_ellipses = []
    if ellipses:
        for sel in ellipses:
            # Store original bbox for ellipse equation
            orig_x1, orig_y1 = sel[0][0], sel[0][1]
            orig_x2, orig_y2 = sel[1][0], sel[1][1]
            
            # Clamp only the pixel access region
            x1 = max(0, min(orig_x1, width))
            y1 = max(0, min(orig_y1, height))
            x2 = max(0, min(orig_x2, width))
            y2 = max(0, min(orig_y2, height))
            
            if x2 > x1 and y2 > y1:
                # Store: clamped region for pixel access, original bbox for ellipse math
                valid_ellipses.append({
                    'region': [[x1, y1], [x2, y2]],
                    'original': [[orig_x1, orig_y1], [orig_x2, orig_y2]]
                })
    
    # Validate lassos (just store points, clamping happens during paste)
    valid_lassos = []
    if lassos:
        for lasso in lassos:
            if len(lasso) >= 3:
                valid_lassos.append(lasso)
                
    if not valid_selections and not valid_ellipses and not valid_lassos:
        return
    
    # Validate negation rectangles
    valid_neg_rects = []
    if neg_rects:
        for sel in neg_rects:
            x1 = max(0, min(sel[0][0], width))
            y1 = max(0, min(sel[0][1], height))
            x2 = max(0, min(sel[1][0], width))
            y2 = max(0, min(sel[1][1], height))
            if x2 > x1 and y2 > y1:
                valid_neg_rects.append([[x1, y1], [x2, y2]])
    
    # Validate negation ellipses
    valid_neg_ellipses = []
    if neg_ellipses:
        for sel in neg_ellipses:
            orig_x1, orig_y1 = sel[0][0], sel[0][1]
            orig_x2, orig_y2 = sel[1][0], sel[1][1]
            x1 = max(0, min(orig_x1, width))
            y1 = max(0, min(orig_y1, height))
            x2 = max(0, min(orig_x2, width))
            y2 = max(0, min(orig_y2, height))
            if x2 > x1 and y2 > y1:
                valid_neg_ellipses.append({
                    'region': [[x1, y1], [x2, y2]],
                    'original': [[orig_x1, orig_y1], [orig_x2, orig_y2]]
                })
    
    # Validate negation lassos
    valid_neg_lassos = []
    if neg_lassos:
        for lasso in neg_lassos:
            if len(lasso) >= 3:
                valid_neg_lassos.append(lasso)
    
    # Get invert mask setting
    wm = context.window_manager
    invert_mask = False
    if hasattr(wm, 'image_edit_properties'):
        invert_mask = wm.image_edit_properties.invert_mask
    
    _layer_paint_mask_data['enabled'] = True
    _layer_paint_mask_data['image_name'] = img.name
    _layer_paint_mask_data['selections'] = valid_selections
    _layer_paint_mask_data['ellipses'] = valid_ellipses
    _layer_paint_mask_data['lassos'] = valid_lassos
    _layer_paint_mask_data['neg_rects'] = valid_neg_rects
    _layer_paint_mask_data['neg_ellipses'] = valid_neg_ellipses
    _layer_paint_mask_data['neg_lassos'] = valid_neg_lassos
    _layer_paint_mask_data['invert_mask'] = invert_mask
    _layer_paint_mask_data['full_cached'] = pixels.copy()
    _layer_paint_mask_data['img_size'] = (width, height)
    
    # Register timer
    if _layer_paint_mask_data['timer'] is None:
        _layer_paint_mask_data['timer'] = bpy.app.timers.register(
            _layer_paint_mask_timer, 
            first_interval=0.8,
            persistent=True
        )

def _layer_paint_mask_timer():
    """Paint mask timer - guaranteed correct algorithm.
    
    Starts with cached image, pastes back only selection regions.
    This ensures paint NEVER escapes outside selections.
    """
    global _layer_paint_mask_data
    
    # Calculate adaptive interval based on image size
    width, height = _layer_paint_mask_data.get('img_size', (1024, 1024))
    pixel_count = width * height
    if pixel_count > 4_000_000:  # > 2K (2048x2048)
        interval = 1.5
    elif pixel_count > 1_000_000:  # > 1K (1024x1024)
        interval = 1.0
    else:
        interval = 0.5
    
    if not _layer_paint_mask_data['enabled']:
        _layer_paint_mask_data['timer'] = None
        return None
    
    img_name = _layer_paint_mask_data['image_name']
    if not img_name:
        return interval
    
    img = bpy.data.images.get(img_name)
    if not img:
        _layer_paint_mask_data['enabled'] = False
        _layer_paint_mask_data['timer'] = None
        return None
    
    # Skip if paused
    if _layer_paint_mask_data.get('paused', False):
        return 0.1  # Fast polling while paused to resume quickly
    
    # Skip if image hasn't changed
    if not img.is_dirty:
        return interval
    
    cached = _layer_paint_mask_data.get('full_cached')
    selections = _layer_paint_mask_data.get('selections', [])
    ellipses = _layer_paint_mask_data.get('ellipses', [])
    lassos = _layer_paint_mask_data.get('lassos', [])
    neg_rects = _layer_paint_mask_data.get('neg_rects', [])
    neg_ellipses = _layer_paint_mask_data.get('neg_ellipses', [])
    neg_lassos = _layer_paint_mask_data.get('neg_lassos', [])
    invert_mask = _layer_paint_mask_data.get('invert_mask', False)
    
    if cached is None or (not selections and not ellipses and not lassos):
        return interval
    
    try:
        # Read current painted image
        if img.size[0] != cached.shape[1] or img.size[1] != cached.shape[0]:
             # Image resized, abort mask
             _layer_paint_mask_data['enabled'] = False
             return None

        current = layer_read_pixels_from_image(img)
        
        if not np.array_equal(current.shape, cached.shape):
             _layer_paint_mask_data['enabled'] = False
             return None

        width, height = _layer_paint_mask_data['img_size']
        
        if invert_mask:
            # INVERTED: Start with current (keeps all paint outside selections)
            result = current.copy()
            
            # Restore cached INSIDE selection regions (removes paint inside)
            for sel in selections:
                x1, y1 = sel[0]
                x2, y2 = sel[1]
                result[y1:y2, x1:x2] = cached[y1:y2, x1:x2]
        else:
            # NORMAL: Start with cached (removes ALL paint)
            result = cached.copy()
            
            # Paste back current ONLY inside selection regions (keeps paint inside)
            for sel in selections:
                x1, y1 = sel[0]
                x2, y2 = sel[1]
                result[y1:y2, x1:x2] = current[y1:y2, x1:x2]
        
        # Paste back ellipses
        for ell in ellipses:
            # Get clamped region for pixel access
            region = ell['region']
            x1, y1 = region[0]
            x2, y2 = region[1]
            
            # Get original bbox for ellipse math
            orig = ell['original']
            orig_x1, orig_y1 = orig[0]
            orig_x2, orig_y2 = orig[1]
            
            # Original ellipse dimensions
            orig_w = orig_x2 - orig_x1
            orig_h = orig_y2 - orig_y1
            
            if orig_w <= 0 or orig_h <= 0:
                continue
            
            # Ellipse center and radii from original bbox
            center_x = orig_x1 + orig_w / 2
            center_y = orig_y1 + orig_h / 2
            radius_x = orig_w / 2
            radius_y = orig_h / 2
            
            # Region dimensions (what we're accessing)
            region_h = y2 - y1
            region_w = x2 - x1
            
            if region_w <= 0 or region_h <= 0:
                continue
            
            # Create mask for the clamped region using ORIGINAL ellipse equation
            cy, cx = np.ogrid[0:region_h, 0:region_w]
            px = x1 + cx
            py = y1 + cy
            
            mask = ((px - center_x) / radius_x) ** 2 + ((py - center_y) / radius_y) ** 2 <= 1
            
            patch_result = result[y1:y2, x1:x2]
            if invert_mask:
                # INVERTED: Restore cached inside ellipse (prevents paint inside)
                patch_cached = cached[y1:y2, x1:x2]
                patch_result[mask] = patch_cached[mask]
            else:
                # NORMAL: Paste current inside ellipse (keeps paint inside)
                patch_current = current[y1:y2, x1:x2]
                patch_result[mask] = patch_current[mask]
            result[y1:y2, x1:x2] = patch_result
        
        # Paste back lassos using vectorized polygon fill
        for lasso in lassos:
            if len(lasso) < 3:
                continue
            
            # Compute bounding box of lasso (clamped to image)
            lx1 = max(0, min(int(min(p[0] for p in lasso)), width))
            ly1 = max(0, min(int(min(p[1] for p in lasso)), height))
            lx2 = max(0, min(int(max(p[0] for p in lasso)) + 1, width))
            ly2 = max(0, min(int(max(p[1] for p in lasso)) + 1, height))
            
            if lx2 <= lx1 or ly2 <= ly1:
                continue
            
            # Use fast vectorized polygon mask
            poly_h = ly2 - ly1
            poly_w = lx2 - lx1
            poly_mask = math_utils.fast_polygon_mask(lasso, lx1, ly1, poly_w, poly_h)
            
            # Apply polygon mask
            patch_result = result[ly1:ly2, lx1:lx2]
            if invert_mask:
                # INVERTED: Restore cached inside lasso (prevents paint inside)
                patch_cached = cached[ly1:ly2, lx1:lx2]
                patch_result[poly_mask] = patch_cached[poly_mask]
            else:
                # NORMAL: Paste current inside lasso (keeps paint inside)
                patch_current = current[ly1:ly2, lx1:lx2]
                patch_result[poly_mask] = patch_current[poly_mask]
            result[ly1:ly2, lx1:lx2] = patch_result
        
        # Apply negation rectangles - RESTORE cached pixels (prevent paint)
        for sel in neg_rects:
            x1, y1 = sel[0]
            x2, y2 = sel[1]
            result[y1:y2, x1:x2] = cached[y1:y2, x1:x2]
        
        # Apply negation ellipses - RESTORE cached pixels (prevent paint)
        for ell in neg_ellipses:
            region = ell['region']
            x1, y1 = region[0]
            x2, y2 = region[1]
            
            orig = ell['original']
            orig_x1, orig_y1 = orig[0]
            orig_x2, orig_y2 = orig[1]
            
            orig_w = orig_x2 - orig_x1
            orig_h = orig_y2 - orig_y1
            
            if orig_w <= 0 or orig_h <= 0:
                continue
            
            center_x = orig_x1 + orig_w / 2
            center_y = orig_y1 + orig_h / 2
            radius_x = orig_w / 2
            radius_y = orig_h / 2
            
            region_h = y2 - y1
            region_w = x2 - x1
            
            if region_w <= 0 or region_h <= 0:
                continue
            
            cy, cx = np.ogrid[0:region_h, 0:region_w]
            px = x1 + cx
            py = y1 + cy
            
            mask = ((px - center_x) / radius_x) ** 2 + ((py - center_y) / radius_y) ** 2 <= 1
            
            # Restore cached (subtract - no paint allowed here)
            patch_cached = cached[y1:y2, x1:x2]
            patch_result = result[y1:y2, x1:x2]
            patch_result[mask] = patch_cached[mask]
            result[y1:y2, x1:x2] = patch_result
        
        # Apply negation lassos - RESTORE cached pixels (prevent paint)
        for lasso in neg_lassos:
            if len(lasso) < 3:
                continue
            
            lx1 = max(0, min(int(min(p[0] for p in lasso)), width))
            ly1 = max(0, min(int(min(p[1] for p in lasso)), height))
            lx2 = max(0, min(int(max(p[0] for p in lasso)) + 1, width))
            ly2 = max(0, min(int(max(p[1] for p in lasso)) + 1, height))
            
            if lx2 <= lx1 or ly2 <= ly1:
                continue
            
            # Use fast vectorized polygon mask
            poly_h = ly2 - ly1
            poly_w = lx2 - lx1
            poly_mask = math_utils.fast_polygon_mask(lasso, lx1, ly1, poly_w, poly_h)
            
            # Restore cached (subtract - no paint allowed here)
            patch_cached = cached[ly1:ly2, lx1:lx2]
            patch_result = result[ly1:ly2, lx1:lx2]
            patch_result[poly_mask] = patch_cached[poly_mask]
            result[ly1:ly2, lx1:lx2] = patch_result

        # Only write if actually different
        if not np.array_equal(current, result):
            layer_write_pixels_to_image(img, result)
    except Exception as e:
        print(f"Paint mask error: {e}")
        pass
    
    return interval

def layer_clear_paint_mask(context):
    """Disable paint mask and clean up."""
    global _layer_paint_mask_data
    
    _layer_paint_mask_data['enabled'] = False
    _layer_paint_mask_data['image_name'] = None
    _layer_paint_mask_data['selections'] = []
    _layer_paint_mask_data['ellipses'] = []
    _layer_paint_mask_data['lassos'] = []
    _layer_paint_mask_data['full_cached'] = None
    _layer_paint_mask_data['img_size'] = None

def layer_pause_paint_mask(context):
    """Pause the paint mask timer."""
    global _layer_paint_mask_data
    _layer_paint_mask_data['paused'] = True

def layer_resume_paint_mask(context):
    """Resume the paint mask timer."""
    global _layer_paint_mask_data
    _layer_paint_mask_data['paused'] = False

def layer_get_selection(context):
    """Get current selection."""
    area_session = layer_get_area_session(context)
    return area_session.selection

def layer_get_target_selection(context):
    """Get selection if no layer is selected."""
    area_session = layer_get_area_session(context)
    selection = area_session.selection
    if not selection:
        return None
    img = context.area.spaces.active.image
    if not img:
        return selection
    img_props = img.image_edit_properties
    layers = img_props.layers
    selected_layer_index = img_props.selected_layer_index
    if selected_layer_index == -1 or selected_layer_index >= len(layers):
        return selection
    return None

def layer_refresh_image(context):
    """Refresh the image in the editor."""
    wm = context.window_manager
    img = context.area.spaces.active.image
    if not img:
        return
    context.area.spaces.active.image = img
    img.update()
    if not hasattr(wm, 'imagelayersnode_api') or wm.imagelayersnode_api.VERSION < (1, 1, 0):
        return
    wm.imagelayersnode_api.update_pasted_layer_nodes(img)

def layer_apply_layer_transform(img, rot, scale):
    """Apply rotation and scale to a layer image."""
    global _layer_session
    if not _layer_session.ui_renderer:
        _layer_session.ui_renderer = UIRenderer()
    buff, width, height = _layer_session.ui_renderer.render_image_offscreen(img, rot, scale)
    pixels = np.array([[pixel for pixel in row] for row in buff], np.float32) / 255.0
    layer_convert_colorspace(pixels, 'Linear', 'Linear' if img.is_float else img.colorspace_settings.name)
    return pixels, width, height

def layer_create_layer(base_img, pixels, img_settings, layer_settings, custom_label=None):
    """Create a new layer from pixels."""
    import os
    base_width, base_height = base_img.size
    target_width, target_height = pixels.shape[1], pixels.shape[0]
    # Strip file extension from base image name for cleaner layer naming
    base_name_no_ext = os.path.splitext(base_img.name)[0]
    # Determine the layer label first
    layer_label = custom_label if custom_label else 'Layer'
    # Name format: (bg image name)_(layer label)
    layer_img_name = base_name_no_ext + '_' + layer_label
    layer_img = bpy.data.images.new(layer_img_name, width=target_width, height=target_height, alpha=True, float_buffer=base_img.is_float)
    layer_img.colorspace_settings.name = base_img.colorspace_settings.name
    pixels = pixels.copy()
    layer_convert_colorspace(pixels, 'Linear' if img_settings['is_float'] else img_settings['colorspace_name'], 'Linear' if base_img.is_float else base_img.colorspace_settings.name)
    layer_write_pixels_to_image(layer_img, pixels)
    layer_img.use_fake_user = True
    layer_img.pack()
    img_props = base_img.image_edit_properties
    layers = img_props.layers
    layer = layers.add()
    layer.name = layer_img.name
    # Center layer (0, 0 is exact center of canvas)
    layer.location = [0.0, 0.0]
    
    # Set layer label
    if custom_label:
        layer.label = custom_label
    else:
        layer.label = 'Layer'
    
    if layer_settings:
        layer.rotation = layer_settings['rotation']
        layer.scale = layer_settings['scale']
        layer.custom_data = layer_settings['custom_data']
    layers.move(len(layers) - 1, 0)
    img_props.selected_layer_index = 0
    layer_rebuild_image_layers_nodes(base_img)

def layer_rebuild_image_layers_nodes(img):
    """Rebuild layer nodes for the image."""
    wm = bpy.context.window_manager
    if not hasattr(wm, 'imagelayersnode_api') or wm.imagelayersnode_api.VERSION < (1, 1, 0):
        return
    wm.imagelayersnode_api.rebuild_image_layers_nodes(img)

def layer_cleanup_scene():
    """Cleanup layer node groups."""
    node_group = bpy.data.node_groups.get('image_edit')
    if node_group:
        bpy.data.node_groups.remove(node_group)

def layer_free_resources():
    """Free global resources."""
    if _layer_session.ui_renderer:
        _layer_session.ui_renderer.free()

@bpy.app.handlers.persistent
def layer_save_pre_handler(args):
    """Handler called before saving to pack dirty images."""
    layer_cleanup_scene()
    for img in bpy.data.images:
        if img.source != 'VIEWER':
            if img.is_dirty:
                if img.packed_files or not img.filepath:
                    img.pack()
                else:
                    img.save()

# Layer-specific pixel read/write functions
def layer_read_pixels_from_image(img):
    """Read pixels from image as numpy array."""
    width, height = img.size[0], img.size[1]
    pixels = np.empty(len(img.pixels), dtype=np.float32)
    img.pixels.foreach_get(pixels)
    return np.reshape(pixels, (height, width, 4))

def layer_write_pixels_to_image(img, pixels):
    """Write pixels to image from numpy array."""
    img.pixels.foreach_set(np.reshape(pixels, -1))
    if img.preview:
        img.preview.reload()
    
    # Update texture cache
    if _layer_session.ui_renderer:
        _layer_session.ui_renderer.update_texture(img)

def layer_convert_colorspace(pixels, src_colorspace, dest_colorspace):
    """Convert pixels between color spaces."""
    if src_colorspace == dest_colorspace:
        return
    if src_colorspace == 'Linear' and dest_colorspace == 'sRGB':
        pixels[:, :, 0:3] = pixels[:, :, :3] ** (1.0 / 2.2)
    elif src_colorspace == 'sRGB' and dest_colorspace == 'Linear':
        pixels[:, :, 0:3] = pixels[:, :, :3] ** 2.2

def layer_get_combined_selection_mask(context, width, height, apply_negation=True):
    """Get a boolean mask representing the current selection state.
    
    Args:
        context: Blender context
        width, height: Image dimensions
        apply_negation: Whether to apply subtraction shapes (true by default)
        
    Returns:
        Numpy boolean array (height, width) where True means selected.
        Returns None if no selection exists.
    """
    area_session = layer_get_area_session(context)
    selections = area_session.selections
    ellipses = area_session.ellipses
    lassos = area_session.lassos
    
    # Check if we have any positive selections
    if not selections and not ellipses and not lassos:
        return None
        
    # Start with empty mask
    mask = np.zeros((height, width), dtype=bool)
    
    # 1. Add Rectangles
    for sel in selections:
        x1 = max(0, min(sel[0][0], width))
        y1 = max(0, min(sel[0][1], height))
        x2 = max(0, min(sel[1][0], width))
        y2 = max(0, min(sel[1][1], height))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = True
            
    # 2. Add Ellipses
    for sel in ellipses:
        # Original bbox for math
        orig_x1, orig_y1 = sel[0]
        orig_x2, orig_y2 = sel[1]
        
        # Clamped bbox for pixel access
        x1 = max(0, min(orig_x1, width))
        y1 = max(0, min(orig_y1, height))
        x2 = max(0, min(orig_x2, width))
        y2 = max(0, min(orig_y2, height))
        
        if x2 <= x1 or y2 <= y1:
            continue
            
        # Ellipse parameters
        orig_w = orig_x2 - orig_x1
        orig_h = orig_y2 - orig_y1
        if orig_w <= 0 or orig_h <= 0:
            continue
            
        center_x = orig_x1 + orig_w / 2
        center_y = orig_y1 + orig_h / 2
        radius_x = orig_w / 2
        radius_y = orig_h / 2
        
        # Create grid for the clamped region
        region_h = y2 - y1
        region_w = x2 - x1
        cy, cx = np.ogrid[0:region_h, 0:region_w]
        px = x1 + cx
        py = y1 + cy
        
        # Ellipse equation: (x-h)^2/a^2 + (y-k)^2/b^2 <= 1
        ellipse_mask = ((px - center_x) / radius_x) ** 2 + ((py - center_y) / radius_y) ** 2 <= 1
        
        # Combine with existing mask (OR operation)
        mask[y1:y2, x1:x2] |= ellipse_mask

    # 3. Add Lassos
    for lasso in lassos:
        if len(lasso) < 3:
            continue
            
        lx1 = max(0, min(int(min(p[0] for p in lasso)), width))
        ly1 = max(0, min(int(min(p[1] for p in lasso)), height))
        lx2 = max(0, min(int(max(p[0] for p in lasso)) + 1, width))
        ly2 = max(0, min(int(max(p[1] for p in lasso)) + 1, height))
        
        if lx2 <= lx1 or ly2 <= ly1:
            continue
            
        poly_h = ly2 - ly1
        poly_w = lx2 - lx1
        poly_mask = math_utils.fast_polygon_mask(lasso, lx1, ly1, poly_w, poly_h)
        
        # Combine with existing mask (OR operation)
        mask[ly1:ly2, lx1:lx2] |= poly_mask

    # 4. Subtract Negations (if requested)
    if apply_negation:
        neg_rects = area_session._neg_rects
        neg_ellipses = area_session._neg_ellipses
        neg_lassos = area_session._neg_lassos
        
        # Subtract Rectangles
        for sel in neg_rects:
            x1 = max(0, min(sel[0][0], width))
            y1 = max(0, min(sel[0][1], height))
            x2 = max(0, min(sel[1][0], width))
            y2 = max(0, min(sel[1][1], height))
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = False
                
        # Subtract Ellipses
        for sel in neg_ellipses:
            orig_x1, orig_y1 = sel[0]
            orig_x2, orig_y2 = sel[1]
            x1 = max(0, min(orig_x1, width))
            y1 = max(0, min(orig_y1, height))
            x2 = max(0, min(orig_x2, width))
            y2 = max(0, min(orig_y2, height))
            
            if x2 <= x1 or y2 <= y1:
                continue
                
            orig_w = orig_x2 - orig_x1
            orig_h = orig_y2 - orig_y1
            if orig_w <= 0 or orig_h <= 0:
                continue
                
            center_x = orig_x1 + orig_w / 2
            center_y = orig_y1 + orig_h / 2
            radius_x = orig_w / 2
            radius_y = orig_h / 2
            
            region_h = y2 - y1
            region_w = x2 - x1
            cy, cx = np.ogrid[0:region_h, 0:region_w]
            px = x1 + cx
            py = y1 + cy
            
            ellipse_mask = ((px - center_x) / radius_x) ** 2 + ((py - center_y) / radius_y) ** 2 <= 1
            
            # Subtract from mask (AND NOT operation)
            mask[y1:y2, x1:x2] &= ~ellipse_mask

        # Subtract Lassos
        for lasso in neg_lassos:
            if len(lasso) < 3:
                continue
                
            lx1 = max(0, min(int(min(p[0] for p in lasso)), width))
            ly1 = max(0, min(int(min(p[1] for p in lasso)), height))
            lx2 = max(0, min(int(max(p[0] for p in lasso)) + 1, width))
            ly2 = max(0, min(int(max(p[1] for p in lasso)) + 1, height))
            
            if lx2 <= lx1 or ly2 <= ly1:
                continue
                
            poly_h = ly2 - ly1
            poly_w = lx2 - lx1
            poly_mask = math_utils.fast_polygon_mask(lasso, lx1, ly1, poly_w, poly_h)
            
            mask[ly1:ly2, lx1:lx2] &= ~poly_mask

    # 5. Handle Invert Mask
    wm = context.window_manager
    if hasattr(wm, 'image_edit_properties') and wm.image_edit_properties.invert_mask:
        mask = ~mask

    return mask

