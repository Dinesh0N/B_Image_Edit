import os
import bpy
from bpy.types import PropertyGroup
from bpy.props import StringProperty, IntProperty, EnumProperty, BoolProperty, FloatVectorProperty, FloatProperty, PointerProperty
from . import utils

def update_gradient_init(self, context):
    """Ensure gradient node tree exists when gradient is enabled."""
    if self.use_gradient:
        # This creates the node tree if missing, which is safe in an operator/update loop
        # but not in a draw loop. By doing it here, we ensure ui.py finds it ready.
        try:
            utils.get_gradient_node()
        except Exception as e:
            print(f"Error initializing gradient node: {e}")

class TextTexProperties(PropertyGroup):
    text: StringProperty(
        name="Text",
        description="Text to paint",
        default="Text"
    )
    use_text_block: BoolProperty(
        name="Use Text Block",
        description="Use a Text Editor data block for multi-line text",
        default=False
    )
    text_block: PointerProperty(
        name="Text Block",
        description="Text data block to use for multi-line text",
        type=bpy.types.Text
    )
    font_file: PointerProperty(
        name="Font",
        description="Font to use",
        type=bpy.types.VectorFont
    )
    font_size: IntProperty(
        name="Font Size",
        description="TTF/OTF font size (pixels)",
        default=64,
        min=8,
        max=512
    )
    color: FloatVectorProperty(
        name="Color",
        description="Text color",
        default=(1.0, 1.0, 1.0, 1.0),
        min=0.0,
        max=1.0,
        size=4,
        subtype='COLOR'
    )

    anchor_horizontal: EnumProperty(
        name="Anchor X",
        description="Horizontal text anchor relative to cursor",
        items=[
            ('LEFT', "Left", "Text starts at cursor (left-aligned)", 'ANCHOR_LEFT', 0),
            ('CENTER', "Center", "Text center at cursor", 'ANCHOR_CENTER', 1),
            ('RIGHT', "Right", "Text ends at cursor (right-aligned)", 'ANCHOR_RIGHT', 2),
        ],
        default='CENTER'
    )
    anchor_vertical: EnumProperty(
        name="Anchor Y",
        description="Vertical text anchor relative to cursor",
        items=[
            ('TOP', "Top", "Text below cursor", 'ANCHOR_TOP', 0),
            ('CENTER', "Center", "Text center at cursor", 'ANCHOR_CENTER', 1),
            ('BOTTOM', "Bottom", "Text above cursor", 'ANCHOR_BOTTOM', 2),
        ],
        default='CENTER'
    )
    text_alignment: EnumProperty(
        name="Alignment",
        description="Text alignment for multi-line text",
        items=[
            ('LEFT', "Left", "Align text to the left", 'ALIGN_LEFT', 0),
            ('CENTER', "Center", "Center align text", 'ALIGN_CENTER', 1),
            ('RIGHT', "Right", "Align text to the right", 'ALIGN_RIGHT', 2),
        ],
        default='CENTER'
    )
    line_spacing: FloatProperty(
        name="Line Spacing",
        description="Spacing between lines (1.0 = normal, 1.5 = 150%)",
        default=1.2,
        min=0.5,
        max=3.0,
        step=10,
        precision=2
    )
    rotation: FloatProperty(
        name="Rotation",
        description="Text rotation in degrees (counter-clockwise)",
        default=0.0,
        min=0.0,
        max=360.0,
        subtype='ANGLE'
    )
    align_to_view: BoolProperty(
        name="Align to View",
        description="Align text rotation to viewport camera (uncheck for UV/3D alignment)",
        default=True
    )
    projection_mode: EnumProperty(
        name="Projection",
        description="How text is projected onto the surface",
        items=[
            ('UV', "UV-Based", "Place text at UV coordinates (fast)"),
            ('VIEW', "3D Projected", "Project text from view like brush strokes (accurate)"),
        ],
        default='VIEW'
    )
    
    # Gradient properties
    use_gradient: BoolProperty(
        name="Use Gradient",
        description="Use gradient instead of solid color",
        default=False,
        update=update_gradient_init
    )
    gradient_type: EnumProperty(
        name="Gradient Type",
        description="Type of gradient to apply",
        items=[
            ('LINEAR', "Linear", "Linear gradient from left to right"),
            ('RADIAL', "Radial", "Radial gradient from center outward"),
        ],
        default='LINEAR'
    )
    gradient_rotation: FloatProperty(
        name="Gradient Rotation",
        description="Rotation of the gradient in degrees",
        default=0.0,
        min=0.0,
        max=360.0
    )
    
    # Outline properties
    use_outline: BoolProperty(
        name="Use Outline",
        description="Add outline/stroke around text",
        default=False
    )
    outline_color: FloatVectorProperty(
        name="Outline Color",
        description="Color of the text outline",
        default=(0.0, 0.0, 0.0, 1.0),
        min=0.0,
        max=1.0,
        size=4,
        subtype='COLOR'
    )
    outline_size: IntProperty(
        name="Outline Size",
        description="Thickness of the text outline in pixels",
        default=2,
        min=1,
        max=20
    )
    
    # Anti-aliasing (for text, gradient, clone tools)
    use_antialiasing: BoolProperty(
        name="Anti-Aliasing",
        description="Smooth the edges of strokes and painted elements",
        default=True
    )
    
    # Crop Tool properties
    crop_show_thirds: BoolProperty(
        name="Show Rule of Thirds",
        description="Display rule of thirds grid overlay on crop selection",
        default=True
    )
    crop_lock_aspect: BoolProperty(
        name="Lock Aspect Ratio",
        description="Constrain crop selection to specified aspect ratio",
        default=False
    )
    crop_aspect_width: FloatProperty(
        name="Width",
        description="Aspect ratio width component",
        default=16.0,
        min=0.1,
        max=100.0
    )
    crop_aspect_height: FloatProperty(
        name="Height",
        description="Aspect ratio height component",
        default=9.0,
        min=0.1,
        max=100.0
    )
    crop_expand_canvas: BoolProperty(
        name="Expand Canvas",
        description="Allow expanding the canvas beyond the original image bounds",
        default=False
    )
    crop_fill_color: FloatVectorProperty(
        name="Fill Color",
        description="Color to fill expanded canvas areas",
        default=(0.0, 0.0, 0.0, 1.0),
        min=0.0,
        max=1.0,
        size=4,
        subtype='COLOR'
    )
    crop_use_resolution: BoolProperty(
        name="Set Resolution",
        description="Scale crop result to a specific resolution",
        default=False
    )
    crop_resolution_x: IntProperty(
        name="Width",
        description="Output width in pixels",
        default=1920,
        min=1,
        max=16384
    )
    crop_resolution_y: IntProperty(
        name="Height",
        description="Output height in pixels",
        default=1080,
        min=1,
        max=16384
    )
    
    # Clone Tool properties
    clone_brush_size: IntProperty(
        name="Brush Size",
        description="Clone brush radius in pixels",
        default=50,
        min=1,
        max=500
    )
    clone_falloff_preset: EnumProperty(
        name="Falloff",
        description="Brush edge falloff curve preset",
        items=[
            ('SMOOTH', "Smooth", "Smooth falloff curve", 'SMOOTHCURVE', 0),
            ('SMOOTHER', "Smoother", "Extra smooth falloff", 'SMOOTHCURVE', 1),
            ('SPHERE', "Sphere", "Spherical falloff", 'SPHERECURVE', 2),
            ('ROOT', "Root", "Root falloff (square root)", 'ROOTCURVE', 3),
            ('SHARP', "Sharp", "Sharp falloff curve", 'SHARPCURVE', 4),
            ('LINEAR', "Linear", "Linear falloff", 'LINCURVE', 5),
            ('CONSTANT', "Constant", "No falloff (hard edge)", 'NOCURVE', 6),
            ('CUSTOM', "Custom", "Use custom curve from active brush", 'RNDCURVE', 7),
        ],
        default='SMOOTH'
    )
    clone_brush_strength: FloatProperty(
        name="Strength",
        description="Opacity/strength of clone painting (1.0 = full, 0.0 = none)",
        default=1.0,
        min=0.0,
        max=1.0
    )
    clone_brush_hardness: FloatProperty(
        name="Hardness",
        description="Brush edge hardness (1.0 = hard edge, 0.0 = soft falloff)",
        default=0.8,
        min=0.0,
        max=1.0
    )
    
    # Pen Tool properties
    pen_use_stroke: BoolProperty(
        name="Stroke",
        description="Enable stroke for the path",
        default=True
    )
    pen_use_fill: BoolProperty(
        name="Fill",
        description="Enable fill for the path",
        default=False
    )
    pen_stroke_color: FloatVectorProperty(
        name="Stroke Color",
        description="Color of the path stroke",
        default=(1.0, 1.0, 1.0, 1.0),
        min=0.0,
        max=1.0,
        size=4,
        subtype='COLOR'
    )
    pen_fill_color: FloatVectorProperty(
        name="Fill Color",
        description="Color of the path fill",
        default=(0.5, 0.5, 0.5, 1.0),
        min=0.0,
        max=1.0,
        size=4,
        subtype='COLOR'
    )
    pen_stroke_width: IntProperty(
        name="Stroke Width",
        description="Width of the stroke in pixels",
        default=3,
        min=1,
        max=50
    )



# ============================================================
# Layer Property Groups 
# ============================================================

# Guard flag to prevent recursive update loops during name sync
_syncing_layer_name = False

def on_layer_label_changed(self, context):
    """When layer label is renamed in the list, rename the image datablock to match."""
    global _syncing_layer_name
    if _syncing_layer_name:
        return
    
    layer = self  # self is the IMAGE_EDIT_LayerPropertyGroup
    new_label = layer.label
    if not new_label:
        return
    
    # Find the image datablock referenced by this layer
    layer_img = bpy.data.images.get(layer.name, None)
    if not layer_img:
        return
    
    # Find the parent (base) image that owns this layer
    base_img = None
    for img in bpy.data.images:
        if not hasattr(img, 'image_edit_properties'):
            continue
        img_props = img.image_edit_properties
        for l in img_props.layers:
            if l == layer:
                base_img = img
                break
        if base_img:
            break
    
    if not base_img:
        return
    
    # Build new image name: (base image name without extension)_(new label)
    base_name_no_ext = os.path.splitext(base_img.name)[0]
    new_img_name = base_name_no_ext + '_' + new_label
    
    # Rename the image datablock
    _syncing_layer_name = True
    try:
        layer_img.name = new_img_name
        # Update layer.name to the actual name (Blender may add .001 suffix)
        layer.name = layer_img.name
    finally:
        _syncing_layer_name = False

def on_layer_placement_changed(self, context):
    area_session = utils.layer_get_area_session(context)
    if area_session.prevent_layer_update_event:
        return
    img = context.area.spaces.active.image
    if not img:
        return
    utils.layer_rebuild_image_layers_nodes(img)

def on_layer_visible_changed(self, context):
    utils.layer_refresh_image(context)

def on_selected_layer_index_changed(self, context):
    if self.selected_layer_index != -1:
        utils.layer_cancel_selection(context)

def on_invert_mask_changed(self, context):
    """Re-apply paint mask when invert toggle changes."""
    area_session = utils.layer_get_area_session(context)
    # Only re-apply if there are active selections
    if area_session.selections or area_session.ellipses or area_session.lassos:
        utils.layer_clear_paint_mask(context)
        utils.layer_apply_selection_as_paint_mask(context)
        context.area.tag_redraw()

class IMAGE_EDIT_WindowPropertyGroup(bpy.types.PropertyGroup):
    foreground_color: bpy.props.FloatVectorProperty(name='Foreground Color', subtype='COLOR_GAMMA', min=0, max=1.0, size=3, default=(1.0, 1.0, 1.0))
    background_color: bpy.props.FloatVectorProperty(name='Background Color', subtype='COLOR_GAMMA', min=0, max=1.0, size=3, default=(0, 0, 0))
    selection_mode: bpy.props.EnumProperty(
        name='Selection Mode',
        description='Mode for box selection tool',
        items=[
            ('SET', 'Normal', 'Replace existing selection', 'SELECT_SET', 0),
            ('ADD', 'Extend', 'Add to existing selection (Shift)', 'SELECT_EXTEND', 1),
            ('SUBTRACT', 'Subtract', 'Remove from selection (Ctrl)', 'SELECT_SUBTRACT', 2),
        ],
        default='SET'
    )
    invert_mask: bpy.props.BoolProperty(
        name='Invert Mask',
        description='Paint outside the selection instead of inside',
        default=False,
        update=on_invert_mask_changed
    )
    # Sculpt tool properties
    sculpt_mode: bpy.props.EnumProperty(
        name='Sculpt Mode',
        description='Mode for image sculpting',
        items=[
            ('GRAB', 'Grab', 'Drag pixels in brush direction', 'ORIENTATION_CURSOR', 0),
            ('PINCH', 'Pinch', 'Compress pixels toward center', 'PINCH', 1),
            ('DRIP', 'Drip', 'Paint dripping effect (gravity)', 'COLORSET_10_VEC', 2),
            ('WAVE', 'Wave', 'Ripple/wave distortion effect', 'MOD_WAVE', 3),
            ('JITTER', 'Jitter', 'Turbulence/noise displacement', 'FORCE_TURBULENCE', 4),
            ('HAZE', 'Haze', 'Heat haze vertical shimmer', 'FORCE_HARMONIC', 5),
            ('ERODE', 'Erode', 'Edge breaking displacement', 'MOD_DECIM', 6),
            ('CREASE', 'Crease', 'Sharp linear deformation', 'LINCURVE', 7),
            ('BRISTLE', 'Bristle', 'Directional streaks and striations', 'BRUSH_DATA', 8),
            ('DRYPULL', 'Dry', 'Broken dry-brush skipped pixels', 'GPBRUSH_PEN', 9),
            ('BLOOM', 'Bloom', 'Soft expanding overlap effect', 'LIGHT_SUN', 10),
            ('INFLATE', 'Inflate', 'Organic uneven bulging', 'META_BALL', 11),
            ('LIQUIFY', 'Liquify', 'Fluid-like warp deformation', 'MOD_FLUID', 12),
            ('SPIRAL', 'Spiral', 'Swirling vortex distortion', 'FORCE_MAGNETIC', 13),
            ('STRETCH', 'Stretch', 'Directional elongation', 'FULLSCREEN_ENTER', 14),
            ('PIXELATE', 'Pixelate', 'Pixelated mosaic effect', 'TEXTURE_DATA', 15),
            ('GLITCH', 'Glitch', 'Digital scan line displacement', 'SEQ_LUMA_WAVEFORM', 16),
        ],
        default='GRAB'
    )
    sculpt_radius: bpy.props.IntProperty(
        name='Radius',
        description='Brush radius in pixels',
        default=50,
        min=5,
        max=500,
        subtype='PIXEL'
    )
    sculpt_strength: bpy.props.FloatProperty(
        name='Strength',
        description='Sculpt effect strength',
        default=0.5,
        min=0.0,
        max=1.0,
        subtype='FACTOR'
    )
    sculpt_falloff_preset: bpy.props.EnumProperty(
        name='Falloff',
        description='Brush edge falloff curve preset',
        items=[
            ('SMOOTH', "Smooth", "Smooth falloff curve", 'SMOOTHCURVE', 0),
            ('SMOOTHER', "Smoother", "Extra smooth falloff", 'SMOOTHCURVE', 1),
            ('SPHERE', "Sphere", "Spherical falloff", 'SPHERECURVE', 2),
            ('ROOT', "Root", "Root falloff (square root)", 'ROOTCURVE', 3),
            ('SHARP', "Sharp", "Sharp falloff curve", 'SHARPCURVE', 4),
            ('LINEAR', "Linear", "Linear falloff", 'LINCURVE', 5),
            ('CONSTANT', "Constant", "No falloff (hard edge)", 'NOCURVE', 6),
            ('CUSTOM', "Custom", "Use brush curve", 'RNDCURVE', 7),
        ],
        default='SMOOTH'
    )

class IMAGE_EDIT_LayerPropertyGroup(bpy.types.PropertyGroup):
    location: bpy.props.FloatVectorProperty(size=2, update=on_layer_placement_changed)
    rotation: bpy.props.FloatProperty(subtype='ANGLE', update=on_layer_placement_changed)
    scale: bpy.props.FloatVectorProperty(size=2, default=(1.0, 1.0), update=on_layer_placement_changed)
    opacity: bpy.props.FloatProperty(name='Opacity', default=1.0, min=0.0, max=1.0, subtype='FACTOR', update=on_layer_visible_changed)
    blend_mode: bpy.props.EnumProperty(
        name='Blend Mode',
        items=[
            ('MIX', 'Mix', 'Normal blend'),
            ('DARKEN', 'Darken', 'Darken blend'),
            ('MULTIPLY', 'Multiply', 'Multiply blend'),
            ('COLOR_BURN', 'Color Burn', 'Color burn blend'),
            ('LIGHTEN', 'Lighten', 'Lighten blend'),
            ('SCREEN', 'Screen', 'Screen blend'),
            ('COLOR_DODGE', 'Color Dodge', 'Color dodge blend'),
            ('ADD', 'Add', 'Add blend'),
            ('OVERLAY', 'Overlay', 'Overlay blend'),
            ('SOFT_LIGHT', 'Soft Light', 'Soft light blend'),
            ('LINEAR_LIGHT', 'Linear Light', 'Linear light blend'),
            ('DIFFERENCE', 'Difference', 'Difference blend'),
            ('EXCLUSION', 'Exclusion', 'Exclusion blend'),
            ('SUBTRACT', 'Subtract', 'Subtract blend'),
            ('DIVIDE', 'Divide', 'Divide blend'),
            ('HUE', 'Hue', 'Hue blend'),
            ('SATURATION', 'Saturation', 'Saturation blend'),
            ('COLOR', 'Color', 'Color blend'),
            ('VALUE', 'Value', 'Value blend'),
        ],
        default='MIX',
        update=on_layer_visible_changed
    )
    label: bpy.props.StringProperty(update=on_layer_label_changed)
    hide: bpy.props.BoolProperty(name='Hide', update=on_layer_visible_changed)
    locked: bpy.props.BoolProperty(name='Lock', default=False, description='Lock layer to prevent editing')
    checked: bpy.props.BoolProperty(name='Select', default=False, description='Select for multi-layer operations')
    custom_data: bpy.props.StringProperty(default='{}')

class IMAGE_EDIT_ImagePropertyGroup(bpy.types.PropertyGroup):
    layers: bpy.props.CollectionProperty(type=IMAGE_EDIT_LayerPropertyGroup)
    selected_layer_index: bpy.props.IntProperty(update=on_selected_layer_index_changed)
    base_image_name: bpy.props.StringProperty(name='Base Image', description='Original base image name when editing a layer')
    editing_layer: bpy.props.BoolProperty(name='Editing Layer', default=False, description='Currently editing a layer')
    
    distribute_use_gap: bpy.props.BoolProperty(
        name="Use Gap",
        description="Distribute with a specific gap value instead of evenly between extremes",
        default=False
    )
    distribute_gap_x: bpy.props.FloatProperty(
        name="Gap X",
        description="Specific horizontal gap value in pixels",
        default=10.0
    )
    distribute_gap_y: bpy.props.FloatProperty(
        name="Gap Y",
        description="Specific vertical gap value in pixels",
        default=10.0
    )
    
    # New properties for Arrange and Align
    align_relative_to: bpy.props.EnumProperty(
        name="Anchor",
        items=[
            ('CANVAS', "Canvas", "Align relative to the canvas"),
            ('BOUNDING_BOX', "Bounding Box", "Align relative to the bounds of the selected layers"),
        ],
        default='BOUNDING_BOX'
    )
    
    distribute_mode: bpy.props.EnumProperty(
        name="Mode",
        items=[
            ('CENTERS', "Centers", "Distribute layers by their centers"),
            ('GAPS', "Even Gaps", "Distribute layers with even gaps between them"),
        ],
        default='CENTERS'
    )
    
    arrange_grid_x: bpy.props.IntProperty(
        name="Grid X",
        description="Number of columns",
        default=3,
        min=1
    )
    arrange_grid_y: bpy.props.IntProperty(
        name="Grid Y",
        description="Number of rows",
        default=3,
        min=1
    )
    arrange_gap_x: bpy.props.FloatProperty(
        name="Gap X",
        description="Horizontal gap between grid cells in pixels",
        default=100.0
    )
    arrange_gap_y: bpy.props.FloatProperty(
        name="Gap Y",
        description="Vertical gap between grid cells in pixels",
        default=100.0
    )
    
    # UI State properties
    ui_align_open: bpy.props.BoolProperty(
        name="Align",
        default=True,
        description="Expand Align panel"
    )
    ui_distribute_open: bpy.props.BoolProperty(
        name="Distribute",
        default=True,
        description="Expand Distribute panel"
    )
    ui_arrange_open: bpy.props.BoolProperty(
        name="Arrange",
        default=True,
        description="Expand Arrange panel"
    )
