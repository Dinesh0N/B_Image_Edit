# Gradient utilities for B_image_edit addon

import bpy


# ----------------------------
# Gradient Node Storage
# ----------------------------
def get_gradient_node(create_if_missing=True):
    """Get or create a hidden node tree with a Color Ramp node.
    
    Args:
        create_if_missing: If False, returns None if node doesn't exist.
                          Set to False when called from draw callbacks.
    """
    tree_name = ".TextTool_Gradient_Storage"
    node_name = "Gradient_Ramp"
    
    # Check if tree exists
    if tree_name not in bpy.data.node_groups:
        if not create_if_missing:
            return None
        # Create new node group (shader type)
        tree = bpy.data.node_groups.new(tree_name, 'ShaderNodeTree')
        tree.use_fake_user = True  # Ensure it persists
    else:
        tree = bpy.data.node_groups[tree_name]
    
    # Check if node exists
    if node_name not in tree.nodes:
        if not create_if_missing:
            return None
        # Find existing color ramp node by type (in case name wasn't set)
        for node in tree.nodes:
            if node.type == 'VALTORGB':
                return node
        # Create new node
        node = tree.nodes.new('ShaderNodeValToRGB')
        node.name = node_name
        node.label = "Gradient"
    else:
        node = tree.nodes[node_name]
        
    return node


# LUT cache to avoid re-evaluating color ramp every frame
_lut_cache = {
    'hash': None,
    'lut': None,
    'samples': 0,
}

def _ramp_hash(ramp):
    """Compute a lightweight hash of a ColorRamp's state."""
    parts = []
    for elem in ramp.elements:
        parts.append(round(elem.position, 6))
        parts.append(round(elem.color[0], 4))
        parts.append(round(elem.color[1], 4))
        parts.append(round(elem.color[2], 4))
        parts.append(round(elem.color[3], 4))
    parts.append(ramp.interpolation)
    return tuple(parts)


def get_gradient_lut(node, samples=256):
    """Evaluate a Color Ramp node into a LUT of RGBA tuples.
    
    Results are cached and only re-evaluated when the ramp changes.
    """
    if not node or not hasattr(node, "color_ramp"):
        return []
    
    ramp = node.color_ramp
    h = _ramp_hash(ramp)
    
    # Return cached LUT if ramp hasn't changed
    if _lut_cache['hash'] == h and _lut_cache['samples'] == samples and _lut_cache['lut'] is not None:
        return _lut_cache['lut']
    
    lut = []
    step = 1.0 / (samples - 1)
    for i in range(samples):
        pos = min(1.0, i * step)
        c = ramp.evaluate(pos)
        lut.append((c[0], c[1], c[2], c[3]))
    
    _lut_cache['hash'] = h
    _lut_cache['lut'] = lut
    _lut_cache['samples'] = samples
    
    return lut
