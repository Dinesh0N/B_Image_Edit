
import sys
import math
import gpu
from gpu.shader import create_from_info
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix, Vector
import numpy as np
from .utils import math_utils





def _fast_connected_components(mask):
    """Two-pass connected component labeling using numpy.
    
    Much faster than Python flood fill for large masks.
    Returns (labels, num_labels).
    """
    labels = np.zeros_like(mask, dtype=np.int32)
    next_label = 1
    equivalences = {}
    
    h, w = mask.shape
    
    # First pass: initial labeling with equivalence tracking
    for y in range(h):
        for x in range(w):
            if not mask[y, x]:
                continue
            
            neighbors = []
            if y > 0 and labels[y-1, x] > 0:
                neighbors.append(labels[y-1, x])
            if x > 0 and labels[y, x-1] > 0:
                neighbors.append(labels[y, x-1])
            
            if not neighbors:
                labels[y, x] = next_label
                next_label += 1
            else:
                min_label = min(neighbors)
                labels[y, x] = min_label
                # Record equivalences
                for n in neighbors:
                    if n != min_label:
                        # Find root of n
                        root = n
                        while root in equivalences:
                            root = equivalences[root]
                        # Find root of min_label
                        min_root = min_label
                        while min_root in equivalences:
                            min_root = equivalences[min_root]
                        if root != min_root:
                            equivalences[max(root, min_root)] = min(root, min_root)
    
    # Build final mapping
    final_map = {}
    for lbl in range(1, next_label):
        root = lbl
        while root in equivalences:
            root = equivalences[root]
        final_map[lbl] = root
    
    # Second pass: resolve equivalences using vectorized operations
    for old_lbl, new_lbl in final_map.items():
        if old_lbl != new_lbl:
            labels[labels == old_lbl] = new_lbl
    
    # Renumber labels to be contiguous
    unique_labels = np.unique(labels[labels > 0])
    for i, lbl in enumerate(unique_labels, 1):
        if lbl != i:
            labels[labels == lbl] = i
    
    return labels, len(unique_labels)


def _trace_boundary_contour(boundary, bx, by, scale, offset_x, offset_y):
    """Trace boundary contour in O(n) using 8-connectivity.
    
    Returns ordered contour points as numpy array.
    """
    if len(bx) == 0:
        return np.array([], dtype=np.float32)
    
    if len(bx) == 1:
        return np.array([[bx[0] * scale + offset_x, by[0] * scale + offset_y]], dtype=np.float32)
    
    h, w = boundary.shape
    
    # Create set of boundary pixels for fast lookup
    boundary_set = set(zip(by, bx))
    
    # Start from first boundary pixel
    start = (by[0], bx[0])
    
    # 8-connectivity directions (clockwise from up)
    dirs = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
    
    contour = [(start[1] * scale + offset_x, start[0] * scale + offset_y)]
    visited = {start}
    current = start
    prev_dir = 0
    
    max_iterations = len(bx) * 2  # Safety limit
    iterations = 0
    
    while iterations < max_iterations:
        iterations += 1
        found = False
        
        # Search neighbors starting from direction opposite to where we came from
        for i in range(8):
            d = (prev_dir + 5 + i) % 8  # Start search from backtrack direction
            dy, dx = dirs[d]
            ny, nx = current[0] + dy, current[1] + dx
            
            if (ny, nx) in visited:
                continue
            if (ny, nx) in boundary_set:
                contour.append((nx * scale + offset_x, ny * scale + offset_y))
                visited.add((ny, nx))
                current = (ny, nx)
                prev_dir = d
                found = True
                break
        
        if not found:
            break
    
    return np.array(contour, dtype=np.float32)

default_vertex_shader = '''
void main()
{
    gl_Position = ModelViewProjectionMatrix * vec4(pos, 0, 1.0);
}
'''

default_fragment_shader = '''
void main()
{
    fragColor = color;
}
'''

dotted_line_vertex_shader = '''
void main()
{
    arcLengthInter = arcLength;

    gl_Position = ModelViewProjectionMatrix * vec4(pos, 0, 1.0);
}
'''

dotted_line_fragment_shader = '''
void main()
{
    if (step(sin((arcLengthInter + offset) * scale), 0.5) == 1) {
        fragColor = color1;
    } else {
        fragColor = color2;
    }
}
'''

image_vertex_shader = '''
void main()
{
    gl_Position = ModelViewProjectionMatrix * vec4(pos, 0, 1.0);
    texCoordOut = texCoord;
}
'''


image_composite_vertex_shader = '''
void main()
{
    gl_Position = ModelViewProjectionMatrix * vec4(pos, 0.0, 1.0);
    texCoordOut = texCoord;
    // clip coords to uv
    baseUV = gl_Position.xy * 0.5 + 0.5;
}
'''

image_composite_fragment_shader = '''
// Blend mode constants
const int BLEND_MIX = 0;
const int BLEND_DARKEN = 1;
const int BLEND_MULTIPLY = 2;
const int BLEND_COLOR_BURN = 3;
const int BLEND_LIGHTEN = 4;
const int BLEND_SCREEN = 5;
const int BLEND_COLOR_DODGE = 6;
const int BLEND_ADD = 7;
const int BLEND_OVERLAY = 8;
const int BLEND_SOFT_LIGHT = 9;
const int BLEND_LINEAR_LIGHT = 10;
const int BLEND_DIFFERENCE = 11;
const int BLEND_EXCLUSION = 12;
const int BLEND_SUBTRACT = 13;
const int BLEND_DIVIDE = 14;
const int BLEND_HUE = 15;
const int BLEND_SATURATION = 16;
const int BLEND_COLOR = 17;
const int BLEND_VALUE = 18;

vec3 rgb_to_hsv(vec3 rgb) {
    float cmax = max(rgb.r, max(rgb.g, rgb.b));
    float cmin = min(rgb.r, min(rgb.g, rgb.b));
    float delta = cmax - cmin;
    vec3 hsv = vec3(0.0, 0.0, cmax);
    if (delta > 0.0001) {
        hsv.y = delta / cmax;
        if (rgb.r >= cmax) hsv.x = (rgb.g - rgb.b) / delta;
        else if (rgb.g >= cmax) hsv.x = 2.0 + (rgb.b - rgb.r) / delta;
        else hsv.x = 4.0 + (rgb.r - rgb.g) / delta;
        hsv.x = mod(hsv.x / 6.0, 1.0);
    }
    return hsv;
}

vec3 hsv_to_rgb(vec3 hsv) {
    float h = hsv.x * 6.0;
    float s = hsv.y;
    float v = hsv.z;
    float c = v * s;
    float x = c * (1.0 - abs(mod(h, 2.0) - 1.0));
    float m = v - c;
    vec3 rgb;
    if (h < 1.0) rgb = vec3(c, x, 0.0);
    else if (h < 2.0) rgb = vec3(x, c, 0.0);
    else if (h < 3.0) rgb = vec3(0.0, c, x);
    else if (h < 4.0) rgb = vec3(0.0, x, c);
    else if (h < 5.0) rgb = vec3(x, 0.0, c);
    else rgb = vec3(c, 0.0, x);
    return rgb + m;
}

vec3 blend_colors(vec3 base, vec3 layer, int mode) {
    vec3 result = layer;
    if (mode == BLEND_MIX) result = layer;
    else if (mode == BLEND_DARKEN) result = min(base, layer);
    else if (mode == BLEND_MULTIPLY) result = base * layer;
    else if (mode == BLEND_COLOR_BURN) result = vec3(layer.r > 0.0 ? 1.0 - min(1.0, (1.0 - base.r) / layer.r) : 0.0, layer.g > 0.0 ? 1.0 - min(1.0, (1.0 - base.g) / layer.g) : 0.0, layer.b > 0.0 ? 1.0 - min(1.0, (1.0 - base.b) / layer.b) : 0.0);
    else if (mode == BLEND_LIGHTEN) result = max(base, layer);
    else if (mode == BLEND_SCREEN) result = 1.0 - (1.0 - base) * (1.0 - layer);
    else if (mode == BLEND_COLOR_DODGE) result = vec3(layer.r < 1.0 ? min(1.0, base.r / (1.0 - layer.r)) : 1.0, layer.g < 1.0 ? min(1.0, base.g / (1.0 - layer.g)) : 1.0, layer.b < 1.0 ? min(1.0, base.b / (1.0 - layer.b)) : 1.0);
    else if (mode == BLEND_ADD) result = min(base + layer, 1.0);
    else if (mode == BLEND_OVERLAY) result = vec3(base.r < 0.5 ? 2.0 * base.r * layer.r : 1.0 - 2.0 * (1.0 - base.r) * (1.0 - layer.r), base.g < 0.5 ? 2.0 * base.g * layer.g : 1.0 - 2.0 * (1.0 - base.g) * (1.0 - layer.g), base.b < 0.5 ? 2.0 * base.b * layer.b : 1.0 - 2.0 * (1.0 - base.b) * (1.0 - layer.b));
    else if (mode == BLEND_SOFT_LIGHT) result = vec3(layer.r < 0.5 ? base.r - (1.0 - 2.0 * layer.r) * base.r * (1.0 - base.r) : base.r + (2.0 * layer.r - 1.0) * (sqrt(base.r) - base.r), layer.g < 0.5 ? base.g - (1.0 - 2.0 * layer.g) * base.g * (1.0 - base.g) : base.g + (2.0 * layer.g - 1.0) * (sqrt(base.g) - base.g), layer.b < 0.5 ? base.b - (1.0 - 2.0 * layer.b) * base.b * (1.0 - base.b) : base.b + (2.0 * layer.b - 1.0) * (sqrt(base.b) - base.b));
    else if (mode == BLEND_LINEAR_LIGHT) result = clamp(base + 2.0 * layer - 1.0, 0.0, 1.0);
    else if (mode == BLEND_DIFFERENCE) result = abs(base - layer);
    else if (mode == BLEND_EXCLUSION) result = base + layer - 2.0 * base * layer;
    else if (mode == BLEND_SUBTRACT) result = max(base - layer, 0.0);
    else if (mode == BLEND_DIVIDE) result = vec3(layer.r > 0.0 ? min(base.r / layer.r, 1.0) : 1.0, layer.g > 0.0 ? min(base.g / layer.g, 1.0) : 1.0, layer.b > 0.0 ? min(base.b / layer.b, 1.0) : 1.0);
    else if (mode == BLEND_HUE) { vec3 baseHSV = rgb_to_hsv(base); vec3 layerHSV = rgb_to_hsv(layer); result = hsv_to_rgb(vec3(layerHSV.x, baseHSV.y, baseHSV.z)); }
    else if (mode == BLEND_SATURATION) { vec3 baseHSV = rgb_to_hsv(base); vec3 layerHSV = rgb_to_hsv(layer); result = hsv_to_rgb(vec3(baseHSV.x, layerHSV.y, baseHSV.z)); }
    else if (mode == BLEND_COLOR) { vec3 baseHSV = rgb_to_hsv(base); vec3 layerHSV = rgb_to_hsv(layer); result = hsv_to_rgb(vec3(layerHSV.x, layerHSV.y, baseHSV.z)); }
    else if (mode == BLEND_VALUE) { vec3 baseHSV = rgb_to_hsv(base); vec3 layerHSV = rgb_to_hsv(layer); result = hsv_to_rgb(vec3(baseHSV.x, baseHSV.y, layerHSV.z)); }
    return result;
}

void main()
{
    vec4 baseColorFromTex = texture(baseImage, baseUV);
    vec4 texColor = texture(image, texCoordOut);
    
    vec3 blended = blend_colors(baseColorFromTex.rgb, texColor.rgb, blendMode);
    float finalAlpha = texColor.a * opacity;
    
    vec3 outColor = mix(baseColorFromTex.rgb, blended, finalAlpha);
    float outAlpha = baseColorFromTex.a + finalAlpha * (1.0 - baseColorFromTex.a);
    
    fragColor = vec4(outColor, outAlpha);
}
'''

image_fragment_shader = '''
// Blend mode constants
const int BLEND_MIX = 0;
const int BLEND_DARKEN = 1;
const int BLEND_MULTIPLY = 2;
const int BLEND_COLOR_BURN = 3;
const int BLEND_LIGHTEN = 4;
const int BLEND_SCREEN = 5;
const int BLEND_COLOR_DODGE = 6;
const int BLEND_ADD = 7;
const int BLEND_OVERLAY = 8;
const int BLEND_SOFT_LIGHT = 9;
const int BLEND_LINEAR_LIGHT = 10;
const int BLEND_DIFFERENCE = 11;
const int BLEND_EXCLUSION = 12;
const int BLEND_SUBTRACT = 13;
const int BLEND_DIVIDE = 14;
const int BLEND_HUE = 15;
const int BLEND_SATURATION = 16;
const int BLEND_COLOR = 17;
const int BLEND_VALUE = 18;

vec3 rgb_to_hsv(vec3 rgb) {
    float cmax = max(rgb.r, max(rgb.g, rgb.b));
    float cmin = min(rgb.r, min(rgb.g, rgb.b));
    float delta = cmax - cmin;
    vec3 hsv = vec3(0.0, 0.0, cmax);
    if (delta > 0.0001) {
        hsv.y = delta / cmax;
        if (rgb.r >= cmax) hsv.x = (rgb.g - rgb.b) / delta;
        else if (rgb.g >= cmax) hsv.x = 2.0 + (rgb.b - rgb.r) / delta;
        else hsv.x = 4.0 + (rgb.r - rgb.g) / delta;
        hsv.x = mod(hsv.x / 6.0, 1.0);
    }
    return hsv;
}

vec3 hsv_to_rgb(vec3 hsv) {
    float h = hsv.x * 6.0;
    float s = hsv.y;
    float v = hsv.z;
    float c = v * s;
    float x = c * (1.0 - abs(mod(h, 2.0) - 1.0));
    float m = v - c;
    vec3 rgb;
    if (h < 1.0) rgb = vec3(c, x, 0.0);
    else if (h < 2.0) rgb = vec3(x, c, 0.0);
    else if (h < 3.0) rgb = vec3(0.0, c, x);
    else if (h < 4.0) rgb = vec3(0.0, x, c);
    else if (h < 5.0) rgb = vec3(x, 0.0, c);
    else rgb = vec3(c, 0.0, x);
    return rgb + m;
}

vec3 blend_colors(vec3 base, vec3 layer, int mode) {
    vec3 result = layer;
    
    if (mode == BLEND_MIX) {
        result = layer;
    } else if (mode == BLEND_DARKEN) {
        result = min(base, layer);
    } else if (mode == BLEND_MULTIPLY) {
        result = base * layer;
    } else if (mode == BLEND_COLOR_BURN) {
        result = vec3(
            layer.r > 0.0 ? 1.0 - min(1.0, (1.0 - base.r) / layer.r) : 0.0,
            layer.g > 0.0 ? 1.0 - min(1.0, (1.0 - base.g) / layer.g) : 0.0,
            layer.b > 0.0 ? 1.0 - min(1.0, (1.0 - base.b) / layer.b) : 0.0
        );
    } else if (mode == BLEND_LIGHTEN) {
        result = max(base, layer);
    } else if (mode == BLEND_SCREEN) {
        result = 1.0 - (1.0 - base) * (1.0 - layer);
    } else if (mode == BLEND_COLOR_DODGE) {
        result = vec3(
            layer.r < 1.0 ? min(1.0, base.r / (1.0 - layer.r)) : 1.0,
            layer.g < 1.0 ? min(1.0, base.g / (1.0 - layer.g)) : 1.0,
            layer.b < 1.0 ? min(1.0, base.b / (1.0 - layer.b)) : 1.0
        );
    } else if (mode == BLEND_ADD) {
        result = min(base + layer, 1.0);
    } else if (mode == BLEND_OVERLAY) {
        result = vec3(
            base.r < 0.5 ? 2.0 * base.r * layer.r : 1.0 - 2.0 * (1.0 - base.r) * (1.0 - layer.r),
            base.g < 0.5 ? 2.0 * base.g * layer.g : 1.0 - 2.0 * (1.0 - base.g) * (1.0 - layer.g),
            base.b < 0.5 ? 2.0 * base.b * layer.b : 1.0 - 2.0 * (1.0 - base.b) * (1.0 - layer.b)
        );
    } else if (mode == BLEND_SOFT_LIGHT) {
        result = vec3(
            layer.r < 0.5 ? base.r - (1.0 - 2.0 * layer.r) * base.r * (1.0 - base.r) : base.r + (2.0 * layer.r - 1.0) * (sqrt(base.r) - base.r),
            layer.g < 0.5 ? base.g - (1.0 - 2.0 * layer.g) * base.g * (1.0 - base.g) : base.g + (2.0 * layer.g - 1.0) * (sqrt(base.g) - base.g),
            layer.b < 0.5 ? base.b - (1.0 - 2.0 * layer.b) * base.b * (1.0 - base.b) : base.b + (2.0 * layer.b - 1.0) * (sqrt(base.b) - base.b)
        );
    } else if (mode == BLEND_LINEAR_LIGHT) {
        result = clamp(base + 2.0 * layer - 1.0, 0.0, 1.0);
    } else if (mode == BLEND_DIFFERENCE) {
        result = abs(base - layer);
    } else if (mode == BLEND_EXCLUSION) {
        result = base + layer - 2.0 * base * layer;
    } else if (mode == BLEND_SUBTRACT) {
        result = max(base - layer, 0.0);
    } else if (mode == BLEND_DIVIDE) {
        result = vec3(
            layer.r > 0.0 ? min(base.r / layer.r, 1.0) : 1.0,
            layer.g > 0.0 ? min(base.g / layer.g, 1.0) : 1.0,
            layer.b > 0.0 ? min(base.b / layer.b, 1.0) : 1.0
        );
    } else if (mode == BLEND_HUE) {
        vec3 baseHSV = rgb_to_hsv(base);
        vec3 layerHSV = rgb_to_hsv(layer);
        result = hsv_to_rgb(vec3(layerHSV.x, baseHSV.y, baseHSV.z));
    } else if (mode == BLEND_SATURATION) {
        vec3 baseHSV = rgb_to_hsv(base);
        vec3 layerHSV = rgb_to_hsv(layer);
        result = hsv_to_rgb(vec3(baseHSV.x, layerHSV.y, baseHSV.z));
    } else if (mode == BLEND_COLOR) {
        vec3 baseHSV = rgb_to_hsv(base);
        vec3 layerHSV = rgb_to_hsv(layer);
        result = hsv_to_rgb(vec3(layerHSV.x, layerHSV.y, baseHSV.z));
    } else if (mode == BLEND_VALUE) {
        vec3 baseHSV = rgb_to_hsv(base);
        vec3 layerHSV = rgb_to_hsv(layer);
        result = hsv_to_rgb(vec3(baseHSV.x, baseHSV.y, layerHSV.z));
    }
    
    return result;
}

void main()
{
    vec4 texColor = texture(image, texCoordOut);
    vec3 blended = blend_colors(baseColor.rgb, texColor.rgb, blendMode);
    float finalAlpha = texColor.a * opacity;
    fragColor = vec4(blended, finalAlpha);
}
'''

def make_scale_matrix(scale):
    return Matrix([
        [scale[0], 0, 0, 0],
        [0, scale[1], 0, 0],
        [0, 0, 1.0, 0],
        [0, 0, 0, 1.0]
    ])

class UIRenderer:
    def __init__(self):
        self.shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        self.image_shader = gpu.shader.from_builtin('IMAGE')
        
        default_shader_info = gpu.types.GPUShaderCreateInfo()
        default_shader_info.push_constant('MAT4', 'ModelViewProjectionMatrix')
        default_shader_info.push_constant('VEC4', 'color')
        default_shader_info.vertex_in(0, 'VEC2', 'pos')
        default_shader_info.fragment_out(0, 'VEC4', 'fragColor')
        default_shader_info.vertex_source(default_vertex_shader)
        default_shader_info.fragment_source(default_fragment_shader)
        self.default_shader = create_from_info(default_shader_info)
        self.default_shader_u_color = self.default_shader.uniform_from_name('color')
        
        self._texture_cache = {}
        self._offscreen = None

        dotted_line_shader_inter = gpu.types.GPUStageInterfaceInfo("dotted_line")
        dotted_line_shader_inter.smooth('FLOAT', "arcLengthInter")

        dotted_line_shader_info = gpu.types.GPUShaderCreateInfo()
        dotted_line_shader_info.push_constant('MAT4', 'ModelViewProjectionMatrix')
        dotted_line_shader_info.push_constant('FLOAT', 'scale')
        dotted_line_shader_info.push_constant('FLOAT', 'offset')
        dotted_line_shader_info.push_constant('VEC4', 'color1')
        dotted_line_shader_info.push_constant('VEC4', 'color2')
        dotted_line_shader_info.vertex_in(0, 'VEC2', 'pos')
        dotted_line_shader_info.vertex_in(1, 'FLOAT', 'arcLength')
        dotted_line_shader_info.vertex_out(dotted_line_shader_inter)
        dotted_line_shader_info.fragment_out(0, 'VEC4', 'fragColor')
        dotted_line_shader_info.vertex_source(dotted_line_vertex_shader)
        dotted_line_shader_info.fragment_source(dotted_line_fragment_shader)
        self.dotted_line_shader = create_from_info(dotted_line_shader_info)
        self.dotted_line_shader_u_color1 = self.dotted_line_shader.uniform_from_name("color1")
        self.dotted_line_shader_u_color2 = self.dotted_line_shader.uniform_from_name("color2")

        image_shader_inter = gpu.types.GPUStageInterfaceInfo("image_shader")
        image_shader_inter.smooth('VEC2', "texCoordOut")

        image_shader_info = gpu.types.GPUShaderCreateInfo()
        image_shader_info.push_constant('MAT4', 'ModelViewProjectionMatrix')
        image_shader_info.push_constant('FLOAT', 'opacity')
        image_shader_info.push_constant('INT', 'blendMode')
        image_shader_info.push_constant('VEC4', 'baseColor')
        image_shader_info.sampler(0, 'FLOAT_2D', 'image')
        image_shader_info.vertex_in(0, 'VEC2', 'pos')
        image_shader_info.vertex_in(1, 'VEC2', 'texCoord')
        image_shader_info.vertex_out(image_shader_inter)
        image_shader_info.fragment_out(0, 'VEC4', 'fragColor')
        image_shader_info.vertex_source(image_vertex_shader)
        image_shader_info.fragment_source(image_fragment_shader)
        self.image_shader = create_from_info(image_shader_info)

        image_comp_shader_inter = gpu.types.GPUStageInterfaceInfo("image_composite_shader")
        image_comp_shader_inter.smooth('VEC2', "texCoordOut")
        image_comp_shader_inter.smooth('VEC2', "baseUV")

        image_comp_shader_info = gpu.types.GPUShaderCreateInfo()
        image_comp_shader_info.push_constant('MAT4', 'ModelViewProjectionMatrix')
        image_comp_shader_info.push_constant('FLOAT', 'opacity')
        image_comp_shader_info.push_constant('INT', 'blendMode')
        image_comp_shader_info.sampler(0, 'FLOAT_2D', 'image')
        image_comp_shader_info.sampler(1, 'FLOAT_2D', 'baseImage')
        image_comp_shader_info.vertex_in(0, 'VEC2', 'pos')
        image_comp_shader_info.vertex_in(1, 'VEC2', 'texCoord')
        image_comp_shader_info.vertex_out(image_comp_shader_inter)
        image_comp_shader_info.fragment_out(0, 'VEC4', 'fragColor')
        image_comp_shader_info.vertex_source(image_composite_vertex_shader)
        image_comp_shader_info.fragment_source(image_composite_fragment_shader)
        self.image_composite_shader = create_from_info(image_comp_shader_info)
        
        self._ofs1 = None
        self._ofs2 = None

    def render_selection_frame(self, pos, size, rot=0, scale=(1.0, 1.0)):
        width, height = size[0], size[1]

        prev_blend = gpu.state.blend_get()
        gpu.state.blend_set('ALPHA')

        gpu.state.line_width_set(2.0)

        with gpu.matrix.push_pop():
            verts = [[0, 0], [0, height], [width, height], [width, 0], [0, 0]]
            # T <= R <= S <= centering
            mat = Matrix.Translation([pos[0] + width / 2.0, pos[1] + height / 2.0, 0]) \
                    @ Matrix.Rotation(rot, 4, 'Z') \
                    @ make_scale_matrix(scale) \
                    @ Matrix.Translation([-width / 2.0, -height / 2.0, 0])

            for i, vert in enumerate(verts):
                verts[i] = (mat @ Vector(vert + [0, 1]))[:2]

            verts = np.array(verts, 'f')

            arc_lengths = [0]
            for a, b in zip(verts[:-1], verts[1:]):
                arc_lengths.append(arc_lengths[-1] + np.linalg.norm(a - b))
        
            batch = batch_for_shader(self.dotted_line_shader, 'LINE_STRIP',
                {"pos": verts, "arcLength": arc_lengths})

            self.dotted_line_shader.bind()

            self.dotted_line_shader.uniform_float("scale", 0.6)
            self.dotted_line_shader.uniform_float("offset", 0)
            self.dotted_line_shader.uniform_vector_float(self.dotted_line_shader_u_color1,
                    np.array([1.0, 1.0, 1.0, 0.5], 'f'), 4)
            self.dotted_line_shader.uniform_vector_float(self.dotted_line_shader_u_color2,
                    np.array([0.0, 0.0, 0.0, 0.5], 'f'), 4)

            batch.draw(self.dotted_line_shader)

        gpu.state.blend_set(prev_blend)
    
    def render_ellipse_selection(self, pos, size, rot=0, scale=(1.0, 1.0)):
        import math
        width, height = size[0], size[1]
        cx, cy = width / 2, height / 2
        
        # Generate ellipse vertices
        segments = 64
        vertices = []
        arc_lengths = [0]
        total_length = 0.0
        
        # We need local vertices first
        local_verts = []
        for i in range(segments + 1):
            angle = 2 * math.pi * i / segments
            x = cx + cx * math.cos(angle)
            y = cy + cy * math.sin(angle)
            local_verts.append([x, y])
            
        prev_blend = gpu.state.blend_get()
        gpu.state.blend_set('ALPHA')
        gpu.state.line_width_set(2.0)

        with gpu.matrix.push_pop():
            # Transform vertices
            mat = Matrix.Translation([pos[0] + width / 2.0, pos[1] + height / 2.0, 0]) \
                    @ Matrix.Rotation(rot, 4, 'Z') \
                    @ make_scale_matrix(scale) \
                    @ Matrix.Translation([-width / 2.0, -height / 2.0, 0])

            transformed_verts = []
            for vert in local_verts:
                transformed_verts.append((mat @ Vector(vert + [0, 1]))[:2])
            
            verts = np.array(transformed_verts, 'f')

            # Calculate arc lengths for dotted line
            for a, b in zip(verts[:-1], verts[1:]):
                arc_lengths.append(arc_lengths[-1] + np.linalg.norm(a - b))
        
            batch = batch_for_shader(self.dotted_line_shader, 'LINE_STRIP',
                {"pos": verts, "arcLength": arc_lengths})

            self.dotted_line_shader.bind()

            self.dotted_line_shader.uniform_float("scale", 0.6)
            self.dotted_line_shader.uniform_float("offset", 0)
            self.dotted_line_shader.uniform_vector_float(self.dotted_line_shader_u_color1,
                    np.array([1.0, 1.0, 1.0, 0.5], 'f'), 4)
            self.dotted_line_shader.uniform_vector_float(self.dotted_line_shader_u_color2,
                    np.array([0.0, 0.0, 0.0, 0.5], 'f'), 4)

            batch.draw(self.dotted_line_shader)

        gpu.state.blend_set(prev_blend)
    
    def render_merged_selection(self, rects):
        """Render merged selection contour like Krita - highly optimized.
        
        Args:
            rects: List of (x1, y1, x2, y2) rectangles in screen coords
        """
        if not rects:
            return
        
        # For single rectangle, just draw it directly
        if len(rects) == 1:
            x1, y1, x2, y2 = rects[0]
            self.render_selection_frame([x1, y1], [x2 - x1, y2 - y1])
            return
        
        # Compute merged contour edges using numpy-optimized algorithm
        all_verts, all_arc_lengths = self._compute_merged_contour_fast(rects)
        
        if len(all_verts) == 0:
            return
        
        prev_blend = gpu.state.blend_get()
        gpu.state.blend_set('ALPHA')
        gpu.state.line_width_set(2.0)
        
        # Single batch draw for all edges
        batch = batch_for_shader(self.dotted_line_shader, 'LINES',
            {"pos": all_verts, "arcLength": all_arc_lengths})
        
        self.dotted_line_shader.bind()
        self.dotted_line_shader.uniform_float("scale", 0.6)
        self.dotted_line_shader.uniform_float("offset", 0)
        self.dotted_line_shader.uniform_vector_float(self.dotted_line_shader_u_color1,
                np.array([1.0, 1.0, 1.0, 0.5], 'f'), 4)
        self.dotted_line_shader.uniform_vector_float(self.dotted_line_shader_u_color2,
                np.array([0.0, 0.0, 0.0, 0.5], 'f'), 4)
        batch.draw(self.dotted_line_shader)
        
        gpu.state.blend_set(prev_blend)
    
    def render_merged_ellipses(self, ellipses):
        """Render merged ellipse contour like Krita.
        
        Args:
            ellipses: List of (x1, y1, x2, y2) bounding boxes in screen coords
        """
        if not ellipses:
            return
        
        # For single ellipse, just draw it directly
        if len(ellipses) == 1:
            x1, y1, x2, y2 = ellipses[0]
            self.render_ellipse_selection((x1, y1), (x2 - x1, y2 - y1))
            return
        
        # Compute bounding box of all ellipses
        all_x1 = min(e[0] for e in ellipses)
        all_y1 = min(e[1] for e in ellipses)
        all_x2 = max(e[2] for e in ellipses)
        all_y2 = max(e[3] for e in ellipses)
        
        total_w = all_x2 - all_x1
        total_h = all_y2 - all_y1
        
        if total_w <= 0 or total_h <= 0:
            return
        
        # Create a downsampled mask for performance (max 256x256)
        max_res = 256
        scale = min(1.0, max_res / max(total_w, total_h))
        mask_w = max(1, int(total_w * scale))
        mask_h = max(1, int(total_h * scale))
        
        # Rasterize all ellipses to the mask
        mask = np.zeros((mask_h, mask_w), dtype=bool)
        
        # Create coordinate grids once
        cy, cx = np.ogrid[0:mask_h, 0:mask_w]
        
        for e in ellipses:
            # Convert ellipse coords to mask space
            ex1 = (e[0] - all_x1) * scale
            ey1 = (e[1] - all_y1) * scale
            ex2 = (e[2] - all_x1) * scale
            ey2 = (e[3] - all_y1) * scale
            
            ew = ex2 - ex1
            eh = ey2 - ey1
            
            if ew <= 0 or eh <= 0:
                continue
            
            # Center of ellipse
            ecx = ex1 + ew / 2
            ecy = ey1 + eh / 2
            
            # Ellipse equation: ((x - cx) / rx)^2 + ((y - cy) / ry)^2 <= 1
            rx = ew / 2
            ry = eh / 2
            
            if rx > 0 and ry > 0:
                ellipse_mask = ((cx - ecx) / rx) ** 2 + ((cy - ecy) / ry) ** 2 <= 1
                mask = mask | ellipse_mask
        
        # Trace the contour using edge detection
        # Pad mask to detect edges at boundaries
        padded = np.pad(mask, 1, constant_values=False)
        
        # Find boundary pixels (pixels that are inside but have at least one outside neighbor)
        boundary = mask & (
            ~padded[:-2, 1:-1] |   # top neighbor outside
            ~padded[2:, 1:-1] |    # bottom neighbor outside  
            ~padded[1:-1, :-2] |   # left neighbor outside
            ~padded[1:-1, 2:]      # right neighbor outside
        )
        
        # Get boundary pixel coordinates
        by, bx = np.where(boundary)
        
        if len(bx) == 0:
            return
        
        # Convert back to screen space and create vertices
        inv_scale = 1.0 / scale
        verts = []
        arc_lengths = []
        
        # Sort boundary pixels to form a continuous contour
        # Simple approach: use the boundary pixel centers
        points = np.column_stack([
            bx * inv_scale + all_x1,
            by * inv_scale + all_y1
        ])
        
        # Order points for continuous drawing using nearest neighbor
        if len(points) > 1:
            ordered = [points[0]]
            remaining = list(range(1, len(points)))
            
            while remaining:
                last = ordered[-1]
                # Find nearest remaining point
                dists = np.sum((points[remaining] - last) ** 2, axis=1)
                nearest_idx = np.argmin(dists)
                ordered.append(points[remaining[nearest_idx]])
                remaining.pop(nearest_idx)
            
            ordered = np.array(ordered, dtype=np.float32)
            
            # Create LINE_STRIP vertices
            verts = ordered.tolist()
            verts.append(verts[0])  # Close the loop
            
            # Calculate arc lengths
            arc_lengths = [0.0]
            for i in range(1, len(verts)):
                dx = verts[i][0] - verts[i-1][0]
                dy = verts[i][1] - verts[i-1][1]
                arc_lengths.append(arc_lengths[-1] + np.sqrt(dx*dx + dy*dy))
        
        if not verts:
            return
        
        # Draw the contour
        prev_blend = gpu.state.blend_get()
        gpu.state.blend_set('ALPHA')
        gpu.state.line_width_set(2.0)
        
        verts_np = np.array(verts, dtype=np.float32)
        arc_np = np.array(arc_lengths, dtype=np.float32)
        
        batch = batch_for_shader(self.dotted_line_shader, 'LINE_STRIP',
            {"pos": verts_np, "arcLength": arc_np})
        
        self.dotted_line_shader.bind()
        self.dotted_line_shader.uniform_float("scale", 0.6)
        self.dotted_line_shader.uniform_float("offset", 0)
        self.dotted_line_shader.uniform_vector_float(self.dotted_line_shader_u_color1,
                np.array([1.0, 1.0, 1.0, 0.5], 'f'), 4)
        self.dotted_line_shader.uniform_vector_float(self.dotted_line_shader_u_color2,
                np.array([0.0, 0.0, 0.0, 0.5], 'f'), 4)
        batch.draw(self.dotted_line_shader)
        
        gpu.state.blend_set(prev_blend)
    
    def render_lasso_preview(self, points):
        """Render lasso preview while user is drawing."""
        if len(points) < 2:
            return
        
        verts = np.array(points, dtype=np.float32)
        
        # Calculate arc lengths
        arc_lengths = [0.0]
        for i in range(1, len(verts)):
            dx = verts[i][0] - verts[i-1][0]
            dy = verts[i][1] - verts[i-1][1]
            arc_lengths.append(arc_lengths[-1] + np.sqrt(dx*dx + dy*dy))
        
        arc_np = np.array(arc_lengths, dtype=np.float32)
        
        prev_blend = gpu.state.blend_get()
        gpu.state.blend_set('ALPHA')
        gpu.state.line_width_set(2.0)
        
        batch = batch_for_shader(self.dotted_line_shader, 'LINE_STRIP',
            {"pos": verts, "arcLength": arc_np})
        
        self.dotted_line_shader.bind()
        self.dotted_line_shader.uniform_float("scale", 0.6)
        self.dotted_line_shader.uniform_float("offset", 0)
        self.dotted_line_shader.uniform_vector_float(self.dotted_line_shader_u_color1,
                np.array([1.0, 1.0, 1.0, 0.5], 'f'), 4)
        self.dotted_line_shader.uniform_vector_float(self.dotted_line_shader_u_color2,
                np.array([0.0, 0.0, 0.0, 0.5], 'f'), 4)
        batch.draw(self.dotted_line_shader)
        
        gpu.state.blend_set(prev_blend)
    
    def render_merged_all(self, rects, ellipses, lassos=None, neg_rects=None, neg_ellipses=None, neg_lassos=None):
        """Render merged contour for rectangles, ellipses, AND lassos like Krita.
        
        Args:
            rects: List of (x1, y1, x2, y2) rectangles in screen coords
            ellipses: List of (x1, y1, x2, y2) ellipse bounding boxes in screen coords
            lassos: List of polygon point lists in screen coords
            neg_rects: Rectangles to subtract (negation)
            neg_ellipses: Ellipses to subtract (negation)
            neg_lassos: Lassos to subtract (negation)
        """
        if lassos is None:
            lassos = []
        if neg_rects is None:
            neg_rects = []
        if neg_ellipses is None:
            neg_ellipses = []
        if neg_lassos is None:
            neg_lassos = []
        
        if not rects and not ellipses and not lassos:
            return
        
        # Simple case: only one shape with no negations
        if len(rects) == 1 and not ellipses and not lassos and not neg_ellipses and not neg_lassos:
            x1, y1, x2, y2 = rects[0]
            self.render_selection_frame([x1, y1], [x2 - x1, y2 - y1])
            return
        if len(ellipses) == 1 and not rects and not lassos and not neg_ellipses and not neg_lassos:
            x1, y1, x2, y2 = ellipses[0]
            self.render_ellipse_selection((x1, y1), (x2 - x1, y2 - y1))
            return
        if len(lassos) == 1 and not rects and not ellipses and not neg_ellipses and not neg_lassos:
            self.render_lasso_preview(lassos[0])
            return
        
        # Only rectangles with no negations - use fast grid-based algorithm
        if not ellipses and not lassos and not neg_ellipses and not neg_lassos and rects:
            self.render_merged_selection(rects)
            return
        
        # Mixed shapes - rasterize all to a common mask
        # Compute bounding box of all shapes (including negations)
        all_x1 = float('inf')
        all_y1 = float('inf')
        all_x2 = float('-inf')
        all_y2 = float('-inf')
        
        for r in rects:
            all_x1 = min(all_x1, r[0])
            all_y1 = min(all_y1, r[1])
            all_x2 = max(all_x2, r[2])
            all_y2 = max(all_y2, r[3])
        
        for e in ellipses:
            all_x1 = min(all_x1, e[0])
            all_y1 = min(all_y1, e[1])
            all_x2 = max(all_x2, e[2])
            all_y2 = max(all_y2, e[3])
        
        for lasso in lassos:
            for pt in lasso:
                all_x1 = min(all_x1, pt[0])
                all_y1 = min(all_y1, pt[1])
                all_x2 = max(all_x2, pt[0])
                all_y2 = max(all_y2, pt[1])
        
        for r in neg_rects:
            all_x1 = min(all_x1, r[0])
            all_y1 = min(all_y1, r[1])
            all_x2 = max(all_x2, r[2])
            all_y2 = max(all_y2, r[3])
        
        for e in neg_ellipses:
            all_x1 = min(all_x1, e[0])
            all_y1 = min(all_y1, e[1])
            all_x2 = max(all_x2, e[2])
            all_y2 = max(all_y2, e[3])
        
        for lasso in neg_lassos:
            for pt in lasso:
                all_x1 = min(all_x1, pt[0])
                all_y1 = min(all_y1, pt[1])
                all_x2 = max(all_x2, pt[0])
                all_y2 = max(all_y2, pt[1])
        
        total_w = all_x2 - all_x1
        total_h = all_y2 - all_y1
        
        if total_w <= 0 or total_h <= 0:
            return
        
        # Create a downsampled mask for performance (max 256x256)
        max_res = 256
        scale = min(1.0, max_res / max(total_w, total_h))
        mask_w = max(1, int(total_w * scale))
        mask_h = max(1, int(total_h * scale))
        
        # Rasterize all shapes to the mask
        mask = np.zeros((mask_h, mask_w), dtype=bool)
        
        # Create coordinate grids once
        cy, cx = np.ogrid[0:mask_h, 0:mask_w]
        
        # Rasterize rectangles
        for r in rects:
            rx1 = int((r[0] - all_x1) * scale)
            ry1 = int((r[1] - all_y1) * scale)
            rx2 = int((r[2] - all_x1) * scale)
            ry2 = int((r[3] - all_y1) * scale)
            
            rx1 = max(0, min(rx1, mask_w))
            ry1 = max(0, min(ry1, mask_h))
            rx2 = max(0, min(rx2, mask_w))
            ry2 = max(0, min(ry2, mask_h))
            
            if rx2 > rx1 and ry2 > ry1:
                mask[ry1:ry2, rx1:rx2] = True
        
        # Rasterize ellipses
        for e in ellipses:
            ex1 = (e[0] - all_x1) * scale
            ey1 = (e[1] - all_y1) * scale
            ex2 = (e[2] - all_x1) * scale
            ey2 = (e[3] - all_y1) * scale
            
            ew = ex2 - ex1
            eh = ey2 - ey1
            
            if ew <= 0 or eh <= 0:
                continue
            
            ecx = ex1 + ew / 2
            ecy = ey1 + eh / 2
            rx = ew / 2
            ry = eh / 2
            
            if rx > 0 and ry > 0:
                ellipse_mask = ((cx - ecx) / rx) ** 2 + ((cy - ecy) / ry) ** 2 <= 1
                mask = mask | ellipse_mask
        
        # Rasterize lassos using vectorized polygon mask (fast)
        for lasso in lassos:
            if len(lasso) < 3:
                continue
            # Convert to mask space
            pts = [(int((pt[0] - all_x1) * scale), int((pt[1] - all_y1) * scale)) for pt in lasso]
            
            # Compute bounding box
            lx1 = max(0, min(p[0] for p in pts))
            ly1 = max(0, min(p[1] for p in pts))
            lx2 = min(mask_w, max(p[0] for p in pts) + 1)
            ly2 = min(mask_h, max(p[1] for p in pts) + 1)
            
            if lx2 > lx1 and ly2 > ly1:
                poly_mask = math_utils.fast_polygon_mask(pts, lx1, ly1, lx2 - lx1, ly2 - ly1)
                mask[ly1:ly2, lx1:lx2] |= poly_mask
        
        # SUBTRACT negation rectangles from mask
        for r in neg_rects:
            rx1 = int((r[0] - all_x1) * scale)
            ry1 = int((r[1] - all_y1) * scale)
            rx2 = int((r[2] - all_x1) * scale)
            ry2 = int((r[3] - all_y1) * scale)
            
            rx1 = max(0, min(rx1, mask_w))
            ry1 = max(0, min(ry1, mask_h))
            rx2 = max(0, min(rx2, mask_w))
            ry2 = max(0, min(ry2, mask_h))
            
            if rx2 > rx1 and ry2 > ry1:
                mask[ry1:ry2, rx1:rx2] = False  # Subtract
        
        # SUBTRACT negation ellipses from mask
        for e in neg_ellipses:
            ex1 = (e[0] - all_x1) * scale
            ey1 = (e[1] - all_y1) * scale
            ex2 = (e[2] - all_x1) * scale
            ey2 = (e[3] - all_y1) * scale
            
            ew = ex2 - ex1
            eh = ey2 - ey1
            
            if ew <= 0 or eh <= 0:
                continue
            
            ecx = ex1 + ew / 2
            ecy = ey1 + eh / 2
            rx = ew / 2
            ry = eh / 2
            
            if rx > 0 and ry > 0:
                neg_ellipse_mask = ((cx - ecx) / rx) ** 2 + ((cy - ecy) / ry) ** 2 <= 1
                mask = mask & ~neg_ellipse_mask  # Subtract
        
        # SUBTRACT negation lassos from mask (using vectorized polygon mask)
        for lasso in neg_lassos:
            if len(lasso) < 3:
                continue
            pts = [(int((pt[0] - all_x1) * scale), int((pt[1] - all_y1) * scale)) for pt in lasso]
            
            # Compute bounding box
            lx1 = max(0, min(p[0] for p in pts))
            ly1 = max(0, min(p[1] for p in pts))
            lx2 = min(mask_w, max(p[0] for p in pts) + 1)
            ly2 = min(mask_h, max(p[1] for p in pts) + 1)
            
            if lx2 > lx1 and ly2 > ly1:
                neg_poly_mask = math_utils.fast_polygon_mask(pts, lx1, ly1, lx2 - lx1, ly2 - ly1)
                mask[ly1:ly2, lx1:lx2] &= ~neg_poly_mask  # Subtract
        
        # Find connected components using fast two-pass algorithm
        labels, current_label = _fast_connected_components(mask)
        
        if current_label == 0:
            return
        
        inv_scale = 1.0 / scale
        
        # Draw each component separately
        prev_blend = gpu.state.blend_get()
        gpu.state.blend_set('ALPHA')
        gpu.state.line_width_set(2.0)
        
        for label in range(1, current_label + 1):
            # Get boundary for this component
            component_mask = (labels == label)
            padded = np.pad(component_mask, 1, constant_values=False)
            
            boundary = component_mask & (
                ~padded[:-2, 1:-1] |
                ~padded[2:, 1:-1] |
                ~padded[1:-1, :-2] |
                ~padded[1:-1, 2:]
            )
            
            by, bx = np.where(boundary)
            
            if len(bx) == 0:
                continue
            
            # Use fast contour tracing instead of O(n²) nearest neighbor
            contour = _trace_boundary_contour(boundary, bx, by, inv_scale, all_x1, all_y1)
            
            if len(contour) > 1:
                verts = contour.tolist()
                verts.append(verts[0])  # Close the loop
                
                arc_lengths = [0.0]
                for i in range(1, len(verts)):
                    dx = verts[i][0] - verts[i-1][0]
                    dy = verts[i][1] - verts[i-1][1]
                    arc_lengths.append(arc_lengths[-1] + np.sqrt(dx*dx + dy*dy))
                
                verts_np = np.array(verts, dtype=np.float32)
                arc_np = np.array(arc_lengths, dtype=np.float32)
                
                batch = batch_for_shader(self.dotted_line_shader, 'LINE_STRIP',
                    {"pos": verts_np, "arcLength": arc_np})
                
                self.dotted_line_shader.bind()
                self.dotted_line_shader.uniform_float("scale", 0.6)
                self.dotted_line_shader.uniform_float("offset", 0)
                self.dotted_line_shader.uniform_vector_float(self.dotted_line_shader_u_color1,
                        np.array([1.0, 1.0, 1.0, 0.5], 'f'), 4)
                self.dotted_line_shader.uniform_vector_float(self.dotted_line_shader_u_color2,
                        np.array([0.0, 0.0, 0.0, 0.5], 'f'), 4)
                batch.draw(self.dotted_line_shader)
        
        gpu.state.blend_set(prev_blend)
    
    def _compute_merged_contour_fast(self, rects):
        """Compute contour edges using numpy - vectorized for performance.
        
        Returns (verts, arc_lengths) arrays ready for GPU batch.
        """
        # Collect unique coordinates
        x_set = set()
        y_set = set()
        for x1, y1, x2, y2 in rects:
            x_set.add(x1)
            x_set.add(x2)
            y_set.add(y1)
            y_set.add(y2)
        
        x_sorted = np.array(sorted(x_set), dtype=np.float32)
        y_sorted = np.array(sorted(y_set), dtype=np.float32)
        
        grid_w = len(x_sorted) - 1
        grid_h = len(y_sorted) - 1
        if grid_w <= 0 or grid_h <= 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
        
        # Create grid using numpy (much faster than Python lists)
        inside = np.zeros((grid_h, grid_w), dtype=bool)
        
        # Build index maps
        x_to_idx = {x: i for i, x in enumerate(x_sorted)}
        y_to_idx = {y: i for i, y in enumerate(y_sorted)}
        
        # Mark cells inside each rectangle
        for x1, y1, x2, y2 in rects:
            xi1, xi2 = x_to_idx[x1], x_to_idx[x2]
            yi1, yi2 = y_to_idx[y1], y_to_idx[y2]
            inside[yi1:yi2, xi1:xi2] = True
        
        # Pad grid with False border for edge detection
        padded = np.pad(inside, 1, constant_values=False)
        
        # Find horizontal edges (y direction transitions)
        h_diff = padded[1:, 1:-1] != padded[:-1, 1:-1]  # Shape: (grid_h+1, grid_w)
        h_edges_y, h_edges_x = np.where(h_diff)
        
        # Find vertical edges (x direction transitions)  
        v_diff = padded[1:-1, 1:] != padded[1:-1, :-1]  # Shape: (grid_h, grid_w+1)
        v_edges_y, v_edges_x = np.where(v_diff)
        
        # Build vertex arrays for horizontal edges
        h_verts = []
        for yi, xi in zip(h_edges_y, h_edges_x):
            h_verts.append([x_sorted[xi], y_sorted[yi]])
            h_verts.append([x_sorted[xi + 1], y_sorted[yi]])
        
        # Build vertex arrays for vertical edges
        v_verts = []
        for yi, xi in zip(v_edges_y, v_edges_x):
            v_verts.append([x_sorted[xi], y_sorted[yi]])
            v_verts.append([x_sorted[xi], y_sorted[yi + 1]])
        
        all_verts = h_verts + v_verts
        if not all_verts:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
        
        all_verts = np.array(all_verts, dtype=np.float32)
        
        # Compute arc lengths for each edge (pairs of vertices)
        arc_lengths = []
        for i in range(0, len(all_verts), 2):
            arc_lengths.append(0.0)
            if i + 1 < len(all_verts):
                length = np.linalg.norm(all_verts[i+1] - all_verts[i])
                arc_lengths.append(length)
        
        return all_verts, np.array(arc_lengths, dtype=np.float32)

    def render_image_sub(self, img, pos, size, rot, scale, opacity=1.0, blend_mode=0, base_color=(0.5, 0.5, 0.5, 1.0)):
        width, height = size[0], size[1]

        texture = self._texture_cache.get(img.name)
        if texture is None:
            try:
                texture = gpu.texture.from_image(img)
                self._texture_cache[img.name] = texture
            except Exception:
                return

        with gpu.matrix.push_pop():
            gpu.matrix.translate([pos[0] + width / 2.0, pos[1] + height / 2.0])
            gpu.matrix.multiply_matrix(
                    Matrix.Rotation(rot, 4, 'Z'))
            gpu.matrix.scale(scale)
            gpu.matrix.translate([-width / 2.0, -height / 2.0])

            batch = batch_for_shader(self.image_shader, 'TRI_FAN',
                {
                    "pos": [
                        (0, 0),
                        (width, 0),
                        size,
                        (0, height)
                    ],
                    "texCoord": [(0, 0), (1, 0), (1, 1), (0, 1)]
                })

            self.image_shader.bind()

            self.image_shader.uniform_float('opacity', opacity)
            self.image_shader.uniform_int('blendMode', blend_mode)
            self.image_shader.uniform_float('baseColor', base_color)
            self.image_shader.uniform_sampler('image', texture)

            batch.draw(self.image_shader)


    def render_layer_composite_pass(self, img, pos, size, rot, scale, opacity, blend_mode, base_texture):
        width, height = size[0], size[1]
        texture = self._texture_cache.get(img.name)
        if texture is None:
            try:
                texture = gpu.texture.from_image(img)
                self._texture_cache[img.name] = texture
            except Exception:
                return

        with gpu.matrix.push_pop():
            gpu.matrix.translate([pos[0] + width / 2.0, pos[1] + height / 2.0])
            gpu.matrix.multiply_matrix(Matrix.Rotation(rot, 4, 'Z'))
            gpu.matrix.scale(scale)
            gpu.matrix.translate([-width / 2.0, -height / 2.0])

            batch = batch_for_shader(self.image_composite_shader, 'TRI_FAN',
                {
                    "pos": [(0, 0), (width, 0), size, (0, height)],
                    "texCoord": [(0, 0), (1, 0), (1, 1), (0, 1)]
                })

            self.image_composite_shader.bind()
            self.image_composite_shader.uniform_float('opacity', opacity)
            self.image_composite_shader.uniform_int('blendMode', blend_mode)
            self.image_composite_shader.uniform_sampler('image', texture)
            self.image_composite_shader.uniform_sampler('baseImage', base_texture)

            batch.draw(self.image_composite_shader)

    def render_texture_fullscreen(self, texture, width, height, alpha=1.0):
        with gpu.matrix.push_pop():
            batch = batch_for_shader(self.image_shader, 'TRI_FAN',
                {
                    "pos": [(0, 0), (width, 0), (width, height), (0, height)],
                    "texCoord": [(0, 0), (1, 0), (1, 1), (0, 1)]
                })
            self.image_shader.bind()
            self.image_shader.uniform_float('opacity', alpha)
            self.image_shader.uniform_int('blendMode', 0) # MIX
            self.image_shader.uniform_float('baseColor', (0,0,0,0))
            self.image_shader.uniform_sampler('image', texture)
            batch.draw(self.image_shader)

    def render_image(self, img, pos, size, rot=0, scale=(1.0, 1.0), opacity=1.0, blend_mode=0, base_color=(0.5, 0.5, 0.5, 1.0)):
        prev_blend = gpu.state.blend_get()
        gpu.state.blend_set('ALPHA')

        self.render_image_sub(img, pos, size, rot, scale, opacity, blend_mode, base_color)

        gpu.state.blend_set(prev_blend)

    def render_image_offscreen(self, img, rot=0, scale=(1.0, 1.0)):
        width, height = img.size[0], img.size[1]

        box = [[0, 0], [width, 0], [0, height], [width, height]]
        mat = Matrix.Rotation(rot, 4, 'Z') \
                @ make_scale_matrix(scale) \
                @ Matrix.Translation([-width / 2.0, -height / 2.0, 0])
        min_x, min_y = sys.float_info.max, sys.float_info.max
        max_x, max_y = -sys.float_info.max, -sys.float_info.max

        # calculate bounding box
        for pos in box:
            pos = mat @ Vector(pos + [0, 1])
            min_x = min(min_x, pos[0])
            min_y = min(min_y, pos[1])
            max_x = max(max_x, pos[0])
            max_y = max(max_y, pos[1])

        ofs_width = math.ceil(max_x - min_x)
        ofs_height = math.ceil(max_y - min_y)

        if self._offscreen is None or self._offscreen.width < ofs_width or self._offscreen.height < ofs_height:
             # Create new offscreen if needed (or if too small)
             # Free existing if any
             if self._offscreen:
                 self._offscreen.free()
             self._offscreen = gpu.types.GPUOffScreen(ofs_width, ofs_height)
        
        # Use cached offscreen (viewport handles using only the part we need?)
        # Wait, GPUOffScreen size is fixed. If we use a larger one, we need to be careful about glViewport?
        # gpu.state.active_framebuffer_get() usually respects the bound framebuffer size.
        # But we want to read back specific size.
        
        with self._offscreen.bind():
            # We must set viewport to match our desired output size??
            # Actually, `fb.clear` clears the whole thing.
            # We only care about the region (0, 0, ofs_width, ofs_height).
            
            fb = gpu.state.active_framebuffer_get()
            fb.clear(color=(0.0, 0.0, 0.0, 0.0))

            with gpu.matrix.push_pop():
                gpu.matrix.load_projection_matrix(Matrix.Identity(4))

                gpu.matrix.load_identity()
                gpu.matrix.scale([1.0 / (ofs_width / 2.0), 1.0 / (ofs_height / 2.0)])
                gpu.matrix.translate([-width / 2.0, -height / 2.0])

                self.render_image_sub(img, [0, 0], [width, height], rot, scale)

            buff = fb.read_color(0, 0, ofs_width, ofs_height, 4, 0, 'UBYTE')

        # Do NOT free offscreen
        # ofs.free() 


        return buff, ofs_width, ofs_height

    def render_info_box(self, pos1, pos2):
        prev_blend = gpu.state.blend_get()
        gpu.state.blend_set('ALPHA')

        verts = [
            pos1,
            (pos2[0], pos1[1]),
            (pos1[0], pos2[1]),
            pos2
        ]

        indices = [
            (0, 1, 2),
            (2, 1, 3)
        ]

        batch = batch_for_shader(self.default_shader, 'TRIS',
            {"pos": verts}, indices=indices)

        self.default_shader.bind()

        self.default_shader.uniform_vector_float(self.default_shader_u_color,
                np.array([0, 0, 0, 0.7], 'f'), 4)

        batch.draw(self.default_shader)

        gpu.state.blend_set(prev_blend)

    def update_texture(self, img):
        """Force update of a cached texture."""
        if img.name in self._texture_cache:
            try:
                # Replace with new texture
                self._texture_cache[img.name] = gpu.texture.from_image(img)
            except Exception:
                del self._texture_cache[img.name]

    def free(self):
        """Free GPU resources."""
        self._texture_cache.clear()
        if self._offscreen:
            self._offscreen.free()
            self._offscreen = None
