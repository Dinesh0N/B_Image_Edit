import math
import array

import bpy
import bmesh
from bpy.types import Operator
from bpy.props import StringProperty
from bpy_extras.view3d_utils import (
    region_2d_to_origin_3d,
    region_2d_to_vector_3d,
)
from mathutils.bvhtree import BVHTree
from mathutils import Vector
from .. import utils
from .. import ui


def get_text_content(props):
    """Get text content from either text property or text block.
    
    Returns the text block content if use_text_block is enabled and a text block is selected,
    otherwise returns the simple text property.
    """
    if props.use_text_block and props.text_block:
        return props.text_block.as_string()
    return props.text


def _sample_texture_pixel(t_pixels, tw, th, x, y):
    """Sample a single pixel from texture buffer.
    
    Args:
        t_pixels: Flat list of RGBA values
        tw, th: Texture width and height
        x, y: Integer pixel coordinates
    
    Returns:
        Tuple of (r, g, b, a) values
    """
    if 0 <= x < tw and 0 <= y < th:
        idx = (y * tw + x) * 4
        return t_pixels[idx:idx+4]
    return (0, 0, 0, 0)

# ----------------------------
# Operators (3D Texture Paint)
# ----------------------------
class TEXTURE_PAINT_OT_text_tool(Operator):
    bl_idname = "paint.text_tool_ttf"
    bl_label = "Text Tool (TTF/OTF)"
    bl_options = {'REGISTER', 'UNDO'}
    
    _draw_handler = None
    
    # Inline adjustment mode state
    _adjust_mode = None  # None, 'SIZE', or 'ROTATION'
    _adjust_start_value = 0
    _adjust_start_mouse_x = 0

    @classmethod
    def poll(cls, context):
        return (context.mode == 'PAINT_TEXTURE' and
                context.active_object and
                context.active_object.type == 'MESH')

    def modal(self, context, event):
        context.area.tag_redraw()
        
        # Allow viewport navigation to pass through
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE'}:
            return {'PASS_THROUGH'}
            
        props = context.scene.text_tool_properties
        
        # Handle inline adjustment mode
        if self._adjust_mode:
            if event.type == 'MOUSEMOVE':
                delta = event.mouse_x - self._adjust_start_mouse_x
                if self._adjust_mode == 'SIZE':
                    new_size = int(self._adjust_start_value + delta * 0.5)
                    props.font_size = max(8, min(512, new_size))
                    context.area.header_text_set(f"Font Size: {props.font_size}  |  LMB/Enter: Confirm  |  RMB/Esc: Cancel")
                elif self._adjust_mode == 'ROTATION':
                    new_rotation = self._adjust_start_value + delta * 0.01
                    new_rotation = new_rotation % (2 * math.pi)
                    if new_rotation < 0:
                        new_rotation += 2 * math.pi
                    props.rotation = new_rotation
                    context.area.header_text_set(f"Rotation: {math.degrees(props.rotation):.1f}°  |  LMB/Enter: Confirm  |  RMB/Esc: Cancel")
                return {'RUNNING_MODAL'}
            elif event.type in {'LEFTMOUSE', 'RET', 'NUMPAD_ENTER'} and event.value == 'PRESS':
                # Confirm adjustment
                context.area.header_text_set(None)
                self._adjust_mode = None
                return {'RUNNING_MODAL'}
            elif event.type in {'RIGHTMOUSE', 'ESC'} and event.value == 'PRESS':
                # Cancel adjustment
                if self._adjust_mode == 'SIZE':
                    props.font_size = int(self._adjust_start_value)
                elif self._adjust_mode == 'ROTATION':
                    props.rotation = self._adjust_start_value
                context.area.header_text_set(None)
                self._adjust_mode = None
                return {'RUNNING_MODAL'}
            return {'RUNNING_MODAL'}
        
        # Check for adjustment shortcut keys
        if event.type == 'F' and event.value == 'PRESS':
            if event.ctrl:
                # Ctrl+F: Rotation adjustment
                self._adjust_mode = 'ROTATION'
                self._adjust_start_value = props.rotation
                self._adjust_start_mouse_x = event.mouse_x
                context.area.header_text_set(f"Rotation: {math.degrees(props.rotation):.1f}°  |  Drag Left/Right  |  LMB/Enter: Confirm  |  RMB/Esc: Cancel")
            else:
                # F: Font size adjustment
                self._adjust_mode = 'SIZE'
                self._adjust_start_value = props.font_size
                self._adjust_start_mouse_x = event.mouse_x
                context.area.header_text_set(f"Font Size: {props.font_size}  |  Drag Left/Right  |  LMB/Enter: Confirm  |  RMB/Esc: Cancel")
            return {'RUNNING_MODAL'}
        
        if event.type == 'MOUSEMOVE':
            utils.cursor_pos = (event.mouse_region_x, event.mouse_region_y)
            utils.show_cursor = True
            
            # Calculate 3D preview scale (WYSIWYG)
            # We need to know how big 'N' pixels on the texture look on screen.
            obj = context.active_object
            if obj and obj.type == 'MESH':
                try:
                    from bpy_extras.view3d_utils import region_2d_to_origin_3d, region_2d_to_vector_3d, location_3d_to_region_2d
                    from mathutils import Vector
                    
                    region = context.region
                    rv3d = context.region_data
                    coord = (event.mouse_region_x, event.mouse_region_y)
                    
                    view_vector = region_2d_to_vector_3d(region, rv3d, coord)
                    view_origin = region_2d_to_origin_3d(region, rv3d, coord)
                    
                    # Convert to local space
                    mat_inv = obj.matrix_world.inverted()
                    ray_origin = mat_inv @ view_origin
                    ray_target = mat_inv @ (view_origin + view_vector * 1000)
                    ray_dir = ray_target - ray_origin
                    
                    success, location, normal, face_index = obj.ray_cast(ray_origin, ray_dir)
                    
                    if success:
                        # Calculate Texel Density
                        mesh = obj.data
                        poly = mesh.polygons[face_index]
                        
                        # Get area of polygon in world space (approx/fast)
                        v0 = mesh.vertices[poly.vertices[0]].co
                        v1 = mesh.vertices[poly.vertices[1]].co
                        v2 = mesh.vertices[poly.vertices[2]].co
                        
                        # Apply world scale 
                        s = obj.scale
                        v0 = Vector((v0.x * s.x, v0.y * s.y, v0.z * s.z))
                        v1 = Vector((v1.x * s.x, v1.y * s.y, v1.z * s.z))
                        v2 = Vector((v2.x * s.x, v2.y * s.y, v2.z * s.z))
                        
                        area_world = 0.5 * ((v1 - v0).cross(v2 - v0)).length
                        
                        # Get UV area
                        uv_layer = mesh.uv_layers.active
                        if uv_layer:
                            loop0 = poly.loop_indices[0]
                            loop1 = poly.loop_indices[1]
                            loop2 = poly.loop_indices[2]
                            
                            u0 = uv_layer.data[loop0].uv
                            u1 = uv_layer.data[loop1].uv
                            u2 = uv_layer.data[loop2].uv
                            
                            # 2D cross product
                            area_uv = 0.5 * abs((u1.x - u0.x) * (u2.y - u0.y) - (u1.y - u0.y) * (u2.x - u0.x))
                            
                            if area_uv > 0.000001:
                                ratio = math.sqrt(area_world) / math.sqrt(area_uv) # Meters per 1.0 UV Unit
                                
                                # Get Texture Resolution
                                mat = obj.active_material
                                tex_res = 1024 # Default
                                if mat and mat.use_nodes:
                                    image_node = self.get_active_image_node(mat)
                                    if image_node and image_node.image:
                                        if image_node.image.size[0] > 0:
                                            tex_res = image_node.image.size[0]
                                
                                # Meters per Pixel
                                meters_per_pixel = ratio / tex_res
                                
                                # Font size is Texture Pixels
                                test_size_px = 100.0
                                world_size = test_size_px * meters_per_pixel
                                
                                # Project to screen to measure size
                                world_pos = obj.matrix_world @ location
                                
                                # Project Center
                                p2d_center = location_3d_to_region_2d(region, rv3d, world_pos)
                                
                                # Project Offset (perpendicular to view)
                                view_quat = rv3d.view_rotation
                                camera_right = view_quat @ Vector((1.0, 0.0, 0.0))
                                world_pos_offset = world_pos + camera_right * world_size
                                p2d_offset = location_3d_to_region_2d(region, rv3d, world_pos_offset)
                                
                                if p2d_center and p2d_offset:
                                    dist_screen = (p2d_offset - p2d_center).length
                                    utils.cursor_pixel_scale = dist_screen / test_size_px
                                else:
                                    utils.cursor_pixel_scale = 1.0
                            else:
                                utils.cursor_pixel_scale = 1.0
                        else:
                            utils.cursor_pixel_scale = 1.0
                    else:
                        pass # Keep previous scale or reset? Keeping helps smoothness
                except Exception as e:
                    # Keep silent to avoid spam
                    pass
            
            return {'RUNNING_MODAL'}
        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            success = self.place_text_at_surface(context, event)
            if success:
                self.report({'INFO'}, f"Text '{props.text}' placed.")
                self.remove_handler(context)
                return {'FINISHED'}
            else:
                self.remove_handler(context)
                return {'CANCELLED'}
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            utils.show_cursor = False
            self.remove_handler(context)
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        if context.area.type == 'VIEW_3D':
            args = ()
            self._draw_handler = bpy.types.SpaceView3D.draw_handler_add(
                ui.draw_cursor_callback_3d, args, 'WINDOW', 'POST_PIXEL')
            utils.show_cursor = True
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            self.report({'WARNING'}, "View3D not found, cannot run operator")
            return {'CANCELLED'}
            
    def remove_handler(self, context):
        if self._draw_handler:
            bpy.types.SpaceView3D.draw_handler_remove(self._draw_handler, 'WINDOW')
            self._draw_handler = None
        utils.show_cursor = False
        context.area.tag_redraw()

    def place_text_at_surface(self, context, event):
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            return False
        mat = obj.active_material
        if not mat or not mat.use_nodes:
            self.report({'WARNING'}, "No active material with nodes found")
            return False
        image_node = self.get_active_image_node(mat)
        if not image_node or not image_node.image:
            self.report({'WARNING'}, "No active image texture found")
            return False

        props = context.scene.text_tool_properties
        
        hit_location, face_index, hit_uv, tangent_world, bitangent_world = self.view3d_raycast_uv(context, event, obj)
        if hit_location is None:
            self.report({'WARNING'}, "No surface intersection found")
            return False

        # Check projection mode
        if props.projection_mode == 'VIEW':
            # 3D Projected mode - project text from screen space onto texture
            self.render_text_projected_3d(context, event, obj, image_node.image, hit_location, face_index)
        else:
            # UV-based mode (original behavior)
            # Calculate view-aware rotation for the text in UV space
            uv_rotation = self.calculate_uv_rotation(context, event, tangent_world, bitangent_world)
            
            # Calculate screen-space size adjustment
            tex_font_size = props.font_size
            if utils.cursor_pixel_scale > 0.001:
                tex_font_size = int(props.font_size / utils.cursor_pixel_scale)
            
            self.render_text_to_image(context, image_node.image, hit_uv, uv_rotation, override_font_size=tex_font_size)
        
        # Force refresh to ensure 3D viewport updates immediately
        utils.force_texture_refresh(context, image_node.image)
        return True
        return True
    
    def calculate_uv_rotation(self, context, event, tangent_world, bitangent_world):
        """Calculate the rotation needed in UV space so text appears correctly oriented from the current view.
        
        The idea: Project the UV tangent (U direction) onto the screen plane.
        The angle of this projection tells us how text should be rotated in UV space
        to appear horizontal (or with user's desired rotation) on screen.
        """
        from bpy_extras.view3d_utils import location_3d_to_region_2d
        
        props = context.scene.text_tool_properties
        user_rotation = math.degrees(props.rotation)
        
        if tangent_world is None:
            # Fallback: just use user rotation
            return user_rotation
        
        region = context.region
        rv3d = context.region_data
        
        if not rv3d:
            return user_rotation
        
        # Get a reference point (use view origin projected towards scene)
        coord = (event.mouse_region_x, event.mouse_region_y)
        view_origin = region_2d_to_origin_3d(region, rv3d, coord)
        view_dir = region_2d_to_vector_3d(region, rv3d, coord).normalized()
        
        # Create a point on the surface (approximate)
        ref_point = view_origin + view_dir * 5.0  # Arbitrary distance
        
        # Project tangent endpoint onto screen
        tangent_end = ref_point + tangent_world * 0.1
        
        ref_2d = location_3d_to_region_2d(region, rv3d, ref_point)
        tangent_2d = location_3d_to_region_2d(region, rv3d, tangent_end)
        
        if not props.align_to_view:
            # 3D Coordinate / Raw UV alignment
            # Blender internal images are bottom-up, UVs are bottom-up 0..1.
            # But visually, text might appear inverted depending on UV mapping.
            # Usually, standard UV mapping requires 180 flip or 0 depending on unwrapping convention.
            # Let's assume standard behavior: return user_rotation directly (maybe with coordinate flip).
            # Pillow renders text top-down. Blender UVs are bottom-up.
            # render_text_to_image implementation might already handle flipping or not.
            # Let's restart with user_rotation. 
            # Note: In view-aligned mode, we calculated compensation.
            # Here we just want strict UV mapping.
            # Ideally: 0 degrees = Text "Up" aligns with UV "V" direction.
            # Based on testing, 0 degrees might need offset.
            # Let's keep it simple: return user_rotation.
            return user_rotation

        if ref_2d is None or tangent_2d is None:
            return user_rotation
        
        # Calculate screen-space angle of the UV U-direction
        dx = tangent_2d.x - ref_2d.x
        dy = tangent_2d.y - ref_2d.y
        
        # Angle in degrees (atan2 gives angle from positive X axis)
        screen_angle = math.degrees(math.atan2(dy, dx))
        
        # The text in UV space should be rotated by the NEGATIVE of this angle
        # to appear horizontal on screen, plus any user-specified rotation
        uv_rotation = -screen_angle + user_rotation
        
        return uv_rotation

    def get_active_image_node(self, material):
        for node in material.node_tree.nodes:
            if node.type == 'TEX_IMAGE' and node.select:
                return node
        for node in material.node_tree.nodes:
            if node.type == 'TEX_IMAGE':
                return node
        return None

    def build_bvh(self, obj):
        bm = bmesh.new()
        bm.from_object(obj, bpy.context.evaluated_depsgraph_get())
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        bvh = BVHTree.FromBMesh(bm)
        bm.free()
        return bvh

    def view3d_raycast_uv(self, context, event, obj):
        region = context.region
        rv3d = context.region_data
        if not rv3d:
            return None, None, None, None, None

        # Build ray from mouse in world space
        coord = (event.mouse_region_x, event.mouse_region_y)
        view_origin = region_2d_to_origin_3d(region, rv3d, coord)
        view_dir = region_2d_to_vector_3d(region, rv3d, coord).normalized()

        # Define near/far along the ray
        near = view_origin + view_dir * 0.001
        far = view_origin + view_dir * 1e6

        # Transform to object local space
        inv = obj.matrix_world.inverted()
        ro_local = inv @ near
        rf_local = inv @ far
        rd_local = (rf_local - ro_local).normalized()

        # Create Evaluated BMesh
        bm = bmesh.new()
        depsgraph = context.evaluated_depsgraph_get()
        bm.from_object(obj, depsgraph)
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()

        # Build BVH
        bvh = BVHTree.FromBMesh(bm)
        hit = bvh.ray_cast(ro_local, rd_local)
        
        if not hit or not hit[0]:
            bm.free()
            return None, None, None, None, None

        hit_loc_local, hit_normal_local, face_index, distance = hit
        
        # Get UV from BMesh
        uv_layer = bm.loops.layers.uv.active
        if not uv_layer:
            bm.free()
            return hit_loc_local, face_index, None, None, None

        face = bm.faces[face_index]
        p = hit_loc_local
        
        # Use Blender's built-in polygon interpolation for accurate UV calculation
        from mathutils.interpolate import poly_3d_calc
        
        # Get vertex coordinates and UV coordinates from the face
        vert_coords = [v.co for v in face.verts]
        loop_uvs = [loop[uv_layer].uv for loop in face.loops]
        
        # Calculate interpolation weights for point p on the polygon
        weights = poly_3d_calc(vert_coords, p)
        
        # Interpolate UV using the calculated weights
        u_interp = sum(w * uv.x for w, uv in zip(weights, loop_uvs))
        v_interp = sum(w * uv.y for w, uv in zip(weights, loop_uvs))
        best_uv = Vector((u_interp, v_interp))

        result_uv = (best_uv.x, best_uv.y)
        
        # Calculate UV tangent vectors (dP/dU and dP/dV) for rotation
        # We need at least 2 edges to compute tangent/bitangent
        tangent_local = None
        bitangent_local = None
        
        loops = list(face.loops)
        if len(loops) >= 3:
            # Use first triangle of the face for tangent calculation
            p0 = loops[0].vert.co
            p1 = loops[1].vert.co
            p2 = loops[2].vert.co
            
            uv0 = loops[0][uv_layer].uv
            uv1 = loops[1][uv_layer].uv
            uv2 = loops[2][uv_layer].uv
            
            # Edge vectors in 3D
            edge1 = p1 - p0
            edge2 = p2 - p0
            
            # Edge vectors in UV
            duv1 = uv1 - uv0
            duv2 = uv2 - uv0
            
            # Tangent and bitangent calculation
            denom = duv1.x * duv2.y - duv2.x * duv1.y
            if abs(denom) > 1e-8:
                r = 1.0 / denom
                # Tangent: direction of increasing U in 3D space
                tangent_local = (edge1 * duv2.y - edge2 * duv1.y) * r
                # Bitangent: direction of increasing V in 3D space  
                bitangent_local = (edge2 * duv1.x - edge1 * duv2.x) * r
                tangent_local.normalize()
                bitangent_local.normalize()
        
        bm.free()
        
        # Transform tangent/bitangent to world space
        tangent_world = None
        bitangent_world = None
        if tangent_local and bitangent_local:
            # Use the rotation part of the matrix (ignore translation/scale)
            mat_rot = obj.matrix_world.to_3x3().normalized()
            tangent_world = (mat_rot @ tangent_local).normalized()
            bitangent_world = (mat_rot @ bitangent_local).normalized()
        
        return hit_loc_local, face_index, result_uv, tangent_world, bitangent_world

    def render_text_to_image(self, context, image, uv_coord, view_rotation=None, override_font_size=None):
        props = context.scene.text_tool_properties
        width, height = image.size

        if uv_coord is None:
            self.report({'WARNING'}, "UV not found on that face")
            return

        u, v = uv_coord
        # Handle repeating textures (wrap UVs to 0-1 range)
        u = u % 1.0
        v = v % 1.0
        
        # Convert UV to pixel coordinates
        # UV (0,0) = bottom-left of image, UV (1,1) = top-right
        # Blender image pixels: row 0 = bottom, row (height-1) = top
        x = int(u * width)
        y = int(v * height)

        # Use view_rotation if provided (3D viewport), otherwise use user rotation (2D/fallback)
        if view_rotation is not None:
            rotation_degrees = view_rotation
        else:
            rotation_degrees = math.degrees(props.rotation)
            
        font_size = override_font_size if override_font_size is not None else props.font_size
        
        # Build gradient info if gradient is enabled
        gradient_data = None
        if props.use_gradient:
            grad_node = utils.get_gradient_node()
            if grad_node:
                lut = utils.get_gradient_lut(grad_node)
                gradient_data = {
                    'type': props.gradient_type,
                    'lut': lut,
                    'angle': props.gradient_rotation,
                    'font_rotation': rotation_degrees
                }
        
        # Build outline info if outline is enabled
        outline_info = None
        if props.use_outline:
            outline_info = {
                'enabled': True,
                'color': tuple(props.outline_color),
                'size': props.outline_size,
            }
        
        # Get font path from the vector font object
        font_path = props.font_file.filepath if props.font_file else ""
        text_content = get_text_content(props)
        t_pixels, size = utils.FontManager.create_text_image(text_content, font_path, font_size, props.color, rotation_degrees=rotation_degrees, gradient_lut=gradient_data, outline_info=outline_info, alignment=props.text_alignment, line_spacing=props.line_spacing)
        if t_pixels is None or size is None:
            self.report({'ERROR'}, "Failed to render text image.")
            return
        tw, th = size

        # Apply horizontal anchor offset
        if props.anchor_horizontal == 'CENTER':
            x -= tw // 2
        elif props.anchor_horizontal == 'RIGHT':
            x -= tw
        # LEFT: no offset
        
        # Apply vertical anchor offset
        if props.anchor_vertical == 'CENTER':
            y -= th // 2
        elif props.anchor_vertical == 'TOP':
            y -= th
        # BOTTOM: no offset

        # Save state for undo before modifying
        utils.ImageUndoStack.get().push_state(image)
        
        base = list(image.pixels)
        
        # Get blend mode from active brush
        blend_mode = 'MIX'
        if context.tool_settings.image_paint.brush:
            blend_mode = context.tool_settings.image_paint.brush.blend

        # Native rendering is bottom-up, matching Blender's image format
        for ty in range(th):
            by = y + ty
            if by < 0 or by >= height:
                continue
            for tx in range(tw):
                bx = x + tx
                if bx < 0 or bx >= width:
                    continue
                t_idx = (ty * tw + tx) * 4
                b_idx = (by * width + bx) * 4
                tr, tg, tb, ta = t_pixels[t_idx:t_idx + 4]
                if ta > 0:
                    utils.blend_pixel(base, b_idx, tr, tg, tb, ta, blend_mode)
        image.pixels = base

    def render_text_projected_3d(self, context, event, obj, image, hit_location_local, hit_face_index):
        """Render text projected from screen space onto the texture.
        
        Optimized version with:
        - Screen-space face bounding box culling
        - Bilinear texture sampling for quality
        - Pre-computed matrices
        - Reduced per-pixel overhead
        """
        from bpy_extras.view3d_utils import location_3d_to_region_2d
        
        props = context.scene.text_tool_properties
        region = context.region
        rv3d = context.region_data
        
        width, height = image.size
        if width == 0 or height == 0:
            return
        
        # 1. Get cursor position in screen space (center of text)
        cursor_x, cursor_y = event.mouse_region_x, event.mouse_region_y
        
        # 2. Render text to a buffer
        rotation_degrees = math.degrees(props.rotation)
        
        # Build gradient info if enabled
        gradient_data = None
        if props.use_gradient:
            grad_node = utils.get_gradient_node()
            if grad_node:
                lut = utils.get_gradient_lut(grad_node)
                gradient_data = {
                    'type': props.gradient_type,
                    'lut': lut,
                    'angle': props.gradient_rotation,
                    'font_rotation': rotation_degrees
                }
        
        # Build outline info if enabled
        outline_info = None
        if props.use_outline:
            outline_info = {
                'enabled': True,
                'color': tuple(props.outline_color),
                'size': props.outline_size,
            }
        
        font_path = props.font_file.filepath if props.font_file else ""
        text_content = get_text_content(props)
        t_pixels, t_size = utils.FontManager.create_text_image(
            text_content, font_path, props.font_size, props.color,
            rotation_degrees=rotation_degrees, gradient_lut=gradient_data, outline_info=outline_info,
            alignment=props.text_alignment, line_spacing=props.line_spacing
        )
        if t_pixels is None or t_size is None:
            self.report({'ERROR'}, "Failed to render text image.")
            return
        
        tw, th = t_size
        
        # Text bounding box in screen space (using anchor properties)
        # Horizontal anchor
        if props.anchor_horizontal == 'CENTER':
            text_left = cursor_x - tw // 2
        elif props.anchor_horizontal == 'RIGHT':
            text_left = cursor_x - tw
        else:  # LEFT
            text_left = cursor_x
        
        # Vertical anchor
        if props.anchor_vertical == 'CENTER':
            text_bottom = cursor_y - th // 2
        elif props.anchor_vertical == 'TOP':
            text_bottom = cursor_y - th
        else:  # BOTTOM
            text_bottom = cursor_y
        text_right = text_left + tw
        text_top = text_bottom + th
        
        # 3. Build BMesh from evaluated object 
        depsgraph = context.evaluated_depsgraph_get()
        bm = bmesh.new()
        bm.from_object(obj, depsgraph)
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        
        uv_layer = bm.loops.layers.uv.active
        if not uv_layer:
            bm.free()
            self.report({'WARNING'}, "No active UV layer found")
            return
        
        # 4. Pre-compute view info
        view_origin = region_2d_to_origin_3d(region, rv3d, (cursor_x, cursor_y))
        view_dir = region_2d_to_vector_3d(region, rv3d, (cursor_x, cursor_y)).normalized()
        mat_world = obj.matrix_world
        mat_world_3x3 = mat_world.to_3x3()
        
        # 5. Save undo state
        utils.ImageUndoStack.get().push_state(image)
        
        # Performance: Use array.array for pixel buffer
        # 'f' is for float (each pixel is 4 floats: R, G, B, A)
        num_pixels = width * height * 4
        base = array.array('f', [0.0] * num_pixels)
        image.pixels.foreach_get(base)
        
        # Get blend mode
        blend_mode = 'MIX'
        if context.tool_settings.image_paint.brush:
            blend_mode = context.tool_settings.image_paint.brush.blend
        
        # 6. Pre-filter faces by screen-space bounding box overlap & backface culling
        candidate_faces = []
        for face in bm.faces:
            # Better Backface Culling: Use face center to camera vector
            face_center_world = mat_world @ face.calc_center_median()
            face_to_camera = (view_origin - face_center_world).normalized()
            face_normal_world = (mat_world_3x3 @ face.normal).normalized()
            
            # Dot product for culling and grazing angle check
            dot = face_normal_world.dot(face_to_camera)
            
            # Reject if backfacing OR if at a very grazing angle (splatter prevention)
            if dot < 0.1:  # Threshold of ~84 degrees
                continue
            
            # Project all face vertices to screen and compute screen bounding box
            screen_coords = []
            valid_verts = 0
            for loop in face.loops:
                world_pos = mat_world @ loop.vert.co
                screen_pos = location_3d_to_region_2d(region, rv3d, world_pos)
                if screen_pos:
                    screen_coords.append(screen_pos)
                    valid_verts += 1
            
            if valid_verts < 3:
                continue
            
            # Get screen bounding box
            face_screen_left = min(s.x for s in screen_coords)
            face_screen_right = max(s.x for s in screen_coords)
            face_screen_bottom = min(s.y for s in screen_coords)
            face_screen_top = max(s.y for s in screen_coords)
            
            # Check overlap with text bounds
            if (face_screen_right < text_left or face_screen_left > text_right or
                face_screen_top < text_bottom or face_screen_bottom > text_top):
                continue
            
            # This face overlaps - collect data
            face_uvs = [loop[uv_layer].uv.copy() for loop in face.loops]
            face_verts = [loop.vert.co.copy() for loop in face.loops]
            
            # Store distance for depth sorting
            dist = (face_center_world - view_origin).length
            
            candidate_faces.append({
                'uvs': face_uvs,
                'verts': face_verts,
                'screen': screen_coords,
                'dist': dist,
                'dot': dot
            })
            
        # Sort candidate faces front-to-back
        candidate_faces.sort(key=lambda f: f['dist'])
        
        # 7. Process candidate faces
        processed_pixels = set()
        
        # Determine depth threshold based on initial hit
        primary_depth = candidate_faces[0]['dist'] if candidate_faces else 0
        depth_threshold = 0.5  # Max allowed depth spread (in Blender units)
        
        for f_data in candidate_faces:
            face_uvs = f_data['uvs']
            face_verts = f_data['verts']
            n_verts = len(face_uvs)
            
            # Depth culling: Skip faces too far behind the primary hit
            if f_data['dist'] > primary_depth + depth_threshold:
                continue
            
            # Get UV bounding box
            uv_min_u = min(uv.x for uv in face_uvs)
            uv_max_u = max(uv.x for uv in face_uvs)
            uv_min_v = min(uv.y for uv in face_uvs)
            uv_max_v = max(uv.y for uv in face_uvs)
            
            px_min_x = max(0, int(uv_min_u * width))
            px_max_x = min(width, int(uv_max_u * width) + 1)
            px_min_y = max(0, int(uv_min_v * height))
            px_max_y = min(height, int(uv_max_v * height) + 1)
            
            uv_coords = [(uv.x, uv.y) for uv in face_uvs]
            vert_coords = [(v.x, v.y, v.z) for v in face_verts]
            
            for py in range(px_min_y, px_max_y):
                for px in range(px_min_x, px_max_x):
                    pixel_key = (px, py)
                    
                    if pixel_key in processed_pixels:
                        continue
                    
                    tex_u = (px + 0.5) / width
                    tex_v = (py + 0.5) / height
                    
                    inside = False
                    j = n_verts - 1
                    for i in range(n_verts):
                        if ((uv_coords[i][1] > tex_v) != (uv_coords[j][1] > tex_v)) and \
                           (tex_u < (uv_coords[j][0] - uv_coords[i][0]) * (tex_v - uv_coords[i][1]) / (uv_coords[j][1] - uv_coords[i][1]) + uv_coords[i][0]):
                            inside = not inside
                        j = i
                    
                    if not inside:
                        continue
                    
                    weights = self._uv_to_barycentric_fast(tex_u, tex_v, uv_coords, n_verts)
                    if weights is None:
                        continue
                    
                    lx = ly = lz = 0.0
                    for i, w in enumerate(weights):
                        lx += w * vert_coords[i][0]
                        ly += w * vert_coords[i][1]
                        lz += w * vert_coords[i][2]
                    
                    world_pos = mat_world @ Vector((lx, ly, lz))
                    screen_pos = location_3d_to_region_2d(region, rv3d, world_pos)
                    if screen_pos is None:
                        continue
                    
                    sx, sy = screen_pos.x, screen_pos.y
                    if sx < text_left or sx >= text_right or sy < text_bottom or sy >= text_top:
                        continue
                    
                    # Mark pixel as processed as soon as we know it's under the text area
                    # Since we sort by depth, the first face to cover this UV pixel wins.
                    processed_pixels.add(pixel_key)
                    
                    tx_f = sx - text_left
                    ty_f = sy - text_bottom
                    
                    tx0 = int(tx_f)
                    ty0 = int(ty_f)
                    tx1 = min(tx0 + 1, tw - 1)
                    ty1 = min(ty0 + 1, th - 1)
                    fx, fy = tx_f - tx0, ty_f - ty0
                    
                    # Bilinear sampling using module-level function
                    c00 = _sample_texture_pixel(t_pixels, tw, th, tx0, ty0)
                    c10 = _sample_texture_pixel(t_pixels, tw, th, tx1, ty0)
                    c01 = _sample_texture_pixel(t_pixels, tw, th, tx0, ty1)
                    c11 = _sample_texture_pixel(t_pixels, tw, th, tx1, ty1)
                    
                    tr = c00[0]*(1-fx)*(1-fy) + c10[0]*fx*(1-fy) + c01[0]*(1-fx)*fy + c11[0]*fx*fy
                    tg = c00[1]*(1-fx)*(1-fy) + c10[1]*fx*(1-fy) + c01[1]*(1-fx)*fy + c11[1]*fx*fy
                    tb = c00[2]*(1-fx)*(1-fy) + c10[2]*fx*(1-fy) + c01[2]*(1-fx)*fy + c11[2]*fx*fy
                    ta = c00[3]*(1-fx)*(1-fy) + c10[3]*fx*(1-fy) + c01[3]*(1-fx)*fy + c11[3]*fx*fy
                    
                    if ta > 0.001:
                        # INLINED BLENDING for performance
                        b_idx = (py * width + px) * 4
                        dr, dg, db, da = base[b_idx], base[b_idx+1], base[b_idx+2], base[b_idx+3]
                        
                        inv_ta = 1.0 - ta
                        
                        # Default MIX mode logic
                        if blend_mode == 'MIX':
                            base[b_idx]   = tr * ta + dr * inv_ta
                            base[b_idx+1] = tg * ta + dg * inv_ta
                            base[b_idx+2] = tb * ta + db * inv_ta
                            base[b_idx+3] = ta + da * inv_ta
                        else:
                            # Fallback to function for complex modes
                            utils.blend_pixel(base, b_idx, tr, tg, tb, ta, blend_mode)
        
        bm.free()
        image.pixels.foreach_set(base)
    
    def _uv_to_barycentric_fast(self, u, v, uv_coords, n):
        """Optimized barycentric calculation using tuples/inline logic."""
        if n == 3:
            u0, v0 = uv_coords[0]
            u1, v1 = uv_coords[1]
            u2, v2 = uv_coords[2]
            
            denom = (v1 - v2) * (u0 - u2) + (u2 - u1) * (v0 - v2)
            if abs(denom) < 1e-10:
                return None
            
            w0 = ((v1 - v2) * (u - u2) + (u2 - u1) * (v - v2)) / denom
            w1 = ((v2 - v0) * (u - u2) + (u0 - u2) * (v - v2)) / denom
            w2 = 1.0 - w0 - w1
            
            return [w0, w1, w2]
        
        elif n == 4:
            # Fast quad split
            u0, v0 = uv_coords[0]
            u1, v1 = uv_coords[1]
            u2, v2 = uv_coords[2]
            u3, v3 = uv_coords[3]
            
            # Triangle 0-1-2
            denom = (v1 - v2) * (u0 - u2) + (u2 - u1) * (v0 - v2)
            if abs(denom) > 1e-10:
                w0 = ((v1 - v2) * (u - u2) + (u2 - u1) * (v - v2)) / denom
                w1 = ((v2 - v0) * (u - u2) + (u0 - u2) * (v - v2)) / denom
                w2 = 1.0 - w0 - w1
                if w0 >= -0.01 and w1 >= -0.01 and w2 >= -0.01:
                    return [w0, w1, w2, 0.0]
            
            # Triangle 0-2-3
            denom = (v2 - v3) * (u0 - u3) + (u3 - u2) * (v0 - v3)
            if abs(denom) > 1e-10:
                w0 = ((v2 - v3) * (u - u3) + (u3 - u2) * (v - v3)) / denom
                w2 = ((v3 - v0) * (u - u3) + (u0 - u3) * (v - v3)) / denom
                w3 = 1.0 - w0 - w2
                if w0 >= -0.01 and w2 >= -0.01 and w3 >= -0.01:
                    return [w0, 0.0, w2, w3]
            
        # Fallback to slower but robust method
        from mathutils import Vector
        from mathutils.interpolate import poly_3d_calc
        pts = [Vector((uv[0], uv[1], 0.0)) for uv in uv_coords]
        p = Vector((u, v, 0.0))
        try:
            return list(poly_3d_calc(pts, p))
        except Exception:
            return [1.0 / n] * n
    
    def _point_in_polygon_2d(self, point, polygon):
        """Check if a 2D point is inside a 2D polygon using ray casting."""
        x, y = point
        n = len(polygon)
        inside = False
        
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i].x, polygon[i].y
            xj, yj = polygon[j].x, polygon[j].y
            
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        
        return inside
    
    def _uv_to_barycentric(self, u, v, face_uvs):
        """Convert UV coordinates to barycentric weights for the face.
        
        For triangles, use standard barycentric. For quads/ngons, use 
        generalized barycentric coordinates (mean value coordinates).
        """
        n = len(face_uvs)
        
        if n == 3:
            # Triangle: standard barycentric
            u0, v0 = face_uvs[0].x, face_uvs[0].y
            u1, v1 = face_uvs[1].x, face_uvs[1].y
            u2, v2 = face_uvs[2].x, face_uvs[2].y
            
            denom = (v1 - v2) * (u0 - u2) + (u2 - u1) * (v0 - v2)
            if abs(denom) < 1e-10:
                return None
            
            w0 = ((v1 - v2) * (u - u2) + (u2 - u1) * (v - v2)) / denom
            w1 = ((v2 - v0) * (u - u2) + (u0 - u2) * (v - v2)) / denom
            w2 = 1.0 - w0 - w1
            
            return [w0, w1, w2]
        
        else:
            # Poly-based interpolation (more robust for Quads and N-gons)
            from mathutils import Vector
            from mathutils.interpolate import poly_3d_calc
            pts = [Vector((uv.x, uv.y, 0.0)) for uv in face_uvs]
            p = Vector((u, v, 0.0))
            try:
                weights = poly_3d_calc(pts, p)
                return list(weights)
            except Exception:
                return [1.0 / n] * n

class IMAGE_PAINT_OT_text_tool(Operator):
    bl_idname = "image_paint.text_tool_ttf"
    bl_label = "Image Text Tool (TTF/OTF)"
    bl_options = {'REGISTER', 'UNDO'}
    
    _draw_handler = None
    
    # Inline adjustment mode state
    _adjust_mode = None  # None, 'SIZE', or 'ROTATION'
    _adjust_start_value = 0
    _adjust_start_mouse_x = 0

    @classmethod
    def poll(cls, context):
        sima = context.space_data
        return (context.area.type == 'IMAGE_EDITOR' and sima.mode == 'PAINT' and sima.image is not None)

    def modal(self, context, event):
        context.area.tag_redraw()
        
        # Allow viewport navigation to pass through
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE'}:
            return {'PASS_THROUGH'}
            
        props = context.scene.text_tool_properties
        
        # Handle inline adjustment mode
        if self._adjust_mode:
            if event.type == 'MOUSEMOVE':
                delta = event.mouse_x - self._adjust_start_mouse_x
                if self._adjust_mode == 'SIZE':
                    new_size = int(self._adjust_start_value + delta * 0.5)
                    props.font_size = max(8, min(512, new_size))
                    context.area.header_text_set(f"Font Size: {props.font_size}  |  LMB/Enter: Confirm  |  RMB/Esc: Cancel")
                elif self._adjust_mode == 'ROTATION':
                    new_rotation = self._adjust_start_value + delta * 0.01
                    new_rotation = new_rotation % (2 * math.pi)
                    if new_rotation < 0:
                        new_rotation += 2 * math.pi
                    props.rotation = new_rotation
                    context.area.header_text_set(f"Rotation: {math.degrees(props.rotation):.1f}°  |  LMB/Enter: Confirm  |  RMB/Esc: Cancel")
                return {'RUNNING_MODAL'}
            elif event.type in {'LEFTMOUSE', 'RET', 'NUMPAD_ENTER'} and event.value == 'PRESS':
                # Confirm adjustment
                context.area.header_text_set(None)
                self._adjust_mode = None
                return {'RUNNING_MODAL'}
            elif event.type in {'RIGHTMOUSE', 'ESC'} and event.value == 'PRESS':
                # Cancel adjustment
                if self._adjust_mode == 'SIZE':
                    props.font_size = int(self._adjust_start_value)
                elif self._adjust_mode == 'ROTATION':
                    props.rotation = self._adjust_start_value
                context.area.header_text_set(None)
                self._adjust_mode = None
                return {'RUNNING_MODAL'}
            return {'RUNNING_MODAL'}
        
        # Check for adjustment shortcut keys
        if event.type == 'F' and event.value == 'PRESS':
            if event.ctrl:
                # Ctrl+F: Rotation adjustment
                self._adjust_mode = 'ROTATION'
                self._adjust_start_value = props.rotation
                self._adjust_start_mouse_x = event.mouse_x
                context.area.header_text_set(f"Rotation: {math.degrees(props.rotation):.1f}°  |  Drag Left/Right  |  LMB/Enter: Confirm  |  RMB/Esc: Cancel")
            else:
                # F: Font size adjustment
                self._adjust_mode = 'SIZE'
                self._adjust_start_value = props.font_size
                self._adjust_start_mouse_x = event.mouse_x
                context.area.header_text_set(f"Font Size: {props.font_size}  |  Drag Left/Right  |  LMB/Enter: Confirm  |  RMB/Esc: Cancel")
            return {'RUNNING_MODAL'}
        
        if event.type == 'MOUSEMOVE':
            utils.cursor_pos = (event.mouse_region_x, event.mouse_region_y)
            utils.show_cursor = True
            return {'RUNNING_MODAL'}
        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            ok = self.place_text_in_image(context, event)
            if ok:
                self.report({'INFO'}, f"Text '{props.text}' placed in image.")
                self.remove_handler(context)
                return {'FINISHED'}
            self.remove_handler(context)
            return {'CANCELLED'}
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            utils.show_cursor = False
            self.remove_handler(context)
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        if context.area.type == 'IMAGE_EDITOR':
            args = ()
            self._draw_handler = bpy.types.SpaceImageEditor.draw_handler_add(
                ui.draw_cursor_callback_image, args, 'WINDOW', 'POST_PIXEL')
            utils.show_cursor = True
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            self.report({'WARNING'}, "Image Editor not found, cannot run operator")
            return {'CANCELLED'}

    def remove_handler(self, context):
        if self._draw_handler:
            bpy.types.SpaceImageEditor.draw_handler_remove(self._draw_handler, 'WINDOW')
            self._draw_handler = None
        utils.show_cursor = False
        context.area.tag_redraw()

    def place_text_in_image(self, context, event):
        sima = context.space_data
        if not sima.image:
            self.report({'WARNING'}, "No active image found")
            return False
        region = context.region
        coord = self.region_to_image_coord(sima, region, event.mouse_region_x, event.mouse_region_y)
        if coord is None:
            self.report({'WARNING'}, "Click outside image bounds")
            return False
        self.render_text_to_image_direct(context, sima.image, coord)
        
        # Force refresh to ensure 3D viewport updates immediately
        utils.force_texture_refresh(context, sima.image)
        
        context.area.tag_redraw()
        return True

    def region_to_image_coord(self, sima, region, mouse_x, mouse_y):
        """Convert region (screen) coordinates to image pixel coordinates.
        
        Properly handles pan and zoom by using the view2d transformation.
        """
        iw, ih = sima.image.size
        if iw == 0 or ih == 0:
            return None
        
        # Use view2d to convert screen coords to UV coords (0-1 range when image fills view)
        # view2d.region_to_view returns coordinates in "view" space
        # For Image Editor, this is in UV units where (0,0) is bottom-left of image
        view_x, view_y = region.view2d.region_to_view(mouse_x, mouse_y)
        
        # Convert UV to pixel coordinates
        # view coords are already in image-normalized space (0 to 1 for the image area)
        x = int(view_x * iw)
        y = int(view_y * ih)
        
        if 0 <= x < iw and 0 <= y < ih:
            return (x, y)
        return None

    def render_text_to_image_direct(self, context, image, coord):
        props = context.scene.text_tool_properties
        width, height = image.size
        x, y = coord
        rotation_degrees = math.degrees(props.rotation)
        
        # Build gradient info if gradient is enabled
        gradient_data = None
        if props.use_gradient:
            grad_node = utils.get_gradient_node()
            if grad_node:
                lut = utils.get_gradient_lut(grad_node)
                gradient_data = {
                    'type': props.gradient_type,
                    'lut': lut,
                    'angle': props.gradient_rotation,
                    'font_rotation': rotation_degrees
                }
        
        # Build outline info if outline is enabled
        outline_info = None
        if props.use_outline:
            outline_info = {
                'enabled': True,
                'color': tuple(props.outline_color),
                'size': props.outline_size,
            }
        
        # Get font path from the vector font object
        font_path = props.font_file.filepath if props.font_file else ""
        text_content = get_text_content(props)
        t_pixels, size = utils.FontManager.create_text_image(text_content, font_path, props.font_size, props.color, rotation_degrees=rotation_degrees, gradient_lut=gradient_data, outline_info=outline_info, alignment=props.text_alignment, line_spacing=props.line_spacing)
        if t_pixels is None or size is None:
            self.report({'ERROR'}, "Failed to render text image.")
            return
        tw, th = size
        
        # Apply horizontal anchor offset
        if props.anchor_horizontal == 'CENTER':
            x -= tw // 2
        elif props.anchor_horizontal == 'RIGHT':
            x -= tw
        # LEFT: no offset
        
        # Apply vertical anchor offset
        if props.anchor_vertical == 'CENTER':
            y -= th // 2
        elif props.anchor_vertical == 'TOP':
            y -= th
        # BOTTOM: no offset

        # Save state for undo before modifying
        utils.ImageUndoStack.get().push_state(image)
        
        # Use NumPy for fast pixel manipulation
        import numpy as np
        base = np.empty(width * height * 4, dtype=np.float32)
        image.pixels.foreach_get(base)
        base = base.reshape((height, width, 4))
        
        # Convert text pixels to numpy array
        text_np = np.array(t_pixels, dtype=np.float32).reshape((th, tw, 4))
        
        # Get blend mode from active brush
        blend_mode = 'MIX'
        if context.tool_settings.image_paint.brush:
            blend_mode = context.tool_settings.image_paint.brush.blend

        # Calculate valid bounds (clipped to image)
        src_y1 = max(0, -y)
        src_y2 = min(th, height - y)
        src_x1 = max(0, -x)
        src_x2 = min(tw, width - x)
        
        dst_y1 = max(0, y)
        dst_y2 = min(height, y + th)
        dst_x1 = max(0, x)
        dst_x2 = min(width, x + tw)
        
        if src_y2 > src_y1 and src_x2 > src_x1:
            # Extract source and destination regions
            src = text_np[src_y1:src_y2, src_x1:src_x2]
            dst = base[dst_y1:dst_y2, dst_x1:dst_x2]
            
            # Get alpha channel
            alpha = src[:, :, 3:4]
            mask = alpha > 0.001
            
            if blend_mode == 'MIX':
                # Vectorized alpha blending
                inv_alpha = 1.0 - alpha
                blended = np.empty_like(dst)
                blended[:, :, :3] = src[:, :, :3] * alpha + dst[:, :, :3] * inv_alpha
                blended[:, :, 3:4] = alpha + dst[:, :, 3:4] * inv_alpha
                
                # Apply only where alpha > 0
                mask_3d = np.broadcast_to(mask, dst.shape)
                dst[mask_3d] = blended[mask_3d]
            else:
                # Fallback to per-pixel for complex blend modes
                flat_base = base.ravel()
                for ty in range(src_y2 - src_y1):
                    for tx in range(src_x2 - src_x1):
                        if src[ty, tx, 3] > 0.001:
                            b_idx = ((dst_y1 + ty) * width + (dst_x1 + tx)) * 4
                            tr, tg, tb, ta = src[ty, tx]
                            utils.blend_pixel(flat_base, b_idx, tr, tg, tb, ta, blend_mode)
                base = flat_base.reshape((height, width, 4))
        
        # Write back using foreach_set (much faster than assignment)
        image.pixels.foreach_set(base.ravel())


class TEXTURE_PAINT_OT_input_text(Operator):
    bl_idname = "paint.input_text_ttf"
    bl_label = "Input Text"
    bl_options = {'REGISTER', 'UNDO'}

    text_input: StringProperty(
        name="Enter Text",
        description="Text to be painted",
        default=""
    )

    def execute(self, context):
        props = context.scene.text_tool_properties
        props.text = self.text_input
        return {'FINISHED'}

    def invoke(self, context, event):
        props = context.scene.text_tool_properties
        self.text_input = props.text
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "text_input")
