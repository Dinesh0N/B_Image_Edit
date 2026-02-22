
import bpy
from . import utils

class API:
    VERSION = (0, 0, 0)

    @staticmethod
    def read_pixels_from_image(img):
        return utils.layer_read_pixels_from_image(img)

    @staticmethod
    def write_pixels_to_image(img, pixels):
        utils.layer_write_pixels_to_image(img, pixels)

    @staticmethod
    def refresh_image(context):
        utils.layer_refresh_image(context)

    @staticmethod
    def select_layer(img, layer):
        img_props = img.image_edit_properties
        layers = img_props.layers
        if layer and layer in layers:
            img_props.selected_layer_index = layers.index(layer)
        else:
            img_props.selected_layer_index = -1

    @staticmethod
    def get_selected_layer(img):
        img_props = img.image_edit_properties
        layers = img_props.layers
        selected_layer_index = img_props.selected_layer_index
        if selected_layer_index == -1 or selected_layer_index >= len(layers):
            return None
        return layers[selected_layer_index]

    @staticmethod
    def create_layer(base_img, pixels, img_settings={}, layer_settings={}):
        img_settings_mod = {
            'is_float': img_settings.get('is_float', True),
            'colorspace_name': img_settings.get('colorspace_name', 'Linear')
        }
        layer_settings_mod = {
            'rotation': layer_settings.get('rotation', 0),
            'scale': layer_settings.get('scale', [1.0, 1.0]),
            'custom_data': layer_settings.get('custom_data', '{}')
        }
        utils.layer_create_layer(base_img, pixels, img_settings_mod, layer_settings_mod)

    @staticmethod
    def read_pixels_from_layer(layer):
        layer_img = bpy.data.images.get(layer.name, None)
        if not layer_img:
            return 0, 0, None
        layer_width, layer_height = layer_img.size[0], layer_img.size[1]
        layer_pixels = utils.layer_read_pixels_from_image(layer_img)
        return layer_width, layer_height, layer_pixels

    @staticmethod
    def write_pixels_to_layer(layer, pixels):
        layer_img = bpy.data.images.get(layer.name, None)
        if not layer_img:
            return
        utils.layer_write_pixels_to_image(layer_img, pixels)

    @staticmethod
    def scale_layer(layer, width, height):
        layer_img = bpy.data.images.get(layer.name, None)
        if not layer_img:
            return
        layer_img.scale(width, height)

    @staticmethod
    def update_layers(img):
        utils.layer_rebuild_image_layers_nodes(img)
