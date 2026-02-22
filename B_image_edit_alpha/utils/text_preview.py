# Text preview cache for B_image_edit addon

import math
import bpy
import gpu

from .fonts import FontManager


class TextPreviewCache:
    """Cache for text preview to avoid re-rendering on every frame."""
    
    _instance = None
    PREVIEW_IMAGE_NAME = "_TextTool_Preview_"
    
    def __init__(self):
        self.blender_image = None
        self.gpu_texture = None
        self.preview_width = 0
        self.preview_height = 0
        # Cache key to detect when settings change
        self.cache_key = None
    
    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = TextPreviewCache()
        return cls._instance
    
    def _make_cache_key(self, text, font_path, font_size, color, rotation, gradient_data):
        """Create a hashable key from current settings."""
        # Color is a tuple/list, convert to tuple for hashing
        color_key = tuple(round(c, 3) for c in color)
        
        # Hash gradient data
        grad_key = None
        if gradient_data:
            grad_key = (
                gradient_data.get('type', 'LINEAR'),
                tuple(tuple(round(c, 3) for c in col) for col in gradient_data.get('lut', [])),
                gradient_data.get('angle', 0.0)
            )
        
        return (text, font_path, font_size, color_key, round(rotation, 2), grad_key)
    
    def update_preview(self, text, font_path, font_size, color, rotation, gradient_data=None):
        """Update the preview texture if settings have changed."""
        if not text:
            return False
        
        new_key = self._make_cache_key(text, font_path, font_size, color, rotation, gradient_data)
        
        # Check if we need to regenerate
        if new_key == self.cache_key and self.gpu_texture is not None:
            return True  # Already up to date
        
        # Generate new preview image using native rendering
        rotation_degrees = math.degrees(rotation) if isinstance(rotation, float) else rotation
        
        pixels, size = FontManager.create_text_image(
            text, font_path, font_size, color, 
            rotation_degrees=rotation_degrees,
            gradient_lut=gradient_data
        )
        
        if pixels is None or size is None:
            self.invalidate()
            return False
        
        self.preview_width, self.preview_height = size
        
        # Create or update Blender image
        try:
            # Remove old preview image if exists with different size
            if self.PREVIEW_IMAGE_NAME in bpy.data.images:
                old_img = bpy.data.images[self.PREVIEW_IMAGE_NAME]
                if old_img.size[0] != self.preview_width or old_img.size[1] != self.preview_height:
                    bpy.data.images.remove(old_img)
                    self.blender_image = None
                    self.gpu_texture = None
            
            # Create new image if needed
            if self.PREVIEW_IMAGE_NAME not in bpy.data.images:
                self.blender_image = bpy.data.images.new(
                    self.PREVIEW_IMAGE_NAME,
                    width=self.preview_width,
                    height=self.preview_height,
                    alpha=True
                )
                self.blender_image.colorspace_settings.name = 'sRGB'
            else:
                self.blender_image = bpy.data.images[self.PREVIEW_IMAGE_NAME]
            
            # Set pixels directly (already in correct format from GPUOffScreen)
            self.blender_image.pixels.foreach_set(pixels)
            self.blender_image.update()
            
            # Force recreate GPU texture after every pixel update
            self.gpu_texture = gpu.texture.from_image(self.blender_image)
            
            self.cache_key = new_key
            return True
            
        except Exception as e:
            print(f"[TextTool] Failed to create preview texture: {e}")
            import traceback
            traceback.print_exc()
            self.invalidate()
            return False
    
    def invalidate(self):
        """Clear the cache."""
        self.gpu_texture = None
        self.preview_width = 0
        self.preview_height = 0
        self.cache_key = None
        # Don't remove the Blender image here to avoid issues during drawing
    
    def get_texture_and_size(self):
        """Return the GPU texture and its size."""
        return self.gpu_texture, self.preview_width, self.preview_height
    
    def cleanup(self):
        """Remove the preview image from Blender. Call on addon unregister."""
        if self.PREVIEW_IMAGE_NAME in bpy.data.images:
            bpy.data.images.remove(bpy.data.images[self.PREVIEW_IMAGE_NAME])
        self.blender_image = None
        self.gpu_texture = None
