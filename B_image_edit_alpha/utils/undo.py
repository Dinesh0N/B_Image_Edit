# Undo/Redo stack for image pixel modifications

import array


class ImageUndoStack:
    """Manages undo/redo history for image pixel modifications.
    
    Blender's built-in undo system does NOT track direct pixel modifications,
    so we need a custom undo stack for text paint operations.
    """
    
    _instance = None
    
    def __init__(self, max_undo_steps=20):
        # Dictionary: image_name -> {"undo": [...], "redo": [...]}
        self._stacks = {}
        self.max_undo_steps = max_undo_steps
    
    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = ImageUndoStack()
        return cls._instance
    
    def _get_stack(self, image_name):
        """Get or create the undo/redo stack for an image."""
        if image_name not in self._stacks:
            self._stacks[image_name] = {"undo": [], "redo": []}
        return self._stacks[image_name]
    
    def push_state(self, image):
        """Save current image state before modification."""
        if image is None:
            return
        
        stack = self._get_stack(image.name)
        
        # Save current pixels as an array (much faster than list)
        width, height = image.size
        num_pixels = width * height * 4
        
        # Create float array
        pixels = array.array('f', [0.0] * num_pixels)
        image.pixels.foreach_get(pixels)
        
        stack["undo"].append({
            "pixels": pixels,
            "size": (width, height)
        })
        
        # Clear redo stack when new action is performed
        stack["redo"].clear()
        
        # Limit undo history size
        while len(stack["undo"]) > self.max_undo_steps:
            stack["undo"].pop(0)
    
    def push_state_from_array(self, image, pixels_array):
        """Save pre-cached pixel array as undo state (for realtime preview)."""
        if image is None or pixels_array is None:
            return
        
        stack = self._get_stack(image.name)
        width, height = image.size
        
        # Make a copy of the array
        pixels_copy = array.array('f', pixels_array)
        
        stack["undo"].append({
            "pixels": pixels_copy,
            "size": (width, height)
        })
        
        stack["redo"].clear()
        
        while len(stack["undo"]) > self.max_undo_steps:
            stack["undo"].pop(0)
    
    def push_state_region(self, image, x, y, w, h):
        """Save only a region of the image as undo state (memory-efficient).
        
        For a 4K image, this stores ~KB instead of ~256MB per step.
        
        Args:
            image: Blender image
            x, y: Top-left corner of dirty region (pixel coords)
            w, h: Width and height of dirty region
        """
        if image is None or w <= 0 or h <= 0:
            return
        
        import numpy as np
        
        stack = self._get_stack(image.name)
        img_w, img_h = image.size
        
        # Clamp region to image bounds
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(img_w, x + w)
        y2 = min(img_h, y + h)
        rw = x2 - x1
        rh = y2 - y1
        
        if rw <= 0 or rh <= 0:
            return
        
        # Read full image into numpy, extract region
        num_pixels = img_w * img_h * 4
        full_px = np.empty(num_pixels, dtype=np.float32)
        image.pixels.foreach_get(full_px)
        full_px = full_px.reshape(img_h, img_w, 4)
        
        # Extract the dirty region
        region_data = full_px[y1:y2, x1:x2, :].copy()
        
        stack["undo"].append({
            "region": True,
            "region_pixels": region_data,
            "region_bounds": (x1, y1, rw, rh),
            "size": (img_w, img_h),
        })
        
        stack["redo"].clear()
        
        while len(stack["undo"]) > self.max_undo_steps:
            stack["undo"].pop(0)

    
    def push_state_from_numpy(self, image, np_array):
        """Save numpy array as undo state efficiently (avoids slow .tolist())."""
        if image is None or np_array is None:
            return
        
        stack = self._get_stack(image.name)
        width, height = image.size
        
        # Efficient copy: use numpy's buffer interface
        # Flatten if needed and ensure contiguous float32
        flat = np_array.ravel().astype('float32', copy=False)
        pixels_copy = array.array('f')
        pixels_copy.frombytes(flat.tobytes())
        
        stack["undo"].append({
            "pixels": pixels_copy,
            "size": (width, height)
        })
        
        stack["redo"].clear()
        
        while len(stack["undo"]) > self.max_undo_steps:
            stack["undo"].pop(0)
    
    def undo(self, image):
        """Restore previous image state (including size changes from crop)."""
        if image is None:
            return False
        
        stack = self._get_stack(image.name)
        
        if not stack["undo"]:
            return False
        
        state = stack["undo"].pop()
        width, height = image.size
        
        if state.get("region"):
            # Region-based undo: only restore the dirty bounding box
            import numpy as np
            x1, y1, rw, rh = state["region_bounds"]
            
            # Save current region to redo
            num_pixels = width * height * 4
            full_px = np.empty(num_pixels, dtype=np.float32)
            image.pixels.foreach_get(full_px)
            full_px = full_px.reshape(height, width, 4)
            
            current_region = full_px[y1:y1+rh, x1:x1+rw, :].copy()
            stack["redo"].append({
                "region": True,
                "region_pixels": current_region,
                "region_bounds": (x1, y1, rw, rh),
                "size": (width, height),
            })
            
            # Restore saved region
            full_px[y1:y1+rh, x1:x1+rw, :] = state["region_pixels"]
            image.pixels.foreach_set(full_px.ravel())
            image.update()
            return True
        
        # Full-image undo (original behavior)
        num_pixels = width * height * 4
        current_pixels = array.array('f', [0.0] * num_pixels)
        image.pixels.foreach_get(current_pixels)
        
        stack["redo"].append({
            "pixels": current_pixels,
            "size": (width, height)
        })
        
        old_width, old_height = state["size"]
        
        # Handle size change (from crop operations)
        if (old_width, old_height) != (width, height):
            image.scale(old_width, old_height)
        
        # Fast restore
        image.pixels.foreach_set(state["pixels"])
        image.update()
        return True
    
    def redo(self, image):
        """Restore next image state (including size changes from crop)."""
        if image is None:
            return False
        
        stack = self._get_stack(image.name)
        
        if not stack["redo"]:
            return False
        
        state = stack["redo"].pop()
        width, height = image.size
        
        if state.get("region"):
            # Region-based redo
            import numpy as np
            x1, y1, rw, rh = state["region_bounds"]
            
            num_pixels = width * height * 4
            full_px = np.empty(num_pixels, dtype=np.float32)
            image.pixels.foreach_get(full_px)
            full_px = full_px.reshape(height, width, 4)
            
            # Save current region to undo
            current_region = full_px[y1:y1+rh, x1:x1+rw, :].copy()
            stack["undo"].append({
                "region": True,
                "region_pixels": current_region,
                "region_bounds": (x1, y1, rw, rh),
                "size": (width, height),
            })
            
            # Restore redo region
            full_px[y1:y1+rh, x1:x1+rw, :] = state["region_pixels"]
            image.pixels.foreach_set(full_px.ravel())
            image.update()
            return True
        
        # Full-image redo (original behavior)
        num_pixels = width * height * 4
        current_pixels = array.array('f', [0.0] * num_pixels)
        image.pixels.foreach_get(current_pixels)
        
        stack["undo"].append({
            "pixels": current_pixels,
            "size": (width, height)
        })
        
        new_width, new_height = state["size"]
        
        # Handle size change (from crop operations)
        if (new_width, new_height) != (width, height):
            image.scale(new_width, new_height)
        
        image.pixels.foreach_set(state["pixels"])
        image.update()
        return True
    
    def can_undo(self, image):
        if image is None:
            return False
        return len(self._get_stack(image.name)["undo"]) > 0
    
    def can_redo(self, image):
        if image is None:
            return False
        return len(self._get_stack(image.name)["redo"]) > 0
    
    def clear(self, image=None):
        if image is None:
            self._stacks.clear()
        elif image.name in self._stacks:
            del self._stacks[image.name]
