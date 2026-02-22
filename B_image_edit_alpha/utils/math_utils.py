import numpy as np

def fast_polygon_mask(polygon_points, bbox_x1, bbox_y1, bbox_w, bbox_h):
    """Generate polygon mask using vectorized ray-casting algorithm.
    
    Much faster than Python scanline loop for large or complex polygons.
    """
    if len(polygon_points) < 3 or bbox_w <= 0 or bbox_h <= 0:
        return np.zeros((bbox_h, bbox_w), dtype=bool)
    
    # Create coordinate grids
    y_coords, x_coords = np.ogrid[0:bbox_h, 0:bbox_w]
    x_coords = x_coords + bbox_x1 + 0.5  # Center of pixel
    y_coords = y_coords + bbox_y1 + 0.5
    
    # Broadcast to full grid
    x_grid = np.broadcast_to(x_coords, (bbox_h, bbox_w))
    y_grid = np.broadcast_to(y_coords, (bbox_h, bbox_w))
    
    # Ray-casting: count edge crossings
    n = len(polygon_points)
    mask = np.zeros((bbox_h, bbox_w), dtype=bool)
    
    for i in range(n):
        p1 = polygon_points[i]
        p2 = polygon_points[(i + 1) % n]
        y1, y2 = p1[1], p2[1]
        x1, x2 = p1[0], p2[0]
        
        # Check if edge crosses horizontal ray from point
        if y1 == y2:
            continue
            
        cond1 = (y1 > y_grid) != (y2 > y_grid)
        
        # Calculate x-intersection of edge with horizontal ray at y_grid
        with np.errstate(divide='ignore', invalid='ignore'):
            slope = (x2 - x1) / (y2 - y1)
            x_intersect = x1 + (y_grid - y1) * slope
        
        cond2 = x_grid < x_intersect
        
        # Toggle mask where both conditions are true
        mask ^= (cond1 & cond2)
    
    return mask
