ORIG_W = 320.0

def determine_spatial_zone(bbox, image_width=ORIG_W):
    """
    Determines whether a bounding box [xmin, ymin, xmax, ymax]
    is in 'ben trai', 'o giua', or 'ben phai' based on box horizontal center.
    Zone 1 (Left):   x < W / 3
    Zone 2 (Middle): W / 3 <= x <= 2 * W / 3
    Zone 3 (Right):  x > 2 * W / 3
    """
    xmin, ymin, xmax, ymax = bbox
    cx = (xmin + xmax) / 2.0

    zone_boundary_1 = image_width / 3.0       # ~106.67 px
    zone_boundary_2 = (2.0 * image_width) / 3.0 # ~213.33 px

    if cx < zone_boundary_1:
        return "ben trai"
    elif cx <= zone_boundary_2:
        return "o giua"
    else:
        return "ben phai"
