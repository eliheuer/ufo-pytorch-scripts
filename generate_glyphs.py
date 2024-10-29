import torch
from train_ufo_model import GlyphGenerator
from defcon import Font
import numpy as np
from scipy import ndimage
from scipy.spatial.distance import cdist

def array_to_glyph(array, glyph, width=650):
    threshold = 0.5
    points = []
    
    # Find contours using better connected components analysis
    labeled_array, num_features = ndimage.label(array > threshold)
    
    for feature in range(1, num_features + 1):
        feature_mask = labeled_array == feature
        # Get boundary points using binary erosion
        boundary = feature_mask & ~ndimage.binary_erosion(feature_mask)
        boundary_points = np.where(boundary)
        
        contour_points = []
        # Convert to font coordinates with proper scaling
        for y, x in zip(boundary_points[0], boundary_points[1]):
            # Scale x from 0-64 to 0-width
            scaled_x = round(x * (width / 64))
            # Scale y from 0-64 to 0-700, and flip coordinate system
            scaled_y = round((64 - y) * (700 / 64))
            # Add some variation to y-coordinates based on x position
            y_offset = int(20 * np.sin(x * np.pi / 32))  # Creates natural-looking variation
            scaled_y += y_offset
            contour_points.append((scaled_x, scaled_y))
        
        if contour_points:
            # Order points to form a continuous contour
            ordered_points = order_contour_points(contour_points)
            # Simplify with higher tolerance for smoother curves
            simplified_points = simplify_points(ordered_points, tolerance=32)
            points.append(simplified_points)
    
    # Create contours with proper Bézier curves
    if points:
        pen = glyph.getPen()
        for contour in points:
            if len(contour) < 3:
                continue
                
            pen.moveTo((int(contour[0][0]), int(contour[0][1])))  # Convert to integers
            # Use curve fitting to create smooth Bézier curves
            i = 1
            while i < len(contour):
                if i + 2 < len(contour):
                    # Create smooth curve through three points
                    p1 = contour[i]
                    p2 = contour[i + 1]
                    p3 = contour[i + 2]
                    # Calculate control points for smooth curve and round to integers
                    c1 = (round((2*p1[0] + p2[0])/3), round((2*p1[1] + p2[1])/3))
                    c2 = (round((p1[0] + 2*p2[0])/3), round((p1[1] + 2*p2[1])/3))
                    pen.curveTo(c1, c2, (int(p2[0]), int(p2[1])))
                    i += 2
                else:
                    # Use line for remaining points
                    pen.lineTo((int(contour[i][0]), int(contour[i][1])))
                    i += 1
            pen.closePath()

def order_contour_points(points):
    """Order points to form a continuous contour"""
    if not points:
        return points
        
    ordered = [points[0]]
    remaining = points[1:]
    
    while remaining:
        current = ordered[-1]
        # Find nearest point
        distances = [((p[0]-current[0])**2 + (p[1]-current[1])**2) for p in remaining]
        nearest_idx = distances.index(min(distances))
        ordered.append(remaining.pop(nearest_idx))
    
    return ordered

def simplify_points(points, tolerance=32):
    # Implement Douglas-Peucker algorithm to simplify contours
    if len(points) < 3:
        return points
    
    distances = cdist(np.array(points), np.array([points[0], points[-1]]))
    max_dist_idx = np.argmax(distances.min(axis=1))
    
    if distances[max_dist_idx].min() > tolerance:
        left = simplify_points(points[:max_dist_idx+1], tolerance)
        right = simplify_points(points[max_dist_idx:], tolerance)
        return left[:-1] + right
    else:
        return [points[0], points[-1]]

def expand_font(input_ufo_path, output_ufo_path, model_path):
    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GlyphGenerator().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load input font
    input_font = Font(input_ufo_path)
    output_font = Font()

    # Copy existing glyphs
    for glyph_name in input_font.keys():
        output_font.newGlyph(glyph_name)
        output_font[glyph_name].copyDataFromGlyph(input_font[glyph_name])

    # Generate new glyphs for missing characters
    # This is where you'd define which new glyphs to generate
    #new_glyphs = ['á', 'é', 'í', 'ó', 'ú', 'ñ', 'ü']  # Example set
    new_glyphs = ['minus', 'A_grave', 'agrave', 'ain-ar', 'A_E_', 'ae', 'mem-hb']  # Example set

    # Dictionary mapping glyph names to their unicode values
    unicode_map = {
        'minus': 0x2212,
        'A_grave': 0x00C0,
        'agrave': 0x00E0,
        'ain-ar': 0x0639,
        'A_E_': 0x00C6,
        'ae': 0x00E6,
        'mem-hb': 0x05DE,
    }

    with torch.no_grad():
        for glyph_name in new_glyphs:
            if glyph_name not in output_font:
                new_glyph = output_font.newGlyph(glyph_name)
                
                if glyph_name in unicode_map:
                    new_glyph.unicode = unicode_map[glyph_name]
                
                new_glyph.width = 650
                
                # Create input tensor - THIS SECTION NEEDS IMPROVEMENT
                glyph_array = np.zeros((64, 64))
                if glyph_name in input_font:
                    # If the glyph exists in input font, use it as reference
                    reference_glyph = input_font[glyph_name]
                    glyph_array = UFODataset.glyph_to_array(reference_glyph)
                else:
                    # Add more comprehensive shape hints based on glyph type
                    if 'grave' in glyph_name.lower():
                        # Add a more substantial grave accent hint
                        for i in range(20):
                            glyph_array[i+20:i+25, i+20:i+25] = 1
                    elif 'minus' in glyph_name.lower():
                        # Add a horizontal line hint
                        glyph_array[30:34, 10:54] = 1  # Make line thicker
                    elif 'ain' in glyph_name.lower():
                        # Add circular hint for ain
                        center = (32, 32)
                        radius = 15
                        y, x = np.ogrid[-32:32, -32:32]
                        mask = x*x + y*y <= radius*radius
                        glyph_array[mask] = 1
                    elif 'mem' in glyph_name.lower():
                        # Add basic shape for Hebrew mem
                        glyph_array[20:44, 20:44] = 1
                    elif 'ae' in glyph_name.lower():
                        # Add basic shape for ae ligature
                        glyph_array[20:44, 15:30] = 1  # Left part
                        glyph_array[20:44, 35:50] = 1  # Right part
                
                tensor = torch.FloatTensor(glyph_array).unsqueeze(0).unsqueeze(0).to(device)
                generated = model(tensor)
                generated_array = generated.squeeze().cpu().numpy()
                
                # Add threshold check to ensure we have meaningful data
                if np.max(generated_array) > 0.1:  # Adjust threshold as needed
                    array_to_glyph(generated_array, new_glyph, width=new_glyph.width)
                else:
                    print(f"Warning: No meaningful outline generated for {glyph_name}")

    # Save the expanded font
    output_font.save(output_ufo_path)

if __name__ == "__main__":
    expand_font("input.ufo", "output.ufo", "glyph_generator.pth")
