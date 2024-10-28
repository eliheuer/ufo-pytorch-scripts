import torch
from train_ufo_model import GlyphGenerator
from defcon import Font
import numpy as np
from scipy import ndimage
from scipy.spatial.distance import cdist

def array_to_glyph(array, glyph, width=650):
    # Improved conversion that creates proper outlines
    threshold = 0.5
    points = []
    
    # Find connected components in the array
    labeled_array, num_features = ndimage.label(array > threshold)
    
    for feature in range(1, num_features + 1):
        feature_points = np.where(labeled_array == feature)
        contour_points = []
        
        # Convert to font coordinates
        for y, x in zip(feature_points[0], feature_points[1]):
            scaled_x = x * (width / 64)
            scaled_y = y * (700 / 64)  # Assuming 700 units height
            contour_points.append((scaled_x, scaled_y))
        
        if contour_points:
            # Simplify points to create smoother outlines
            simplified_points = simplify_points(contour_points)
            points.append(simplified_points)
    
    # Create contours with proper point types
    if points:
        pen = glyph.getPen()
        for contour in points:
            pen.moveTo(contour[0])
            # Create smooth curves between points
            for i in range(1, len(contour)):
                if i % 3 == 1:  # Add control points for curves
                    pen.curveTo(contour[i], 
                              contour[min(i+1, len(contour)-1)],
                              contour[min(i+2, len(contour)-1)])
                elif i % 3 == 0:
                    continue  # Skip points used as control points
                else:
                    pen.lineTo(contour[i])
            pen.closePath()

def simplify_points(points, tolerance=5):
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
                # Create new glyph with proper metadata
                new_glyph = output_font.newGlyph(glyph_name)
                
                # Set unicode value
                if glyph_name in unicode_map:
                    new_glyph.unicode = unicode_map[glyph_name]
                
                # Set width (you might want to adjust this based on the glyph)
                new_glyph.width = 650  # Default width, adjust as needed
                
                # Generate and convert the glyph outline
                glyph_array = np.zeros((64, 64))
                tensor = torch.FloatTensor(glyph_array).unsqueeze(0).unsqueeze(0).to(device)
                generated = model(tensor)
                generated_array = generated.squeeze().cpu().numpy()
                
                # Convert array to proper outline with curves
                array_to_glyph(generated_array, new_glyph, width=new_glyph.width)

    # Save the expanded font
    output_font.save(output_ufo_path)

if __name__ == "__main__":
    expand_font("input.ufo", "output.ufo", "glyph_generator.pth")
