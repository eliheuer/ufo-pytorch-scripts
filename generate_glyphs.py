import torch
from train_ufo_model import GlyphGenerator
from defcon import Font
import numpy as np

def array_to_glyph(array, glyph):
    # Convert the generated array back to glyph contours
    # This is a simplified conversion - you'll want to improve this
    threshold = 0.5
    points = []
    
    for y in range(array.shape[0]):
        for x in range(array.shape[1]):
            if array[y, x] > threshold:
                # Scale back to font coordinates
                scaled_x = x * (glyph.width / 64)
                scaled_y = y * (glyph.height / 64)
                points.append((scaled_x, scaled_y))
    
    # Create contours from points
    if points:
        pen = glyph.getPen()
        pen.moveTo(points[0])
        for point in points[1:]:
            pen.lineTo(point)
        pen.closePath()

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

    with torch.no_grad():
        for glyph_name in new_glyphs:
            if glyph_name not in output_font:
                # Use similar existing glyph as reference
                base_glyph = input_font['a']  # Use appropriate base glyph
                
                # Create tensor from base glyph
                glyph_array = np.zeros((64, 64))  # Similar to training conversion
                tensor = torch.FloatTensor(glyph_array).unsqueeze(0).unsqueeze(0).to(device)
                
                # Generate new glyph
                generated = model(tensor)
                generated_array = generated.squeeze().cpu().numpy()
                
                # Create new glyph in output font
                new_glyph = output_font.newGlyph(glyph_name)
                array_to_glyph(generated_array, new_glyph)

    # Save the expanded font
    output_font.save(output_ufo_path)

if __name__ == "__main__":
    expand_font("input.ufo", "output.ufo", "glyph_generator.pth")
