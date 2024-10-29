import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from defcon import Font
import numpy as np
from pathlib import Path

# Custom dataset for UFO fonts
class UFODataset(Dataset):
    def __init__(self, root_dir):
        self.ufo_paths = []
        for path in Path(root_dir).rglob("*.ufo"):
            self.ufo_paths.append(str(path))
        
        self.glyphs_data = []
        self.process_fonts()
    
    def process_fonts(self):
        for ufo_path in self.ufo_paths:
            font = Font(ufo_path)
            for glyph_name in font.keys():
                glyph = font[glyph_name]
                glyph_array = self.glyph_to_array(glyph)
                
                # Convert None unicode to -1 or another default value
                unicode_value = -1 if glyph.unicode is None else glyph.unicode
                
                # Add unicode and outline data to the dataset
                self.glyphs_data.append({
                    'tensor': torch.FloatTensor(glyph_array),
                    'name': glyph_name,
                    'font': os.path.basename(ufo_path),
                    'unicode': unicode_value,  # Use the converted unicode value
                    'width': glyph.width,
                    'outline': [(point.x, point.y, point.segmentType) 
                              for contour in glyph
                              for point in contour]
                })
    
    def glyph_to_array(self, glyph):
        # Create a 64x64 binary image
        image = np.zeros((64, 64))
        
        if glyph.bounds:
            xmin, ymin, xmax, ymax = glyph.bounds
            # Handle zero-size bounds
            if xmax == xmin:
                xmax = xmin + 1
            if ymax == ymin:
                ymax = ymin + 1
            
            # Draw filled contours instead of just lines
            for contour in glyph:
                points = [(point.x, point.y) for point in contour]
                # Convert points to numpy array format
                points_array = np.array(points)
                # Scale points to fit in our 64x64 grid
                scaled_x = ((points_array[:, 0] - xmin) * 63 / (xmax - xmin)).astype(int)
                scaled_y = ((points_array[:, 1] - ymin) * 63 / (ymax - ymin)).astype(int)
                
                # Create polygon mask
                polygon = list(zip(scaled_y, scaled_x))
                from skimage.draw import polygon as draw_polygon
                rr, cc = draw_polygon(scaled_y, scaled_x)
                valid_mask = (rr >= 0) & (rr < 64) & (cc >= 0) & (cc < 64)
                image[rr[valid_mask], cc[valid_mask]] = 1
                
                # Draw edges to ensure thin features are captured
                for i in range(len(points) - 1):
                    x1, y1 = scaled_x[i], scaled_y[i]
                    x2, y2 = scaled_x[i + 1], scaled_y[i + 1]
                    for x, y in self._bresenham_line(x1, y1, x2, y2):
                        if 0 <= x < 64 and 0 <= y < 64:
                            image[y, x] = 1
        
        return image

    def _bresenham_line(self, x0, y0, x1, y1):
        """Implementation of Bresenham's line algorithm"""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = 1 if x1 > x0 else -1
        sy = 1 if y1 > y0 else -1
        
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                points.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                points.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
            
        points.append((x, y))
        return points

    def __len__(self):
        return len(self.glyphs_data)

    def __getitem__(self, idx):
        return {
            'tensor': self.glyphs_data[idx]['tensor'].unsqueeze(0),
            'unicode': self.glyphs_data[idx]['unicode'],
            'width': self.glyphs_data[idx]['width'],
            # Exclude 'outline', 'name', and 'font' as they have variable lengths
        }

# Define the GlyphGenerator model
class GlyphGenerator(nn.Module):
    def __init__(self):
        super(GlyphGenerator, self).__init__()
        # Increase network capacity
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train_model():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize dataset and dataloader
    dataset = UFODataset(".")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize model
    model = GlyphGenerator().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 500  # Increase epochs
    best_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            inputs = batch['tensor'].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "glyph_generator_best.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print("Early stopping triggered")
            break

if __name__ == "__main__":
    train_model()
