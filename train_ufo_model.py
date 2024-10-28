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
            
            # Draw lines between points for each contour
            for contour in glyph:
                points = [(point.x, point.y) for point in contour]
                # Add first point to close the contour
                points.append(points[0])
                
                for i in range(len(points) - 1):
                    x1, y1 = points[i]
                    x2, y2 = points[i + 1]
                    
                    # Normalize and scale coordinates
                    x1_norm = int((x1 - xmin) * 63 / (xmax - xmin))
                    y1_norm = int((y1 - ymin) * 63 / (ymax - ymin))
                    x2_norm = int((x2 - xmin) * 63 / (xmax - xmin))
                    y2_norm = int((y2 - ymin) * 63 / (ymax - ymin))
                    
                    # Draw line between points using Bresenham's algorithm
                    for x, y in self._bresenham_line(x1_norm, y1_norm, x2_norm, y2_norm):
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
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
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
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            inputs = batch['tensor'].to(device)  # Remove .unsqueeze(1) since it's done in __getitem__
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # Save the model
    torch.save(model.state_dict(), "glyph_generator.pth")

if __name__ == "__main__":
    train_model()
