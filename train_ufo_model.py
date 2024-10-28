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
                # Convert glyph to tensor representation
                # We'll use a simplified 64x64 binary image representation
                glyph_array = self.glyph_to_array(glyph)
                self.glyphs_data.append({
                    'tensor': torch.FloatTensor(glyph_array),
                    'name': glyph_name,
                    'font': os.path.basename(ufo_path)
                })
    
    def glyph_to_array(self, glyph):
        # Convert glyph to 64x64 binary image
        # This is a simplified representation - you might want to improve this
        image = np.zeros((64, 64))
        if glyph.bounds:
            # Normalize coordinates to 64x64 grid
            xmin, ymin, xmax, ymax = glyph.bounds
            for contour in glyph:
                for point in contour:
                    x = int((point.x - xmin) * 63 / (xmax - xmin))
                    y = int((point.y - ymin) * 63 / (ymax - ymin))
                    if 0 <= x < 64 and 0 <= y < 64:
                        image[y, x] = 1
        return image

    def __len__(self):
        return len(self.glyphs_data)

    def __getitem__(self, idx):
        return self.glyphs_data[idx]

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
            inputs = batch['tensor'].unsqueeze(1).to(device)  # Add channel dimension
            
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
