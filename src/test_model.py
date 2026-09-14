import torch
from model import EmotionCNN


# Create model
model = EmotionCNN(num_classes=8)

print(model)


# Create a fake input
# Shape:
# batch_size = 4
# channels = 1
# height = 64
# width = 130

x = torch.randn(4, 1, 64, 130)

# Pass through CNN
output = model(x)

print("\nInput shape:", x.shape)
print("Output shape:", output.shape)