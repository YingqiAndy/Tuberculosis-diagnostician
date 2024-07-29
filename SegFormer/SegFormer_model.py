import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor

class SegFormerModel(nn.Module):
    def __init__(self, n_classes):
        super(SegFormerModel, self).__init__()
        # Initialize the SegFormer model from Hugging Face's transformers library
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512"
        )
        # Update the number of labels (classes)
        self.segformer.config.num_labels = n_classes
        # Replace the classifier head to match the number of classes
        self.segformer.decode_head.classifier = nn.Conv2d(
            self.segformer.config.decoder_hidden_size,
            n_classes,
            kernel_size=(1, 1)
        )

    def forward(self, x):
        # Forward pass through the SegFormer model
        outputs = self.segformer(pixel_values=x)
        return outputs.logits

# Initialize model
n_classes = 3  # Set the example number of classes
model = SegFormerModel(n_classes=n_classes)

# Print model architecture
print(model)
