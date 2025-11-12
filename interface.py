import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv_block1 = nn.Sequential(
            # Input: 3 channels (RGB), Output: 16 feature maps
            # Kernel size 3x3 is standard. Padding=1 keeps the image size the same.
            nn.Conv2d(
                in_channels=3, out_channels=16, kernel_size=3, padding=1
            ),  # output: 16x128x128
            nn.ReLU(),  # Activation function to introduce non-linearity
            nn.MaxPool2d(
                kernel_size=2, stride=2
            ),  # Downsamples the image by a factor of 2
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # --- Classifier Head ---

        self.classifier = nn.Sequential(
            nn.Flatten(),  # Flattens the 32x32x32 feature map into a single vector
            nn.Linear(in_features=32 * 32 * 32, out_features=128),
            nn.ReLU(),
            nn.Linear(
                in_features=128, out_features=num_classes
            ),  # Output layer has 2 neurons for our 2 classes
        )

    def forward(self, x):
        # This defines the forward pass: how data flows through the layers.
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.classifier(x)
        return x
    
model = SimpleCNN()
state_dict = torch.load("simple_cnn_skin_lesion.pt", map_location=torch.device("cpu"))
model.load_state_dict(state_dict)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def predict(image):
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][pred_class].item()

        label = "Malignant" if pred_class == 1 else "Benign"

        return f"{label} (Confidence: {confidence:.2f})"

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Skin Lesion Image"),
    outputs=gr.Textbox(label="Prediction"),
    title="Skin Lesion Classifier",
    description="Upload an image of skin lesion to classify it as benign or malignant"
)

demo.launch(share=True)
