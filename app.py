import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np

# Paste your model class here
class CVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28*28 + 10, 400)
        self.fc21 = torch.nn.Linear(400, 20)
        self.fc22 = torch.nn.Linear(400, 20)
        self.fc3 = torch.nn.Linear(20 + 10, 400)
        self.fc4 = torch.nn.Linear(400, 28*28)

    def encode(self, x, y):
        xy = torch.cat([x, y], dim=1)
        h1 = torch.relu(self.fc1(xy))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        zy = torch.cat([z, y], dim=1)
        h3 = torch.relu(self.fc3(zy))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar

def one_hot(labels, num_classes=10):
    return F.one_hot(labels, num_classes).float()

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CVAE().to(device)
    model.load_state_dict(torch.load("cvae_mnist.pth", map_location=device))
    model.eval()
    return model, device

def generate_images(model, device, digit, num_samples=5):
    with torch.no_grad():
        y = torch.tensor([digit]*num_samples).to(device)
        y_onehot = one_hot(y).to(device)
        z = torch.randn(num_samples, 20).to(device)
        samples = model.decode(z, y_onehot).cpu()
        samples = samples.view(-1, 28, 28).numpy()
    return samples

def main():
    st.title("Handwritten Digit Generator")
    digit = st.slider("Select digit", 0, 9, 0)
    model, device = load_model()
    images = generate_images(model, device, digit, 5)

    cols = st.columns(5)
    for i, img in enumerate(images):
        with cols[i]:
            st.image(img, width=100, clamp=True)

if __name__ == "__main__":
    main()
