# AI-Image-Caption-Generator
This project employs the BLIP transformer model to generate descriptive captions for images. It preprocesses visual input, performs inference using a pretrained vision-language model, and presents results via an interactive Gradio interface for seamless user engagement.

#🚀 Features
✅ Pretrained BLIP model from Hugging Face

🧠 Zero-shot image captioning (no training required)

⚡ Automatic device selection (GPU/CPU)

🌐 Interactive Gradio web interface

📦 Lightweight and easy to deploy

#🛠️ Installation
bash
pip install torch torchvision transformers gradio
📸 Usage
Run the script in your Python environment or Google Colab:

python
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import gradio as gr

# Load model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def preprocess_image(image):
    return processor(images=image, return_tensors="pt").to(device)

def generate_caption(image):
    inputs = preprocess_image(image)
    with torch.no_grad():
        output = model.generate(**inputs)
    return processor.decode(output[0], skip_special_tokens=True)

def predict(image):
    return generate_caption(image)

interface = gr.Interface(fn=predict, inputs=gr.Image(type="pil"), outputs="text", title="AI Image Caption Generator")
interface.launch()
#🧪 Example
  Upload an image of a dog playing in the park, and the model might return: "A dog running through a grassy field on a sunny day."
  <img width="1633" height="810" alt="Screenshot 2025-09-05 113124" src="https://github.com/user-attachments/assets/5805e1f9-1fe2-4d9b-966a-0cd5c437a328" />


#📚 Model Details
  Model: Salesforce/blip-image-captioning-base

  Architecture: Vision-language transformer

  Source: Hugging Face Model Card
#Summary
        This project is an AI-driven image captioning system that utilizes the BLIP (Bootstrapped Language Image Pretraining) transformer model to generate descriptive text from visual input. By leveraging a pretrained vision-language model from Hugging Face, it processes images and produces natural language captions without requiring additional training. The integration of a Gradio interface allows users to interact with the model seamlessly, enabling real-time image uploads and instant caption generation in a user-friendly web environment.
