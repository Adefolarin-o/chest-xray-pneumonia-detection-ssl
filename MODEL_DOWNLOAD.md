# Pneumonia Detection Model Files

This folder contains the pretrained and fine-tuned models for pneumonia detection using self-supervised learning.

### Download Instructions:
1. **Click the link below** to access the models:
   pneumonia _detection https://drive.google.com/drive/folders/18ZVlHUDXu0JdfUDbS66muOyjwOE1inJ6?usp=drive_link

2. **Download the models** and place them in your working directory.

3. **Load the models** in Python:
   ```python
   import torch
   from model import init_ssl_model  # Adjust based on your code
   
   ssl_model = init_ssl_model()
   ssl_model.load_state_dict(torch.load("path/to/best_ssl_model.pth"))
   ssl_model.eval()
