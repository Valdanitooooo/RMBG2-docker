import gradio as gr
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
import requests
from io import BytesIO
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from typing import Optional
import uvicorn
from contextlib import asynccontextmanager

# Initialize FastAPI application
app = FastAPI()

# Global loading model
model = None
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"


def load_model():
    """
    Load the model and set up the device
    """
    global model, device
    if model is None or device is None:
        model = AutoModelForImageSegmentation.from_pretrained('/models/RMBG-2.0', trust_remote_code=True)
        device = torch.device(device)
        model.to(device)
        model.eval()


def process_image(input_image, model, device):
    """
    Process the image and remove the background
    Parameters:
        input_image:  The input PIL image
        model:  Pre trained model
        device:  Computing devices
    return:
        PIL image with transparent background
    """
    if input_image is None:
        return None

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Convert to tensor and move to device
    img_tensor = transform(input_image).unsqueeze(0).to(device)

    # Predict segmentation mask
    with torch.no_grad():
        pred = model(img_tensor)[-1].sigmoid().cpu()

    # Convert mask to PIL image
    mask = transforms.ToPILImage()(pred[0].squeeze())
    mask = mask.resize(input_image.size)

    # Create a new image with a transparent background
    result = Image.new('RGBA', input_image.size, (0, 0, 0, 0))
    # Convert input image to RGBA mode
    input_image = input_image.convert('RGBA')
    # Merge images using masks
    result.paste(input_image, (0, 0), mask=mask)

    return result


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/api/remove-bg")
async def remove_bg_api(file: Optional[UploadFile] = None, url: Optional[str] = None):
    """
    API endpoint for removing image background
    Parameters:
        file:  Upload image files
        url:  The URL of the image
    return:
        Remove background image
    """
    if url:
        try:
            response = requests.get(url)
            input_image = Image.open(BytesIO(response.content)).convert('RGB')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Unable to load image from URL: {e}")
    elif file:
        try:
            input_image = Image.open(BytesIO(await file.read())).convert('RGB')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Unable to read uploaded file: {e}")
    else:
        raise HTTPException(status_code=400, detail="No image file or URL provided")

    result_image = process_image(input_image, model, device)

    # Save the results to BytesIO for streaming response
    output_buffer = BytesIO()
    result_image.save(output_buffer, format="PNG")
    output_buffer.seek(0)

    return StreamingResponse(output_buffer, media_type="image/png")


def create_gradio_interface():
    """
    Create Gradio interface for removing background
    """

    def remove_bg(image=None, url=None):
        if url:
            try:
                response = requests.get(url)
                input_image = Image.open(BytesIO(response.content)).convert('RGB')
            except:
                return None
        elif image is not None:
            input_image = Image.fromarray(image).convert('RGB')
        else:
            return None

        return process_image(input_image, model, device)

    demo = gr.Interface(
        title="RMBG-2.0 App",
        description="Upload image or provide URL to remove background",
        allow_flagging="never",
        fn=remove_bg,
        inputs=[
            gr.Image(label="Upload image", type="numpy"),
            gr.Textbox(label="Image URL")
        ],
        outputs=gr.Image(label="Output", type="pil"),
    )
    return demo


# Create Gradio interface
gradio_app = create_gradio_interface()

# Mount Gradio application into FastAPI application
app = gr.mount_gradio_app(app, gradio_app, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
