import gradio as gr
from PIL import Image, ImageEnhance, ImageOps, ImageDraw, ImageFilter, ImageFont
import cv2
import numpy as np
import pytesseract
from rembg import remove
import tempfile
import os
import io
import pilgram
from diffusers import StableDiffusionPipeline
import torch

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"
stable_diffusion = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)


# Function to load an image
def load_image(image):
    return image

# Function to save an edited image to a file and return the path for downloading
def save_image(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            image.save(temp_file.name)
        return temp_file.name
    return None

# Function to convert RGBA to RGB
def rgba_to_rgb(image):
    if image.mode == 'RGBA':
        return image.convert('RGB')
    return image

# Function to extract text using Tesseract
def extract_text(image):
    if image is not None:
        return pytesseract.image_to_string(image)
    return "No text detected"

# Basic Image editing functions
def rotate_image(image, angle):
    return image.rotate(angle, expand=True)

def adjust_brightness(image, brightness):
    image = rgba_to_rgb(image)
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(brightness)

def adjust_contrast(image, contrast):
    image = rgba_to_rgb(image)
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(contrast)

def adjust_saturation(image, saturation):
    image = rgba_to_rgb(image)
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(saturation)

def adjust_hue(image, hue):
    image = rgba_to_rgb(image)
    img = np.array(image.convert('HSV'))
    img[..., 0] = (img[..., 0].astype(int) + int(hue * 180)) % 180
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_HSV2RGB))

def resize_image(image, scale):
    w, h = image.size
    return image.resize((int(w * scale), int(h * scale)))

def apply_blur(image, blur_radius):
    return image.filter(ImageFilter.GaussianBlur(blur_radius))

def apply_sharpening(image, sharpness):
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(sharpness)

def flip_image(image):
    return image.transpose(Image.FLIP_LEFT_RIGHT)

def apply_grayscale(image):
    image = rgba_to_rgb(image)
    return ImageOps.grayscale(image)

def apply_sepia(image):
    image = rgba_to_rgb(image)
    sepia_filter = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])
    img = np.array(image).dot(sepia_filter.T)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return Image.fromarray(img)

def apply_edge_detection(image):
    image = rgba_to_rgb(image)
    return image.filter(ImageFilter.FIND_EDGES)

def remove_background(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    result = remove(img_byte_arr)
    return Image.open(io.BytesIO(result))

def add_watermark(image, text):
    # Create a new transparent image for the watermark
    watermark = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(watermark)
    font = ImageFont.load_default()

    # Get text bounding box
    left, top, right, bottom = font.getbbox(text)
    text_width = right - left
    text_height = bottom - top

    # Calculate position
    x = image.width - text_width - 10
    y = image.height - text_height - 10

    # Draw the text on the watermark image
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 128))

    # Combine the original image with the watermark
    return Image.alpha_composite(image.convert('RGBA'), watermark)

def add_border(image, border_width, color):
    return ImageOps.expand(image, border=border_width, fill=color)

def apply_color_filter(image, color_filter):
    image = rgba_to_rgb(image)
    if color_filter == 'red':
        r, g, b = image.split()
        return Image.merge("RGB", (r, g.point(lambda p: 0), b.point(lambda p: 0)))
    elif color_filter == 'green':
        r, g, b = image.split()
        return Image.merge("RGB", (r.point(lambda p: 0), g, b.point(lambda p: 0)))
    elif color_filter == 'blue':
        r, g, b = image.split()
        return Image.merge("RGB", (r.point(lambda p: 0), g.point(lambda p: 0), b))
    else:
        return image

def apply_vignette(image, intensity):
    width, height = image.size
    vignette = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(vignette)
    draw.ellipse([(intensity, intensity), (width - intensity, height - intensity)], fill=255)
    vignette = vignette.resize(image.size)
    image = image.convert('RGBA')
    r, g, b, a = image.split()
    r = r.point(lambda i: i * vignette.getpixel((0, 0)) / 255)
    g = g.point(lambda i: i * vignette.getpixel((0, 0)) / 255)
    b = b.point(lambda i: i * vignette.getpixel((0, 0)) / 255)
    return Image.merge('RGBA', (r, g, b, a))

def apply_posterize(image, level):
    return ImageOps.posterize(image, level)

# New function to clear the image
def clear_image():
    return None, "Image cleared"

# New function to undo the last change
def undo_change(image, history):
    if len(history) > 1:
        history.pop()  # Remove the current image
        return history[-1], "Last change undone"
    return image, "No changes to undo"

# New function for Pilgram filters
def apply_pilgram_filter(image, filter_name):
    if image is not None and filter_name != 'none':
        filter_func = getattr(pilgram, filter_name, None)
        if filter_func:
            return filter_func(image)
    return image

def process_background(image, prompt):
    """Remove background, generate new background, and combine images."""
    # Remove background
    foreground = remove(image)

    # Generate background
    background_prompt = f"A scenic background based on: {prompt}"
    with torch.no_grad():
        background = stable_diffusion(background_prompt).images[0]

    # Combine images
    background = background.resize(foreground.size, Image.LANCZOS)
    combined = Image.new("RGBA", foreground.size)
    combined.paste(background, (0, 0))
    combined.paste(foreground, (0, 0), foreground)

    return combined
def face_swap(dest_img, target_img):
    if dest_img is None or target_img is None:
        return None, "Please provide both images for face swapping."

    # Convert PIL images to cv2 format
    dest_img = cv2.cvtColor(np.array(dest_img), cv2.COLOR_RGB2BGR)
    target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)

    dest_xyz_landmark_points, dest_landmark_points = face_utils.get_face_landmark(dest_img)
    dest_convexhull = cv2.convexHull(np.array(dest_landmark_points))

    target_img_hist_match = ImageProcessing.hist_match(target_img, dest_img)

    _, target_landmark_points = face_utils.get_face_landmark(target_img)
    target_convexhull = cv2.convexHull(np.array(target_landmark_points))

    new_face, result = face_utils.face_swapping(dest_img, dest_landmark_points, dest_xyz_landmark_points, dest_convexhull, target_img, target_landmark_points, target_convexhull, return_face=True)

    # Convert result back to PIL image
    result_pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    return result_pil, "Face swap completed successfully."

# Modified apply_edits function to include background processing
def apply_edits(image, pilgram_filter, rotate, brightness, contrast, saturation, hue, scale, blur, sharpness, flip, grayscale, sepia, edge, remove_bg, watermark_text, border_width, border_color, color_filter, vignette, posterize, bg_prompt, face_swap_image, apply_face_swap):
    if image is not None:
        # Apply face swap if requested
        if apply_face_swap and face_swap_image is not None:
            image, _ = face_swap(image, face_swap_image)

        # Apply Pilgram filter first
        if pilgram_filter != 'none':
            filter_func = getattr(pilgram, pilgram_filter, None)
            if filter_func:
                image = filter_func(image)

        # Perform other image edits
        edited_image = image.copy()
        edited_image = rotate_image(edited_image, rotate)
        edited_image = adjust_brightness(edited_image, brightness)
        edited_image = adjust_contrast(edited_image, contrast)
        edited_image = adjust_saturation(edited_image, saturation)
        edited_image = adjust_hue(edited_image, hue)
        edited_image = resize_image(edited_image, scale)
        edited_image = apply_blur(edited_image, blur)
        edited_image = apply_sharpening(edited_image, sharpness)
        if flip:
            edited_image = flip_image(edited_image)
        if grayscale:
            edited_image = apply_grayscale(edited_image)
        if sepia:
            edited_image = apply_sepia(edited_image)
        if edge:
            edited_image = apply_edge_detection(edited_image)
        if remove_bg:
            edited_image = remove_background(edited_image)
        if watermark_text:
            edited_image = add_watermark(edited_image, watermark_text)
        if border_width > 0:
            edited_image = add_border(edited_image, border_width, border_color)
        if color_filter != 'none':
            edited_image = apply_color_filter(edited_image, color_filter)
        if vignette > 0:
            edited_image = apply_vignette(edited_image, vignette)
        if posterize > 1:
            edited_image = apply_posterize(edited_image, posterize)

        # Apply background processing if prompt is provided
        if bg_prompt:
            edited_image = process_background(edited_image, bg_prompt)

        return edited_image
    return None
pilgram_filters = [
    "1977", "aden", "brannan", "brooklyn", "clarendon", "earlybird",
    "gingham", "hudson", "inkwell", "kelvin", "lark", "lofi", "maven",
    "mayfair", "moon", "nashville", "perpetua", "reyes", "rise",
    "slumber", "stinson", "toaster", "valencia", "walden", "willow", "xpro2"
]
# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# Advanced Image Editor with Face Swap")

    with gr.Row():
        with gr.Column(scale=2):
            image_input = gr.Image(type="pil", label="Upload/Edit Image")
            with gr.Row():
                undo_button = gr.Button("Undo")
                clear_button = gr.Button("Clear Image")

        with gr.Column(scale=1):
            with gr.Accordion("Pilgram Filters"):
                pilgram_filter_dropdown = gr.Dropdown(choices=pilgram_filters, label="Select a Pilgram Filter", value='none')

            with gr.Accordion("Basic Effects", open=False):
                rotate_slider = gr.Slider(minimum=0, maximum=360, label="Rotate")
                brightness_slider = gr.Slider(minimum=0.5, maximum=2.0, step=0.1, label="Brightness", value=1.0)
                contrast_slider = gr.Slider(minimum=0.5, maximum=2.0, step=0.1, label="Contrast", value=1.0)
                saturation_slider = gr.Slider(minimum=0.5, maximum=2.0, step=0.1, label="Saturation", value=1.0)
                hue_slider = gr.Slider(minimum=-1.0, maximum=1.0, step=0.1, label="Hue", value=0)
                scale_slider = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, label="Scale", value=1.0)
                flip_checkbox = gr.Checkbox(label="Flip Image")
                grayscale_checkbox = gr.Checkbox(label="Grayscale")
                sepia_checkbox = gr.Checkbox(label="Sepia")

            with gr.Accordion("Advanced Effects", open=False):
                blur_slider = gr.Slider(minimum=0, maximum=10, step=0.1, label="Blur")
                sharpness_slider = gr.Slider(minimum=0.5, maximum=2.0, step=0.1, label="Sharpness", value=1.0)
                edge_checkbox = gr.Checkbox(label="Edge Detection")
                remove_bg_checkbox = gr.Checkbox(label="Remove Background")
                watermark_textbox = gr.Textbox(label="Add Watermark Text")
                border_width_slider = gr.Slider(minimum=0, maximum=20, step=1, label="Border Width")
                border_color_picker = gr.ColorPicker(label="Border Color")
                color_filter_dropdown = gr.Dropdown(choices=["none", "red", "green", "blue"], label="Color Filter")
                vignette_slider = gr.Slider(minimum=0, maximum=100, step=5, label="Vignette")
                posterize_slider = gr.Slider(minimum=1, maximum=8, step=1, label="Posterize", value=8)

            with gr.Accordion("Background Editing", open=False):
                bg_prompt_input = gr.Textbox(label="Enter AI Prompt for Background", placeholder="e.g., sunset over the mountains")

            with gr.Accordion("Face Swap", open=False):
                face_swap_image = gr.Image(type="pil", label="Upload Face Image for Swap")
                apply_face_swap = gr.Checkbox(label="Apply Face Swap")

    # List to store edit history
    edit_history = []

    def update_image(*args):
        global edit_history
        edited_image = apply_edits(*args)
        if edited_image is not None:
            edit_history.append(edited_image)
            if len(edit_history) > 10:  # Keep only last 10 edits
                edit_history.pop(0)
        return edited_image

    def undo_edit():
        global edit_history
        if len(edit_history) > 1:
            edit_history.pop()  # Remove the current state
            return edit_history[-1]  # Return the previous state
        elif len(edit_history) == 1:
            return edit_history[0]  # Return the original image
        return None

    def clear_image():
        global edit_history
        edit_history = []
        return None

    # Connect all inputs to the update_image function
    all_inputs = [
        image_input, pilgram_filter_dropdown,
        rotate_slider, brightness_slider, contrast_slider, saturation_slider, hue_slider, scale_slider,
        blur_slider, sharpness_slider, flip_checkbox, grayscale_checkbox, sepia_checkbox,
        edge_checkbox, remove_bg_checkbox, watermark_textbox, border_width_slider,
        border_color_picker, color_filter_dropdown, vignette_slider, posterize_slider,
        bg_prompt_input, face_swap_image, apply_face_swap
    ]

    for input_component in all_inputs[1:]:  # Exclude image_input
        input_component.change(update_image, inputs=all_inputs, outputs=image_input)

    undo_button.click(undo_edit, outputs=image_input)
    clear_button.click(clear_image, outputs=image_input)

    # Add download button
    gr.Markdown("## Download Edited Image")
    gr.File(label="Click to Download", value=lambda: save_image(image_input.value))

demo.launch()