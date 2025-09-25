import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

watermarked_image = None

def show_watermark_page():
   
    st.set_page_config(page_title="Image Watermark App", page_icon="🖼️", layout="wide")
    
    st.title("🖼️ Image Watermark App")
    st.markdown("Upload an image and add custom watermarks using OpenCV!")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=['png', 'jpg'],
        help="Upload the image you want to add a watermark to"
    )

    if uploaded_file is not None:

        # Read and display original image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        original_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), use_container_width=True)

        # Watermark options
        st.sidebar.header("Watermark Settings")
        watermark_type = st.sidebar.radio("Watermark Type", ["Text", "Image"])

        if watermark_type == "Text":
            render_watermark_text_UI(original_image, col2)
        else:
            render_watermark_image_UI(original_image, col2)

        print('here ', watermarked_image)


        # Download button
        if watermarked_image is not None:
            # Convert to bytes for download
            print('here')
            is_success, buffer = cv2.imencode(".png", watermarked_image)
            if is_success:
                st.download_button(
                    label="Download Watermarked Image",
                    data=buffer.tobytes(),
                    file_name="watermarked_image.png",
                    mime="image/png"
                )

def render_watermark_image_UI(original_image, col2):
    st.sidebar.subheader("Image Watermark Settings")
    watermark_file = st.sidebar.file_uploader(
    "Choose watermark image",
    type=['png', 'jpg'],
    help="Upload the image to use as watermark"
    )

    if watermark_file is not None:
        # Read watermark image
        wm_bytes = np.asarray(bytearray(watermark_file.read()), dtype=np.uint8)
        watermark_img = cv2.imdecode(wm_bytes, cv2.IMREAD_COLOR)
        scale = st.sidebar.slider("Scale", 0.1, 1.0, 0.3, 0.05)
        opacity = st.sidebar.slider("Opacity", 0.1, 1.0, 0.5, 0.1)
        position = st.sidebar.selectbox("Position",["Top Left", "Top Right", "Bottom Left", "Bottom Right", "Center"])
        
        # Apply image watermark
        global watermarked_image
        watermarked_image = add_image_watermark(original_image, watermark_img, position, scale, opacity)
        with col2:
            st.subheader("Watermarked Image")
            st.image(cv2.cvtColor(watermarked_image, cv2.COLOR_BGR2RGB), use_container_width =True)
    


def render_watermark_text_UI(original_image, col2):
    # Text watermark settings
    st.sidebar.subheader("Text Settings")
    watermark_text = st.sidebar.text_input("Watermark Text", value="Your Name")
    font_size = st.sidebar.slider("Font Size", 10, 100, 40)
    opacity = st.sidebar.slider("Opacity", 0.1, 1.0, 0.7, 0.1)
    
    # Color picker
    color = st.sidebar.color_picker("Text Color", "#94175C")
    # Convert hex to RGB
    color_rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
    
    position = st.sidebar.selectbox(
        "Position", 
        ["Top Left", "Top Right", "Bottom Left", "Bottom Right", "Center"]
    )
    
    if watermark_text:
        global watermarked_image
        # Apply text watermark
        watermarked_image = add_text_watermark(
            original_image, watermark_text, position, font_size, opacity, color_rgb
        )
        
        with col2:
            st.subheader("Watermarked Image")
            st.image(cv2.cvtColor(watermarked_image, cv2.COLOR_BGR2RGB), use_container_width =True)


   


def add_text_watermark(image, text, position, font_size, opacity, color):
    """Add text watermark to image using OpenCV with masks and logical operations"""
    # Create a copy of the original image
    result = image.copy()
    
    # Convert font size to cv2 scale (approximate conversion)
    font_scale = font_size / 30.0
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(font_size / 20))
    
    # Get text dimensions
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Get image dimensions
    img_height, img_width = image.shape[:2]
    
    # Calculate position
    if position == "Top Left":
        x, y = 20, 20 + text_height
    elif position == "Top Right":
        x, y = img_width - text_width - 20, 20 + text_height
    elif position == "Bottom Left":
        x, y = 20, img_height - 20
    elif position == "Bottom Right":
        x, y = img_width - text_width - 20, img_height - 20
    else:  # Center
        x, y = (img_width - text_width) // 2, (img_height + text_height) // 2
    
    # Create a black mask for the text
    text_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    
    # Draw white text on black mask
    cv2.putText(text_mask, text, (x, y), font, font_scale, 255, thickness, cv2.LINE_AA)
    
    # Create inverse mask (black text on white background)
    text_mask_inv = cv2.bitwise_not(text_mask)
    
    # Convert RGB color to BGR for OpenCV (color picker returns RGB)
    bgr_color = (color[2], color[1], color[0])
    
    # Create colored text image
    text_img = np.zeros_like(image)
    text_img[:] = bgr_color
    
    # Apply opacity to the text color
    text_img = (text_img * opacity).astype(np.uint8)
    
    # Extract text region using mask
    text_fg = cv2.bitwise_and(text_img, text_img, mask=text_mask)
    
    # Apply opacity to original image in text region
    original_dimmed = (result * (1 - opacity)).astype(np.uint8)
    text_bg = cv2.bitwise_and(original_dimmed, original_dimmed, mask=text_mask)
    
    # Extract non-text region from original image
    result_bg = cv2.bitwise_and(result, result, mask=text_mask_inv)
    
    # Combine text foreground with dimmed background in text area
    text_combined = cv2.add(text_fg, text_bg)
    
    # Combine final text region with non-text background
    result = cv2.add(result_bg, text_combined)
    
    return result

def add_image_watermark(main_image, watermark_image, position, scale, opacity):
    """Add image watermark to main image using masks and logical operations"""
    # Resize watermark
    h_main, w_main = main_image.shape[:2]
    h_wm, w_wm = watermark_image.shape[:2]
    
    # Scale watermark
    new_width = int(w_wm * scale)
    new_height = int(h_wm * scale)
    watermark_resized = cv2.resize(watermark_image, (new_width, new_height))
    
    # Calculate position
    if position == "Top Left":
        x, y = 20, 20
    elif position == "Top Right":
        x, y = w_main - new_width - 20, 20
    elif position == "Bottom Left":
        x, y = 20, h_main - new_height - 20
    elif position == "Bottom Right":
        x, y = w_main - new_width - 20, h_main - new_height - 20
    else:  # Center
        x, y = (w_main - new_width) // 2, (h_main - new_height) // 2
    
    # Ensure watermark fits within image bounds
    if x + new_width > w_main:
        x = w_main - new_width
    if y + new_height > h_main:
        y = h_main - new_height
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    
    # Adjust watermark size if it extends beyond bounds
    end_x = min(x + new_width, w_main)
    end_y = min(y + new_height, h_main)
    actual_width = end_x - x
    actual_height = end_y - y
    
    # Crop watermark if necessary
    watermark_cropped = watermark_resized[:actual_height, :actual_width]
    
    # Create a copy of the main image
    result = main_image.copy()
    
    # Create a mask for the watermark region (all white where watermark will be placed)
    watermark_mask = np.ones((actual_height, actual_width), dtype=np.uint8) * 255
    
    # Create inverse mask
    watermark_mask_inv = cv2.bitwise_not(watermark_mask)
    
    # Extract the ROI (Region of Interest) from the main image
    roi = result[y:end_y, x:end_x]
    
    # Apply opacity to watermark
    watermark_with_opacity = (watermark_cropped * opacity).astype(np.uint8)
    
    # Apply inverse opacity to original ROI
    roi_dimmed = (roi * (1 - opacity)).astype(np.uint8)
    
    # Extract watermark region using mask
    watermark_fg = cv2.bitwise_and(watermark_with_opacity, watermark_with_opacity, mask=watermark_mask)
    
    # Extract dimmed background region using mask
    roi_bg = cv2.bitwise_and(roi_dimmed, roi_dimmed, mask=watermark_mask)
    
    # Combine watermark with dimmed background
    watermark_combined = cv2.add(watermark_fg, roi_bg)
    
    # Place the combined result back into the main image
    result[y:end_y, x:end_x] = watermark_combined
    
    return result


        

