import streamlit as st
import numpy as np
import cv2
import fitz  # PyMuPDF
import io
from datetime import datetime
import base64


def show_pdf_sign_page():
    st.set_page_config(
        page_title="PDF Digital Signature App", 
        page_icon="📄", 
        layout="wide"
    )

    st.title("📄 PDF Digital Signature App")
    st.markdown("Upload a PDF and add your digital signature using OpenCV!")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📄 Upload PDF Document")
        pdf_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload the PDF document to sign"
        )
    
    with col2:
        st.subheader("✍️ Upload Signature Image")
        signature_file = st.file_uploader(
            "Choose signature image",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            help="Upload your signature image (preferably on white background)"
        )


    if pdf_file is None or signature_file is None:
        st.info("👆 Upload both a PDF document and a signature image to get started!")
        return

    # Read files
    pdf_bytes = pdf_file.read()
    sig_bytes = np.asarray(bytearray(signature_file.read()), dtype=np.uint8)
    signature_img = cv2.imdecode(sig_bytes, cv2.IMREAD_COLOR)
    
    # Signature settings
    st.sidebar.header("Signature Settings")
    
    # Page selection
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(doc)
        doc.close()
        
        if total_pages > 1:
            page_option = st.sidebar.radio(
                "Apply signature to:",
                ["All pages", "Specific page", "Last page only"]
            )
            
            if page_option == "Specific page":
                selected_page = st.sidebar.number_input(
                    "Page number", 
                    min_value=1, 
                    max_value=total_pages, 
                    value=1
                ) - 1  # Convert to 0-based index
            elif page_option == "Last page only":
                selected_page = total_pages - 1
            else:
                selected_page = None  # All pages
        else:
            selected_page = 0
            page_option = "Single page"
        
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return
    
    # Signature customization
    position = st.sidebar.selectbox(
        "Position",
        ["Bottom Right", "Bottom Left", "Top Right", "Top Left", "Center"]
    )
    
    scale = st.sidebar.slider("Size", 0.1, 1.0, 0.3, 0.05)
    opacity = st.sidebar.slider("Opacity", 0.3, 1.0, 0.8, 0.1)
    
    # Add signature info
    add_info = st.sidebar.checkbox("Add signature info", value=True)
    if add_info:
        signer_name = st.sidebar.text_input("Signer Name", value="Your Name")
        add_date = st.sidebar.checkbox("Add date", value=True)
    
    # Preview section
    if st.sidebar.button("Generate Preview", type="primary"):
        try:
            with st.spinner("Processing PDF..."):
                # Convert PDF to images
                page_images = pdf_to_images(pdf_bytes)
                
                # Apply signature to selected pages
                signed_images = []
                preview_pages = []
                
                for i, page_img in enumerate(page_images):
                    if selected_page is None or i == selected_page:
                        # Add signature to this page
                        signed_img = add_signature_to_image(
                            page_img, signature_img, position, scale, opacity
                        )
                        
                        # Add text info if requested
                        if add_info and signer_name:
                            # Calculate text position near signature
                            h, w = signed_img.shape[:2]
                            if position == "Bottom Right":
                                text_x, text_y = w - 300, h - 30
                            elif position == "Bottom Left":
                                text_x, text_y = 50, h - 30
                            elif position == "Top Right":
                                text_x, text_y = w - 300, 100
                            elif position == "Top Left":
                                text_x, text_y = 50, 100
                            else:  # Center
                                text_x, text_y = w//2 - 100, h//2 + 100
                            
                            # Add signer name
                            cv2.putText(signed_img, f"Signed by: {signer_name}", 
                                    (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.6, (0, 0, 0), 1, cv2.LINE_AA)
                            
                            # Add date if requested
                            if add_date:
                                date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
                                cv2.putText(signed_img, f"Date: {date_str}", 
                                        (text_x, text_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 
                                        0.5, (0, 0, 0), 1, cv2.LINE_AA)
                        
                        signed_images.append(signed_img)
                        
                        # Add to preview (limit to first few pages for display)
                        if len(preview_pages) < 3:
                            preview_pages.append((i+1, signed_img))
                    else:
                        signed_images.append(page_img)
            
            
            # Convert back to PDF
            with st.spinner("Generating signed PDF..."):
                signed_pdf_bytes = images_to_pdf(signed_images)

            st.pdf(signed_pdf_bytes, height=500, key=None)
            


            # Download button
            st.markdown("---")
            st.download_button(
                label="📥 Download Signed PDF",
                data=signed_pdf_bytes,
                file_name=f"signed_{pdf_file.name}",
                mime="application/pdf",
                type="primary"
            )
            
            st.success("✅ PDF signed successfully!")
            
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            st.info("Make sure your PDF is not password-protected and your signature image has a clear background.")



def pdf_to_images(pdf_bytes):
    """Convert PDF pages to images"""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        # Get page as image (higher DPI for better quality)
        mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        
        # Convert to OpenCV format
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        images.append(img)
    
    doc.close()
    return images 

def preprocess_signature_image(signature_img):
    """Preprocess signature image for better integration"""
    # Convert to grayscale
    gray = cv2.cvtColor(signature_img, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to make background white and signature dark
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # Invert if needed (signature should be dark on white background)
    if np.mean(thresh) < 127:  # If mostly dark
        thresh = cv2.bitwise_not(thresh)
    
    # Convert back to BGR
    processed = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    
    # Create transparency mask (white pixels become transparent)
    mask = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    
    return processed, mask

def images_to_pdf(images):
    """Convert images back to PDF"""
    doc = fitz.open()
    
    for img in images:
        # Convert OpenCV image to bytes
        is_success, buffer = cv2.imencode(".png", img)
        if is_success:
            img_bytes = buffer.tobytes()
            
            # Create new page with image
            img_doc = fitz.open(stream=img_bytes, filetype="png")
            page = img_doc[0]
            
            # Get image dimensions and create page
            rect = page.rect
            pdf_page = doc.new_page(width=rect.width, height=rect.height)
            pdf_page.insert_image(rect, stream=img_bytes)
            
            img_doc.close()
    
    # Get PDF bytes
    pdf_bytes = doc.tobytes()
    doc.close()
    
    return pdf_bytes

def add_signature_to_image(page_img, signature_img, position, scale, opacity):
    """Add signature to a page image using CV2 with masks"""
    # Get dimensions
    page_h, page_w = page_img.shape[:2]
    sig_h, sig_w = signature_img.shape[:2]
    
    # Scale signature
    new_width = int(sig_w * scale)
    new_height = int(sig_h * scale)
    signature_resized = cv2.resize(signature_img, (new_width, new_height))
    
    # Preprocess signature
    signature_processed, signature_mask = preprocess_signature_image(signature_resized)
    
    # Calculate position
    margin = 50
    if position == "Bottom Right":
        x, y = page_w - new_width - margin, page_h - new_height - margin
    elif position == "Bottom Left":
        x, y = margin, page_h - new_height - margin
    elif position == "Top Right":
        x, y = page_w - new_width - margin, margin
    elif position == "Top Left":
        x, y = margin, margin
    else:  # Center
        x, y = (page_w - new_width) // 2, (page_h - new_height) // 2
    
    # Ensure signature fits within bounds
    x = max(0, min(x, page_w - new_width))
    y = max(0, min(y, page_h - new_height))
    
    # Adjust if extends beyond bounds
    end_x = min(x + new_width, page_w)
    end_y = min(y + new_height, page_h)
    actual_width = end_x - x
    actual_height = end_y - y
    
    # Crop signature if necessary
    signature_cropped = signature_processed[:actual_height, :actual_width]
    mask_cropped = signature_mask[:actual_height, :actual_width]
    
    # Create result image
    result = page_img.copy()
    
    # Extract ROI
    roi = result[y:end_y, x:end_x]
    
    # Create binary mask (white areas become transparent)
    gray_mask = cv2.cvtColor(mask_cropped, cv2.COLOR_BGR2GRAY)
    binary_mask = cv2.threshold(gray_mask, 240, 255, cv2.THRESH_BINARY)[1]
    signature_mask_final = cv2.bitwise_not(binary_mask)  # Invert: signature areas are white
    
    # Apply opacity to signature
    signature_with_opacity = (signature_cropped * opacity).astype(np.uint8)
    
    # Apply masks using logical operations
    # Extract signature region (non-white areas)
    signature_fg = cv2.bitwise_and(signature_with_opacity, signature_with_opacity, mask=signature_mask_final)
    
    # Extract background region (white areas of signature)
    roi_bg = cv2.bitwise_and(roi, roi, mask=binary_mask)
    
    # Dim the background under signature
    roi_under_sig = cv2.bitwise_and(roi, roi, mask=signature_mask_final)
    roi_dimmed = (roi_under_sig * (1 - opacity * 0.3)).astype(np.uint8)  # Slight dimming
    
    # Combine all regions
    final_roi = cv2.add(roi_bg, cv2.add(signature_fg, roi_dimmed))
    
    # Place back into result
    result[y:end_y, x:end_x] = final_roi
    
    return result 