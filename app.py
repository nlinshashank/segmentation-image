
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from scipy import ndimage as ndi
from skimage.segmentation import slic, mark_boundaries
from skimage.color import label2rgb
from skimage.morphology import remove_small_objects
from streamlit_image_coordinates import streamlit_image_coordinates
from io import BytesIO

st.set_page_config(
    page_title="Image Segmentation System",
    page_icon="🖼️",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #f4f7fb;
}
h1 {
    color: #183153;
    text-align: center;
}
.stButton > button {
    width: 100%;
    background-color: #176b4d;
    color: white;
    font-weight: bold;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

st.title("IMAGE SEGMENTATION SYSTEM")
st.caption("Thresholding • Region Growing • Region Merging • Region Splitting • Point, Line and Edge Detection")
st.divider()

if "seed_point" not in st.session_state:
    st.session_state.seed_point = None

if "upload_id" not in st.session_state:
    st.session_state.upload_id = None

if "result" not in st.session_state:
    st.session_state.result = None

if "result_title" not in st.session_state:
    st.session_state.result_title = ""

with st.sidebar:
    st.header("Segmentation Options")

    method = st.selectbox(
        "Select Technique",
        [
            "Thresholding",
            "Region Growing",
            "Region Merging",
            "Region Splitting",
            "Point Detection",
            "Line Detection",
            "Edge Detection"
        ]
    )

    threshold = st.slider(
        "Threshold Value",
        min_value=0,
        max_value=255,
        value=128,
        step=1
    )

    st.metric("Current Threshold", threshold)

    uploaded_file = st.file_uploader(
        "Choose Input Image",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"]
    )

    process_button = st.button("Process Image")

if uploaded_file is None:
    st.info("Upload an image using the sidebar to get started.")
    st.stop()

try:
    pil_image = Image.open(uploaded_file).convert("RGB")
    rgb = np.array(pil_image, dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    upload_id = f"{uploaded_file.name}_{uploaded_file.size}"

    if st.session_state.upload_id != upload_id:
        st.session_state.upload_id = upload_id
        st.session_state.seed_point = None
        st.session_state.result = None
        st.session_state.result_title = ""

except Exception as error:
    st.error(f"Unable to load image: {error}")
    st.stop()

st.subheader("Input Image")

if method == "Region Growing":
    st.write("Click a point inside the object to select the region-growing seed.")

    clicked = streamlit_image_coordinates(
        pil_image,
        key=f"seed_selector_{uploaded_file.name}_{uploaded_file.size}"
    )

    if clicked is not None:
        x = int(clicked["x"])
        y = int(clicked["y"])

        if 0 <= x < rgb.shape[1] and 0 <= y < rgb.shape[0]:
            st.session_state.seed_point = (x, y)

    if st.session_state.seed_point is not None:
        x, y = st.session_state.seed_point
        st.success(f"Seed point selected: X = {x}, Y = {y}")
else:
    st.image(rgb, caption="Original Image", use_container_width=True)

st.divider()


def to_uint8(image):
    if image.dtype == bool:
        return image.astype(np.uint8) * 255

    if np.issubdtype(image.dtype, np.floating):
        image = np.nan_to_num(image)
        if image.min() >= 0 and image.max() <= 1:
            image = image * 255

    return np.clip(image, 0, 255).astype(np.uint8)


def process_image(rgb, gray, method, threshold, seed_point):
    t = int(threshold)

    if method == "Thresholding":
        output = gray >= t
        return output, "Thresholding", "Binary mask using the selected intensity threshold."

    elif method == "Region Growing":
        if seed_point is None:
            raise ValueError(
                "Click inside the image to select a seed point first."
            )

        x, y = seed_point
        seed_value = int(gray[y, x])

        candidate = np.abs(
            gray.astype(np.int16) - seed_value
        ) <= t

        seed_mask = np.zeros(gray.shape, dtype=bool)
        seed_mask[y, x] = True

        output = ndi.binary_propagation(
            seed_mask,
            structure=np.ones((3, 3), dtype=bool),
            mask=candidate
        )

        return output, "Region Growing", "Connected pixels within the selected seed intensity tolerance."

    elif method == "Region Merging":
        labels = slic(
            rgb,
            n_segments=200,
            compactness=10,
            start_label=1,
            channel_axis=-1
        )

        output = mark_boundaries(
            rgb.astype(np.float32) / 255.0,
            labels,
            color=(1, 1, 1)
        )

        return output, "Region Merging - Superpixel Regions", (
            "Shows initial superpixel regions. Actual similarity-based region merging is not implemented."
        )

    elif method == "Region Splitting":
        output = gray >= t
        output = remove_small_objects(output, min_size=20)

        return output, "Region Splitting - Threshold Partition", (
            "Simplified threshold-based partitioning with small components removed; not recursive quadtree splitting."
        )

    elif method == "Point Detection":
        kernel = np.array([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ], dtype=np.float32)

        response = cv2.filter2D(
            gray.astype(np.float32),
            cv2.CV_32F,
            kernel
        )

        output = np.abs(response) > t

        return output, "Point Detection", "Laplacian-based detection of strong local intensity changes."

    elif method == "Line Detection":
        kernels = [
            np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], dtype=np.float32),
            np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]], dtype=np.float32),
            np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], dtype=np.float32),
            np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]], dtype=np.float32)
        ]

        responses = [
            np.abs(cv2.filter2D(
                gray.astype(np.float32),
                cv2.CV_32F,
                kernel
            ))
            for kernel in kernels
        ]

        response = np.maximum.reduce(responses)
        output = response > t

        return output, "Line Detection", "Directional filters detect horizontal, vertical and diagonal structures."

    elif method == "Edge Detection":
        sensitivity = max(0.01, min(0.99, t / 255.0))

        output = cv2.Canny(
            gray,
            threshold1=int(sensitivity * 0.4 * 255),
            threshold2=int(sensitivity * 255)
        )

        return output > 0, "Canny Edge Detection", (
            "Canny detector identifies intensity boundaries. The slider adjusts its thresholds."
        )


if process_button:
    try:
        with st.spinner("Processing image..."):
            result, result_title, description = process_image(
                rgb,
                gray,
                method,
                threshold,
                st.session_state.seed_point
            )

            st.session_state.result = to_uint8(result)
            st.session_state.result_title = result_title
            st.session_state.description = description
            st.session_state.processed_method = method

    except Exception as error:
        st.error(str(error))

if st.session_state.result is not None:
    st.subheader("Processing Output")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            rgb,
            caption="Original Image",
            use_container_width=True
        )

    with col2:
        st.image(
            st.session_state.result,
            caption=st.session_state.result_title,
            use_container_width=True,
            clamp=True
        )

    st.success(f"Completed: {st.session_state.result_title}")
    st.write(st.session_state.description)

    buffer = BytesIO()
    Image.fromarray(st.session_state.result).save(buffer, format="PNG")

    st.download_button(
        label="Download Segmentation Result",
        data=buffer.getvalue(),
        file_name="segmentation_result.png",
        mime="image/png"
    )
