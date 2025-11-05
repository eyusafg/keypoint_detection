import numpy as np
import cv2
import os

def overlay_single_channel_to_red_channel(single_channel_image, rgb_image):
    """
    Overlays a single-channel image onto the red channel of an RGB image.
    
    Parameters:
    - single_channel_image: A 2D NumPy array representing the single-channel image.
    - rgb_image: A 3D NumPy array representing the RGB image.
    
    Returns:
    - A 3D NumPy array with the single-channel image overlaid onto the red channel.
    """
    # Ensure the dimensions match
    if single_channel_image.shape != rgb_image.shape[:2]:
        raise ValueError("The dimensions of the single-channel image and the RGB image must match.")
    
    # Zero out the red channel
    rgb_image[:, :, 0] = 0
    
    # Overlay the single-channel image onto the red channel
    rgb_image[:, :, 0] = single_channel_image
    
    return rgb_image

heatmap_path = os.listdir(r'pred\heatmap')
roi_path = os.listdir(r'pred\roi_image')
for i, heatmap_name in enumerate(heatmap_path):
    if heatmap_name.endswith('.png'):
        heatmap_path = os.path.join(r'pred\heatmap', heatmap_name)
        roi_name = roi_path[i]
        print('heatmap_name', heatmap_name)
        print('roi_name', roi_name)
        roi_path_ = os.path.join(r'pred\roi_image', roi_name)
        heatmap = cv2.imread(heatmap_path, cv2.IMREAD_GRAYSCALE)
        roi_image = cv2.imread(roi_path_)
        roi_image = cv2.resize(roi_image, (96, 96))
        heatmap = np.uint8(np.clip(heatmap * 255, 0, 255))

        # Ensure roi_image is in the correct format (H, W, 3)
        assert roi_image.ndim == 3 and roi_image.shape[2] == 3, "roi_image must be a 3-channel image."

        # Overlay the heatmap onto the red channel of the roi_image
        roi_image_with_heatmap = overlay_single_channel_to_red_channel(heatmap, roi_image)
        
        cv2.namedWindow('roi_with_heatmap', cv2.WINDOW_NORMAL)
        cv2.imshow('roi_with_heatmap', roi_image_with_heatmap)
        cv2.waitKey(0)
        # # Save the resulting image
        # cv2.imwrite('roi_with_heatmap.png', roi_image_with_heatmap)
