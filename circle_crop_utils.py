import numpy as np
import torch
import torch.nn.functional as F
from skimage import measure


def detect_and_crop_circle(hsi_image, octa_image=None, padding=10):
    """
    Detect the circular field of view in an HSI image and crop both HSI and OCTA
    images to that region with some padding. Always applies a circular mask to the OCTA image.

    Args:
        hsi_image (torch.Tensor): HSI image tensor of shape [C, H, W] or [B, C, H, W]
        octa_image (torch.Tensor, optional): OCTA image tensor of shape [C, H, W] or [B, C, H, W]
        padding (int): Padding around the detected circle (in pixels)

    Returns:
        tuple: Cropped HSI image and OCTA image (if provided)
    """
    # Handle batch dimension if present
    has_batch_dim = len(hsi_image.shape) == 4
    batch_size = hsi_image.shape[0] if has_batch_dim else 1

    # Process each image in the batch
    cropped_hsi_list = []
    cropped_octa_list = []

    # Store circle parameters for masking
    circle_params = []

    for b in range(batch_size):
        # Get a single image from the batch if necessary
        if has_batch_dim:
            hsi_single = hsi_image[b]
            octa_single = octa_image[b] if octa_image is not None else None
        else:
            hsi_single = hsi_image
            octa_single = octa_image

        # Get dimensions
        _, H, W = hsi_single.shape

        # Create a binary mask from the HSI image (average across spectral channels)
        mask = hsi_single.mean(dim=0).cpu().numpy()

        # Normalize to [0, 1] if not already
        if mask.max() > 1.0:
            mask = mask / mask.max()

        # Threshold to create binary mask
        # Otsu thresholding might be more robust, but we'll use a simple threshold for now
        threshold = 0.1  # This may need adjustment based on your data
        binary_mask = (mask > threshold).astype(np.uint8)

        # Find contours in the binary mask
        contours = measure.find_contours(binary_mask, 0.5)

        if not contours:
            print("No contours found, using original image")
            cropped_hsi_list.append(hsi_single)
            if octa_single is not None:
                cropped_octa_list.append(octa_single)
            # Add placeholder circle parameters for the whole image
            circle_params.append({"center_x": W // 2, "center_y": H // 2, "radius": min(H, W) // 2})
            continue

        # Find the largest contour (should be the circle)
        largest_contour = max(contours, key=len)

        # Find the bounding box of the contour
        min_y, min_x = np.floor(np.min(largest_contour, axis=0)).astype(int)
        max_y, max_x = np.ceil(np.max(largest_contour, axis=0)).astype(int)

        # Calculate the center and radius of the circle
        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2
        radius = max((max_x - min_x) // 2, (max_y - min_y) // 2)

        # Store circle parameters relative to cropped image
        # Adjust parameters to account for cropping
        adjusted_center_x = center_x - min_x + padding
        adjusted_center_y = center_y - min_y + padding

        # Add padding
        min_y = max(0, min_y - padding)
        min_x = max(0, min_x - padding)
        max_y = min(H, max_y + padding)
        max_x = min(W, max_x + padding)

        # Store adjusted circle parameters
        circle_params.append({
            "center_x": adjusted_center_x,
            "center_y": adjusted_center_y,
            "radius": radius
        })

        # Crop the HSI image
        cropped_hsi = hsi_single[:, min_y:max_y, min_x:max_x]

        # Crop the OCTA image if provided
        cropped_octa = None
        if octa_single is not None:
            cropped_octa = octa_single[:, min_y:max_y, min_x:max_x]

        # Append to lists
        cropped_hsi_list.append(cropped_hsi)
        if octa_single is not None:
            cropped_octa_list.append(cropped_octa)

    # Reshape back into batches if necessary
    if has_batch_dim:
        cropped_hsi = torch.stack(cropped_hsi_list)
        cropped_octa = torch.stack(cropped_octa_list) if octa_image is not None else None
    else:
        cropped_hsi = cropped_hsi_list[0]
        cropped_octa = cropped_octa_list[0] if octa_image is not None else None

    # Apply circular mask to OCTA (always)
    if octa_image is not None:
        cropped_octa = apply_circular_mask(cropped_octa, circle_params, has_batch_dim)

    return cropped_hsi, cropped_octa


def apply_circular_mask(image, circle_params, has_batch_dim=False):
    """
    Apply a circular mask to the image(s), setting pixels outside the circle to black.

    Args:
        image (torch.Tensor): Image tensor of shape [C, H, W] or [B, C, H, W]
        circle_params (list): List of dictionaries containing circle parameters
        has_batch_dim (bool): Whether the image has a batch dimension

    Returns:
        torch.Tensor: Image with circular mask applied
    """
    # Handle batch dimension
    batch_size = image.shape[0] if has_batch_dim else 1
    masked_images = []

    for b in range(batch_size):
        # Get single image
        if has_batch_dim:
            single_img = image[b]
        else:
            single_img = image

        # Get image dimensions
        C, H, W = single_img.shape

        # Get circle parameters
        circle = circle_params[b]
        center_x = circle["center_x"]
        center_y = circle["center_y"]
        radius = circle["radius"]

        # Create a coordinate grid
        y_grid, x_grid = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')

        # Calculate distance from center
        distance = torch.sqrt((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2)

        # Create circular mask (1 inside circle, 0 outside)
        circular_mask = (distance <= radius).float()

        # Move to same device as image if needed
        if single_img.device.type != 'cpu':
            circular_mask = circular_mask.to(single_img.device)

        # Apply mask to all channels
        masked_img = single_img.clone()
        for c in range(C):
            masked_img[c] = single_img[c] * circular_mask

        masked_images.append(masked_img)

    # Reshape back into batches if necessary
    if has_batch_dim:
        return torch.stack(masked_images)
    else:
        return masked_images[0]


def resize_to_square(image, target_size=500):
    """
    Resize an image to a square of the specified size while maintaining aspect ratio.

    Args:
        image (torch.Tensor): Image tensor of shape [C, H, W] or [B, C, H, W]
        target_size (int): Target size for both height and width

    Returns:
        torch.Tensor: Resized image
    """
    # Check if batch dimension exists
    has_batch_dim = len(image.shape) == 4

    if not has_batch_dim:
        # Add batch dimension for processing
        image = image.unsqueeze(0)

    batch_size, channels, height, width = image.shape

    # Resize to target_size x target_size
    resized_image = F.interpolate(
        image,
        size=(target_size, target_size),
        mode='bilinear',
        align_corners=False
    )

    # Remove batch dimension if it wasn't there originally
    if not has_batch_dim:
        resized_image = resized_image.squeeze(0)

    return resized_image


def crop_and_resize(hsi_image, octa_image=None, target_size=500, padding=10):
    """
    Detect circle, crop both HSI and OCTA images, and resize to target size.
    Always applies circular mask to make the OCTA have black edges like HSI.

    Args:
        hsi_image (torch.Tensor): HSI image tensor
        octa_image (torch.Tensor, optional): OCTA image tensor
        target_size (int): Target size for both height and width
        padding (int): Padding around the detected circle (in pixels)

    Returns:
        tuple: Processed HSI and OCTA images
    """
    # Detect circle and crop (masking is now always applied)
    cropped_hsi, cropped_octa = detect_and_crop_circle(hsi_image, octa_image, padding)

    # Resize to target size
    resized_hsi = resize_to_square(cropped_hsi, target_size)
    resized_octa = resize_to_square(cropped_octa, target_size) if octa_image is not None else None

    return resized_hsi, resized_octa