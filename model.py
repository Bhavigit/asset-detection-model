import cv2
import numpy as np
from collections import Counter
from ultralytics import YOLO
from sklearn.cluster import KMeans

# Load the YOLO model.
model = YOLO("runs/detect/train/weights/best.pt")
LABELS = model.names


def get_color_name(rgb_tuple):
    """
    Converts an RGB tuple to a common color name using Euclidean distance.
    This method is more robust than using fixed HSV ranges or simple thresholding,
    especially for varied lighting conditions.

    Args:
        rgb_tuple (tuple): A tuple containing RGB values (R, G, B).

    Returns:
        str: The name of the closest color from the predefined palette.
    """
    r, g, b = rgb_tuple

    # --- Step 1: Check for achromatic colors (Black, White, Gray) based on brightness ---
    # This helps in correctly classifying shades of gray, white, and black,
    # which can be ambiguous in RGB space.
    if max(r, g, b) - min(r, g, b) < 45:
        avg_brightness = (r + g + b) / 3
        if avg_brightness < 40:
            return "Black"
        elif avg_brightness > 220:
            return "White"
        else:
            return "Gray"

    # --- Step 2: Define a palette of primary and secondary colors for traffic signs ---
    # These RGB values are chosen to be representative of common sign colors.
    colors = {
        "Red": (255, 0, 0),
        "Yellow": (255, 220, 0), 
        "Blue": (0, 0, 180),
        "Green": (0, 128, 0),
        "Orange": (255, 165, 0),
        "Purple": (128, 0, 128),
        "Pink": (255, 192, 203)
    }

    min_distance = float('inf')
    closest_color_name = "Unknown"

    # --- Step 3: Find the closest color in the palette using Euclidean distance ---
    # This calculates the straight-line distance between the input RGB and each palette color.
    for color_name, color_rgb in colors.items():
        distance = np.sqrt((r - color_rgb[0]) ** 2 + (g - color_rgb[1]) ** 2 + (b - color_rgb[2]) ** 2)
        if distance < min_distance:
            min_distance = distance
            closest_color_name = color_name

    # --- Step 4: Refine classification for Yellow vs. Orange ---
    # This is a specific heuristic to handle the ambiguity between yellow and orange shades
    # commonly found on traffic signs. It re-classifies borderline orange as yellow.
    if closest_color_name == "Orange" and g > 180 and r > 200:
        return "Yellow"

    # --- Step 5: Use a distance threshold to avoid misclassification ---
    # If the closest color is too far away, it is considered an 'Other' color.
    if min_distance > 100:
        return "Other"

    return closest_color_name


def detect_objects(image_path: str):
    """
    Detects objects using the YOLO model and extracts their primary and secondary colors
    from the detected bounding boxes. This function uses KMeans clustering to find
    dominant colors and applies a heuristic logic for consistent color naming.

    Args:
        image_path (str): The file path to the input image.

    Returns:
        list: A list of dictionaries, where each dictionary contains the class,
              primary color, secondary colors, and bounding box for each detected object.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return []

    # Run object detection on the image.
    results = model(image_path)
    detections = results[0].boxes

    output = []

    for box in detections:
        # Extract detection information.
        cls_id = int(box.cls[0])
        class_name = LABELS[cls_id]
        xyxy = box.xyxy[0].numpy().astype(int)

        x1, y1, x2, y2 = xyxy

        # Ensure bounding box is within image boundaries to avoid errors.
        height, width = image.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)

        # Crop the image to the detected bounding box.
        cropped = image[y1:y2, x1:x2]

        # Handle very tiny or empty cropped regions.
        if cropped.size == 0 or cropped.shape[0] < 5 or cropped.shape[1] < 5:
            continue

        # Convert the cropped image to RGB and flatten the pixels for clustering.
        rgb_crop = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        flat_crop = rgb_crop.reshape((-1, 3))

        # Use KMeans to find the dominant colors (clusters) in the cropped region.
        # The number of clusters is limited to avoid over-segmentation.
        n_colors = min(8, len(np.unique(flat_crop, axis=0)))
        if n_colors < 2:
            n_colors = 2

        kmeans = KMeans(n_clusters=n_colors, n_init=10, random_state=42)
        labels = kmeans.fit_predict(flat_crop)
        counts = Counter(labels)

        # Get cluster centers and sort them by pixel count (size of the cluster).
        sorted_clusters = sorted(counts.items(), key=lambda item: item[1], reverse=True)

        primary_color_name = "Unknown"
        secondary_color_names = []

        # Convert cluster centers (RGB values) to color names and calculate their percentage.
        color_data = []
        total_pixels = len(flat_crop)
        for idx, count in sorted_clusters:
            rgb_color = tuple(map(int, kmeans.cluster_centers_[idx]))
            name = get_color_name(rgb_color)
            percentage = (count / total_pixels) * 100
            color_data.append({'name': name, 'percentage': percentage})

        # Filter for colors that occupy a significant portion of the area (e.g., > 5%).
        significant_colors = [c for c in color_data if c['name'] not in ["Other", "Unknown"] and c['percentage'] > 5]

        if significant_colors:
            # --- Refined Logic to handle specific sign color schemes for consistency ---

            # Find the most dominant color by percentage.
            most_dominant_color_name = significant_colors[0]['name']

            # Check for the presence of White, Red, and Black.
            white_present = 'White' in [c['name'] for c in significant_colors]
            red_present = 'Red' in [c['name'] for c in significant_colors]
            black_present = 'Black' in [c['name'] for c in significant_colors]

            # --- Heuristic 1: For signs with a white background, red border, and black symbol ---
            # If White is the most dominant color and both Red and Black are also present,
            # we classify White as the primary and Red/Black as secondary. This ensures consistency
            # for signs like speed limit and right/left turn signs.
            if most_dominant_color_name == "White" and red_present and black_present:
                primary_color_name = "White"
                secondary_color_names = ["Red", "Black"]

            # --- Heuristic 2: For yellow warning signs ---
            # If Yellow is the most dominant color, it's the primary.
            elif most_dominant_color_name == "Yellow":
                primary_color_name = "Yellow"
                # Add Black to secondary if present
                if black_present:
                    secondary_color_names.append("Black")

            # --- Heuristic 3: For red regulatory signs (e.g., stop signs) ---
            # If Red is the most dominant color, it's the primary.
            elif most_dominant_color_name == "Red":
                primary_color_name = "Red"
                # Add White and Black to secondary if present
                if white_present:
                    secondary_color_names.append("White")
                if black_present:
                    secondary_color_names.append("Black")

            else:
                # --- Fallback Logic: For all other cases ---
                # If none of the above specific heuristics apply, use the most dominant color
                # as the primary and others as secondary.
                primary_color_name = most_dominant_color_name
                # Collect all other significant colors as secondary.
                for color_info in significant_colors:
                    if color_info['name'] != primary_color_name and color_info['name'] not in secondary_color_names:
                        secondary_color_names.append(color_info['name'])

        # Append the results to the output list.
        output.append({
            "class": class_name,
            "primary": primary_color_name,
            "secondary": secondary_color_names,
            "bbox": [x1, y1, x2, y2]
        })

    return output


def visualize_colors(image_path: str, detections: list):
    """
    Draws bounding boxes and labels on an image for visualization purposes.

    Args:
        image_path (str): The file path to the input image.
        detections (list): The list of detected objects with their bounding boxes and colors.

    Returns:
        numpy.ndarray: The image with visualizations.
    """
    image = cv2.imread(image_path)

    for detection in detections:
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{detection['class']}: {detection['primary']}"
        cv2.putText(image, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image


def get_color_palette(image_path: str, detection_bbox: list, n_colors: int = 5):
    """
    Generates a color palette from a specific region of an image.

    Args:
        image_path (str): The file path to the input image.
        detection_bbox (list): The bounding box coordinates [x1, y1, x2, y2].
        n_colors (int): The number of colors to include in the palette.

    Returns:
        list: A list of dictionaries with color information.
    """
    image = cv2.imread(image_path)
    x1, y1, x2, y2 = detection_bbox
    cropped = image[y1:y2, x1:x2]

    rgb_crop = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    flat_crop = rgb_crop.reshape((-1, 3))

    # Use KMeans to find the most representative colors.
    kmeans = KMeans(n_clusters=n_colors, n_init=10, random_state=42)
    labels = kmeans.fit_predict(flat_crop)
    counts = Counter(labels)

    colors = []
    for i in range(n_colors):
        color_rgb = tuple(map(int, kmeans.cluster_centers_[i]))
        color_name = get_color_name(color_rgb)
        pixel_count = counts.get(i, 0)
        colors.append({
            'rgb': color_rgb,
            'name': color_name,
            'pixel_count': pixel_count,
            'percentage': (pixel_count / len(flat_crop)) * 100
        })

    # Sort colors by their pixel count in descending order.
    colors.sort(key=lambda x: x['pixel_count'], reverse=True)
    return colors
