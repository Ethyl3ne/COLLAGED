import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

# failed/defunct function, but I'm too lazy to remove it from the other components
def remove_white_background(image, threshold=240):
    img_array = np.array(image)
    if img_array.shape[2] == 3:
        img_rgba = np.zeros((img_array.shape[0], img_array.shape[1], 4), dtype=np.uint8)
        img_rgba[:, :, :3] = img_array
        img_rgba[:, :, 3] = 255  
    else:
        img_rgba = img_array.copy()
    white_mask = np.all(img_rgba[:, :, :3] > threshold, axis=2)
    img_rgba[white_mask, 3] = 0
    
    return Image.fromarray(img_rgba, 'RGBA')

def resize_image(image, target_size=(16, 16)):
    """
    Resize image to target size, works w/ upscaling and downscaling
    """
    current_size = image.size
    
    if current_size[0] <= target_size[0] and current_size[1] <= target_size[1]:
        print(f"Upscaling from {current_size} to {target_size} using nearest neighbor")
        return image.resize(target_size, Image.NEAREST)
    else:
        print(f"Downsampling from {current_size} to {target_size} using LANCZOS")
        return image.resize(target_size, Image.LANCZOS)

def extract_palette_from_image(image, n_colors=8):
    """
    Extract color palette
    
    Args:
        image (PIL.Image): Input image
        n_colors (int): Number of colors to extract
        
    Returns:
        numpy.ndarray: Array of RGB colors (n_colors, 3)
    """
    
    img_array = np.array(image)
    
    if img_array.shape[2] == 4:
        
        rgb = img_array[:, :, :3]
        alpha = img_array[:, :, 3]
        
        non_transparent = alpha > 0
        if not np.any(non_transparent):
            return np.array([[0, 0, 0]] * n_colors)  
        
        pixels = rgb[non_transparent].reshape(-1, 3)
    else:
        pixels = img_array.reshape(-1, 3)
    
    n_clusters = min(n_colors, len(pixels))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    palette = kmeans.cluster_centers_.astype(np.uint8)

    if len(palette) < n_colors:
        padding = np.zeros((n_colors - len(palette), 3), dtype=np.uint8)
        palette = np.vstack([palette, padding])
    
    return palette

def quantize_colors(image, n_colors=8):
    """
    Reduce number of colors by K-means clustering
    """
    
    img_array = np.array(image)
    
    
    if img_array.shape[2] == 4:
        
        rgb = img_array[:, :, :3]
        alpha = img_array[:, :, 3]
        
        non_transparent = alpha > 0
        if not np.any(non_transparent):
            return image  
        
        pixels = rgb[non_transparent].reshape(-1, 3)
    else:
        pixels = img_array.reshape(-1, 3)
        alpha = None
    
    kmeans = KMeans(n_clusters=min(n_colors, len(pixels)), random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    new_colors = kmeans.cluster_centers_.astype(np.uint8)
    labels = kmeans.labels_
    
    if alpha is not None:
        quantized = img_array.copy()
        quantized[non_transparent, :3] = new_colors[labels]
    else:
        quantized = new_colors[labels].reshape(img_array.shape)
    
    return Image.fromarray(quantized, 'RGBA' if alpha is not None else 'RGB')

def apply_palette_to_image(image, palette_image, pixel_size=(16, 16)):
    """
    Apply an existing color palette to a new image for consistency
    
    Args:
        image (PIL.Image): Input image to recolor
        palette_image (PIL.Image): Image containing the reference palette
        pixel_size (tuple): Target pixel dimensions
        
    Returns:
        PIL.Image: Image with applied palette
    """
    try:
        palette_colors = extract_palette_from_reference_image(palette_image)
        
        if len(palette_colors) == 0:
            print("Could not extract palette, fall back: standard quantization")
            return create_pixel_art_from_image(image, pixel_size=pixel_size, n_colors=16)
        
        print(f"Applying palette with {len(palette_colors)} colors")
        
        if image.mode not in ['RGB', 'RGBA']:
            image = image.convert('RGB')
        
        resized_image = resize_image(image, pixel_size)
        
        recolored_image = map_image_to_palette(resized_image, palette_colors)
        
        return recolored_image
        
    except Exception as e:
        print(f"Error applying palette: {e}")
        print("Fall back: standard pixel art processing")
        return create_pixel_art_from_image(image, pixel_size=pixel_size)

def extract_palette_from_reference_image(palette_image):
    """
    Extract unique colors from a palette reference image
    
    Args:
        palette_image (PIL.Image): Reference image containing the palette
        
    Returns:
        numpy.ndarray: Array of unique RGB colors
    """
    
    if palette_image.mode == 'RGBA':
        
        background = Image.new('RGB', palette_image.size, (255, 255, 255))
        palette_image = Image.alpha_composite(background.convert('RGBA'), palette_image).convert('RGB')
    elif palette_image.mode != 'RGB':
        palette_image = palette_image.convert('RGB')
    
    img_array = np.array(palette_image)
    
    pixels = img_array.reshape(-1, 3)
    unique_colors = np.unique(pixels, axis=0)
    
    non_white_colors = []
    for color in unique_colors:
        if not (color[0] > 240 and color[1] > 240 and color[2] > 240):
            non_white_colors.append(color)
    
    return np.array(non_white_colors) if non_white_colors else unique_colors

def map_image_to_palette(image, palette_colors):
    """
    Map all colors in an image to the closest colors in a given palette
    
    Args:
        image (PIL.Image): Input image
        palette_colors (numpy.ndarray): Array of RGB palette colors
        
    Returns:
        PIL.Image: Image with colors mapped to palette
    """
    
    img_array = np.array(image)
    original_shape = img_array.shape
    
    if len(original_shape) == 3 and original_shape[2] == 4:  
        rgb_array = img_array[:, :, :3]
        alpha_array = img_array[:, :, 3]
        has_alpha = True
    else:  
        rgb_array = img_array
        alpha_array = None
        has_alpha = False
    
    pixels = rgb_array.reshape(-1, 3)
    
    nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
    nn.fit(palette_colors)
    
    distances, indices = nn.kneighbors(pixels) # wait what was distances for again?
    mapped_pixels = palette_colors[indices.flatten()]
    mapped_array = mapped_pixels.reshape(rgb_array.shape).astype(np.uint8)

    if has_alpha:
        result_array = np.zeros(original_shape, dtype=np.uint8)
        result_array[:, :, :3] = mapped_array
        result_array[:, :, 3] = alpha_array
        return Image.fromarray(result_array, 'RGBA')
    else:
        return Image.fromarray(mapped_array, 'RGB')

def create_pixel_art_from_image(image, pixel_size=(16, 16), n_colors=8, 
                               remove_bg=True, bg_threshold=240):
    """
    Convert PIL Image to pixel art
    
    Args:
        image: PIL Image 
        pixel_size: Target pixel dimensions (width, height) <- FINAL output size
        n_colors: Number of colors in final palette
        remove_bg: Whether to remove white background
        bg_threshold: Threshold for considering pixels as background (0-255)
        
    Returns:
        PIL.Image: Pixel art version
    """
    
    if image.mode not in ['RGB', 'RGBA']:
        image = image.convert('RGB')
    
    print(f"Processing image size: {image.size}")
    
    if remove_bg:
        image = remove_white_background(image, bg_threshold)
        print("Background removed")

    pixel_art = resize_image(image, pixel_size)
    print(f"Resized to: {pixel_art.size}")
    
    pixel_art = quantize_colors(pixel_art, n_colors)
    print(f"Colors quantized to: {n_colors}")
    
    print(f"Final output size: {pixel_art.size}")
    
    return pixel_art

def create_palette_preview(image):
    """
    Create a visual preview of the color palette used
    
    Args:
        image (PIL.Image): Image to extract palette from
        
    Returns:
        PIL.Image: Palette preview image
    """
    img_array = np.array(image)
    
    if img_array.shape[2] == 4:
        rgb = img_array[:, :, :3]
        alpha = img_array[:, :, 3]
        non_transparent = alpha > 0
        if not np.any(non_transparent):
            return None
        pixels = rgb[non_transparent].reshape(-1, 3)
    else:
        pixels = img_array.reshape(-1, 3)
    
    unique_colors = np.unique(pixels, axis=0)
    
    filtered_colors = []
    for color in unique_colors:
        if not (color[0] > 240 and color[1] > 240 and color[2] > 240):
            filtered_colors.append(color)
    
    display_colors = np.array(filtered_colors) if filtered_colors else unique_colors
    
    if len(display_colors) > 32:
        display_colors = display_colors[:32]
    
    palette_height = 64
    color_width = max(50, 400 // len(display_colors))  
    palette_width = len(display_colors) * color_width
    palette = np.zeros((palette_height, palette_width, 3), dtype=np.uint8)
    
    for i, color in enumerate(display_colors):
        palette[:, i*color_width:(i+1)*color_width] = color
    
    palette_image = Image.fromarray(palette)
    
    return palette_image

def create_enhanced_palette_preview(image, save_path=None):
    """
    Create an enhanced palette preview with color swatches and information
    
    Args:
        image (PIL.Image): Image to extract palette from
        save_path (str): Optional path to save the palette preview
        
    Returns:
        PIL.Image: Enhanced palette preview image
    """
    try:
        palette_image = create_palette_preview(image)
        
        if palette_image and save_path:
            palette_image.save(save_path)
            print(f"Enhanced palette preview saved: {save_path}")
        
        return palette_image
        
    except Exception as e:
        print(f"Could not create enhanced palette preview: {e}")
        return None

