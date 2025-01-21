import os
from tqdm import tqdm
from PIL import Image

def crop_image_to_tiles(input_file, tile_size):
    """
    Crop an image to dimensions divisible by the specified tile size.

    Parameters:
        input_file (str): Path to the input image file.
        tile_size (int): Size of the tiles (e.g., 64 for 64x64 tiles).

    Returns:
        PIL.Image.Image: Cropped image with dimensions divisible by tile_size.
    """
    # Allow processing of very large images
    Image.MAX_IMAGE_PIXELS = None

    # Load the image
    image = Image.open(input_file)

    # Get original dimensions
    orig_width, orig_height = image.size

    # Calculate new dimensions divisible by tile_size
    new_width = (orig_width // tile_size) * tile_size
    new_height = (orig_height // tile_size) * tile_size

    # Calculate cropping box (center crop)
    crop_left = (orig_width - new_width) // 2
    crop_top = (orig_height - new_height) // 2
    crop_right = orig_width - crop_left - new_width
    crop_bottom = orig_height - crop_top - new_height

    # Perform the crop
    print("Performing cropping. This may take a while for large files...")
    cropped_image = image.crop((
        crop_left, 
        crop_top, 
        orig_width - crop_right, 
        orig_height - crop_bottom
    ))

    return cropped_image



def create_image_tiles(image, input_file, output_folder, tile_size=256):
    """
    Create equal-sized image tiles from a cropped image and save them as PNG files with coordinate-based filenames.
    
    Parameters:
        image (PIL.Image.Image): Cropped PIL image object.
        input_file (str): Path to the input cropped image file (used for naming tiles).
        output_folder (str): Path to the folder where the tiles will be saved.
        tile_size (int): Size of the square tiles (default is 256x256).

    Returns:
        int: Total number of tiles created.
    """
    try:
        # Ensure the image is loaded
        img_width, img_height = image.size

        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Generate unique identifier (based on input filename without extension)
        unique_id = os.path.splitext(os.path.basename(input_file))[0]

        # Calculate total number of tiles
        total_tiles = (img_width // tile_size) * (img_height // tile_size)

        # Progress bar setup
        print(f"Generating tiles for image '{input_file}'...")
        with tqdm(total=total_tiles, desc="Processing Tiles", unit="tile") as pbar:
            tile_count = 0
            for y in range(0, img_height, tile_size):
                for x in range(0, img_width, tile_size):
                    # Crop the tile
                    tile = image.crop((x, y, x + tile_size, y + tile_size))
                    
                    # Construct the filename: {unique_id}_{x}_{y}.png
                    tile_filename = f"{unique_id}_{x}_{y}.png"
                    tile_path = os.path.join(output_folder, tile_filename)
                    
                    # Save the tile
                    tile.save(tile_path, format="PNG")
                    
                    tile_count += 1
                    pbar.update(1)  # Update progress bar

        print(f"Created {tile_count} tiles in '{output_folder}'.")
        return tile_count

    except Exception as e:
        print(f"Error creating tiles: {e}")
        return 0
