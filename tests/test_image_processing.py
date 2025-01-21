from dinov2_pipeline.image_processing import crop_image_to_tiles

# Example usage
if __name__ == "__main__":
    input_file = "C:/Users/rrai/dinov2/WSI/TCGA_EB_COMP.tif"  # Replace with your image path
    tile_size = 256  # Example tile size

    cropped_image = crop_image_to_tiles(input_file, tile_size)
    if cropped_image:
        print(f"Cropped image size: {cropped_image.size}")
        # Save the cropped image
        cropped_image.save("cropped_image.tif")
