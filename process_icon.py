from PIL import Image, ImageFilter
import os
import shutil

def remove_background_and_create_ico(input_path, output_png_path, output_ico_path):
    print(f"Processing: {input_path}")
    
    try:
        img = Image.open(input_path)
        
        # Resize for better quality if too large
        if img.width > 512:
            img = img.resize((512, 512), Image.Resampling.LANCZOS)
            
        img = img.convert("RGBA")
        
        # Apply slight sharpening
        img = img.filter(ImageFilter.SHARPEN)
        
        datas = img.getdata()
        new_data = []
        
        # Adjusted threshold for clearer cut
        threshold = 20
        
        for item in datas:
            # item is (R, G, B, A)
            if item[0] < threshold and item[1] < threshold and item[2] < threshold:
                new_data.append((0, 0, 0, 0))  # Transparent
            else:
                new_data.append(item)
        
        img.putdata(new_data)
        
        # Save transparent PNG
        img.save(output_png_path, "PNG")
        print(f"Saved transparent PNG: {output_png_path}")
        
        # Create optimized images for each size
        icon_sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
        icons = []
        
        for size in icon_sizes:
            # Resize from original high-res image
            # Use LANCZOS for best downscaling quality
            resized_img = img.resize(size, Image.Resampling.LANCZOS)
            
            # Apply stronger sharpening for small sizes to combat blur
            if size[0] <= 48:
                # Sharpen twice for very small icons
                resized_img = resized_img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
            
            icons.append(resized_img)
            
        img.save(output_ico_path, sizes=icon_sizes, append_images=icons)
        print(f"Saved ICO: {output_ico_path}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    input_file = r"d:\Workspace\SoundBoardTool\src\web\assets\logo_hd.png"
    output_png = r"d:\Workspace\SoundBoardTool\src\web\assets\logo_transparent.png"
    output_ico = r"d:\Workspace\SoundBoardTool\src\web\assets\icon.ico"
    
    remove_background_and_create_ico(input_file, output_png, output_ico)
