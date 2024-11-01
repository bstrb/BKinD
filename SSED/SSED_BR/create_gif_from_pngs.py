import os
from PIL import Image

def create_gif_from_pngs(folder, output_filename='output.gif', duration=1000, resize_factor=0.3, reduce_colors=True, colors=128):
    png_files = sorted([f for f in os.listdir(folder) if f.endswith('.png')])
    
    if not png_files:
        print("No .png files found in the folder.")
        return
    
    images = []
    for file in png_files:
        img_path = os.path.join(folder, file)
        img = Image.open(img_path)
        
        # Resize the image
        if resize_factor < 1.0:
            new_size = (int(img.width * resize_factor), int(img.height * resize_factor))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Reduce color palette
        if reduce_colors:
            img = img.convert("P", palette=Image.ADAPTIVE, colors=colors)
        
        images.append(img)
    
    # Save as GIF
    images[0].save(
        output_filename,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
        optimize=True
    )
    print(f"GIF created successfully: {output_filename}")

# Provide the folder path
folder_path = '/home/buster/UOX1/0-05-step-indexing/41x41_0-05_indexing/cm_heatmaps_gif'
output_gif = '/home/buster/UOX1/0-05-step-indexing/41x41_0-05_indexing/cm_heatmaps.gif'

create_gif_from_pngs(folder_path, output_filename=output_gif, duration=1000, resize_factor=0.3, reduce_colors=True, colors=128)
