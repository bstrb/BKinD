from tkinter import Tk, filedialog, Button, Label, Entry, messagebox
from PIL import Image
import math

def browse_images():
    global image_paths
    image_paths = filedialog.askopenfilenames(
        title="Select PNG Images",
        filetypes=[("PNG Images", "*.png")]
    )
    if image_paths:
        messagebox.showinfo("Selected Images", f"{len(image_paths)} images selected.")
    else:
        messagebox.showwarning("No Images", "No images were selected.")

def browse_save_path():
    global save_path
    save_path = filedialog.asksaveasfilename(
        title="Save Combined Image As",
        defaultextension=".png",
        filetypes=[("PNG Images", "*.png")]
    )
    save_path_entry.delete(0, 'end')
    save_path_entry.insert(0, save_path)

def combine_images():
    if not image_paths:
        messagebox.showerror("Error", "No images selected!")
        return
    
    if not save_path:
        messagebox.showerror("Error", "No save path specified!")
        return

    # Open images
    images = [Image.open(img_path) for img_path in image_paths]

    # Find grid size (square grid)
    num_images = len(images)
    grid_size = math.ceil(math.sqrt(num_images))
    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)

    # Create a blank canvas for the combined image
    combined_width = grid_size * max_width
    combined_height = grid_size * max_height
    combined_image = Image.new("RGBA", (combined_width, combined_height), (255, 255, 255, 0))

    # Paste images into the combined image
    for idx, img in enumerate(images):
        x_offset = (idx % grid_size) * max_width
        y_offset = (idx // grid_size) * max_height
        combined_image.paste(img, (x_offset, y_offset))

    # Save the combined image
    combined_image.save(save_path)
    messagebox.showinfo("Success", f"Combined image saved at: {save_path}")

# Initialize Tkinter
root = Tk()
root.title("Combine PNG Images")
root.geometry("400x300")

# Global variables
image_paths = []
save_path = ""

# GUI Elements
Label(root, text="Step 1: Select PNG Images").pack(pady=10)
Button(root, text="Browse Images", command=browse_images).pack()

Label(root, text="Step 2: Choose Save Path").pack(pady=10)
save_path_entry = Entry(root, width=40)
save_path_entry.pack()
Button(root, text="Browse Save Path", command=browse_save_path).pack()

Button(root, text="Combine Images", command=combine_images, bg="green", fg="white").pack(pady=20)

# Run the GUI
root.mainloop()
