from PIL import Image
import numpy as np

def convert_jpg_to_28x28_raw(jpg_path, raw_output_path):
    # Open image
    img = Image.open(jpg_path).convert("L")  # jpg to grayscale
    img = img.resize((28, 28))  # Resize to 28x28(Normal MNIST data)

    # image to array + Flatten
    img_array = np.array(img).flatten()

    # Write the raw data to the file
    with open(raw_output_path, 'wb') as raw_file:
        img_array.tofile(raw_file)
    
    print(f"JPG image converted to 28x28 size and saved as '{raw_output_path}'.")

# Conversion
convert_jpg_to_28x28_raw("my_pdf_file.jpg", "converted_28x28.raw")
