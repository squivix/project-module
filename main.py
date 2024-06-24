import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
import shutil
import os

if hasattr(os, 'add_dll_directory'):
    # Windows
    OPENSLIDE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "libs/openslide-bin-4.0.0.3-windows-x64/bin")
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide


def downsample_image(image, level):
    """Downsample the image to a manageable size."""
    ds_factor = 2 ** level
    return cv2.resize(image, (image.shape[1] // ds_factor, image.shape[0] // ds_factor))


def display_image(image, title="image"):
    """Display the image with a title."""
    plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()


def split_image_into_grid(image, cell_width, cell_height, output_dir):
    height, width, _ = image.shape
    pad_height = (cell_height - (height % cell_height)) % cell_height
    pad_width = (cell_width - (width % cell_width)) % cell_width

    pad_color = [241, 241, 241, 255]  # White with full opacity
    padded_image = cv2.copyMakeBorder(image, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=pad_color)
    new_height, new_width, _ = padded_image.shape
    rows = new_height // cell_height
    cols = new_width // cell_width
    annotated_image = padded_image.copy()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        for row in range(rows):
            for col in range(cols):
                # Calculate the coordinates of the current cell
                y1 = row * cell_height
                y2 = y1 + cell_height
                x1 = col * cell_width
                x2 = x1 + cell_width

                # Crop the cell from the image
                cell = padded_image[y1:y2, x1:x2]

                # Save the cell as a separate image
                cell_filename = f"{output_dir}/cell_{row:06d}_{col:06d}.png"
                cv2.imwrite(cell_filename, cell)
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                print(f"Saved {cell_filename}")

    return annotated_image


def main():
    svs_file = "input/522934.svs"
    output_dir = "output/segments"
    level = 1

    shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    slide = openslide.OpenSlide(svs_file)
    ds_image = np.array(slide.read_region((0, 0), level, slide.level_dimensions[level]))

    gray_image = cv2.cvtColor(ds_image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY_INV)
    labels = measure.label(binary_image, connectivity=2)
    properties = measure.regionprops(labels)
    del ds_image
    del gray_image
    del binary_image

    W, H = slide.level_dimensions[0]
    w, h = slide.level_dimensions[level]
    scale_factor = W / w
    min_blob_area = 50_000
    large_blobs = [prop for prop in properties if prop.area >= min_blob_area]
    # annotated_image = np.copy(ds_image)
    for i, blob_props in enumerate(large_blobs):
        minr, minc, maxr, maxc = blob_props.bbox
        minr = int(minr * scale_factor)
        maxr = int(maxr * scale_factor)
        minc = int(minc * scale_factor)
        maxc = int(maxc * scale_factor)

        blob_image = np.array(
            slide.read_region((minc, minr), 0, (maxc - minc, maxr - minr)))  # ds_image[minr:maxr, minc:maxc]
        annotated_blob_image = (
            split_image_into_grid(blob_image, 256, 256, os.path.join(output_dir, f"blob_{i:03d}"))
        )

        # cv2.rectangle(annotated_image, (minc, minr), (maxc, maxr), (255, 0, 0), 2)

        # display_image(annotated_blob_image, title=f"{annotated_blob_image.shape}")
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    # display_image(f'Annotated image', annotated_image)


if __name__ == '__main__':
    main()
