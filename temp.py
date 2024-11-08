import math
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from utils import downscale_bbox, draw_bbox

if hasattr(os, 'add_dll_directory'):
    # Windows
    OPENSLIDE_PATH = os.path.join(os.path.abspath(os.getcwd()),
                                  "libs/openslide-bin-4.0.0.3-windows-x64/bin")
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide


def grid_segment_slides(input_dir, callback=None, filter=None, cell_size=256, level=0):
    # if os.path.exists(root_output_dir):
    #     shutil.rmtree(root_output_dir)
    for slide_filename in os.listdir(input_dir):
        if Path(slide_filename).suffix != ".svs":
            continue
        print(slide_filename)

        # output_dir = f"{root_output_dir}/{Path(slide_filename).stem}/{level}/{cell_size}x{cell_size}/"
        # os.makedirs(output_dir, exist_ok=True)
        slide = openslide.OpenSlide(f"{input_dir}/{slide_filename}")
        # full_slide = np.array(slide.read_region((0, 0), level, slide.level_dimensions[level]))

        # downscale cell size
        _, _, ds_cell_size, _ = downscale_bbox((0, 0, cell_size, cell_size), slide.level_downsamples[level])

        full_slide_width, full_slide_height = slide.level_dimensions[0]
        cells_count_x = math.floor(full_slide_width / cell_size)
        cells_count_y = math.floor(full_slide_height / cell_size)
        with tqdm(total=cells_count_x * cells_count_y, desc="Progress") as pbar:
            for i, x in enumerate(range(0, full_slide_width, cell_size)):
                for j, y in enumerate(range(0, full_slide_height, cell_size)):
                    cell = np.array(slide.read_region((x, y), level, (ds_cell_size, ds_cell_size)))

                    if filter is None or filter(cell, x, y, slide_filename) and callback is not None:
                        # draw_bbox(full_slide, downscale_bbox((x, y, cell_size, cell_size), slide.level_downsamples[level]))
                        cell_meta_data = {
                            "cell_row": j,
                            "cell_column": i,
                            "cell_x1": x,
                            "cell_y1": y,
                            "cell_width": ds_cell_size,
                            "cell_height": ds_cell_size,
                            "cell_width_in_level_0": cell_size,
                            "cell_height_in_level_0": cell_size,
                            "slide_filename": Path(slide_filename).stem,
                            "slide_level": level
                        }
                        callback(cell, cell_meta_data)
                    pbar.update(1)
        # cv2.imwrite(f"{root_output_dir}/{Path(slide_filename).stem}/{level}/slide-grid.png", full_slide)


def is_not_mostly_blank(cell, non_blank_percentage=0.5, blank_threshold=240):
    cell_gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    non_white_pixels = np.sum(cell_gray < blank_threshold)
    pure_black_pixels = np.sum(cell_gray == 0)
    return ((non_white_pixels - pure_black_pixels) / cell_gray.size) > non_blank_percentage


def save_cell(cell, cell_meta_data, root_output_dir, extension="png"):
    row, column, x1, y1, slide_filename, slide_level, cell_size = (cell_meta_data['cell_row'], cell_meta_data['cell_column'], cell_meta_data['cell_x1'],
                                                                   cell_meta_data['cell_y1'], cell_meta_data["slide_filename"], cell_meta_data["slide_level"],
                                                                   cell_meta_data["cell_width_in_level_0"])
    output_dir = f"{root_output_dir}/{slide_filename}/{slide_level}/{cell_size}x{cell_size}/"
    os.makedirs(output_dir, exist_ok=True)
    cell_file_path = f"{output_dir}/{row},{column}_{x1}_{y1}.{extension}"
    cv2.imwrite(cell_file_path, cell)


def template_match_cell(cell, cell_meta_data):
    pass


def main():
    grid_segment_slides(
        input_dir="data/whole-slides/gut",
        cell_size=4096,
        level=0,
        filter=lambda cell, x, y, slide_filename: is_not_mostly_blank(cell, non_blank_percentage=0.35),
        callback=lambda cell, cell_meta_data: save_cell(cell, cell_meta_data, "output/tiles"),
    )


main()
