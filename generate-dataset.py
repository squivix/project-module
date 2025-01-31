import json
import math
import os
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils import downscale_bbox, calculate_bbox_overlap, downscale_image, mean_blur_image, relative_bbox_to_absolute, is_bbox_1_center_in_bbox_2, crop_cv_image, \
    upscale_bbox, downscale_points, absolute_points_to_relative, get_polygon_bbox_intersection, crop_bbox

matplotlib.use('qtagg')

if hasattr(os, 'add_dll_directory'):
    # Windows
    OPENSLIDE_PATH = os.path.join(os.path.abspath(os.getcwd()),
                                  "libs/openslide-bin-4.0.0.3-windows-x64/bin")
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide


def grid_segment_slides(slides_input_dir, callback=None, filter=None, cell_size=256, level=0):
    # if os.path.exists(root_output_dir):
    #     shutil.rmtree(root_output_dir)
    only_positive = False
    only_negative = True
    if only_positive and only_negative:
        raise ValueError("only_positive and only_negative are mutually exclusive")
    for slide_filename in os.listdir(slides_input_dir):
        if Path(slide_filename).suffix != ".svs":
            continue
        slide_filepath = f"{slides_input_dir}/{slide_filename}"
        print(slide_filename)
        slide_filename = Path(slide_filename).stem
        with open(f"data/whole-slides/gut/{slide_filename}.json") as f:
            positive_rois = [(roi["x_min"], roi["y_min"], roi["width"], roi["height"]) for roi in json.load(f)]

        # output_dir = f"{root_output_dir}/{Path(slide_filename).stem}/{level}/{cell_size}x{cell_size}/"
        # os.makedirs(output_dir, exist_ok=True)
        slide = openslide.OpenSlide(slide_filepath)
        # full_slide = np.array(slide.read_region((0, 0), level, slide.level_dimensions[level]))

        # downscale cell size
        _, _, ds_cell_size, _ = downscale_bbox((0, 0, cell_size, cell_size), slide.level_downsamples[level])

        full_slide_width, full_slide_height = slide.level_dimensions[0]
        cells_count_x = math.floor(full_slide_width / cell_size)
        cells_count_y = math.floor(full_slide_height / cell_size)
        with tqdm(total=cells_count_x * cells_count_y, desc="Progress") as pbar:
            for j, y in enumerate(range(0, full_slide_height, cell_size)):
                for i, x in enumerate(range(0, full_slide_width, cell_size)):
                    pbar.update(1)
                    cell_bbox_level_0 = (x, y, cell_size, cell_size)
                    positive_rois_in_cell = []
                    for positive_bbox in positive_rois:
                        if is_bbox_1_center_in_bbox_2(positive_bbox, cell_bbox_level_0):
                            positive_rois_in_cell.append(positive_bbox)
                    if only_positive and len(positive_rois_in_cell) == 0:
                        continue
                    if only_negative and len(positive_rois_in_cell) != 0:
                        continue

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
                            "slide_filename": slide_filename,
                            "slide_level": level,
                            "level_downsample": slide.level_downsamples[level],
                            "cell_bbox_level_0": cell_bbox_level_0,
                            "slide": slide
                        }
                        # print(cell_meta_data["cell_bbox_level_0"])
                        callback(cell, cell_meta_data)
        # cv2.imwrite(f"{root_output_dir}/{(slide_filename)}/{level}/slide-grid.png", full_slide)


def is_not_mostly_blank(cell, non_blank_percentage=0.5, blank_threshold=235):
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


def extract_candidates(cell, cell_metadata):
    slide_filename = cell_metadata["slide_filename"]
    candidate_output_dir = f"output/candidates/"
    candidate_size = 256
    ds_candidate_size = round(candidate_size / cell_metadata["level_downsample"])
    outline_width = 5
    extension = "png"
    display = False
    display_copy = cell.copy()

    with open(f"data/whole-slides/gut/{slide_filename}.json") as f:
        positive_rois = [(roi["points"], (roi["x_min"], roi["y_min"], roi["width"], roi["height"])) for roi in json.load(f)]
    positive_rois_indexes_in_cell = []
    for i, positive_roi in enumerate(positive_rois):
        positive_points, positive_bbox = positive_roi
        # TODO fix this mess: you've got the same issue one level higher: the big cells you grid from the whole slide could split a positive roi in half and only one will have the center
        if is_bbox_1_center_in_bbox_2(positive_bbox, cell_metadata["cell_bbox_level_0"]):
            # relative_positive_bbox = absolute_bbox_to_relative(positive_bbox, cell_metadata["cell_bbox_level_0"])
            # relative_positive_bbox = downscale_bbox(relative_positive_bbox, cell_metadata["level_downsample"])
            relative_positive_points = absolute_points_to_relative(positive_points, cell_metadata["cell_bbox_level_0"])
            relative_positive_points = downscale_points(relative_positive_points, cell_metadata["level_downsample"])

            positive_rois_indexes_in_cell.append(i)
            if display:
                cv2.polylines(display_copy, [np.array(relative_positive_points)], isClosed=True, color=(0, 255, 0), thickness=2)
                # cv2.rectangle(display_copy, (relative_positive_bbox[0], relative_positive_bbox[1]),
                #               (relative_positive_bbox[0] + relative_positive_bbox[2], relative_positive_bbox[1] + relative_positive_bbox[3]), (0, 0, 255), outline_width)

    filtered_matches = template_match(cell, match_size=ds_candidate_size, match_threshold=0.4, overlap_threshold=0.25)

    positive_rois_caught = set()
    for candidate_bbox in filtered_matches:
        is_positive = False
        abs_candidate_bbox = relative_bbox_to_absolute(upscale_bbox(candidate_bbox, cell_metadata["level_downsample"]), cell_metadata["cell_bbox_level_0"])

        for i in positive_rois_indexes_in_cell:
            # print(f"{positive_bbox}, {abs_candidate_bbox}")
            positive_points, positive_bbox = positive_rois[i]
            if get_polygon_bbox_intersection(positive_points, abs_candidate_bbox) >= 0.35:
                positive_rois_caught.add(i)
                is_positive = True
                break
        if not is_not_mostly_blank(crop_bbox(cell, candidate_bbox), non_blank_percentage=0.1):
            continue
        if display:
            cv2.rectangle(display_copy, (candidate_bbox[0], candidate_bbox[1]), (candidate_bbox[0] + candidate_bbox[2], candidate_bbox[1] + candidate_bbox[3]),
                          (0, 0, 255) if is_positive else (0, 255, 0), outline_width)
        # crop = cell[candidate_bbox[1]:candidate_bbox[1] + ds_candidate_size, candidate_bbox[0]:candidate_bbox[0] + ds_candidate_size]
        abs_x_min, abs_y_min, _, _ = abs_candidate_bbox
        crop = np.array(cell_metadata["slide"].read_region((abs_x_min, abs_y_min), 0, (candidate_size, candidate_size)))
        if is_positive:
            output_path = f"{candidate_output_dir}/positive/"
        else:
            output_path = f"{candidate_output_dir}/negative/"
        os.makedirs(output_path, exist_ok=True)
        # output_path = f"{candidate_output_dir}/blank/"
        cv2.imwrite(f"{output_path}/{slide_filename}_{abs_x_min}_{abs_y_min}_{candidate_size}_{candidate_size}.{extension}", crop)
    # if len(positive_rois_caught) < len(positive_rois_in_cell):
    #     print(f"{cell_metadata['slide_filename']} {cell_metadata['cell_bbox_level_0']} {len(positive_rois_caught)} / {len(positive_rois_in_cell)}")
    if display:
        # Convert the RGBA image from OpenCV format to a format compatible with Matplotlib
        # If the image has an alpha channel, strip it for proper display in Matplotlib
        if display_copy.shape[2] == 4:  # Check if the image has 4 channels (RGBA)
            display_rgb = cv2.cvtColor(display_copy, cv2.COLOR_BGRA2RGB)
        else:  # Otherwise, directly convert from BGR to RGB
            display_rgb = cv2.cvtColor(display_copy, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 6))  # Adjust figure size as needed
        plt.imshow(display_rgb)
        plt.axis('off')  # Turn off the axis if not needed
        plt.title("Image")  # Add a title if desired
        plt.show()


def template_match(image, ds_factor=2, match_threshold=0.5, match_size=256, overlap_threshold=0.2, templates_dir='data/templates/gastric-glands/1/all/', filter=None):
    image = mean_blur_image(image)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    matches = []
    for template_name in os.listdir(templates_dir):
        if Path(template_name).suffix.lower() not in [".png", ".jpg", ".jpeg"]:
            continue
        template_path = os.path.join(templates_dir, template_name)
        # print(template_path)
        raw_template = downscale_image(cv2.imread(template_path, cv2.IMREAD_UNCHANGED), ds_factor)

        if raw_template.shape[2] == 4:
            template_mask = np.where(cv2.resize(raw_template[:, :, 3], (match_size, match_size)) > 0, 255, 0).astype(np.uint8)
            template = mean_blur_image(cv2.resize(raw_template[:, :, :3], (match_size, match_size)))
        else:
            template_mask = None
            template = mean_blur_image(cv2.resize(raw_template, (match_size, match_size)))

        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        result = cv2.matchTemplate(image_gray, template_gray, cv2.TM_CCOEFF_NORMED, mask=template_mask)
        locations = np.where(result >= match_threshold)
        for point, score in zip(zip(*locations[::-1]), result[locations]):
            matches.append((point, score))

    matches = sorted(matches, key=lambda x: x[1], reverse=True)

    filtered_matches = []
    for match in matches:
        point, score = match
        match_bbox = point[0].item(), point[1].item(), match_size, match_size

        overlap = False
        for existing_bbox in filtered_matches:
            overlap_percentage = calculate_bbox_overlap(match_bbox, existing_bbox)
            if overlap_percentage > overlap_threshold:
                overlap = True
                break

        if not overlap and (filter is None or filter(crop_cv_image(image, match_bbox))):
            filtered_matches.append(match_bbox)
    return filtered_matches


def main():
    grid_segment_slides(
        slides_input_dir="data/whole-slides/gut",
        cell_size=4096,
        level=1,
        filter=lambda cell, x, y, slide_filename: is_not_mostly_blank(cell, non_blank_percentage=0.35),
        # callback=lambda cell, cell_meta_data: save_cell(cell, cell_meta_data, "output/tiles"),
        callback=extract_candidates,

    )


main()
