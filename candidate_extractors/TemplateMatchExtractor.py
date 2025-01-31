import json
import math
import os
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from tqdm import tqdm

from utils import is_not_mostly_blank, downscale_bbox, calculate_bbox_overlap, crop_cv_image, absolute_points_to_relative, downscale_points, relative_bbox_to_absolute, \
    get_polygon_bbox_intersection, crop_bbox, mean_blur_image, upscale_bbox, show_cv2_image, show_cv2_images

matplotlib.use('qtagg')

if hasattr(os, 'add_dll_directory'):
    # Windows
    OPENSLIDE_PATH = os.path.join(os.path.abspath(os.getcwd()), "libs/openslide-bin-4.0.0.3-windows-x64/bin")
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide


class TemplateMatchExtractor:
    def __init__(self, level=1, candidate_size=256, match_threshold=0.4, cell_size=4096, candidate_overlap_threshold=0.4, display=True, outline_thickness=2):
        self.cell_size = cell_size
        self.level = level
        self.candidate_size = candidate_size
        self.match_threshold = match_threshold
        self.display = display
        self.outline_thickness = outline_thickness
        self.candidate_overlap_threshold = candidate_overlap_threshold
        self.templates_dir = f'data/templates/{self.level}'

    def extract_candidates(self, slide_filepath):
        slide_filename = Path(slide_filepath).stem
        with open(f"data/annotations/json/{slide_filename}.json") as f:
            positive_rois = json.load(f)
        slide = openslide.OpenSlide(slide_filepath)
        level_downsample = slide.level_downsamples[self.level]
        _, _, ds_cell_size, _ = downscale_bbox((0, 0, self.cell_size, self.cell_size), level_downsample)
        full_slide_width, full_slide_height = slide.level_dimensions[0]
        cells_count_x = math.floor(full_slide_width / self.cell_size)
        cells_count_y = math.floor(full_slide_height / self.cell_size)
        candidates = []
        with tqdm(total=cells_count_x * cells_count_y, desc="Progress") as pbar:
            for j, y in enumerate(range(0, full_slide_height, self.cell_size)):
                for i, x in enumerate(range(0, full_slide_width, self.cell_size)):
                    pbar.update(1)
                    cell_bbox_level_0 = (x, y, self.cell_size, self.cell_size)
                    cell = np.array(slide.read_region((x, y), self.level, (ds_cell_size, ds_cell_size)))

                    if is_not_mostly_blank(cell, non_blank_percentage=0.35):
                        ds_candidate_size = round(self.candidate_size / level_downsample)

                        display_copy = cell.copy()

                        positive_rois_indexes_in_cell = []
                        for i, positive_roi in enumerate(positive_rois):
                            positive_points = positive_roi
                            # TODO fix this mess: you've got the same issue one level higher: the big cells you grid from the whole slide could split a positive roi in half and only one will have the center
                            if get_polygon_bbox_intersection(positive_points, cell_bbox_level_0) > 0.1:
                                relative_positive_points = absolute_points_to_relative(positive_points, cell_bbox_level_0)
                                relative_positive_points = downscale_points(relative_positive_points, level_downsample)

                                positive_rois_indexes_in_cell.append(i)
                                if self.display:
                                    cv2.polylines(display_copy, [np.array(relative_positive_points)], isClosed=True, color=(0, 255, 0), thickness=self.outline_thickness)

                        filtered_matches = self.template_match(cell, match_size=ds_candidate_size)

                        for candidate_bbox in filtered_matches:
                            is_positive = False
                            abs_candidate_bbox = relative_bbox_to_absolute(upscale_bbox(candidate_bbox, level_downsample), cell_bbox_level_0)

                            for i in positive_rois_indexes_in_cell:
                                positive_points = positive_rois[i]
                                if get_polygon_bbox_intersection(positive_points, abs_candidate_bbox) >= 0.35:
                                    is_positive = True
                                    break
                            if not is_not_mostly_blank(crop_bbox(cell, candidate_bbox), non_blank_percentage=0.1):
                                continue
                            if self.display:
                                cv2.rectangle(display_copy, (candidate_bbox[0], candidate_bbox[1]), (candidate_bbox[0] + candidate_bbox[2], candidate_bbox[1] + candidate_bbox[3]),
                                              (0, 0, 255) if is_positive else (0, 255, 0), self.outline_thickness)
                            abs_x_min, abs_y_min, _, _ = abs_candidate_bbox
                            candidates.append({"slide_filepath": slide_filepath, "candidate_bbox": (abs_x_min, abs_y_min, self.candidate_size, self.candidate_size), "is_positive": is_positive})
                            # crop = np.array(slide.read_region((abs_x_min, abs_y_min), 0, (candidate_size, candidate_size)))

                        if self.display:
                            show_cv2_image(display_copy)
        return candidates

    def template_match(self, image, match_size=256, filter=None):
        image = mean_blur_image(image)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        matches = []
        for template_name in os.listdir(self.templates_dir):
            if Path(template_name).suffix.lower() not in [".png", ".jpg", ".jpeg"]:
                continue
            template_path = os.path.join(self.templates_dir, template_name)
            raw_template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)

            if raw_template.shape[2] == 4:
                template_mask = np.where(cv2.resize(raw_template[:, :, 3], (match_size, match_size)) > 0, 255, 0).astype(np.uint8)
                template = mean_blur_image(cv2.resize(raw_template[:, :, :3], (match_size, match_size)))
            else:
                template_mask = None
                template = mean_blur_image(cv2.resize(raw_template, (match_size, match_size)))

            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

            result = cv2.matchTemplate(image_gray, template_gray, cv2.TM_CCOEFF_NORMED, mask=template_mask)
            locations = np.where(result >= self.match_threshold)
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
                if overlap_percentage > self.candidate_overlap_threshold:
                    overlap = True
                    break

            if not overlap and (filter is None or filter(crop_cv_image(image, match_bbox))):
                filtered_matches.append(match_bbox)
        return filtered_matches
