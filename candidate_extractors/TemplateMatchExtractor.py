import json
import math
import os
import shutil
from fileinput import filename
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from imutils.object_detection import non_max_suppression
from tqdm import tqdm

from utils import is_not_mostly_blank, downscale_bbox, absolute_points_to_relative, downscale_points, relative_bbox_to_absolute, \
    get_polygon_bbox_intersection, mean_blur_image, upscale_bbox, show_cv2_image, rotate_image, crop_cv_image

matplotlib.use('qtagg')

if hasattr(os, 'add_dll_directory'):
    # Windows
    OPENSLIDE_PATH = os.path.join(os.path.abspath(os.getcwd()), "libs/openslide-bin-4.0.0.3-windows-x64/bin")
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide


def generate_dataset_from_slides(slides_root_dir, extractor, output_dir, slide_filenames=None, separate_by_slide=True, extension="jpg"):
    if slide_filenames is None:
        slide_filenames = [file_name for file_name in os.listdir(slides_root_dir) if file_name.endswith(".svs")]
    if os.path.exists(f"{output_dir}/dataset.json"):
        with open(f"{output_dir}/dataset.json", "r") as f:
            cache = json.load(f)
            if cache == {**extractor.to_dict(), "slides": sorted(slide_filenames)}:
                print(f"Found cached candidates dataset {output_dir}")
                return
    if os.path.exists(output_dir):
        print(f"Deleting past dataset {output_dir}")
        shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating candidates dataset from {len(slide_filenames)} slides...")

    def save_candidate(candidate, slide):
        x_min, y_min, w, h = candidate["candidate_bbox"]
        is_positive = candidate["is_positive"]
        crop = np.array(slide.read_region((x_min, y_min), 0, (w, h)))
        slide_name = Path(slide_filename).stem
        output_path = f"{output_dir}/{f'{slide_name}/' if separate_by_slide else ''}/{'positive' if is_positive else 'negative'}/"
        os.makedirs(output_path, exist_ok=True)
        cv2.imwrite(f"{output_path}/{slide_name}_{x_min}_{y_min}_{w}_{h}.{extension}", crop)

    for slide_filename in slide_filenames:
        slide_filepath = os.path.join(slides_root_dir, slide_filename)
        extractor.extract_candidates(slide_filepath, callback=save_candidate)
    with open(f"{output_dir}/dataset.json", "w") as f:
        cache = {**extractor.to_dict(), "slides": sorted(slide_filenames)}
        json.dump(cache, f)


class TemplateMatchExtractor:
    def __init__(self, level=2, candidate_size=256, match_threshold=0.5, cell_size=4096, candidate_overlap_threshold=0.4, verbose=True, display=False):
        self.cell_size = cell_size
        self.level = level
        self.candidate_size = candidate_size
        self.match_threshold = match_threshold
        self.display = display
        self.verbose = verbose
        self.outline_thickness = 2 if level == 1 or level == 0 else 1
        self.candidate_overlap_threshold = candidate_overlap_threshold
        self.templates_dir = f'data/templates/{self.level}'

    def extract_candidates(self, slide_filepath, callback=None):
        slide_filename = Path(slide_filepath).stem
        # if self.verbose:
        #     print(f"Extracting candidates from {slide_filename}")
        with open(f"data/annotations/json/{slide_filename}.json") as f:
            positive_rois = json.load(f)
        slide = openslide.OpenSlide(slide_filepath)
        level_downsample = slide.level_downsamples[self.level]
        _, _, ds_cell_size, _ = downscale_bbox((0, 0, self.cell_size, self.cell_size), level_downsample)
        full_slide_width, full_slide_height = slide.level_dimensions[0]
        cells_count_x = math.floor(full_slide_width / self.cell_size)
        cells_count_y = math.floor(full_slide_height / self.cell_size)
        candidates = []
        if self.verbose:
            pbar = tqdm(total=cells_count_x * cells_count_y, desc=f"Extracting candidates from slide {slide_filename}")

        for j, y in enumerate(range(0, full_slide_height, self.cell_size)):
            for i, x in enumerate(range(0, full_slide_width, self.cell_size)):
                if self.verbose:
                    pbar.update(1)
                cell_bbox_level_0 = (x, y, self.cell_size, self.cell_size)
                cell = np.array(slide.read_region((x, y), self.level, (ds_cell_size, ds_cell_size)))

                if is_not_mostly_blank(cell, non_blank_percentage=0.1):
                    ds_candidate_size = round(self.candidate_size / level_downsample)
                    if self.display:
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
                    filtered_matches = self.template_match(cell, match_size=ds_candidate_size, filter=lambda crop: is_not_mostly_blank(crop, non_blank_percentage=0.1))
                    for candidate_bbox in filtered_matches:
                        is_positive = False
                        abs_candidate_bbox = relative_bbox_to_absolute(upscale_bbox(candidate_bbox, level_downsample), cell_bbox_level_0)

                        for i in positive_rois_indexes_in_cell:
                            positive_points = positive_rois[i]
                            if get_polygon_bbox_intersection(positive_points, abs_candidate_bbox) >= 0.35:
                                is_positive = True
                                break

                        if self.display:
                            cv2.rectangle(display_copy, (candidate_bbox[0], candidate_bbox[1]), (candidate_bbox[0] + candidate_bbox[2], candidate_bbox[1] + candidate_bbox[3]),
                                          (0, 0, 255) if is_positive else (0, 255, 0), self.outline_thickness)
                        abs_x_min, abs_y_min, _, _ = abs_candidate_bbox
                        candidate = {"candidate_bbox": (abs_x_min, abs_y_min, self.candidate_size, self.candidate_size), "is_positive": is_positive}
                        candidates.append(candidate)
                        if callback is not None:
                            callback(candidate, slide)
                    if self.display:
                        show_cv2_image(display_copy)
        if self.verbose:
            pbar.close()
        slide.close()
        return candidates

    def template_match(self, image, match_size=256, filter=None, mean_blur_kernel_size=None):
        image = mean_blur_image(image, mean_blur_kernel_size)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        match_points = []
        match_scores = []
        for template_name in os.listdir(self.templates_dir):
            if Path(template_name).suffix.lower() not in [".png", ".jpg", ".jpeg"]:
                continue
            template_path = os.path.join(self.templates_dir, template_name)
            raw_template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)

            if raw_template.shape[2] == 4:
                template_mask = np.where(cv2.resize(raw_template[:, :, 3], (match_size, match_size)) > 0, 255, 0).astype(np.uint8)
                template = mean_blur_image(cv2.resize(raw_template[:, :, :3], (match_size, match_size)), mean_blur_kernel_size)
            else:
                template_mask = None
                template = mean_blur_image(cv2.resize(raw_template, (match_size, match_size)), mean_blur_kernel_size)

            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            for angle in range(0, 360, 45):
                rotated_template = rotate_image(template, angle)

                result = cv2.matchTemplate(image_gray, rotated_template, cv2.TM_CCOEFF_NORMED, mask=template_mask)
                locations = np.where(result >= self.match_threshold)
                for point, score in zip(zip(*locations[::-1]), result[locations]):
                    match_points.append(point)
                    match_scores.append(score)

        match_bboxes = non_max_suppression(np.array([[x, y, x + match_size, y + match_size] for (x, y) in match_points]), probs=match_scores, overlapThresh=self.candidate_overlap_threshold)
        filtered_matches = []
        for (x_min, y_min, _, _) in match_bboxes:
            filtered_match = x_min, y_min, match_size, match_size
            if filter is None or filter(crop_cv_image(image, filtered_match)):
                filtered_matches.append(filtered_match)
        return filtered_matches

    def to_dict(self):
        return {
            "cell_size": self.cell_size,
            "level": self.level,
            "candidate_size": self.candidate_size,
            "match_threshold": self.match_threshold,
            "candidate_overlap_threshold": self.candidate_overlap_threshold,
            "templates_dir": self.templates_dir,
        }
