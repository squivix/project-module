import json
import pandas as pd
from shapely import Polygon, box, Point


class GroundTruthLabeler:
    def __init__(self, json_file, csv_file, patch_size=256, io_polygon_thresh=0.3, iou_bbox_thresh=0.3):
        self.patch_size = patch_size
        # buffer(0)
        with open(json_file, 'r') as f:
            file_data = json.load(f)
            self.polygons = {
                slide: [Polygon(points) for points in polygons]
                for slide, polygons in file_data.items()
            }

        self.bboxes = pd.read_csv(csv_file)
        self.bboxes["slide_name"], self.bboxes["xmin"], self.bboxes["ymin"] = zip(
            *self.bboxes["file_name"].apply(self._parse_filename)
        )

        self.bboxes["width"] = self.patch_size
        self.bboxes["height"] = self.patch_size
        self.bboxes["xmax"] = self.bboxes["xmin"] + self.patch_size
        self.bboxes["ymax"] = self.bboxes["ymin"] + self.patch_size

        self.bboxes["bbox"] = self.bboxes.apply(
            lambda row: box(row["xmin"], row["ymin"], row["xmax"], row["ymax"]),
            axis=1
        )

        self.bboxes["center"] = self.bboxes.apply(
            lambda row: Point((row["xmin"] + row["xmax"]) / 2, (row["ymin"] + row["ymax"]) / 2),
            axis=1
        )
        self.bboxes["is_positive"] = self.bboxes["classification"].str.lower() == "positive"
        self.io_polygon_thresh = io_polygon_thresh
        self.iou_bbox_thresh = iou_bbox_thresh
        self.positive_regions_summary = self._get_positive_regions_summary()

    def _parse_filename(self, filename):
        parts = filename.rsplit("_", 2)
        return parts[0], int(parts[1]), int(parts[2])

    def intersects_with_gt_positive_polygon(self, slide_name, query_bbox, intersection_threshold=None):
        if intersection_threshold is None:
            intersection_threshold = self.io_polygon_thresh
        if slide_name in self.polygons:
            x_min, y_min, w, h = query_bbox
            query_box = box(x_min, y_min, x_min + w, y_min + h)
            for polygon in self.polygons[slide_name]:
                intersection = polygon.buffer(0).intersection(query_box)
                intersection_over_polygon = intersection.area / polygon.area if polygon.area > 0 else 0
                if intersection_over_polygon > intersection_threshold:
                    return True
        return False

    def intersects_with_gt_positive_patch(self, slide_name, query_bbox, iou_threshold=None):
        if iou_threshold is None:
            iou_threshold = self.iou_bbox_thresh
        x_min, y_min, w, h = query_bbox
        query_box = box(x_min, y_min, x_min + w, y_min + h)
        positive_bboxes = self.bboxes[
            (self.bboxes["is_positive"]) & (self.bboxes["slide_name"] == slide_name)
            ]
        for _, row in positive_bboxes.iterrows():
            intersection = query_box.intersection(row["bbox"]).area
            union = query_box.union(row["bbox"]).area
            iou = intersection / union if union > 0 else 0
            if iou > iou_threshold:
                return True
        return False

    def contains_positive_region(self, slide_name, bbox):
        x_min, y_min, width, height = bbox
        query_bbox = box(x_min, y_min, x_min + width, y_min + height)
        positive_centers = self.bboxes[
            (self.bboxes["is_positive"]) & (self.bboxes["slide_name"] == slide_name)
            ]["center"]

        for center in positive_centers:
            if query_bbox.contains(center):
                return True

        if slide_name in self.polygons:
            for polygon in self.polygons[slide_name]:
                for point in polygon.exterior.coords:
                    if query_bbox.contains(Point(point)):
                        return True

        return False

    def get_positive_regions(self, slide_name, bbox=None):
        positive_bboxes_in_slide = self.bboxes[
            (self.bboxes["is_positive"]) & (self.bboxes["slide_name"] == slide_name)
            ]
        if bbox is None:
            return [list(row["bbox"].exterior.coords) for _, row in positive_bboxes_in_slide.iterrows()] + [list(p.exterior.coords) for p in self.polygons[slide_name]]

        xmin, ymin, width, height = bbox
        query_box = box(xmin, ymin, xmin + width, ymin + height)
        positive_regions = []

        for _, row in positive_bboxes_in_slide.iterrows():
            if query_box.contains(row["center"]):
                bbox_points = list(row["bbox"].exterior.coords)
                positive_regions.append(bbox_points)

        if slide_name in self.polygons:
            for polygon in self.polygons[slide_name]:
                if any(query_box.contains(Point(point)) for point in polygon.exterior.coords):
                    positive_regions.append(list(polygon.exterior.coords))

        return positive_regions

    def _get_positive_regions_summary(self):
        slide_names = list(self.polygons.keys()) + list(self.bboxes["slide_name"].unique())
        slide_names = list(set(slide_names))  # Remove duplicates

        summary_data = []

        for slide_name in slide_names:
            num_regions = len(self.get_positive_regions(slide_name, bbox=None))
            summary_data.append({"slide_name": slide_name, "n_gt_positive_regions": num_regions})

        return pd.DataFrame(summary_data)

    def is_positive_patch(self, slide_name, bbox):
        xmin, ymin, width, height = bbox
        if not (width == height == self.patch_size):
            raise ValueError("Not a valid patch")
        # xmax, ymax = xmin + self.patch_size, ymin + self.patch_size
        # query_box = box(xmin, ymin, xmax, ymax)

        # 1. Check for exact match in CSV
        exact_match = self.bboxes[
            (self.bboxes["slide_name"] == slide_name) &
            (self.bboxes["xmin"] == xmin) &
            (self.bboxes["ymin"] == ymin)
            ]
        if not exact_match.empty:
            return exact_match.iloc[0]["is_positive"]

        # 2. Check for polygon intersection
        if self.intersects_with_gt_positive_polygon(slide_name, bbox):
            return True

        # 3. Check for intersection with positive bounding boxes
        if self.intersects_with_gt_positive_patch(slide_name, bbox):
            return True

        return False
