import json
from pathlib import Path

import numpy as np

from candidate_extractors.TemplateMatchExtractor import TemplateMatchExtractor
from datasets.SlidesDataset import SlideDataset
from utils import get_polygon_bbox_intersection

extractor = TemplateMatchExtractor(verbose=True, display=False, match_threshold=0.5)
dataset = SlideDataset("data/whole-slides/gut/", extractor)
overall_positives_caught = 0
overall_positives = 0
for slide in dataset.slide_candidates.keys():
    positive_candidates = [c["candidate_bbox"] for c in dataset.slide_candidates[slide] if c["is_positive"]]
    slide = Path(slide).stem
    with open(f"data/annotations/json/{slide}.json") as f:
        positive_annotations = json.load(f)
    positives_caught = 0
    for positive_annotation in positive_annotations:
        caught = False
        for positive_candidate_bbox in positive_candidates:
            if get_polygon_bbox_intersection(positive_annotation, positive_candidate_bbox) > 0.0:
                caught = True
                break
        if caught:
            positives_caught += 1
    overall_positives_caught += positives_caught
    overall_positives += len(positive_annotations)
    print(f"{slide}: {positives_caught} caught out of {len(positive_annotations)}")
print()
print(f"Overall: {overall_positives_caught} caught out of {overall_positives} or {overall_positives_caught / overall_positives:.4%}")
labels=np.array(dataset.labels)
print(f"Positives: {labels.sum()}, Negatives: {labels.shape[0]}")

