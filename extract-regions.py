import glob
import json
import os

import cv2
import numpy as np

from utils import get_relative_bbox2_within_bbox1, downscale_bbox, is_bbox2_within_bbox1, draw_sign

if hasattr(os, 'add_dll_directory'):
    # Windows
    OPENSLIDE_PATH = os.path.join(os.path.abspath(os.getcwd()),
                                  "libs/openslide-bin-4.0.0.3-windows-x64/bin")
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide
def is_bbox2_within_bbox1(bbox1, bbox2):
    # Unpacking bbox1 and bbox2
    xmin1, ymin1, width1, height1 = bbox1
    xmin2, ymin2, width2, height2 = bbox2

    # Calculate the bottom-right corners of bbox1 and bbox2
    xmax1, ymax1 = xmin1 + width1, ymin1 + height1
    xmax2, ymax2 = xmin2 + width2, ymin2 + height2

    # Check if bbox2 is inside bbox1
    return (xmin1 <= xmin2 <= xmax1 and
            ymin1 <= ymin2 <= ymax1
            # and
            # xmax1 >= xmax2 and
            # ymax1 >= ymax2
            )

slide_file_path = "data/whole-slides/gut/522934.svs"
output_dir = "output/temp"
os.makedirs(output_dir, exist_ok=True)
for f in glob.glob(f"{output_dir}/*"):
    os.remove(f)

slide = openslide.OpenSlide(slide_file_path)

# with open("data/whole-slides/gut/522934.json", 'r') as file:
#     rois = json.load(file)
big_tiles_level = 0
crops = [
    {"xy": (0, 0), "level": slide.level_count - 1, "wh": slide.level_dimensions[0], "rescale_wh": True,
     "file_name": "thumnmail"},
    # {"xy": (0, 0), "level": slide.level_count - 1, "wh": slide.level_dimensions[0], "rescale_wh": True,
    #  "file_name": "thumnmail_a", "annotate": True},

    # {"xy": (99144, 11477), "level": slide.level_count - 2, "wh": (38367, 44380), "rescale_wh": True, "file_name": "thumnmail"},
    # {"xy": (99144, 11477), "level": slide.level_count - 2, "wh": (38367, 44380), "rescale_wh": True, "file_name": "thumnmail_a", "annotate": True},
    {"xy": (101350, 36000), "level": big_tiles_level, "wh": (4096, 4096), "rescale_wh": True, "file_name": "big01"},
    {"xy": (118968, 21422), "level": big_tiles_level, "wh": (4096, 4096), "rescale_wh": True, "file_name": "big02"},
    {"xy": (127500, 37272), "level": big_tiles_level, "wh": (4096, 4096), "rescale_wh": True, "file_name": "big03"},
    {"xy": (58861, 21961), "level": big_tiles_level, "wh": (4096, 4096), "rescale_wh": True, "file_name": "big04"},
    # {"xy": (101350, 36000), "level": big_tiles_level, "wh": (4096, 4096), "rescale_wh": True, "file_name": "big01_a", "annotate": True},
    # {"xy": (118968, 21422), "level": big_tiles_level, "wh": (4096, 4096), "rescale_wh": True, "file_name": "big02_a", "annotate": True},
    # {"xy": (127500, 37272), "level": big_tiles_level, "wh": (4096, 4096), "rescale_wh": True, "file_name": "big03_a", "annotate": True},
    # {"xy": (58861, 21961), "level": big_tiles_level, "wh": (4096, 4096), "rescale_wh": True, "file_name": "big04_a", "annotate": True},

    # {"xy": (163433, 29388), "level": 0, "wh": (300, 300), "rescale_wh": True, "file_name": "small01"},
    # {"xy": (81958, 41598), "level": 0, "wh": (300, 300), "rescale_wh": True, "file_name": "small02"},
    # {"xy": (109712, 28292), "level": 0, "wh": (300, 300), "rescale_wh": True, "file_name": "small03"},
    # {"xy": (108819, 28553), "level": 0, "wh": (300, 300), "rescale_wh": True, "file_name": "small04"},
    # {"xy": (163433, 29388), "level": 0, "wh": (300, 300), "rescale_wh": True, "file_name": "small01_a", "classify": True},
    # {"xy": (81958, 41598), "level": 0, "wh": (300, 300), "rescale_wh": True, "file_name": "small02_a", "classify": True},
    # {"xy": (109712, 28292), "level": 0, "wh": (300, 300), "rescale_wh": True, "file_name": "small03_a", "classify": True},
    # {"xy": (108819, 28553), "level": 0, "wh": (300, 300), "rescale_wh": True, "file_name": "small04_a", "classify": True},

    {"xy": (114054, 27411), "level": big_tiles_level, "wh": (2048,2048), "rescale_wh": True, "file_name": "temp", "annotate": False}
]
for crop in crops:
    xy = crop["xy"]
    wh = crop["wh"]
    level = crop["level"]
    if "rescale_wh" in crop and crop["rescale_wh"]:
        ds = int(slide.level_downsamples[level])
        wh = (wh[0] // ds, wh[1] // ds)
    crop_image = np.array(slide.read_region(xy, level, wh))
    # print((xy, level, wh))
    if "file_name" in crop:
        file_name = crop["file_name"]
    else:
        file_name = f'{crop["xy"][0]},{crop["xy"][1]}_{crop["level"]}_{crop["wh"][0]},{crop["wh"][1]}'
    if "annotate" in crop and crop["annotate"]:
        for roi in rois:
            roi_bbox = roi["x_min"], roi["y_min"], roi["width"], roi["height"]
            crop_bbox = *xy, *crop["wh"]
            relative_bbox = get_relative_bbox2_within_bbox1(crop_bbox, roi_bbox)
            if relative_bbox is not None:
                relative_bbox = downscale_bbox(relative_bbox, slide.level_downsamples[level])
                cv2.rectangle(crop_image, (relative_bbox[0], relative_bbox[1]), (relative_bbox[0] + relative_bbox[2], relative_bbox[1] + relative_bbox[3],), (0, 0, 255, 255), 8)
    if "classify" in crop and crop["classify"]:
        is_positive = False
        for roi in rois:
            roi_bbox = roi["x_min"], roi["y_min"], roi["width"], roi["height"]
            crop_bbox = *xy, *crop["wh"]
            if is_bbox2_within_bbox1(crop_bbox, roi_bbox):
                is_positive = True
                break
        crop_image = draw_sign(crop_image, is_positive)
    cv2.imwrite(os.path.join(output_dir, f"{file_name}.png"), crop_image)
