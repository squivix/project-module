import json
import os
from pathlib import Path

import pandas as pd

from utils import get_polygon_bbox_intersection

df = pd.read_csv("data/mislabels/all-mislabels.csv")
df['file_name'] = df['file_name'].apply(lambda x: f'{"_".join(x.split("_")[1:])}_256_256')
positive_df = df[df['classification'] == 'positive']
square_mislabels = {}
for index, row in positive_df.iterrows():
    slide, x, y, w, h = row["file_name"].split("_")
    x, y, w, h = int(x), int(y), int(w), int(h)
    if not slide in square_mislabels:
        square_mislabels[slide] = []
    square_mislabels[slide].append((x, y, w, h))

annotations_dir = "data/annotations/json"
slide_df = []
new_regions = 0
total_regions = 0
for annotations_file_name in os.listdir(annotations_dir):
    with open(f"{annotations_dir}/{annotations_file_name}") as f:
        slide_annotations = [annotation["points"] for annotation in json.load(f)]

        slide_filename = Path(annotations_file_name).stem
        slide_fns = square_mislabels.get(slide_filename, [])
        filtered_fns = []
        for slide_fn in slide_fns:
            if not any([get_polygon_bbox_intersection(annotation, slide_fn) > 0.0 for annotation in slide_annotations]):
                x_min, y_min, w, h = slide_fn
                x_max, y_max = x_min + w, y_min + h
                top_left = (x_min, y_min)
                top_right = (x_max, y_min)
                bottom_left = (x_min, y_max)
                bottom_right = (x_max, y_max)
                filtered_fns.append([top_left, top_right, bottom_right, bottom_left, top_left])
        new_regions += len(filtered_fns)
        total_regions += len(slide_fns)
        print(f"{slide_filename}: {len(slide_fns)} positive regions, {len(filtered_fns)} of them new")
        slide_df.append({"slide": slide_filename,
                         "annotations": slide_annotations + filtered_fns,
                         })
print(f"All in all: {total_regions} positive regions, {new_regions} of them new")
with open("data/annotations/all.json", "w") as f:
    json.dump(slide_df, f)
import json
import xml.etree.ElementTree as ET

slide_name_to_new_bboxes = {}
with open("data/annotations/all.json", "r") as f:
    list_data = json.load(f)
for obj in list_data:
    slide_name_to_new_bboxes[obj["slide"]] = obj["annotations"]


def modify_annotations(xml_file, regions):
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Find the <Regions> element
    regions_element = root.find(".//Regions")
    if regions_element is None:
        raise ValueError("No <Regions> element found in the XML.")

    # Remove existing <Region> elements
    for region in list(regions_element):
        regions_element.remove(region)

    # Add new <Region> elements based on the provided list of tuples
    for idx, points in enumerate(regions):
        # Create a new <Region> element
        region_element = ET.SubElement(regions_element, "Region", {
            "Id": str(idx + 1),
            "Type": "0",
            "Zoom": "1.0",
            "Selected": "1",
            "Analyze": "1",
            "DisplayId": str(idx + 1)
        })

        # Add <Vertices> to the region
        vertices_element = ET.SubElement(region_element, "Vertices")
        for point in points:
            x, y = point
            ET.SubElement(vertices_element, "Vertex", {"X": str(x), "Y": str(y), "Z": "0"})

    # Return the modified XML as a string
    return ET.tostring(root, encoding="unicode")


for slide_filename, bboxes in slide_name_to_new_bboxes.items():
    modified_xml = modify_annotations(f"data/whole-slides/gut/{slide_filename}.xml", bboxes)
    with open(f"data/annotations/xml/{slide_filename}.xml", "w") as file:
        file.write(modified_xml)
