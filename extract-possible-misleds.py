import os
import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.LabeledImageDataset import LabeledImageDataset
from utils import reduce_dataset
import xml.etree.ElementTree as ET


def extract_possible_mislabels():
    model = torch.load("pickled_model_20.pth")
    dataset = LabeledImageDataset("data/candidates")
    dataset = reduce_dataset(dataset, discard_ratio=0.0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 256
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=False)
    k = batch_size
    most_confident_probs = []
    most_confident_indexes = []
    for i, batch in enumerate(tqdm(data_loader)):
        with torch.no_grad():
            x_test, y_test, batch_index = batch
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            batch_index = batch_index.numpy(force=True)

            test_probs = model.forward(x_test)[:, 0].detach().cpu()
            confident = torch.topk(test_probs, min(test_probs.shape[0], k))
            most_confident_probs.extend(confident[0].numpy(force=True))
            most_confident_indexes.extend(batch_index[confident[1]])

    sorted_indexes_values = sorted(zip(most_confident_probs, most_confident_indexes), reverse=True)
    sorted_values, sorted_indexes = zip(*sorted_indexes_values)
    return dataset, sorted_values, sorted_indexes


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
    for idx, (min_x, min_y, width, height) in enumerate(regions):
        max_x = min_x + width
        max_y = min_y + height

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
        ET.SubElement(vertices_element, "Vertex", {"X": str(min_x), "Y": str(min_y), "Z": "0"})
        ET.SubElement(vertices_element, "Vertex", {"X": str(max_x), "Y": str(min_y), "Z": "0"})
        ET.SubElement(vertices_element, "Vertex", {"X": str(max_x), "Y": str(max_y), "Z": "0"})
        ET.SubElement(vertices_element, "Vertex", {"X": str(min_x), "Y": str(max_y), "Z": "0"})
        ET.SubElement(vertices_element, "Vertex", {"X": str(min_x), "Y": str(min_y), "Z": "0"})

    # Return the modified XML as a string
    return ET.tostring(root, encoding="unicode")


dataset, sorted_probs, sorted_indexes = extract_possible_mislabels()
candidate_output_dir = 'output/mislabel-candidates/'
if os.path.exists(candidate_output_dir):
    shutil.rmtree(candidate_output_dir)
os.makedirs(f"{candidate_output_dir}/images", exist_ok=True)
os.makedirs(f"{candidate_output_dir}/annotations", exist_ok=True)
threshold = 0.9
slide_name_to_bboxes = {}
for i in range(len(sorted_indexes)):
    dataset_index = sorted_indexes[i]
    prob = sorted_probs[i]
    src_file_path = dataset.dataset.file_paths[dataset_index]
    if prob >= threshold and "negative" in src_file_path:
        path_obj = Path(src_file_path)
        new_file_name = f"{str(prob)[2:]}_{path_obj.stem}{path_obj.suffix}"
        print(new_file_name)
        shutil.copy(src_file_path, f"{candidate_output_dir}/images/{new_file_name}")
        slide_filename, x_min, y_min, width, height = path_obj.stem.split("_")
        if not slide_filename in slide_name_to_bboxes:
            slide_name_to_bboxes[slide_filename] = []
        slide_name_to_bboxes[slide_filename].append((int(x_min), int(y_min), int(width), int(height)))

for slide_filename, bboxes in slide_name_to_bboxes.items():
    modified_xml = modify_annotations(f"data/whole-slides/gut/{slide_filename}.xml", bboxes)
    with open(f"{candidate_output_dir}/annotations/{slide_filename}.xml", "w") as file:
        file.write(modified_xml)
