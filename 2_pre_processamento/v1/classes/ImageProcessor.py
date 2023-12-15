from classes.BoundingBox import BoundingBox
from PIL import Image
import os
import json


class ImageProcessor:
    def __init__(self, image_folder, output_folder, json_file):
        self.image_folder = image_folder
        self.output_folder = output_folder
        self.json_file = json_file

    def get_image_data(self, image_name):
        with open(self.json_file, "r") as f:
            json_data = json.load(f)

        for item in json_data:
            if item['image']['pathname'].split("/")[-1] == image_name:
                return item
        return None

    def create_folders_for_images(self):
        for image_file in os.listdir(self.image_folder):
            image_name = image_file
            image_data = self.get_image_data(image_name)
            if image_data is None:
                continue

            checksum = image_data["image"]["checksum"]
            output_folder = os.path.join(self.output_folder, checksum)
            image_path = os.path.join(self.image_folder, image_file)

            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            self.crop_objects_in_image(image_path, image_data, output_folder)

    def crop_objects_in_image(self, image_path, image_data, output_folder):
        image = Image.open(image_path)

        for i, bbox_data in enumerate(image_data['objects']):
            bbox = BoundingBox(bbox_data['bounding_box']['minimum'], bbox_data['bounding_box']['maximum'])
            r_min, c_min, r_max , c_max = bbox.get_coordinates()

            cropped_image = image.crop((c_min, r_min, c_max, r_max))
            cropped_image.save(os.path.join(output_folder,  f"object_{i}.jpg"))
