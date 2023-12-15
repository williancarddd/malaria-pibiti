from PIL import Image
import matplotlib.pyplot as plt
import os, json, io
import concurrent.futures
from classes.BoundingBoxExtractor import BoundingBoxExtractor
from classes.ImageLoader import ImageLoader


class ImagePlotter:
    def __init__(self, image_data):
        self.image_data = image_data
        image_path = os.path.join(os.getcwd(), 'images', image_data['image']['pathname'].split('/')[-1])
        self.image_loader = ImageLoader(image_path)

    def plot_objects(self, image: Image):
        """
        Plot the image with the bounding boxes
        """
        plt.imshow(image)
        plt.axis('off')
        plt.show()

    def get_plotted_image(self):
        """
        Original Image with areas of interest plotted
        """
        plt.figure(figsize=(10, 8))
        plt.imshow(self.image_loader.get_image())
        plt.axis('off')

        bounding_box_extractor = BoundingBoxExtractor(self.image_data)
        bounding_boxes = bounding_box_extractor.extract_bounding_boxes()

        for bounding_box in bounding_boxes:
            r_min, c_min, r_max, c_max = bounding_box.get_coordinates()
            category = self.image_data['objects'][bounding_boxes.index(bounding_box)]['category']
            color = "blue"
            if category != "red blood cell":
                color = "red"
                plt.text(c_min, r_min - 5, category, fontsize=12, color=color, backgroundcolor='white')
            plt.plot([c_min, c_max], [r_min, r_min], color=color, linewidth=2)
            plt.plot([c_max, c_max], [r_min, r_max], color=color, linewidth=2)
            plt.plot([c_max, c_min], [r_max, r_max], color=color, linewidth=2)
            plt.plot([c_min, c_min], [r_max, r_min], color=color, linewidth=2)

        plt.axis('off')
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        image_buffer = io.BytesIO()
        plt.savefig(image_buffer, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()

        image_buffer.seek(0)
        return Image.open(image_buffer)
    
    def create_galery_for_objects(self, name_folder):
        """
        
        
        path_folder = os.path.join(os.getcwd(), name_folder, self.image_data['image']['checksum']) # create a folder in "name_folder" with the name checksum id


        if not os.path.exists(path_folder):
            os.makedirs(path_folder)
                                                        #clear folder if it is not empty
        else:
            for filename in os.listdir(path_folder):
                file_path = os.path.join(path_folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))
        """
                                                        # for each object in the image, save the croped image in the folder
        bounding_box_extractor = BoundingBoxExtractor(self.image_data)
        bounding_boxes = bounding_box_extractor.extract_bounding_boxes()
        for bounding_box in bounding_boxes:
            r_min, c_min, r_max, c_max = bounding_box.get_coordinates()
            category = self.image_data['objects'][bounding_boxes.index(bounding_box)]['category']
            color = "blue"
            if category != "red blood cell":  color = "red"
            croped_image = self.image_loader.get_image().crop((c_min, r_min, c_max, r_max))
           
            new_image = croped_image.resize((224, 224))  # deixar todos do mesmo tamanho
            category_new = category.replace(" ", "-")
            new_image.save(os.path.join(name_folder, str(self.image_data['image']['checksum'][::6]) + "_" + str(bounding_boxes.index(bounding_box)) + '_' + category_new + '.png'))
            new_image.close()
    
    def destroy(self):
        self.image_loader.destroy()
    

# croped_images_test
# croped_images_training



def process_image_data(image_data, name_folder):
    for i in range(len(image_data)):
        plotter = ImagePlotter(image_data[i])
        plotter.create_galery_for_objects(name_folder)
        plotter.destroy()
        print(f"{i} ", image_data[i]['image']['checksum'])

def main():
    name_folder = 'croped_images_test'
    path_jsons = os.path.join(os.getcwd(), 'test.json')

    with open(path_jsons, 'r') as f:
        image_data = json.load(f)

    # Divide image data into chunks of 10
    chunk_size = 20
    chunks = [image_data[i:i+chunk_size] for i in range(0, len(image_data), chunk_size)]

    # Process chunks of data using thread pool with 10 threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        future_to_chunk = {executor.submit(process_image_data, chunk, name_folder): chunk for chunk in chunks}

        for future in concurrent.futures.as_completed(future_to_chunk):
            chunk = future_to_chunk[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error occurred while processing chunk: {e}")

if __name__ == "__main__":
    main()



    