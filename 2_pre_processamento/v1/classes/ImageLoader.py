from PIL import Image

class ImageLoader:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = None

    def load_image(self):
        try:
            self.image = Image.open(self.image_path)
            return True  # Successfully loaded the image
        except Exception as e:
            print(f"Failed to load the image: {str(e)}")
            return False  # Failed to load the image

    def display_image(self):
        if self.image:
            self.image.show()
        else:
            print("No image loaded. Use load_image() to load an image first.")

    def process_image(self, width, height):
        if self.image:
            try:
                self.image = self.image.resize((width, height))
                return True  # Successfully processed the image
            except Exception as e:
                print(f"Failed to process the image: {str(e)}")
                return False  # Failed to process the image
        else:
            print("No image loaded. Use load_image() to load an image first.")

    def save_image(self, output_path):
        if self.image:
            try:
                self.image.save(output_path)
                print(f"Image saved to {output_path}")
            except Exception as e:
                print(f"Failed to save the image: {str(e)}")
        else:
            print("No image loaded. Use load_image() to load an image first.")

# Example usage:
# image_loader = ImageLoader("path_to_image.jpg")
# image_loader.load_image()
# image_loader.process_image(800, 600)
# image_loader.display_image()
# image_loader.save_image("processed_image.jpg")
