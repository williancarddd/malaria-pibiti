from classes.ImageProcessor import ImageProcessor


def main():
    image_folder = "/media/william/NVME/projects/malaria-pibiti/1_entrada/malaria/malaria/images"
    output_folder = "/media/william/NVME/projects/malaria-pibiti/1_entrada/malaria/malaria/train"
    json_file = "/media/william/NVME/projects/malaria-pibiti/1_entrada/malaria/malaria/training.json"

    processor = ImageProcessor(image_folder, output_folder, json_file)
    processor.create_folders_for_images()

if __name__ == "__main__":
    main()
