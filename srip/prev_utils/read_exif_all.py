from PIL import Image
from PIL.ExifTags import TAGS
import sys

def read_exif(image_path):
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()
        
        if exif_data is None:
            print("No EXIF data found.")
            return

        print(f"EXIF data for {image_path}:")
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            print(f"{tag:25}: {value}")
    
    except Exception as e:
        print(f"Error reading EXIF data: {e}")

if __name__ == "__main__":
    import os
    
    path = os.listdir()
    for i in path: 
        read_exif(i)

