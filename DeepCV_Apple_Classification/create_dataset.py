import os
import random
from PIL import Image

# Set the path of the original apple images
original_image_path = "apple.jpg"

# Set the size of the white background
background_size = (500, 500)

# Set the number of images to generate
num_images = 1000

# Create the directories for the quadrants
os.makedirs("data/quadrant1", exist_ok=True)
os.makedirs("data/quadrant2", exist_ok=True)
os.makedirs("data/quadrant3", exist_ok=True)
os.makedirs("data/quadrant4", exist_ok=True)

# Load the original apple image
original_image = Image.open(original_image_path)
original_image = original_image.resize((200, 200))


for i in range(num_images):
    # Create a new white background
    background = Image.new("RGB", background_size, (255, 255, 255))
    
    # Randomly place the apple image in one of the four quadrants
    x = random.randint(0, background_size[0] // 2 - original_image.size[0])
    y = random.randint(0, background_size[1] // 2 - original_image.size[1])
    
    quadrant = random.randint(1, 4)
    if quadrant == 1:
        background.paste(original_image, (x, y))
        background.save("data/quadrant1/apple_{}.jpg".format(i))
    elif quadrant == 2:
        background.paste(original_image, (x + background_size[0] // 2, y))
        background.save("data/quadrant2/apple_{}.jpg".format(i))
    elif quadrant == 3:
        background.paste(original_image, (x, y + background_size[1] // 2))
        background.save("data/quadrant3/apple_{}.jpg".format(i))
    else:
        background.paste(original_image, (x + background_size[0] // 2, y + background_size[1] // 2))
        background.save("data/quadrant4/apple_{}.jpg".format(i))
