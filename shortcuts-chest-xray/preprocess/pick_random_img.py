import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# Load the data from a CSV file (assuming the CSV file is named 'data.csv')
data = pd.read_csv('../chest_drains_annotations_no_duplicates.csv')

def visualize_and_save_random_image_with_criteria(data, save_path):
    """
    Visualizes and saves a random image from the data frame where Drain == 1 and Pneumothorax == 1.

    Parameters:
    - data (DataFrame): The data frame containing image information.
    - save_path (str): The path where the image will be saved.
    """
    # Filter the data to find rows where Drain == 1 and Pneumothorax == 1
    filtered_data = data[(data['Drain'] == 0) & (data['Pneumothorax'] == 0)]

    # Check if there are any images that meet the criteria
    if filtered_data.empty:
        print("No images found matching the criteria.")
        return

    # Randomly select one image from the filtered data
    image_row = filtered_data.sample(n=1).iloc[0]
    image_index = image_row['Image Index']

    # Assuming the 'Image Index' contains the path to the image
    base_path = '/home/data_shares/purrlab/ChestX-ray14/images/'
    image_path = os.path.join(base_path, image_index)

    # Check if the image file exists
    if not os.path.exists(image_path):
        print(f"No image file found at {image_path}")
        return

    # Load the image
    img = mpimg.imread(image_path)

    # Display the image
    plt.imshow(img, cmap='gray')
    plt.title(f"Image Index: {image_index}\nDrain: {image_row['Drain']}, Pneumothorax: {image_row['Pneumothorax']}")
    plt.axis('off')  # Hide axes
    plt.show()

    # Save the image
    plt.imsave(save_path, img, cmap='gray')
    print(f"Image saved to {save_path}")

# Example usage:
visualize_and_save_random_image_with_criteria(data, 'saved_image.png')
