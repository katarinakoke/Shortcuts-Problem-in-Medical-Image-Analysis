import tensorflow as tf
import pandas as pd

class ImageDataset:
    def __init__(self, cfg, mode='train', y_label_col='Pneumothorax', a_label_col='Support Devices'):
        self.cfg = cfg
        self.mode = mode
        self.csv_file = cfg[mode + '_csv'] if mode + '_csv' in cfg else None
        self.base_path = cfg['base_path']
        self.y_label_col = y_label_col
        self.a_label_col = a_label_col
        self.img_height = cfg.get('height', 512)
        self.img_width = cfg.get('width', 512)

    def preprocess_image(self, image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=1)
        img = tf.image.resize(img, [self.img_height, self.img_width])
        img = img / 255.0  # Normalize to [0,1]
        img = tf.expand_dims(img, axis=-1)  # Make sure the image has shape (height, width, 1)
        return img

    @staticmethod
    def preprocess(images, y_labels, a_labels):
        # Ensure labels are reshaped to be 2-dimensional with shape (1, 1)
        y_labels = tf.reshape(y_labels, (-1, 1))
        a_labels = tf.reshape(a_labels, (-1, 1))
        return images, y_labels, a_labels

    def load_data(self):
        df = pd.read_csv(self.csv_file)
        df['Path'] = df['Path'].apply(lambda x: f"{self.base_path}/{x}")
        image_paths = df['Path'].values
        y_labels = df[self.y_label_col].values.astype(float)
        a_labels = df[self.a_label_col].values.astype(float)

        image_ds = tf.data.Dataset.from_tensor_slices(image_paths).map(self.preprocess_image)
        y_label_ds = tf.data.Dataset.from_tensor_slices(y_labels)
        a_label_ds = tf.data.Dataset.from_tensor_slices(a_labels)

        # Zip the datasets and then apply the preprocess method to reshape labels
        dataset = tf.data.Dataset.zip((image_ds, y_label_ds, a_label_ds))
        dataset = dataset.map(lambda img, y, a: ImageDataset.preprocess(img, y, a))

        return dataset

