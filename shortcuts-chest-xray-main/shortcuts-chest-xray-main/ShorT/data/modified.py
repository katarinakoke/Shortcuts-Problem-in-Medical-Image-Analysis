import pandas as pd
import tensorflow as tf

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
        self.dict = {'1.0': '1', '1': '1', '': '0', '0.0': '0', '0': '0', '-1.0': '0', '-1': '0'}
        
    def preprocess_image(self, image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=1)
        img = tf.image.resize(img, [self.img_height, self.img_width])
        img = img / 255.0  # Normalize to [0,1]
        img = tf.expand_dims(img, axis=-1)  # Ensure image has shape (height, width, 1)
        return img

    @staticmethod
    def preprocess_gender(df):
        df['Sex'] = df['Sex'].replace('Unknown', 'Male')  # Impute 'Unknown' with 'Male'
        df['Sex'] = df['Sex'].map({'Female': 0, 'Male': 1}).astype(int)
        return df

    @staticmethod
    def preprocess_age(df):
        median_age = df.loc[df['Age'] > 0, 'Age'].median()
        df['Age'] = df['Age'].replace(0, median_age)
        max_age = df['Age'].max()
        df['Age'] = df['Age'] / max_age  # Normalize ages to a [0, 1] range
        return df

    @staticmethod
    def encode_categories(df):
        frontal_lateral_dummies = pd.get_dummies(df['Frontal/Lateral'], prefix='Position')
        df = pd.concat([df, frontal_lateral_dummies], axis=1).drop('Frontal/Lateral', axis=1)
        
        ap_pa_dummies = pd.get_dummies(df['AP/PA'], prefix='View')
        df = pd.concat([df, ap_pa_dummies], axis=1).drop('AP/PA', axis=1)
        return df

    @staticmethod
    def preprocess(images, y_labels, a_labels):
        y_labels = tf.reshape(y_labels, (-1, 1))
        a_labels = tf.reshape(a_labels, (-1, 1))
        return images, y_labels, a_labels

    def load_data(self, subsample_size=None):
        df = pd.read_csv(self.csv_file)
        df['Path'] = df['Path'].apply(lambda x: f"{self.base_path}/{x}")

        df = self.preprocess_gender(df)
        df = self.preprocess_age(df)
        df = self.encode_categories(df)

        # Apply label mapping for all relevant columns
        for col in df.columns:
            if col in [self.y_label_col, self.a_label_col] or col.endswith('Finding'):
                df[col] = df[col].astype(str).map(self.dict).fillna('0').astype(float)

        image_paths = df['Path'].values
        y_labels = df[self.y_label_col].values
        a_labels = df[self.a_label_col].values

        image_ds = tf.data.Dataset.from_tensor_slices(image_paths).map(self.preprocess_image)
        y_label_ds = tf.data.Dataset.from_tensor_slices(y_labels)
        a_label_ds = tf.data.Dataset.from_tensor_slices(a_labels)

        dataset = tf.data.Dataset.zip((image_ds, y_label_ds, a_label_ds))
        dataset = dataset.map(self.preprocess)

        if subsample_size is not None:
            dataset = dataset.shuffle(buffer_size=10000, reshuffle_each_iteration=True).take(subsample_size)

        return dataset
