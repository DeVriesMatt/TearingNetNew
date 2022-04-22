import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
from pyntcloud import PyntCloud
import numpy as np
import random


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=1, clip=0.02):
    N, C = pointcloud.shape
    rotation = np.copy(pointcloud)
    rotation += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return rotation


def generate_24_rotations():
    res = []
    for id in [[0, 1, 2], [1, 2, 0], [2, 0, 1]]:
        R = np.identity(3)[:, id].astype(int)
        R1= np.asarray([R[:, 0], R[:, 1], R[:, 2]]).T
        R2 = np.asarray([-R[:, 0], -R[:, 1], R[:, 2]]).T
        R3 = np.asarray([-R[:, 0], R[:, 1], -R[:, 2]]).T
        R4 = np.asarray([R[:, 0], -R[:, 1], -R[:, 2]]).T
        res += [R1, R2, R3, R4]
    for id in [[0, 2, 1], [1, 0, 2], [2, 1, 0]]:
        R = np.identity(3)[:, id].astype(int)
        R1 = np.asarray([-R[:, 0], -R[:, 1], -R[:, 2]]).T
        R2 = np.asarray([-R[:, 0], R[:, 1], R[:, 2]]).T
        R3 = np.asarray([R[:, 0], -R[:, 1], R[:, 2]]).T
        R4 = np.asarray([R[:, 0], R[:, 1], -R[:, 2]]).T
        res += [R1, R2, R3, R4]
    return res


def three_d_rotation(pointcloud, alpha, beta, gamma):
    rotation_matrix = np.array(
        [[np.cos(beta) * np.cos(gamma),
          (np.sin(alpha) * np.sin(beta) * np.cos(gamma)) - (np.cos(alpha) * np.cos(gamma)),
          (np.cos(alpha) * np.sin(beta) * np.cos(gamma)) + (np.sin(alpha) * np.sin(gamma))],

         [np.cos(beta) * np.sin(gamma),
          (np.sin(alpha) * np.sin(beta) * np.sin(gamma)) + (np.cos(alpha) * np.cos(gamma)),
          (np.cos(alpha) * np.sin(beta) * np.sin(gamma)) - (np.sin(alpha) * np.cos(gamma))],

         [-np.sin(beta),
          np.sin(alpha) * np.cos(beta),
          np.cos(alpha) * np.cos(beta)]]
    ).squeeze()

    rotation = pointcloud @ rotation_matrix

    return rotation, (alpha, beta, gamma, rotation_matrix)


class PointCloudDataset(Dataset):
    def __init__(
        self,
        annotations_file,
        img_dir,
        img_size=64,
        label_col="Treatment",
        transform=None,
        target_transform=None,
    ):
        self.annot_df = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.img_size = img_size
        self.label_col = label_col
        self.transform = transform
        self.target_transform = target_transform

        self.new_df = self.annot_df[
            (self.annot_df.xDim <= self.img_size)
            & (self.annot_df.yDim <= self.img_size)
            & (self.annot_df.zDim <= self.img_size)
        ].reset_index(drop=True)

        # encode label
        le = LabelEncoder()
        label_col_enc = self.new_df.loc[:, self.label_col]
        label_col_enc = le.fit_transform(label_col_enc)
        self.new_df["label_col_enc"] = label_col_enc

    def __len__(self):
        return len(self.new_df)

    def __getitem__(self, idx):
        # read the image
        treatment = self.new_df.loc[idx, "Treatment"]
        img_path = os.path.join(
            self.img_dir, treatment, self.new_df.loc[idx, "serialNumber"]
        )
        image = PyntCloud.from_file(img_path + ".ply")
        image = torch.tensor(image.points.values)
        mean = torch.tensor([[13.4828, 26.5144, 24.4187]])
        std = torch.tensor([[9.2821, 20.4512, 18.9049]])
        image = (image - mean) / std
        # return encoded label as tensor
        label = self.new_df.loc[idx, "label_col_enc"]
        label = torch.tensor(label)

        # return the classical features as torch tensor
        feats = self.new_df.iloc[idx, 16:-4]
        feats = torch.tensor(feats)

        serial_number = self.new_df.loc[idx, "serialNumber"]

        return image, label, feats, serial_number


class PointCloudDatasetAll(Dataset):
    def __init__(
        self,
        annotations_file,
        img_dir,
        img_size=400,
        label_col="Treatment",
        transform=None,
        target_transform=None,
        cell_component="cell",
    ):
        self.annot_df = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.img_size = img_size
        self.label_col = label_col
        self.transform = transform
        self.target_transform = target_transform
        self.cell_component = cell_component

        self.new_df = self.annot_df[
            (self.annot_df.xDim <= self.img_size)
            & (self.annot_df.yDim <= self.img_size)
            & (self.annot_df.zDim <= self.img_size)
        ].reset_index(drop=True)

        # encode label
        le = LabelEncoder()
        label_col_enc = self.new_df.loc[:, self.label_col]
        label_col_enc = le.fit_transform(label_col_enc)
        self.new_df["label_col_enc"] = label_col_enc

    def __len__(self):
        return len(self.new_df)

    def __getitem__(self, idx):
        # read the image
        treatment = self.new_df.loc[idx, "Treatment"]
        plate_num = "Plate" + str(self.new_df.loc[idx, "PlateNumber"])
        if self.cell_component == "cell":
            component_path = "stacked_pointcloud"
        else:
            component_path = "stacked_pointcloud_nucleus"

        img_path = os.path.join(
            self.img_dir,
            plate_num,
            component_path,
            treatment,
            self.new_df.loc[idx, "serialNumber"],
        )
        image = PyntCloud.from_file(img_path + ".ply")
        image = torch.tensor(image.points.values)
        mean = torch.mean(image, 0)
        std = torch.tensor([[20., 20., 20.]])
        image = (image - mean) / std
        # return encoded label as tensor
        label = self.new_df.loc[idx, "label_col_enc"]
        label = torch.tensor(label)

        # return the classical features as torch tensor
        feats = self.new_df.iloc[idx, 16:-4]
        feats = torch.tensor(feats)

        serial_number = self.new_df.loc[idx, "serialNumber"]

        return image, treatment, feats, serial_number


class PointCloudDatasetAllAligned(Dataset):
    def __init__(
        self,
        annotations_file,
        img_dir,
        img_size=400,
        label_col="Treatment",
        transform=None,
        target_transform=None,
        cell_component="cell",
            rotation_matrices=generate_24_rotations()
    ):
        self.annot_df = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.img_size = img_size
        self.label_col = label_col
        self.transform = transform
        self.target_transform = target_transform
        self.cell_component = cell_component
        self.rotation_matrices = rotation_matrices

        self.new_df = self.annot_df[
            (self.annot_df.xDim <= self.img_size)
            & (self.annot_df.yDim <= self.img_size)
            & (self.annot_df.zDim <= self.img_size)
        ].reset_index(drop=True)

        # encode label
        le = LabelEncoder()
        label_col_enc = self.new_df.loc[:, self.label_col]
        label_col_enc = le.fit_transform(label_col_enc)
        self.new_df["label_col_enc"] = label_col_enc


    def __len__(self):
        return len(self.new_df)

    def __getitem__(self, idx):
        # read the image
        treatment = self.new_df.loc[idx, "Treatment"]
        alpha = 0
        beta = self.new_df.loc[idx, 'Pitch_cell']
        gamma = self.new_df.loc[idx, 'Azimuth_cell']
        plate_num = "Plate" + str(self.new_df.loc[idx, "PlateNumber"])
        if self.cell_component == "cell":
            component_path = "stacked_pointcloud"
        else:
            component_path = "stacked_pointcloud_nucleus"

        img_path = os.path.join(
            self.img_dir,
            plate_num,
            component_path,
            treatment,
            self.new_df.loc[idx, "serialNumber"],
        )
        image = PyntCloud.from_file(img_path + ".ply")
        image = image.points.values
        aligned_image, _ = three_d_rotation(image,
                                            alpha=-abs(alpha),
                                            beta=-abs(beta),
                                            gamma=-abs(gamma)
                                            )

        image = torch.tensor(image)
        aligned_image = torch.tensor(aligned_image)
        rotation_matrix = torch.tensor(self.rotation_matrices[random.randrange(0, 24)]).type(torch.FloatTensor)

        mean = torch.mean(image, 0)
        std = torch.tensor([[20., 20., 20.]])
        image = (image - mean) / std
        rotated_image = torch.matmul(image, rotation_matrix)

        mean_al = torch.mean(aligned_image, 0)
        std_al = torch.tensor([[20., 20., 20.]])
        aligned_image = (aligned_image - mean_al) / std_al

        # return encoded label as tensor
        label = self.new_df.loc[idx, "label_col_enc"]
        label = torch.tensor(label)

        # return the classical features as torch tensor
        feats = self.new_df.iloc[idx, 16:-4]
        feats = torch.tensor(feats)

        serial_number = self.new_df.loc[idx, "serialNumber"]

        return image, aligned_image, rotated_image, (serial_number, treatment)


class PointCloudDatasetAllRotation(Dataset):
    def __init__(
            self,
            annotations_file,
            img_dir,
            img_size=400,
            label_col="Treatment",
            transform=None,
            target_transform=None,
            cell_component="cell",
            rotation_matrices=generate_24_rotations()

    ):
        self.annot_df = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.img_size = img_size
        self.label_col = label_col
        self.transform = transform
        self.target_transform = target_transform
        self.cell_component = cell_component
        self.rotation_matrices = rotation_matrices

        self.new_df = self.annot_df[
            (self.annot_df.xDim <= self.img_size)
            & (self.annot_df.yDim <= self.img_size)
            & (self.annot_df.zDim <= self.img_size)
            ].reset_index(drop=True)

        # encode label
        le = LabelEncoder()
        label_col_enc = self.new_df.loc[:, self.label_col]
        label_col_enc = le.fit_transform(label_col_enc)
        self.new_df["label_col_enc"] = label_col_enc

    def __len__(self):
        return len(self.new_df)

    def __getitem__(self, idx):
        # read the image
        treatment = self.new_df.loc[idx, "Treatment"]
        plate_num = "Plate" + str(self.new_df.loc[idx, "PlateNumber"])
        if self.cell_component == "cell":
            component_path = "stacked_pointcloud"
        else:
            component_path = "stacked_pointcloud_nucleus"

        img_path = os.path.join(
            self.img_dir,
            plate_num,
            component_path,
            treatment,
            self.new_df.loc[idx, "serialNumber"],
        )
        image = PyntCloud.from_file(img_path + ".ply")
        image = torch.tensor(image.points.values)
        mean = torch.mean(image, 0)
        std = torch.tensor([[20., 20., 20.]])
        image = (image - mean) / std
        rotation_matrix = torch.tensor(self.rotation_matrices[random.randrange(0, 24)]).type(torch.FloatTensor)
        rotated_image = torch.matmul(image, rotation_matrix)
        # return encoded label as tensor
        label = self.new_df.loc[idx, "label_col_enc"]
        label = torch.tensor(label)

        # return the classical features as torch tensor
        feats = self.new_df.iloc[idx, 16:-4]
        feats = torch.tensor(feats)

        serial_number = self.new_df.loc[idx, "serialNumber"]

        return image, treatment, rotated_image, serial_number


class PointCloudDatasetAllDistal(Dataset):
    def __init__(
        self,
        annotations_file,
        img_dir,
        img_size=400,
        label_col="Treatment",
        transform=None,
        target_transform=None,
        cell_component="cell",
    ):
        self.annot_df = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.img_size = img_size
        self.label_col = label_col
        self.transform = transform
        self.target_transform = target_transform
        self.cell_component = cell_component

        self.new_df = self.annot_df[
            (self.annot_df.xDim <= self.img_size)
            & (self.annot_df.yDim <= self.img_size)
            & (self.annot_df.zDim <= self.img_size)
            & (self.annot_df.Proximal == 0)
        ].reset_index(drop=True)

        # encode label
        le = LabelEncoder()
        label_col_enc = self.new_df.loc[:, self.label_col]
        label_col_enc = le.fit_transform(label_col_enc)
        self.new_df["label_col_enc"] = label_col_enc

    def __len__(self):
        return len(self.new_df)

    def __getitem__(self, idx):
        # read the image
        treatment = self.new_df.loc[idx, "Treatment"]
        plate_num = "Plate" + str(self.new_df.loc[idx, "PlateNumber"])
        if self.cell_component == "cell":
            component_path = "stacked_pointcloud"
        else:
            component_path = "stacked_pointcloud_nucleus"

        img_path = os.path.join(
            self.img_dir,
            plate_num,
            component_path,
            treatment,
            self.new_df.loc[idx, "serialNumber"],
        )
        image = PyntCloud.from_file(img_path + ".ply")
        image = torch.tensor(image.points.values)
        mean = torch.mean(image, 0)
        std = torch.tensor([[20., 20., 20.]])
        image = (image - mean) / std
        # return encoded label as tensor
        label = self.new_df.loc[idx, "label_col_enc"]
        label = torch.tensor(label)

        # return the classical features as torch tensor
        feats = self.new_df.iloc[idx, 16:-4]
        feats = torch.tensor(feats)

        serial_number = self.new_df.loc[idx, "serialNumber"]

        return image, treatment, feats, serial_number


class PointCloudDatasetAllDistalRotation(Dataset):
    def __init__(
        self,
        annotations_file,
        img_dir,
        img_size=400,
        label_col="Treatment",
        transform=None,
        target_transform=None,
        cell_component="cell",
            rotation_matrices=generate_24_rotations()

    ):
        self.annot_df = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.img_size = img_size
        self.label_col = label_col
        self.transform = transform
        self.target_transform = target_transform
        self.cell_component = cell_component
        self.rotation_matrices = rotation_matrices

        self.new_df = self.annot_df[
            (self.annot_df.xDim <= self.img_size)
            & (self.annot_df.yDim <= self.img_size)
            & (self.annot_df.zDim <= self.img_size)
            & (self.annot_df.Proximal == 0)
        ].reset_index(drop=True)

        # encode label
        le = LabelEncoder()
        label_col_enc = self.new_df.loc[:, self.label_col]
        label_col_enc = le.fit_transform(label_col_enc)
        self.new_df["label_col_enc"] = label_col_enc

    def __len__(self):
        return len(self.new_df)

    def __getitem__(self, idx):
        # read the image
        treatment = self.new_df.loc[idx, "Treatment"]
        plate_num = "Plate" + str(self.new_df.loc[idx, "PlateNumber"])
        if self.cell_component == "cell":
            component_path = "stacked_pointcloud"
        else:
            component_path = "stacked_pointcloud_nucleus"

        img_path = os.path.join(
            self.img_dir,
            plate_num,
            component_path,
            treatment,
            self.new_df.loc[idx, "serialNumber"],
        )
        image = PyntCloud.from_file(img_path + ".ply")
        image = torch.tensor(image.points.values)
        mean = torch.mean(image, 0)
        std = torch.tensor([[20., 20., 20.]])
        image = (image - mean) / std
        rotation_matrix = torch.tensor(self.rotation_matrices[random.randrange(0, 24)]).type(torch.FloatTensor)
        rotated_image = torch.matmul(image, rotation_matrix)
        # return encoded label as tensor
        label = self.new_df.loc[idx, "label_col_enc"]
        label = torch.tensor(label)

        # return the classical features as torch tensor
        feats = self.new_df.iloc[idx, 16:-4]
        feats = torch.tensor(feats)

        serial_number = self.new_df.loc[idx, "serialNumber"]

        return image, treatment, rotated_image, serial_number


class PointCloudDatasetAllProximal(Dataset):
    def __init__(
        self,
        annotations_file,
        img_dir,
        img_size=400,
        label_col="Treatment",
        transform=None,
        target_transform=None,
        cell_component="cell",
    ):
        self.annot_df = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.img_size = img_size
        self.label_col = label_col
        self.transform = transform
        self.target_transform = target_transform
        self.cell_component = cell_component

        self.new_df = self.annot_df[
            (self.annot_df.xDim <= self.img_size)
            & (self.annot_df.yDim <= self.img_size)
            & (self.annot_df.zDim <= self.img_size)
            & (self.annot_df.Proximal == 1)
        ].reset_index(drop=True)

        # encode label
        le = LabelEncoder()
        label_col_enc = self.new_df.loc[:, self.label_col]
        label_col_enc = le.fit_transform(label_col_enc)
        self.new_df["label_col_enc"] = label_col_enc

    def __len__(self):
        return len(self.new_df)

    def __getitem__(self, idx):
        # read the image
        treatment = self.new_df.loc[idx, "Treatment"]
        plate_num = "Plate" + str(self.new_df.loc[idx, "PlateNumber"])
        if self.cell_component == "cell":
            component_path = "stacked_pointcloud"
        else:
            component_path = "stacked_pointcloud_nucleus"

        img_path = os.path.join(
            self.img_dir,
            plate_num,
            component_path,
            treatment,
            self.new_df.loc[idx, "serialNumber"],
        )
        image = PyntCloud.from_file(img_path + ".ply")
        image = torch.tensor(image.points.values)
        mean = torch.mean(image, 0)
        std = torch.tensor([[20., 20., 20.]])
        image = (image - mean) / std
        # return encoded label as tensor
        label = self.new_df.loc[idx, "label_col_enc"]
        label = torch.tensor(label)

        # return the classical features as torch tensor
        feats = self.new_df.iloc[idx, 16:-4]
        feats = torch.tensor(feats)

        serial_number = self.new_df.loc[idx, "serialNumber"]

        return image, treatment, feats, serial_number


class PointCloudDatasetAllProximalRotation(Dataset):
    def __init__(
        self,
        annotations_file,
        img_dir,
        img_size=400,
        label_col="Treatment",
        transform=None,
        target_transform=None,
        cell_component="cell",
            rotation_matrices=generate_24_rotations()

    ):
        self.annot_df = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.img_size = img_size
        self.label_col = label_col
        self.transform = transform
        self.target_transform = target_transform
        self.cell_component = cell_component
        self.rotation_matrices = rotation_matrices

        self.new_df = self.annot_df[
            (self.annot_df.xDim <= self.img_size)
            & (self.annot_df.yDim <= self.img_size)
            & (self.annot_df.zDim <= self.img_size)
            & (self.annot_df.Proximal == 1)
        ].reset_index(drop=True)

        # encode label
        le = LabelEncoder()
        label_col_enc = self.new_df.loc[:, self.label_col]
        label_col_enc = le.fit_transform(label_col_enc)
        self.new_df["label_col_enc"] = label_col_enc

    def __len__(self):
        return len(self.new_df)

    def __getitem__(self, idx):
        # read the image
        treatment = self.new_df.loc[idx, "Treatment"]
        plate_num = "Plate" + str(self.new_df.loc[idx, "PlateNumber"])
        if self.cell_component == "cell":
            component_path = "stacked_pointcloud"
        else:
            component_path = "stacked_pointcloud_nucleus"

        img_path = os.path.join(
            self.img_dir,
            plate_num,
            component_path,
            treatment,
            self.new_df.loc[idx, "serialNumber"],
        )
        image = PyntCloud.from_file(img_path + ".ply")
        image = torch.tensor(image.points.values)
        mean = torch.mean(image, 0)
        std = torch.tensor([[20., 20., 20.]])
        image = (image - mean) / std
        rotation_matrix = torch.tensor(self.rotation_matrices[random.randrange(0, 24)]).type(torch.FloatTensor)
        rotated_image = torch.matmul(image, rotation_matrix)
        # return encoded label as tensor
        label = self.new_df.loc[idx, "label_col_enc"]
        label = torch.tensor(label)

        # return the classical features as torch tensor
        feats = self.new_df.iloc[idx, 16:-4]
        feats = torch.tensor(feats)

        serial_number = self.new_df.loc[idx, "serialNumber"]

        return image, treatment, rotated_image, serial_number


class PointCloudDatasetAllPRotation(Dataset):
    def __init__(
        self,
        annotations_file,
        img_dir,
        img_size=400,
        label_col="Treatment",
        transform=None,
        target_transform=None,
        cell_component="cell",
            rotation_matrices=generate_24_rotations()

    ):
        self.annot_df = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.img_size = img_size
        self.label_col = label_col
        self.transform = transform
        self.target_transform = target_transform
        self.cell_component = cell_component
        self.rotation_matrices = rotation_matrices

        self.new_df = self.annot_df[
            (self.annot_df.xDim <= self.img_size)
            & (self.annot_df.yDim <= self.img_size)
            & (self.annot_df.zDim <= self.img_size)
        ].reset_index(drop=True)

        # encode label
        le = LabelEncoder()
        label_col_enc = self.new_df.loc[:, self.label_col]
        label_col_enc = le.fit_transform(label_col_enc)
        self.new_df["label_col_enc"] = label_col_enc

    def __len__(self):
        return len(self.new_df)

    def __getitem__(self, idx):
        # read the image
        treatment = self.new_df.loc[idx, "Treatment"]
        plate_num = "Plate" + str(self.new_df.loc[idx, "PlateNumber"])
        if self.cell_component == "cell":
            component_path = "stacked_pointcloud"
        else:
            component_path = "stacked_pointcloud_nucleus"

        img_path = os.path.join(
            self.img_dir,
            plate_num,
            component_path,
            treatment,
            self.new_df.loc[idx, "serialNumber"],
        )
        image = PyntCloud.from_file(img_path + ".ply")
        image = torch.tensor(image.points.values)
        mean = torch.mean(image, 0)
        std = torch.tensor([[20., 20., 20.]])
        image = (image - mean) / std
        rotation_matrix = torch.tensor(self.rotation_matrices[random.randrange(0, 24)]).type(torch.FloatTensor)
        rotated_image = torch.matmul(image, rotation_matrix)
        # return encoded label as tensor
        label = self.new_df.loc[idx, "label_col_enc"]
        label = torch.tensor(label)

        # return the classical features as torch tensor
        feats = self.new_df.iloc[idx, 16:-4]
        feats = torch.tensor(feats)

        serial_number = self.new_df.loc[idx, "serialNumber"]

        return image, treatment, rotated_image, serial_number


class PointCloudDatasetAllBoth(Dataset):
    def __init__(self, annotations_file,
                 img_dir,
                 img_size=400,
                 label_col='Treatment',
                 transform=None,
                 target_transform=None,
                 centring_only=False,
                 cell_component='cell',
                 proximal=1):
        self.annot_df = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.img_size = img_size
        self.label_col = label_col
        self.transform = transform
        self.target_transform = target_transform
        self.cell_component = cell_component
        self.proximal = proximal

        self.new_df = self.annot_df[(self.annot_df.xDim <= self.img_size) &
                                    (self.annot_df.yDim <= self.img_size) &
                                    (self.annot_df.zDim <= self.img_size)
                                    ].reset_index(drop=True)

        # encode label
        le = LabelEncoder()
        label_col_enc = self.new_df.loc[:, self.label_col]
        label_col_enc = le.fit_transform(label_col_enc)
        self.new_df['label_col_enc'] = label_col_enc

    def __len__(self):
        return len(self.new_df)

    def __getitem__(self, idx):
        # read the image
        treatment = self.new_df.loc[idx, 'Treatment']
        plate_num = 'Plate' + str(self.new_df.loc[idx, 'PlateNumber'])
        cell_path = 'stacked_pointcloud'
        nuc_path = 'stacked_pointcloud_nucleus'

        cell_img_path = os.path.join(self.img_dir,
                                     plate_num,
                                     cell_path,
                                     treatment,
                                     self.new_df.loc[idx, 'serialNumber'])

        nuc_img_path = os.path.join(self.img_dir,
                                    plate_num,
                                    nuc_path,
                                    treatment,
                                    self.new_df.loc[idx, 'serialNumber'])

        cell = PyntCloud.from_file(cell_img_path + '.ply')
        nuc = PyntCloud.from_file(nuc_img_path + '.ply')

        cell = torch.tensor(cell.points.values)
        nuc = torch.tensor(nuc.points.values)
        full = torch.tensor(np.concatenate((cell[:1024], nuc[:1024])))
        mean = torch.mean(full, 0)
        std = torch.tensor([[20., 20., 20.]])

        image = (full - mean) / std

        # return encoded label as tensor
        label = self.new_df.loc[idx, 'label_col_enc']
        label = torch.tensor(label)

        # return the classical features as torch tensor
        feats = self.new_df.iloc[idx, 16:-4]
        feats = torch.tensor(feats)

        serial_number = self.new_df.loc[idx, 'serialNumber']

        return image, treatment, feats, serial_number
