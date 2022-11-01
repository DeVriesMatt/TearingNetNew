import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
from pyntcloud import PyntCloud
import random
import numpy as np
from sklearn.decomposition import PCA
from pathlib import Path


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
            num_points=2046
    ):
        self.annot_df = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.img_size = img_size
        self.label_col = label_col
        self.transform = transform
        self.target_transform = target_transform
        self.cell_component = cell_component
        self.num_points = num_points

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

        if self.num_points == 4096:
            num_str = "_4096"
        else: num_str = ""

        if self.cell_component == "cell":
            component_path = "stacked_pointcloud" + num_str
        elif self.cell_component == "smooth":
            component_path = "stacked_pointcloud_smoothed" + num_str
        else:
            component_path = "stacked_pointcloud_nucleus" + num_str

        try:
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

        except FileNotFoundError:
            return 0, "FileNotFound", 0, 0


class PointCloudDatasetAllBlebbNoc(Dataset):
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
            & ((self.annot_df.Treatment == 'Nocodazole') |
               (self.annot_df.Treatment == 'Blebbistatin'))
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
        alpha = self.new_df.loc[idx, 'yaw']
        beta = self.new_df.loc[idx, 'pitch']
        gamma = self.new_df.loc[idx, 'roll']
        plate_num = "Plate" + str(self.new_df.loc[idx, "PlateNumber"])
        if self.cell_component == "cell":
            component_path = "stacked_pointcloud_aligned"
        else:
            component_path = "stacked_pointcloud_nucleus_aligned"

        img_path = os.path.join(
            self.img_dir,
            plate_num,
            component_path,
            treatment,
            self.new_df.loc[idx, "serialNumber"],
        )
        image = PyntCloud.from_file(img_path + ".ply")
        image = image.points.values


        rotation_matrix = torch.tensor(self.rotation_matrices[random.randrange(0, 24)]).type(torch.FloatTensor)

        # image = (image - mean)
        # print(image.shape)
        # from sklearn.decomposition import PCA
        # pca = PCA(n_components=3)
        # pca.fit(image)
        # # aligned_image, _ = three_d_rotation(image.numpy(),
        # #                                     alpha=-alpha,
        # #                                     beta=-beta,
        # #                                     gamma=-gamma
        # #                                     )
        # aligned_image = image @ pca.components_.T
        # aligned_image = torch.tensor(aligned_image).type(torch.FloatTensor)
        # aligned_image = (aligned_image-torch.mean(aligned_image, 0))/torch.tensor([[20., 20., 20.]])

        image = torch.tensor(image)
        mean = torch.mean(image, 0)
        std = torch.tensor([[20., 20., 20.]])

        image = (image-mean)/std
        rotated_image = torch.matmul(image, rotation_matrix)
        rotated_image = (rotated_image - torch.mean(rotated_image, 0))/std

        # mean_al = torch.mean(aligned_image, 0)
        # std_al = torch.tensor([[20., 20., 20.]])
        # aligned_image = (aligned_image - mean_al) / std_al

        # return encoded label as tensor
        label = self.new_df.loc[idx, "label_col_enc"]
        label = torch.tensor(label)

        # return the classical features as torch tensor
        feats = self.new_df.iloc[idx, 16:-4]
        feats = torch.tensor(feats)

        serial_number = self.new_df.loc[idx, "serialNumber"]

        return image, image, rotated_image, (serial_number, treatment)


class PointCloudDatasetAllAlignedPCA(Dataset):
    def __init__(
            self,
            annotations_file,
            img_dir,
            img_size=400,
            label_col="Treatment",
            transform=None,
            target_transform=None,
            cell_component="cell",
            rotation_matrices=generate_24_rotations(),
            proximal=2
    ):
        self.annot_df = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.img_size = img_size
        self.label_col = label_col
        self.transform = transform
        self.target_transform = target_transform
        self.cell_component = cell_component
        self.proximal = proximal
        self.rotation_matrices = rotation_matrices

        if self.proximal == 2:
            self.new_df = self.annot_df[
                (self.annot_df.xDim <= self.img_size)
                & (self.annot_df.yDim <= self.img_size)
                & (self.annot_df.zDim <= self.img_size)
                ].reset_index(drop=True)
        else:
            self.new_df = self.annot_df[
                (self.annot_df.xDim <= self.img_size)
                & (self.annot_df.yDim <= self.img_size)
                & (self.annot_df.zDim <= self.img_size)
                & (self.annot_df.Proximal == self.proximal)
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
        alpha = self.new_df.loc[idx, 'yaw']
        beta = self.new_df.loc[idx, 'pitch']
        gamma = self.new_df.loc[idx, 'roll']
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

        image = torch.tensor(image)
        mean = torch.mean(image, 0)
        std = torch.tensor([[20., 20., 20.]])
        image = (image - mean) / std

        pca = PCA(n_components=3)
        # pca.fit(image.numpy())

        # aligned_image = torch.tensor(image.numpy() @ pca.components_.T)
        aligned_image = torch.tensor(pca.fit_transform(image.numpy()))
        # return encoded label as tensor
        label = self.new_df.loc[idx, "label_col_enc"]
        label = torch.tensor(label)

        # return the classical features as torch tensor
        feats = self.new_df.iloc[idx, 16:-4]
        feats = torch.tensor(feats)

        serial_number = self.new_df.loc[idx, "serialNumber"]

        return image, aligned_image, aligned_image, (serial_number, treatment)


class GefGapAllAlignedPCA(Dataset):
    def __init__(
            self,
            annotations_file,
            img_dir,
            img_size=400,
            label_col="Treatment",
            transform=None,
            target_transform=None,
            cell_component="cell",
            plate_num="1"
    ):
        self.new_df = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.img_size = img_size
        self.label_col = label_col
        self.transform = transform
        self.target_transform = target_transform
        self.cell_component = cell_component
        self.plate_num = plate_num

    def __len__(self):
        return len(self.new_df)

    def __getitem__(self, idx):
        # read the image
        treatment = self.new_df.loc[idx, "treatment"]
        plate = "Plate" + self.plate_num
        if self.cell_component == "cell":
            component_path = "stacked_pointcloud"
        else:
            component_path = "stacked_pointcloud_nucleus"

        img_path = os.path.join(
            self.img_dir,
            plate,
            component_path,
            "Cells",
            self.new_df.loc[idx, "serialNumber"],
        )
        image = PyntCloud.from_file(img_path + ".ply")
        image = image.points.values

        image = torch.tensor(image)
        mean = torch.mean(image, 0)
        std = torch.tensor([[20., 20., 20.]])
        image = (image - mean) / std

        # pca = PCA(n_components=3)
        # pca.fit(image.numpy())

        # aligned_image = torch.tensor(image.numpy() @ pca.components_.T)
        aligned_image = image
        serial_number = self.new_df.loc[idx, "serialNumber"]

        return image, aligned_image, aligned_image, (serial_number, treatment)


class GefGapAlignedBothPlates(Dataset):
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
        self.new_df = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.img_size = img_size
        self.label_col = label_col
        self.transform = transform
        self.target_transform = target_transform
        self.cell_component = cell_component

    def __len__(self):
        return len(self.new_df)

    def __getitem__(self, idx):
        # read the image
        plate_num = self.new_df.loc[idx, "PlateNumber"]
        treatment = self.new_df.loc[idx, "GEF_GAP_GTPase"]
        plate = "Plate" + str(plate_num)
        if self.cell_component == "cell":
            component_path = "stacked_pointcloud"
        else:
            component_path = "stacked_pointcloud_nucleus"

        img_path = os.path.join(
            self.img_dir,
            plate,
            component_path,
            "Cells",
            self.new_df.loc[idx, "serialNumber"],
        )
        image = PyntCloud.from_file(img_path + ".ply")
        image = image.points.values

        image = torch.tensor(image)
        mean = torch.mean(image, 0)
        std = torch.tensor([[20., 20., 20.]])
        image = (image - mean) / std

        # pca = PCA(n_components=3)
        # pca.fit(image.numpy())

        # aligned_image = torch.tensor(image.numpy() @ pca.components_.T)
        aligned_image = image
        serial_number = self.new_df.loc[idx, "serialNumber"]

        return image, aligned_image, aligned_image, (serial_number, treatment)


class PointCloudDatasetAllAlignedDistal(Dataset):
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
        alpha = self.new_df.loc[idx, 'yaw']
        beta = self.new_df.loc[idx, 'pitch']
        gamma = self.new_df.loc[idx, 'roll']
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
                                            alpha=-alpha,
                                            beta=-beta,
                                            gamma=-gamma
                                            )

        image = torch.tensor(image)
        aligned_image = torch.tensor(aligned_image).type(torch.FloatTensor)
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


class PointCloudDatasetAllAlignedProximal(Dataset):
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
        alpha = self.new_df.loc[idx, 'yaw']
        beta = self.new_df.loc[idx, 'pitch']
        gamma = self.new_df.loc[idx, 'roll']
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
                                            alpha=-alpha,
                                            beta=-beta,
                                            gamma=-gamma
                                            )

        image = torch.tensor(image)
        aligned_image = torch.tensor(aligned_image).type(torch.FloatTensor)
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


class ModelNet40(Dataset):
    def __init__(self,
                 img_dir,
                 train='train',
                 transform=None):

        self.img_dir = Path(img_dir)
        self.train = train
        self.transform = transform
        self.files = list(self.img_dir.glob(f"**/{train}/*.ply"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # read the image
        file = self.files[idx]
        image = PyntCloud.from_file(str(file))
        label = str(file.name)[:-9]
        image = (image.points.values - image.points.values.min()) /\
               (image.points.values.max() - image.points.values.min())

        return image, label


class GefGapDataset(Dataset):
    def __init__(
        self,
        annotations_file,
        img_dir,
        img_size=100,
        label_col="Treatment",
        transform=None,
        target_transform=None,
        cell_component="cell",
        norm_std=True,
    ):
        self.annot_df = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.img_size = img_size
        self.label_col = label_col
        self.transform = transform
        self.target_transform = target_transform
        self.cell_component = cell_component
        self.norm_std = norm_std

        self.new_df = self.annot_df[
            (self.annot_df.xDim_cell <= self.img_size)
            & (self.annot_df.yDim_cell <= self.img_size)
            & (self.annot_df.zDim_cell <= self.img_size)
        ].reset_index(drop=True)

    def __len__(self):
        return len(self.new_df)

    def __getitem__(self, idx):
        # read the image
        plate_num = self.new_df.loc[idx, "PlateNumber"]
        treatment = self.new_df.loc[idx, "GEF_GAP_GTPase"]
        plate = "Plate" + str(plate_num)
        if self.cell_component == "cell":
            component_path = "stacked_pointcloud"
            img_path = os.path.join(
                self.img_dir,
                plate,
                component_path,
                self.new_df.loc[idx, "serialNumber"],
            )
        else:
            component_path = "stacked_pointcloud_nucleus"
            img_path = os.path.join(
                self.img_dir,
                plate,
                component_path,
                "Cells",
                self.new_df.loc[idx, "serialNumber"],
            )

        image = PyntCloud.from_file(img_path + ".ply")
        image = image.points.values

        image = torch.tensor(image)
        mean = torch.mean(image, 0)
        if self.norm_std:
            std = torch.tensor([[20.0, 20.0, 20.0]])
        else:
            std = torch.abs(image - mean).max() * 0.9999999

        image = (image - mean) / std

        serial_number = self.new_df.loc[idx, "serialNumber"]

        return image, treatment, 0, serial_number
