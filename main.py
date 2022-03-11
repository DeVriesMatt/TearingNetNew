import torch
from torch.utils.data import DataLoader
from training_functions import train
from encoders.dgcnn import ChamferLoss
from dataset import PointCloudDatasetAllBoth
from autoencoder import GraphAutoEncoder
from chamfer import ChamferLoss1
import argparse


if __name__ == "__main__":
    # PATH_TO_DATASET = (
    #     "/home/mvries/Documents/Datasets/OPM/SingleCellFromNathan_17122021/"
    # )
    # df = "/home/mvries/Documents/Datasets/OPM/SingleCellFromNathan_17122021/all_cell_data.csv"
    # root_dir = "/home/mvries/Documents/Datasets/OPM/SingleCellFromNathan_17122021/"
    parser = argparse.ArgumentParser(description='Use DCEC for clustering')
    parser.add_argument('--dataset_path', default='./', type=str)
    parser.add_argument('--dataframe_path', default='./', type=str)
    parser.add_argument('--output_path', default='./', type=str)
    parser.add_argument('--num_epochs', default=250, type=int)

    args = parser.parse_args()
    df = args.dataframe_path
    root_dir = args.dataset_path
    output_path = args.output_path
    num_epochs = args.num_epochs

    batch_size = 16
    learning_rate = 0.0000001

    model = GraphAutoEncoder(num_features=50, k=20, encoder_type="dgcnn", decoder_type='foldingnet')

    dataset = PointCloudDatasetAllBoth(df, root_dir)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = ChamferLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate * 16 / batch_size,
        betas=(0.9, 0.999),
        weight_decay=1e-6,
    )

    train(model, dataloader, num_epochs, criterion, optimizer, output_path)
