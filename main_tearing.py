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
    parser = argparse.ArgumentParser(description='DGCNN + Tearing')
    parser.add_argument('--dataset_path', default='/home/mvries/Documents/Datasets/OPM/SingleCellFromNathan_17122021/', type=str)
    parser.add_argument('--dataframe_path',
                        default='/home/mvries/Documents/Datasets/OPM/SingleCellFromNathan_17122021/all_cell_data.csv',
                        type=str)
    parser.add_argument('--output_path', default='./', type=str)
    parser.add_argument('--num_epochs', default=250, type=int)
    parser.add_argument('--fold_path',
                        default='/run/user/1128299809/gvfs/smb-share:server=rds.icr.ac.uk,share=data/DBI/DUDBI/DYNCESYS/mvries/ResultsAlma/TearingNetNew/nets/dgcnn_foldingnet_50_002.pt',
                        type=str)

    args = parser.parse_args()
    df = args.dataframe_path
    root_dir = args.dataset_path
    output_path = args.output_path
    num_epochs = args.num_epochs
    fold_path = args.fold_path

    batch_size = 16
    learning_rate = 0.0000001
    fold = GraphAutoEncoder(num_features=50, k=20, encoder_type="dgcnn", decoder_type='foldingnet')
    checkpoint = torch.load(fold_path)
    model = GraphAutoEncoder(num_features=50, k=20, encoder_type="dgcnn", decoder_type='tearingnet')
    print(model)
    # print(checkpoint['model_state_dict'])
    model_dict = model.state_dict()  # load parameters from pre-trained FoldingNet
    for k in checkpoint['model_state_dict']:
        if k in model_dict:
            model_dict[k] = checkpoint['model_state_dict'][k]
            print("    Found weight: " + k)
        elif k.replace('folding1', 'folding') in model_dict:
            model_dict[k.replace('folding1', 'folding')] = checkpoint['model_state_dict'][k]
            print("    Found weight: " + k)
    model.load_state_dict(model_dict)

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
