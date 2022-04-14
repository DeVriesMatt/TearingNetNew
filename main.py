import torch
from torch.utils.data import DataLoader
from training_functions import train
from encoders.dgcnn import ChamferLoss
from dataset import PointCloudDatasetAllBoth
from autoencoder import GraphAutoEncoder
from chamfer import ChamferLoss1
import argparse
import os


def create_dir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    # PATH_TO_DATASET = (
    #     "/home/mvries/Documents/Datasets/OPM/SingleCellFromNathan_17122021/"
    # )
    # df = "/home/mvries/Documents/Datasets/OPM/SingleCellFromNathan_17122021/all_cell_data.csv"
    # root_dir = "/home/mvries/Documents/Datasets/OPM/SingleCellFromNathan_17122021/"
    parser = argparse.ArgumentParser(description='DGCNN + Folding')
    parser.add_argument('--dataset_path', default='/home/mvries/Documents/Datasets/OPM/SingleCellFromNathan_17122021/', type=str)
    parser.add_argument('--dataframe_path',
                        default='/home/mvries/Documents/Datasets/OPM/SingleCellFromNathan_17122021/all_cell_data.csv',
                        type=str)
    parser.add_argument('--output_path', default='./', type=str)
    parser.add_argument('--num_epochs', default=250, type=int)
    parser.add_argument('--fold_path',
                        default='/run/user/1128299809/gvfs/smb-share:server=rds.icr.ac.uk,share=data/DBI/DUDBI/DYNCESYS/mvries/ResultsAlma/TearingNetNew/shapenet/nets/dgcnn_foldingnet_128_001.pt',
                        type=str)
    parser.add_argument('--dgcnn_path',
                        default='/run/user/1128299809/gvfs/smb-share:server=rds.icr.ac.uk,share=data/DBI/DUDBI/DYNCESYS/mvries/Reconstruct_dgcnn_cls_k20_plane/models/shapenetcorev2_250.pkl',
                        type=str)
    parser.add_argument('--num_features',
                        default=128,
                        type=int)
    parser.add_argument('--k',
                        default=20,
                        type=int)
    parser.add_argument('--encoder_type',
                        default="dgcnn",
                        type=str)
    parser.add_argument('--decoder_type',
                        default="foldingnet",
                        type=str)
    parser.add_argument('--learning_rate',
                        default=0.00001,
                        type=float)
    parser.add_argument('--batch_size',
                        default=16,
                        type=int)

    args = parser.parse_args()
    df = args.dataframe_path
    root_dir = args.dataset_path
    output_path = args.output_path
    num_epochs = args.num_epochs
    fold_path = args.fold_path
    dgcnn_path = args.dgcnn_path
    create_dir_if_not_exist(output_path)
    num_features = args.num_features
    k = args.k
    encoder_type = args.encoder_type
    decoder_type = args.decoder_type
    learning_rate = args.learning_rate
    batch_size = args.batch_size

    model = GraphAutoEncoder(num_features=num_features, k=20, encoder_type=encoder_type, decoder_type=decoder_type)
    # model.load_state_dict(checkpoint['model_state_dict'])
    checkpoint = torch.load(fold_path)
    model_dict = model.state_dict()  # load parameters from pre-trained FoldingNet
    for k in checkpoint['model_state_dict']:
        if k in model_dict:
            model_dict[k] = checkpoint['model_state_dict'][k]
            print("    Found weight: " + k)
        elif k.replace('folding1', 'folding') in model_dict:
            model_dict[k.replace('folding1', 'folding')] = checkpoint['model_state_dict'][k]
            print("    Found weight: " + k)
    model.load_state_dict(model_dict)
    print(checkpoint['loss'])

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
