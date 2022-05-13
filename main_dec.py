import torch
from torch import nn
from torch.utils.data import DataLoader
from training_functions import train
from encoders.dgcnn import ChamferLoss
from dataset import PointCloudDatasetAllBoth, \
    PointCloudDatasetAll, \
    PointCloudDatasetAllDistal, \
    PointCloudDatasetAllProximal, \
    PointCloudDatasetAllBlebbNoc
from autoencoder import GraphAutoEncoder
from chamfer import ChamferLoss1
import argparse
import os
from encoders.dgcnn import ClusterlingLayer
from train_DEC import train_DEC_func


def create_dir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    # PATH_TO_DATASET = (
    #     "/home/mvries/Documents/Datasets/OPM/SingleCellFromNathan_17122021/"
    # )
    # df = "/home/mvries/Documents/Datasets/OPM/SingleCellFromNathan_17122021/all_cell_data.csv"
    # root_dir = "/home/mvries/Documents/Datasets/OPM/SingleCellFromNathan_17122021/"
    parser = argparse.ArgumentParser(description='DGCNN + Folding + DEC')
    parser.add_argument('--dataset_path', default='/home/mvries/Documents/Datasets/OPM/'
                                                  'SingleCellFromNathan_17122021/', type=str)
    parser.add_argument('--dataframe_path',
                        default='/home/mvries/Documents/Datasets/OPM/SingleCellFromNathan_17122021/'
                                'all_cell_data.csv',
                        type=str)
    parser.add_argument('--output_path', default='./', type=str)
    parser.add_argument('--num_epochs', default=250, type=int)
    parser.add_argument('--fold_path',
                        default='/run/user/1128299809/gvfs/smb-share:server=rds.icr.ac.uk,share=data/'
                                'DBI/DUDBI/DYNCESYS/mvries/ResultsAlma/TearingNetNew/shapenet/nets/'
                                'dgcnn_foldingnet_128_002.pt',
                        type=str)
    parser.add_argument('--dgcnn_path',
                        default='/run/user/1128299809/gvfs/smb-share:server=rds.icr.ac.uk,share=data/'
                                'DBI/DUDBI/DYNCESYS/mvries/Reconstruct_dgcnn_cls_k20_plane/models/'
                                'shapenetcorev2_250.pkl',
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
    parser.add_argument('--num_clusters',
                        default=None,
                        type=int)
    parser.add_argument('--proximal',
                        default=0,
                        type=int)
    parser.add_argument('--gamma',
                        default=10,
                        type=int)
    parser.add_argument('--divergence_threshold',
                        default=0.001,
                        type=float)
    parser.add_argument('--update_interval',
                        default=5,
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
    num_clusters = args.num_clusters
    proximal = args.proximal
    gamma = args.gamma
    divergence_threshold = args.divergence_threshold
    update_interval = args.update_interval

    checkpoint = torch.load(fold_path)

    ae = GraphAutoEncoder(num_features=num_features, k=20, encoder_type=encoder_type, decoder_type=decoder_type)
    ae.load_state_dict(checkpoint['model_state_dict'])
    # model_dict = ae.state_dict()  # load parameters from pre-trained FoldingNet
    # for k in checkpoint['model_state_dict']:
    #     if k in model_dict:
    #         model_dict[k] = checkpoint['model_state_dict'][k]
    #         print("    Found weight: " + k)
    #     elif k.replace('folding1', 'folding') in model_dict:
    #         model_dict[k.replace('folding1', 'folding')] = checkpoint['model_state_dict'][k]
    #         print("    Found weight: " + k)
    # ae.load_state_dict(model_dict)
    # print(checkpoint['loss'])

    if proximal == 0:
        dataset = PointCloudDatasetAllDistal(df, root_dir)
    elif proximal == 1:
        dataset = PointCloudDatasetAllProximal(df, root_dir)
    elif proximal == 2:
        dataset = PointCloudDatasetAll(df, root_dir)
    else:
        dataset = PointCloudDatasetAllBlebbNoc(df, root_dir)

    # TODO: Imperative that shuffle=False
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    dataloader_ind = DataLoader(dataset, batch_size=1, shuffle=False)

    criterion_rec = ChamferLoss()
    criterion_cluster = torch.nn.KLDivLoss(reduction="batchmean")

    train_DEC_func(autoencoder=ae,
                   dataloader=dataloader,
                   dataloader_ind=dataloader_ind,
                   num_epochs=num_epochs,
                   criterion_rec=criterion_rec,
                   criterion_cluster=criterion_cluster,
                   output_dir=output_path,
                   update_interval=update_interval,
                   divergence_tolerance=divergence_threshold,
                   gamma=gamma,
                   learning_rate=learning_rate,
                   batch_size=batch_size,
                   proximal=proximal,
                   num_clusters=num_clusters)



