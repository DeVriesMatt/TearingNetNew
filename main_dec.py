import torch
from torch import nn
from torch.utils.data import DataLoader
from training_functions import train
from encoders.dgcnn import ChamferLoss
from dataset import PointCloudDatasetAllBoth, \
    PointCloudDatasetAll, \
    PointCloudDatasetAllDistal, \
    PointCloudDatasetAllProximal, \
    PointCloudDatasetAllBlebbNoc, \
    GefGapDataset, \
    ShapeNetDataset, \
    OPMDataset

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import logging

from cellshape_cloud.lightning_autoencoder import CloudAutoEncoderPL

from cellshape_cloud.reports import get_experiment_name
from cellshape_cloud.cloud_autoencoder import CloudAutoEncoder

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
    parser.add_argument('--dataset_path',
                        default="/home/mvries/Documents/Datasets/OPM/VickyPlates_010922",
                        type=str)
    parser.add_argument('--dataframe_path',
                        default="/home/mvries/Documents/Datasets/OPM/VickyCellshape/" 
                                "cn_allFeatures_withGeneNames_updated_removedwrong.csv",
                        type=str)
    parser.add_argument('--output_path', default='./', type=str)
    parser.add_argument('--num_epochs', default=250, type=int)
    parser.add_argument('--fold_path',
                        default="/run/user/1128299809/gvfs/smb-share:server=rds"
                                ".icr.ac.uk,share=data/DBI/DUDBI/DYNCESYS/"
                                "mvries/ResultsAlma/cellshape-cloud/Vicky/"
                                "dgcnn_foldingnetbasic_128_pretrained_001/"
                                "lightning_logs/version_10952234/checkpoints/"
                                "epoch=149-step=52950.ckpt",
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
                        default=3,
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
    parser.add_argument('--is_lightning',
                        default=1,
                        type=int)
    parser.add_argument('--std',
                        default=3.0,
                        type=int)
    parser.add_argument('--shape',
                        default="plane",
                        type=str)
    parser.add_argument('--sphere_path',
                        default="/home/mvries/Documents/GiHub/cellshape-cloud/cellshape_cloud/vendor/sphere.npy",
                        type=str)
    parser.add_argument('--gaussian_path',
                        default="/home/mvries/Documents/GiHub/cellshape-cloud/cellshape_cloud/vendor/gaussian.npy",
                        type=str)
    parser.add_argument('--learning_rate_autoencoder',
                        default=0.00001,
                        type=float)
    parser.add_argument(
        "--single_path",
        default="./",
        type=str,
        help="Standard deviation of sampled points.",
    )
    parser.add_argument(
        "--gef_path",
        default="./",
        type=str,
        help="Standard deviation of sampled points.",
    )


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

    ae = GraphAutoEncoder(num_features=num_features,
                          k=20,
                          encoder_type=encoder_type,
                          decoder_type=decoder_type)

    if args.is_lightning:
        model = CloudAutoEncoder(
            num_features=num_features,
            k=k,
            encoder_type=encoder_type,
            decoder_type="foldingnetbasic",
            std=args.std,
            shape=args.shape
        )

        autoencoder = CloudAutoEncoderPL(args=args, model=model).cuda()
        autoencoder.load_state_dict(checkpoint['state_dict'])
        model_sd = autoencoder.model.state_dict()
        # try:
        #     ae.load_state_dict(model_sd)
        # except Exception as e:
        #     print(e)
        #     print("Trying to load model another way")
        try:
            print("Trying to load Model")
            model_dict = ae.state_dict()  # load parameters from pre-trained FoldingNet
            print(model_dict.keys())
            print(checkpoint['state_dict'].keys())
            for k in checkpoint['state_dict']:
                if k in model_dict:
                    model_dict[k] = checkpoint['state_dict'][k]
                    print("    Found weight: " + k)
                elif k.replace('model.decoder', 'decoder.folding') in model_dict:
                    model_dict[k.replace('model.decoder', 'decoder.folding')] = checkpoint['state_dict'][k]
                    print("    Found weight: " + k)
                elif k.replace('model.', '') in model_dict:
                    model_dict[k.replace('model.', '')] = checkpoint['state_dict'][k]
                    print("    Found weight: " + k)
            ae.load_state_dict(model_dict)

        except Exception as e:
            print("there was an error")
            print(e)
            try:
                print("Trying again to load Model")
                model_dict = ae.state_dict()  # load parameters from pre-trained FoldingNet
                print(model_dict.keys())
                print(checkpoint['state_dict'].keys())
                for k in checkpoint['state_dict']:
                    if k in model_dict:
                        model_dict[k] = checkpoint['state_dict'][k]
                        print("    Found weight: " + k)
                    elif k.replace('model.decoder', 'model.decoder.folding') in model_dict:
                        model_dict[k.replace('model.decoder', 'model.decoder.folding')] = checkpoint['state_dict'][k]
                        print("    Found weight: " + k)
                    elif k.replace('model.', '') in model_dict:
                        model_dict[k.replace('model.', '')] = checkpoint['state_dict'][k]
                        print("    Found weight: " + k)
                ae.load_state_dict(model_dict)

            except Exception as e:
                print("there was a new error")
                print(e)




    else:
        print("not a lightning model")
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
    elif proximal == 3:
        dataset = GefGapDataset(df, root_dir)
    elif proximal == 4:
        dataset = OPMDataset(
            args.dataframe_path,
            args.cloud_dataset_path,
            norm_std=args.norm_std,
            cell_component=args.cell_component,
            single_path=args.single_path,
            gef_path=args.gef_path,
        )
    else:
        dataset = ShapeNetDataset(
            root=args.dataset_path,
            dataset_name="modelnet40",
            random_rotate=False,
            random_jitter=False,
            random_translate=False
        )

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
                   num_clusters=num_clusters,
                   args=args)



