import torch
from torch.utils.data import DataLoader
from training_functions import train
from encoders.dgcnn import ChamferLoss
from dataset import PointCloudDatasetAllBoth, PointCloudDatasetAll
from autoencoder import GraphAutoEncoder
from chamfer import ChamferLoss1
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DGCNN + Tearing')
    parser.add_argument('--dataset_path', default='/home/mvries/Documents/Datasets/OPM/SingleCellFromNathan_17122021/', type=str)
    parser.add_argument('--dataframe_path',
                        default='/home/mvries/Documents/Datasets/OPM/SingleCellFromNathan_17122021/all_cell_data.csv',
                        type=str)
    parser.add_argument('--output_path', default='./', type=str)
    parser.add_argument('--num_epochs', default=250, type=int)
    parser.add_argument('--fold_path',
                        default='/run/user/1128299809/gvfs/smb-share:server=rds.icr.ac.uk,share=data/DBI/DUDBI/DYNCESYS'
                                '/mvries/ResultsAlma/TearingNetNew/nets/dgcnn_foldingnet_50_002.pt',
                        type=str)
    parser.add_argument('--full_path',
                        default='/run/user/1128299809/gvfs'
                                '/smb-share:server=rds.icr.ac.uk,share=data/DBI'
                                '/DUDBI/DYNCESYS/mvries/ResultsAlma/TearingNetNew/nets/dgcnn_tearingnet_50_001.pt',
                        type=str)

    args = parser.parse_args()
    df = args.dataframe_path
    root_dir = args.dataset_path
    output_path = args.output_path
    num_epochs = args.num_epochs
    fold_path = args.fold_path
    full_path = args.full_path

    checkpoint = torch.load(full_path)
    print(checkpoint['epoch'])
    print(checkpoint['loss'])
    model = GraphAutoEncoder(num_features=50, k=20, encoder_type="dgcnn", decoder_type='tearingnet')
    model.load_state_dict(checkpoint['model_state_dict'])

    model.cuda()

    print('Extracting cell features')
    inputs_test = []
    outputs_test = []
    features_test = []
    embeddings_test = []
    clusterings_test = []
    labels_test = []
    serial_numbers = []
    model.eval()
    device = "cuda"
    criterion = ChamferLoss()

    dataset = PointCloudDatasetAll(
        df,
        root_dir,
        transform=None,
        img_size=400,
        target_transform=True,
        cell_component="cell",
    )
    loss = 0
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    for data in tqdm(dataloader):
        with torch.no_grad():
            pts, lab, _, serial_num = data

            labels_test.append(lab.detach().numpy())
            inputs = pts.to(device)
            output, embedding = model(inputs)
            embeddings_test.append(torch.squeeze(embedding).cpu().detach().numpy())
            serial_numbers.append(serial_num)
            loss += criterion(inputs, output)

    print(loss / len(dataloader))
    features = pd.DataFrame(np.asarray(embeddings_test))
    features['serialNumber'] = np.asarray(serial_numbers)
    all_data = pd.read_csv(df)
    all_data_labels = all_data[['serialNumber', 'Treatment', 'Proximal', 'nucleusCoverslipDistance', 'erkRatio',
                                'erkIntensityNucleus', 'erkIntensityCell']]
    features_new = features.join(all_data_labels.set_index('serialNumber'), on='serialNumber')

    features_new.to_csv(
        '/home/mvries/Documents/Datasets/'
        'OPM/SingleCellFromNathan_17122021/dgcnn_tearing_cell.csv')

    # ======================================= Extracting nucleus
    dataset = PointCloudDatasetAll(
        df,
        root_dir,
        transform=None,
        img_size=400,
        target_transform=True,
        cell_component="nuc",
    )
    loss = 0
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    for data in tqdm(dataloader):
        with torch.no_grad():
            pts, lab, _, serial_num = data

            labels_test.append(lab.detach().numpy())
            inputs = pts.to(device)
            output, embedding = model(inputs)
            embeddings_test.append(torch.squeeze(embedding).cpu().detach().numpy())
            serial_numbers.append(serial_num)
            loss += criterion(inputs, output)

    print(loss / len(dataloader))
    features = pd.DataFrame(np.asarray(embeddings_test))
    features['serialNumber'] = np.asarray(serial_numbers)
    all_data = pd.read_csv(df)
    all_data_labels = all_data[['serialNumber', 'Treatment', 'Proximal', 'nucleusCoverslipDistance', 'erkRatio',
                                'erkIntensityNucleus', 'erkIntensityCell']]
    features_new = features.join(all_data_labels.set_index('serialNumber'), on='serialNumber')

    features_new.to_csv(
        '/home/mvries/Documents/Datasets/'
        'OPM/SingleCellFromNathan_17122021/dgcnn_tearing_nuc.csv')

    # ======================================= Extracting both
    dataset = PointCloudDatasetAllBoth(
        df,
        root_dir,
        transform=None,
        img_size=400,
        target_transform=True,
        cell_component="nuc",
    )
    loss = 0
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    for data in tqdm(dataloader):
        with torch.no_grad():
            pts, lab, _, serial_num = data

            labels_test.append(lab.detach().numpy())
            inputs = pts.to(device)
            output, embedding = model(inputs)
            embeddings_test.append(torch.squeeze(embedding).cpu().detach().numpy())
            serial_numbers.append(serial_num)
            loss += criterion(inputs, output)

    print(loss/len(dataloader))
    features = pd.DataFrame(np.asarray(embeddings_test))
    features['serialNumber'] = np.asarray(serial_numbers)
    all_data = pd.read_csv(df)
    all_data_labels = all_data[['serialNumber', 'Treatment', 'Proximal', 'nucleusCoverslipDistance', 'erkRatio',
                                'erkIntensityNucleus', 'erkIntensityCell']]
    features_new = features.join(all_data_labels.set_index('serialNumber'), on='serialNumber')

    features_new.to_csv(
        '/home/mvries/Documents/Datasets/'
        'OPM/SingleCellFromNathan_17122021/dgcnn_tearing_both.csv')
