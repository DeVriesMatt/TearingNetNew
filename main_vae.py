import torch
from torch.utils.data import DataLoader
from training_functions import train_vae
from encoders.dgcnn import ChamferLoss
from dataset import PointCloudDatasetAllBoth
from autoencoder import GraphAutoEncoder
from torch import nn
import argparse


class VAE(nn.Module):
    def __init__(self, encoder, decoder, encoder_type, decoder_type, latent_dim=8):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.latent_dim = latent_dim
        self.enc_mu = nn.Linear(50, self.latent_dim)
        self.enc_log_var = nn.Linear(50, self.latent_dim)
        self.fc_dec = nn.Linear(self.latent_dim, 50)
        self.decoder = decoder
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type + "vae"

    def reparametrize(self, mus, log_vars):
        sigma = torch.exp(0.5*log_vars)
        z = torch.randn(size=(mus.size(0), mus.size(1)))
        z = z.type_as(mus)
        return mus + sigma * z

    def forward(self, x):
        embeddings = self.encoder(x)
        mu = self.enc_mu(embeddings)
        log_var = self.enc_log_var(embeddings)
        z = self.reparametrize(mu, log_var)
        upsamples = self.fc_dec(z)
        outs, grid = self.decoder(upsamples)
        return outs, mu, log_var, embeddings, z


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
                        default='/run/user/1128299809/gvfs/smb-share:server=rds.icr.ac.uk,share=data/DBI/DUDBI/DYNCESYS/mvries/ResultsAlma/TearingNetNew/nets/dgcnn_foldingnet_50_003.pt',
                        type=str)
    parser.add_argument('--dgcnn_path',
                        default='/home/mvries/Documents/GitHub/FoldingNetNew/nets/FoldingNetNew_50feats_planeshape_foldingdecoder_trainallTrue_centringonlyTrue_train_bothTrue_003.pt',
                        type=str)

    args = parser.parse_args()
    df = args.dataframe_path
    root_dir = args.dataset_path
    output_path = args.output_path
    num_epochs = args.num_epochs
    fold_path = args.fold_path
    dgcnn_path = args.dgcnn_path

    checkpoint = torch.load(fold_path)
    batch_size = 16
    learning_rate = 0.00000001

    ae = GraphAutoEncoder(num_features=50, k=20, encoder_type="dgcnn", decoder_type='foldingnet')
    model = VAE(encoder=ae.encoder,
                decoder=ae.decoder,
                encoder_type=ae.encoder_type,
                decoder_type=ae.decoder_type)
    # model.load_state_dict(checkpoint['model_state_dict'])
    print(checkpoint['loss'])

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

    train_vae(model, dataloader, num_epochs, criterion, optimizer, output_path)
