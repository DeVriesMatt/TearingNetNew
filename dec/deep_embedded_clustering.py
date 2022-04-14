import torch
from torch import nn
from encoders.dgcnn import ClusterlingLayer


class DEC(nn.Module):
    def __init__(
        self,
            autoencoder,
            num_clusters,
    ):
        super(DEC, self).__init__()
        self.autoencoder = autoencoder
        self.encoder = autoencoder.encoder
        self.decoder = autoencoder.decoder
        self.num_clusters = num_clusters
        self.num_features = autoencoder.num_features
        self.encoder_type = autoencoder.encoder_type
        self.decoder_type = autoencoder.decoder_type + "DEC"

        self.clustering_layer = ClusterlingLayer(self.num_features, self.num_clusters)

    def forward(self, x):
        features = self.encoder(x)
        q = self.clustering_layer(features)
        output, grid = self.decoder(features)
        return output, features, q
