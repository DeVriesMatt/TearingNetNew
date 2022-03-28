import torch
from torch import nn

from encoders.dgcnn import FoldNetEncoder, DGCNNEncoder
from decoders.folding_decoder import FoldingNetBasicDecoder
from decoders.tearing_decoder import TearingNetDecoder


class GraphAutoEncoder(nn.Module):
    def __init__(
        self,
        num_features,
        k=20,
        encoder_type="dgcnn",
        decoder_type="foldingnet",
    ):
        super(GraphAutoEncoder, self).__init__()
        self.k = k
        self.num_features = num_features
        assert encoder_type.lower() in [
            "foldingnet",
            "dgcnn",
        ], "Please select an encoder type from either foldingnet or dgcnn."

        assert decoder_type.lower() in [
            "foldingnet",
            "tearingnet",
        ], "Please select an decoder type from either foldingnet, tearingnet."

        self.encoder_type = encoder_type.lower()
        self.decoder_type = decoder_type.lower()
        if self.encoder_type == "dgcnn":
            self.encoder = DGCNNEncoder(num_features=self.num_features, k=self.k)
        else:
            self.encoder = FoldNetEncoder(num_features=self.num_features, k=self.k)

        if self.decoder_type == 'foldingnet':
            self.decoder = FoldingNetBasicDecoder(
                num_features=self.num_features, num_clusters=10
            )
        else:
            self.decoder = TearingNetDecoder(
                num_features=self.num_features, num_clusters=10
            )

    def forward(self, x):
        features = self.encoder(x)
        output, _ = self.decoder(x=features)
        return output, features
