import sys,os,glob
import torch.nn as nn
import torch

class VertTransformerLayers(nn.Module):
    def __init__(self,
                 in_dim=512,
                 model_dim=512,
                 position_encoding_dim=5,
                 seq_encoding_dim=5,
                 dropout=0.1):
        super().__init__()
        self.position_encoding_dim = position_encoding_dim
        self.in_dim = in_dim
        self.model_dim = model_dim
        self.in_layer = nn.Linear(position_encoding_dim, model_dim)
        self.seq_encodings_layer = nn.Linear(seq_encoding_dim, model_dim)
        transformer_encoding_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=2, batch_first=True, dropout=dropout)

        self.transformer_encoder = nn.TransformerEncoder(transformer_encoding_layer, num_layers=2)
        self.classifier_sequence_embedding = nn.Embedding(1, model_dim)
        # transformer_decoding_layer = nn.TransformerDecoderLayer(d_model=model_dim, nhead=4, batch_first=True)
        # self.transformer_decoder = nn.TransformerDecoder(transformer_decoding_layer, num_layers=1)


    def forward(self, x, position_encodings,seq_encodings, use_seq_encodings=False):
        position_encodings = torch.stack([position_encodings]*x.shape[-2],dim=-2)
        transformer_input = x + self.in_layer(position_encodings)
        if use_seq_encodings: transformer_input += self.seq_encodings_layer(seq_encodings)
        batch_size, num_verts, sequences, embedding_dim = transformer_input.shape
        transformer_input = transformer_input.flatten(start_dim=1,end_dim=2)
        transformer_output = self.transformer_encoder(transformer_input)
        transformer_output = transformer_output.reshape(batch_size,num_verts,sequences,self.model_dim)
        return transformer_output
