from itertools import count
from re import A
from tkinter import E
from turtle import end_fill
from matplotlib import use
from numpy import expand_dims
import torch
import torch.nn as nn
import torch.nn.functional as F


from torchvision.models import resnet18, ResNet18_Weights

class VertTransformerEncoder(nn.Module):
    def __init__(self,
                 embedding_dim=512,
                 use_resnet_encoder=False,
                 ) -> None:

        super().__init__()
        self.embedding_dim = embedding_dim
        weights = ResNet18_Weights.IMAGENET1K_V2
        self.encoder = resnet18(weights=weights)
        # fix number of output classes from resnet
        self.encoder.fc = nn.Linear(self.encoder.fc.in_features, self.embedding_dim)
        self.preprocess = weights.transforms()

        self.fc_met = EmbeddingClassifier(embedding_dim, 1)
        self.fc_col = EmbeddingClassifier(embedding_dim, 1)
        self.fc_mscc = EmbeddingClassifier(embedding_dim, 1)

        # if testing just the max-pooled resnet embeddings, set use_resnet_encoder=True
        if use_resnet_encoder:
            self.aggregator = lambda x: torch.max(x,dim=-2)[0]
        else:
            self.aggregator = AttentionAggregator(embedding_dim)

    def embed_verts(self, vert_vols : torch.Tensor, return_attention_vals=False,aggregate=True) -> torch.Tensor:
        dim_sizes = vert_vols.shape

        # input has shape [batch_size, num_seqs, num_slices, height,width]
        # reshape for input to resnet
        x = vert_vols.flatten(start_dim=0, end_dim=-3).unsqueeze(1).expand(-1,3,-1,-1)

        # has shape [batch_size, 3, heights, widths]
        slice_encodings = self.encoder(x)

        # reshape to original shape
        slice_encodings =slice_encodings.view(list(dim_sizes[:-2])+[-1])
        # attention aggregation along slices dimension
        if return_attention_vals:
            vert_vol_encodings,attention_vals = self.aggregator.forward_with_attention_vals(slice_encodings)
            if not aggregate: vert_vol_encodings = slice_encodings
            return vert_vol_encodings,attention_vals
        else:
            vert_vol_encodings = self.aggregator(slice_encodings)
            if not aggregate: vert_vol_encodings = slice_encodings
            return vert_vol_encodings

    def classify_embeddings(self,embeddings: torch.Tensor):
        x_met = self.fc_met(embeddings).squeeze(2)
        x_col = self.fc_col(embeddings).squeeze(2)
        x_mscc =  self.fc_mscc(embeddings).squeeze(2)
        return x_met,x_col,x_mscc

    def forward(self, x : torch.Tensor) -> torch.Tensor:
            x = self.embed_verts(x)
            out = self.classify_embeddings(x)
            return out


    def forward_with_attention_vals(self, x : torch.Tensor) -> torch.Tensor:
            x,attn_vals = self.embed_verts(x,return_attention_vals=True)
        
            out = self.classify_embeddings(x)
            return out,attn_vals

    def make_slicewise_predictions(self, x):
            x = self.embed_verts(x,return_attention_vals=False, aggregate=False)
            out = self.classify_embeddings(x)
            return out

    def _get_classification_layers(self):
        disease_categories = ['met', 'col', 'mscc']
        is_classification_layer = lambda x: torch.Tensor(['fc_'+disease_category in x[0] for disease_category in disease_categories]).bool().any()
        for child in self.named_children():
            if is_classification_layer(child):
                yield child
            else:
                continue

    def finetune(self, reset_weights=True):
        ''' Freeze all layers except classification, and reset weights 
        for classification_layer '''
        for parameter in self.parameters():
            parameter.requires_grad = False
        for classification_layer in self._get_classification_layers():
            for parameter in classification_layer[1].parameters():
                parameter.requires_grad = True
            if reset_weights:
                classification_layer[1].reset_parameters()




class AttentionAggregator(nn.Module):
    def __init__(self,
                 embedding_dim=512,
                 agg_dim=64) -> None:
            
            super().__init__()
            self.agg_dim = agg_dim
            self.embedding_dim = embedding_dim 
            self.query = nn.Sequential(
                nn.Linear(embedding_dim,agg_dim),
                nn.ReLU(),
                nn.Linear(agg_dim,1)
            )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        query = self.query(x)
        attention_vals = torch.softmax(query, dim=-2)
        return (attention_vals*x).sum(dim=-2)

    def forward_with_attention_vals(self, x : torch.Tensor) -> torch.Tensor:
        query = self.query(x)
        attention_vals = torch.softmax(query, dim=-2)
        return (attention_vals*x).sum(dim=-2), attention_vals

class EmbeddingClassifier(nn.Module):
    def __init__(self,
                 embedding_dim=512,
                 num_classes=1,
                 ) -> None:

        super().__init__()
        self.embedding_dim = embedding_dim 
        # self.layer1 = nn.Linear(embedding_dim,classification_dim,bias=True)
        self.attention_layer = nn.Linear(embedding_dim,embedding_dim,bias=False)
        self.layer2 = nn.Linear(embedding_dim,num_classes,bias=False)


    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # x = torch.relu(self.layer1(x))
        # calculate attention values then softmax along sequence dimension
        attn_vals = torch.softmax(self.attention_layer(x),dim=2)
        x = (attn_vals*x).sum(dim=2, keepdim=True)
        return self.layer2(x)
        


    def reset_parameters(self):
        self.weight.reset_parameters()
