from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import Linear


from .vert_transformer_encoder import VertTransformerEncoder
from .vert_transformer_layers import VertTransformerLayers


class VertTransformer(nn.Module):
    def __init__(self,
                 embedding_dim=512,
                 transformer_dropout=0.1,
                 use_resnet_encoder=False,
                 ) -> None:

        super().__init__()
        self.encoder = VertTransformerEncoder(embedding_dim=embedding_dim,use_resnet_encoder=use_resnet_encoder)
        self.embedding_dim = embedding_dim
        self.use_resnet_encoder = use_resnet_encoder

        self.fc_met =  EmbeddingClassifier(embedding_dim, 1)
        self.fc_col =  EmbeddingClassifier(embedding_dim, 1)
        self.fc_mscc = EmbeddingClassifier(embedding_dim, 1)
        self.transformer = VertTransformerLayers(in_dim=embedding_dim, dropout=transformer_dropout)

    def forward(self, x : torch.Tensor, pos_encodings : torch.Tensor,
                seq_encodings : torch.Tensor, use_transformer=True,
                use_seq_encodings=False) -> torch.Tensor:
        init_shape = x.shape
        # should have shape [batch_size, num_verts, num_channels, num_slices, num_slices, height,width]
        x = x.flatten(start_dim=0, end_dim=2).unsqueeze(1)
        # embed verts and then reshape
        x = self.encoder.embed_verts(x)

        x = x.view(init_shape[0],init_shape[1],init_shape[2],-1)

        # apply transformer layers
        if use_transformer:
            x = self.transformer(x,pos_encodings, seq_encodings,use_seq_encodings=use_seq_encodings)
        # apply classification output
        # average embeddings over sequence dimension
        if self.use_resnet_encoder:
            x = x.mean(dim=2,keepdim=True)
        x = self.classify_embeddings(x)
        return x

    def classify_embeddings(self,embeddings: torch.Tensor):
        x_met = self.fc_met(embeddings).squeeze(2).squeeze(-1)
        x_col = self.fc_col(embeddings).squeeze(2).squeeze(-1)
        x_mscc =  self.fc_mscc(embeddings).squeeze(2).squeeze(-1)
        return x_met,x_col,x_mscc


    def finetune(self, reset_weights=True):
        ''' Freeze all layers except classification, and reset weights 
        for classification_layer '''
        for parameter in self.parameters():
            parameter.requires_grad = False
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        for parameter in self.transformer.parameters():
            parameter.requires_grad = False
        for classification_layer in self._get_classification_layers():
            for parameter in classification_layer[1].parameters():
                parameter.requires_grad = True

    def _get_classification_layers(self):
        disease_categories = ['met', 'col', 'mscc']
        is_classification_layer = lambda x: torch.Tensor(['fc_'+disease_category in x[0] for disease_category in disease_categories]).bool().any()
        for child in self.named_children():
            if is_classification_layer(child):
                yield child
            else:
                continue


class AttentionAggregator():
    def __init__(self,
                 embedding_dim=512,
                 agg_dim=64) -> None:
            
            super().__init__()
            self.agg_dim = agg_dim
            self.embedding_dim = embedding_dim 
            self.query = nn.Linear(embedding_dim,agg_dim,bias=False)
            self.key = nn.Linear(embedding_dim,1,bias=False)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        query = self.query(x)
        key = self.key(x)
        return torch.matmul(query,key.transpose(-1,-2))


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
