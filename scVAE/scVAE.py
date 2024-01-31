import os
import anndata as ad
from typing import Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from anndata import AnnData
from pickle import dump, load
from scipy.sparse import issparse, csr_matrix

class scVAE(nn.Module):
    def __init__(self, adata:AnnData, num_layers:int=1, hidden_size:int=128, latent_size:int=10, dropout_rate:float=0.5):
        """single-cell Variational AutoEncoder(scVAE)
           Learn a low-dimensional latent representation and denoise the data.

        Args:
            adata (AnnData) - Input single-cell data.
            num_layers (int, default: 1) - Number of hidden layers.
            hidden_size (int, default: 128) - Number of neurons per layer.
            latent_size (int, default: 10) - Number of latent space.
            dropout_rate (float, default: 0.5) - Dropout rate in each layer.
        """
        super(scVAE, self).__init__()
        self.adata = adata
        self.input_size = adata.shape[1]
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.dropout_rate = dropout_rate
        self.is_prepared = False
        self.is_trained = False
        
        # Encoder
        self.encoder = nn.Sequential(nn.Linear(self.input_size, self.hidden_size),nn.ReLU(),nn.Dropout(p=self.dropout_rate))
        for i in range(self.num_layers):
            self.encoder.append(nn.Linear(self.hidden_size, self.hidden_size))
            self.encoder.append(nn.ReLU())
            self.encoder.append(nn.Dropout(p=self.dropout_rate))
        self.encoder.append(nn.Linear(self.hidden_size, self.latent_size * 2))

        # Decoder
        self.decoder = nn.Sequential(nn.Linear(self.latent_size, self.hidden_size),nn.ReLU(),nn.Dropout(p=self.dropout_rate))
        for i in range(self.num_layers):
            self.decoder.append(nn.Linear(self.hidden_size, self.hidden_size))
            self.decoder.append(nn.ReLU())
            self.decoder.append(nn.Dropout(p=self.dropout_rate))
        self.decoder.append(nn.Linear(self.hidden_size, self.input_size))
        self.decoder.append(nn.ReLU())

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def __str__(self):
        return(f'scVAE model with params:\n\
        num_layers: {self.num_layers}, hidden_size: {self.hidden_size}, latent_size: {self.latent_size}, dropout_rate: {self.dropout_rate}\n\
        is_prepared: {self.is_prepared}\n\
        is_trained: {self.is_trained}')

    def forward(self, x):
        # Encode
        enc_output = self.encoder(x)
        mu, logvar = enc_output[:, :self.latent_size], enc_output[:, self.latent_size:]

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode
        recon_x = self.decoder(z)

        return recon_x, mu, logvar

    def fit(self, num_epochs:int=100, optimizer=torch.optim.Adam, loss_function=nn.MSELoss(reduce='sum'), lr:float=0.001, device='cpu'):
        """Fit the model to the data.

        Args:
            num_epochs (int, default: 100) - Number of epochs to train.
            optimizer (function, default: torch.optim.Adam) - Optimizer from torch.nn.optim or custom optimizer function (must be callable).
            loss_function (function, default: nn.MSELoss(reduce='sum')) - Loss function from torch.nn or custom loss function (must be callable).
            lr (float, default: 0.001) - Learning rate during training.
            device (str, default: 'cpu') - Device for training ['cpu','cuda','mps'].
        """
        if not self.is_prepared:
            raise ValueError('Please prepare data first by calling prepare_adata()')
        
        self.to(device)
        self.optimizer = optimizer(self.parameters(), lr=lr)
        self.loss_function = loss_function
        self.history = {'loss': []}
        
        for epoch in range(num_epochs):
            self.train()
            for batch_data in self.data_loader:
                batch_data = batch_data.float().to(device)  # Ensure data type is float
                self.optimizer.zero_grad()

                recon_batch, mu, logvar = self(batch_data)
                loss = self.loss_function(recon_batch, batch_data)

                loss.backward()
                self.optimizer.step()
            self.history['loss'].append(loss.item())
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            
        self.is_trained = True

    def prepare_adata(self, adata:AnnData=None, batch_size=32, device='cpu', layer=None):
        """Register AnnData object to the model and prepare the data for training by creating the DataLoader object.

        Args:
            adata (AnnData, default: None) - AnnData object to compute the latent representation.
                                             If None, the input AnnData object during the model initialization will be used.
                                             
            batch_size (int, default: 32) - Number of batch size to train on per epoch.
            device (str, default: 'cpu') - Device for training ['cpu','cuda','mps'].
            layer (str, default: None) - Layer of AnnData object to use. If None, use the count matrix found in .X.
        """
        if adata is not None:
            self.adata = adata
        
        if layer is not None:
            self.adata.X = self.adata.layers[layer].copy()
            
        if issparse(self.adata.X):
            self.adata.X = self.adata.X.toarray()
        
        # Extract necessary information from AnnData and create DataLoader
        data_loader = DataLoader(torch.tensor(self.adata.X).float().to(device),
                                    batch_size=batch_size, shuffle=True)
        
        self.is_prepared = True
        self.data_loader = data_loader
        
    def get_latent(self, adata:AnnData=None, device='cpu'):
        """Get the latent representation of the data.

        Args:
            adata (AnnData, default:None) - AnnData object to compute the latent representation.
                                            If None, the input AnnData object during the model initialization will be used.
            device (str, default: 'cpu') - Device to compute the latent representation ['cpu','cuda','mps'].
            
        Returns:
            np.ndarray: The latent representation of the data.
        """
        if not self.is_trained:
            raise ValueError('Please train the model first by calling fit()')
        if adata is not None:
            adata_ = adata.copy()
            if issparse(adata_.X):
                adata_.X = adata_.X.toarray()
            X = torch.tensor(adata_.X).to_sparse_csr().to(device)   
        else:    
            X = torch.tensor(self.adata.X).to_sparse_csr().to(device)   
        
        self.eval()
        with torch.no_grad():
            mu, logvar = self.encoder(X)[:, :self.latent_size], self.encoder(X)[:, self.latent_size:]
            z = self.reparameterize(mu, logvar)
        return z.cpu().numpy()
    
    def reconstruc_counts(self, adata:AnnData=None, device='cpu', return_sparse:bool=True):
        """Reconstruct the denoised count matrix from the latent representation.

        Args:
            adata (AnnData, default: None) - AnnData object to reconstruct the count matrix on.
                                             If None, the input AnnData object during the model initialization will be used.
            device (str, default: 'cpu') - Device to compute the reconstructed matrix ['cpu','cuda','mps'].
            return_sparse (bool, default: True) - Sparse matrix as return.

        Returns:
            csr_matrix or np.ndarray: The denoised reconstructed count matrix.
        """
        if not self.is_trained:
            raise ValueError('Please train the model first by calling fit()')
        if adata is not None:
            adata_ = adata.copy()
        else:
            adata_ = self.adata.copy()
            
        self.eval()
        with torch.no_grad():
            recon_counts = self.decoder(torch.tensor(self.get_latent(adata=adata_, device=device)).to(device))
        return csr_matrix(recon_counts.cpu().numpy()) if return_sparse else recon_counts.cpu().numpy()
    
    def save(self, save_dir):
        """
        Save the VAE model to a file.

        Args:
            save_dir (str) - Directory where the model will be saved.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save both model state and optimizer state
        save_path = os.path.join(save_dir, 'model.pk')
        dump(self, open(save_path, 'wb'))
        print(f'Model saved to {save_path}')


def load_model(save_dir:str):
    """
    Load the VAE model from a file.

    Parameters:
        save_dir (str) - Directory where the model will be saved.

    Returns:
        model (scVAE): The loaded model.
    """
    if not os.path.exists(save_dir):
        raise ValueError(f'{save_dir} does not exist')
    # Load the model
    save_path = os.path.join(save_dir, 'model.pk')
    with open(save_path, 'rb') as f:
        model = load(f)
    print(f'Model loaded from {save_path}')

    return model
