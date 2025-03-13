import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from model import Autoencoder
import argparse

torch.manual_seed(666)
if torch.cuda.is_available():
    torch.cuda.manual_seed(666)

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def cos_loss(network_output, gt):
    return 1 - F.cosine_similarity(network_output, gt, dim=0).mean()

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def calculate_pairwise_euclidean_distances(vectors):
  
    squared_sums = torch.sum(torch.square(vectors), dim=1)

    distances_squared = squared_sums.unsqueeze(1) - 2 * torch.mm(vectors, vectors.t()) + squared_sums.unsqueeze(0)
    
    distances = torch.sqrt(distances_squared+1e-6)
    
    return distances
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="input sence name") 
    parser.add_argument('--sence_name', type=str, metavar='T',default='room_0')

    args = parser.parse_args()
    sence_name=args.sence_name

    print(f'------Doing train data compress on sence {sence_name}')
    train_data=f'Sences/{sence_name}/f_bank_trained.pt'
    train_data=torch.load(train_data).to("cuda:0")


    encoder_hidden_dims = [256, 128, 64, 32, 9]
    decoder_hidden_dims = [32, 64, 128, 256, 1024, 512]
    num_epochs=5000
    lr=0.0001

    model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to("cuda:0")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=0.000005)

    for epoch in tqdm(range(num_epochs)):
        model.train()
        
        data=train_data
        outputs_dim3 = model.encode(data)
        outputs = model.decode(outputs_dim3)
        l2loss = l2_loss(outputs, data) 

        loss = l2loss 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), f'Sences/{sence_name}/latest_AE_ckpt.pth')
