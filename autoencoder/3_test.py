import os
import torch
import argparse
from model import Autoencoder


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="input sence name") 
    parser.add_argument('--sence_name', type=str, metavar='T',default='room_0')

    args = parser.parse_args()
    sence_name=args.sence_name

    print(f'------Doing inference data compress on sence {sence_name}')
    train_data=f'Sences/{sence_name}/f_bank_trained.pt'

    train_data=torch.load(train_data).to("cuda:0")
     
    print(train_data.shape)
    encoder_hidden_dims = [256, 128, 64, 32, 9]
    decoder_hidden_dims = [32, 64, 128, 256, 1024, 512]
    
    no_AE=False  ### Whether to use dimensionality reduction
    checkpoint=torch.load(f'Sences/{sence_name}/latest_AE_ckpt.pth')
    model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to("cuda:0")
    model.load_state_dict(checkpoint)
    model.eval()

    if no_AE:
        outputs = train_data
        os.makedirs(f'Sences/{sence_name}/encode_feature_512', exist_ok=True)
    else:
        outputs = model.encode(train_data)
        os.makedirs(f'Sences/{sence_name}/encode_feature_9', exist_ok=True)

    indexs=torch.load(f'Sences/{sence_name}/findx.pt')

    
    for ii in indexs.keys():
        idx=indexs[ii]
        mask=outputs[idx]
        if no_AE:
            mask=mask.reshape(360,480,decoder_hidden_dims[-1]).permute(2,0,1).to(torch.float16)
            torch.save(mask,f'Sences/{sence_name}/encode_feature_512/{ii}_f.pth')
        else:
            mask=mask.reshape(360,480,encoder_hidden_dims[-1]).permute(2,0,1)
            torch.save(mask,f'Sences/{sence_name}/encode_feature_9/{ii}_f.pth')
    print('Done!!!')

        


        
    





    



