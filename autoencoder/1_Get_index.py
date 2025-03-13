import numpy as np
import torch
import os
import argparse

def get_findexs(features, fbank):

    fnorm = torch.norm(features, dim=-1, keepdim=True)
    normalized_features = features / (fnorm + 1e-8)

    fbank_norm = torch.norm(fbank, dim=-1, keepdim=True)
    normalized_fbank = fbank / (fbank_norm + 1e-8)

    cosine_similarity_matrix = torch.mm(normalized_features, normalized_fbank.T)

    indexs = torch.argmax(cosine_similarity_matrix, dim=1)

    return indexs

if __name__=="__main__":

    parser = argparse.ArgumentParser(description="input sence name") 
    parser.add_argument('--sence_name', type=str, metavar='T',default='room_0')

    args = parser.parse_args()
    sence_name=args.sence_name

    print(f'------Doing get findex on sence {sence_name}')
    data_path=f'../dataset/{sence_name}/rgb_feature_langseg'
    print(data_path)
    data_filt=[(i,torch.load(os.path.join(data_path,i)).permute(1,2,0))  for i in os.listdir(data_path) if 'fmap' in i]

    fbank=f'Sences/{sence_name}/f_bank_trained.pt'
    save_p=f'Sences/{sence_name}/findx.pt'
    fbank=torch.load(fbank)
    findx={}

    num=0
    for ii in data_filt:
        num+=1
        feature=ii[1].reshape(-1,ii[1].shape[-1]).to(torch.float32)
        idx=get_findexs(feature.cuda(0),fbank.cuda(0))
        name=ii[0][:-14]
        findx[name]=idx
    
    torch.save(findx,save_p)
    print('Done!!!')

