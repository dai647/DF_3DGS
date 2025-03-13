import os
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import argparse

def compute_cosine_similarity(features):
    norm = torch.norm(features, dim=-1, keepdim=True)
    normalized_features = features / (norm + 1e-10)
    cosine_similarity_matrix = torch.mm(normalized_features, normalized_features.T)

    return cosine_similarity_matrix


def fps_cosine_sampling(features, num_samples):

    fsum = torch.sum(torch.abs(features), dim=-1)

    features = features[fsum != 0]

    cos_sim = compute_cosine_similarity(features)

    first = torch.argmin(torch.min(cos_sim, dim=0).values)
    selected_indices = [first]
    selected_features = [features[first]]

    second = torch.argmin(cos_sim[first])
    selected_indices.append(second)
    selected_features.append(features[second])

    cos_sim[:, first] = -torch.inf
    cos_sim[:, second] = -torch.inf

    for i in range(1, num_samples - 1):

        min_distances = torch.max(cos_sim[selected_indices,:], dim=0).values
        min_distances[min_distances == -torch.inf] = torch.inf
        farthest_index = torch.argmin(min_distances)
        min_distances[min_distances == torch.inf] = -torch.inf
        cos_sim[:, farthest_index] = -torch.inf

        selected_indices.append(farthest_index)
        selected_features.append(features[farthest_index])

    selected_features = torch.stack(selected_features, dim=0)

    return selected_features


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
    parser.add_argument('--epoch', type=int, default=28)
    parser.add_argument('--fps_num', type=int, default=8)
    parser.add_argument('--seed', type=int, default=21)
    

    args = parser.parse_args()
    sence_name=args.sence_name

    os.makedirs(f'Sences/{sence_name}', exist_ok=True)

    print(f'------Doing semantic compress on sence {sence_name}')
    data_path=f'../dataset/{sence_name}/rgb_feature_langseg'
    image_path=f'../dataset/{sence_name}/train_images'
    data_files=os.listdir(image_path)
    data_files=sorted(data_files,key=lambda x:int(x.split('_')[1][:-4]))
    print(data_files)

    data_filt=[torch.load(os.path.join(data_path,i[:-4]+'_fmap_CxHxW.pt')).permute(1,2,0)  for i in  data_files]
    print('Data have loaded !!!')
    print(len(data_filt))
    print(data_filt[0].shape) #torch.Size([360, 480, 512 ])

    save_p=f'Sences/{sence_name}/f_bank_trained.pt'
    epoch=args.epoch

    remove_threhold=0.01
    join_threhold=0.90
    join_remove_lock=1e10
    fps_num=args.fps_num
    print(f'----------------------------epoch:{epoch} fps_num:{fps_num} seed{args.seed}-------------------------------')
    max_bank_len=100
    random.seed(args.seed) 
    init_item=data_filt[0].to(torch.float32).cuda()
    init_item=init_item[1::2,1::3,:]
    init_item=init_item.reshape(-1,data_filt[0].shape[-1])
    print(init_item.shape)
    feature_bank=fps_cosine_sampling(init_item,fps_num)
    print(feature_bank.shape)
    fbank=nn.Parameter(feature_bank.requires_grad_(True)).cuda()
    print(fbank.shape)

    lr=0.0008
    optimizer4fbank = torch.optim.Adam([{'params': [fbank], 'lr':lr, "name": "feature_bank"}], eps=1e-9)

    batch_size=8
    
    ids=list(range(len(data_filt)))
    floss_list=[[] for i in range(len(ids)+1)]

    
    for ep in range(epoch):
        random.shuffle(ids)
        print(ids)
        numm=0
        for id in ids:
            numm+=1
            print('-----epoch:',ep,'--------num:',numm,'------id:',id)
            ff_init=data_filt[id].clone()

            ff_init=ff_init[1::2,1::3,:] 
            ff_init=ff_init.reshape(-1,data_filt[0].shape[-1]).to(torch.float32).cuda()

            idxs=get_findexs(ff_init,fbank) 
            
            f_map=fbank[idxs,:]
            
            sim0 = F.cosine_similarity(f_map, ff_init, dim=-1, eps=1e-8)

            sim=sim0.clone()

            need_new_sim=False

            if ep<join_remove_lock:

                ########join new semantic discriptor
                sim[sim>join_threhold]=10
                if len(sim[sim!=10])!=0 :
                    print('----join----')
                    rank_k=max_bank_len-len(fbank)
                    if rank_k>0:
                        rank_v=torch.sort(sim)[0][rank_k]
                        if len(sim[sim!=10])>rank_k:
                            sim[sim>rank_v]=10
                        new_f=ff_init[sim!=10]
                        print(fbank.shape)
                        fbank=torch.cat([fbank,new_f],dim=0)
                        fbank=nn.Parameter(fbank.requires_grad_(True))
                        optimizer4fbank = torch.optim.Adam([{'params': [fbank], 'lr':lr, "name": "feature_bank"}], eps=1e-9)
                        print(fbank.shape)
                        need_new_sim=True
                
                ########remove similar semantic discriptor
                sim_m=compute_cosine_similarity(fbank)
                sim_m_tril=torch.tril(sim_m,-1)
                flag=torch.zeros_like(sim_m_tril)
                flag[sim_m_tril>(1-remove_threhold)]=1
                flag=torch.sum(flag,0)
                if sum(flag)!=0:
                    print('----remove----')
                    print(fbank.shape)
                    fbank=fbank[flag==0].clone()
                    fbank=nn.Parameter(fbank.requires_grad_(True))
                    optimizer4fbank = torch.optim.Adam([{'params': [fbank], 'lr':lr, "name": "feature_bank"}], eps=1e-9)
                    print(fbank.shape)
                    need_new_sim=True
            
            similarity=1-sim0
            if need_new_sim:
                idxs=get_findexs(ff_init,fbank) 
                f_map=fbank[idxs,:]
                similarity = 1-F.cosine_similarity(f_map, ff_init, dim=-1, eps=1e-8)

            f_loss=torch.tensor(0.0).cuda(0)
            for ii in torch.unique(idxs):
                f_loss+=torch.mean(similarity[idxs==ii])
            f_loss/=len(torch.unique(idxs))
            

            floss_list[id].append(f"{ep} : {float(f_loss.cpu().detach()):.4f}")
            print(f'----feature loss: \n', floss_list[id])
            f_loss.backward()

            if ep <2  or numm%batch_size ==0 :

                optimizer4fbank.step()
                optimizer4fbank.zero_grad()

    torch.save(fbank,save_p)
    print('Feature bank have saved to: ',save_p)



