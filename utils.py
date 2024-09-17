import copy
import torch
from torch import nn
import numpy as np
from resnet_with_mix_style import resnet18_with_mix_style
from models import Classifier
from torch.utils.data import DataLoader
from tqdm import tqdm

def FedAvg(w, weight_avg=None):
    """
    Federated averaging
    :param w: list of client model parameters
    :return: updated server model parameters
    """
    if weight_avg is None:
        weight_avg = [1/len(w) for i in range(len(w))]
        
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = w_avg[k].cuda() * weight_avg[0]
        
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] = w_avg[k].cuda() + w[i][k].cuda() * weight_avg[i]
        #w_avg[k] = torch.div(w_avg[k].cuda(), len(w)) 
    return w_avg

def calculating_adjacency(clients_idxs, U): 
        
    nclients = len(clients_idxs)
    
    sim_mat = np.zeros([nclients, nclients])
    for idx1 in range(nclients):
        for idx2 in range(nclients):
            #print(idx1)
            #print(U)
            #print(idx1)
            U1 = copy.deepcopy(U[clients_idxs[idx1]])
            U2 = copy.deepcopy(U[clients_idxs[idx2]])
            
            #sim_mat[idx1,idx2] = np.where(np.abs(U1.T@U2) > 1e-2)[0].shape[0]
            #sim_mat[idx1,idx2] = 10*np.linalg.norm(U1.T@U2 - np.eye(15), ord='fro')
            #sim_mat[idx1,idx2] = 100/np.pi*(np.sort(np.arccos(U1.T@U2).reshape(-1))[0:4]).sum()
            mul = np.clip(U1.T@U2 ,a_min =-1.0, a_max=1.0)
            sim_mat[idx1,idx2] = np.trace(mul)
           
    return sim_mat

def eval_test_glob(net_glob, clf_glob, test_ldr, device):
    net_glob.to(device)
    clf_glob.to(device)

    net_glob.eval()
    clf_glob.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_ldr:
            data, target = data.to(device), target.to(device)
            output = clf_glob(net_glob(data))
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
    test_loss /= len(test_ldr.dataset)
    accuracy = 100. * correct / len(test_ldr.dataset)
    return test_loss, accuracy

def get_basis_vec(dataset, K):
    images = []
    labels = []

    dataloader = DataLoader(dataset, batch_size = 61, shuffle = False)
    for img, lbl in tqdm(dataloader):
        images.extend(img)
        labels.extend(lbl)
    
    images = torch.stack(images)
    labels = torch.tensor(labels)
    
    idxs_local = np.arange(len(dataset))
    labels_local = np.array(labels)
    # Sort Labels Train 
    idxs_labels_local = np.vstack((idxs_local, labels_local))
    idxs_labels_local = idxs_labels_local[:, idxs_labels_local[1, :].argsort()]
    idxs_local = idxs_labels_local[0, :]
    labels_local = idxs_labels_local[1, :]
    
    uni_labels, cnt_labels = np.unique(labels_local, return_counts=True)
    
    print(f'Labels: {uni_labels}')
    print(f'Counts: {cnt_labels}')
    
    nlabels = len(uni_labels)
    cnt = 0
    U_temp = []
    for j in range(nlabels):
        local_ds1 = images[idxs_local[cnt:cnt+cnt_labels[j]]]
        local_ds1 = local_ds1.reshape(cnt_labels[j], -1)
        local_ds1 = local_ds1.T
        if type(labels[idxs_local[cnt:cnt+cnt_labels[j]]]) == torch.Tensor:
            label1 = list(set(labels[idxs_local[cnt:cnt+cnt_labels[j]]].numpy()))
        else:
            label1 = list(set(labels[idxs_local[cnt:cnt+cnt_labels[j]]]))
        #assert len(label1) == 1
        
        #print(f'Label {j} : {label1}')
        
       
        if K > 0: 
            u1_temp, sh1_temp, vh1_temp = np.linalg.svd(local_ds1, full_matrices=False)
            u1_temp=u1_temp/np.linalg.norm(u1_temp, ord=2, axis=0)
            U_temp.append(u1_temp[:, 0:K])
            
        cnt+=cnt_labels[j]
        
    #U_temp = [u1_temp[:, 0:K], u2_temp[:, 0:K]]
    return U_temp  

def init_models(n_clients, in_dim, num_classes):
    src_models = []
    src_clfs = []

    for i in range(-1, n_clients ):
        model = resnet18_with_mix_style(mix_layers=['layer1', 'layer2', 'layer3'], mix_p=0.5, mix_alpha=0.1, pretrained=True)
        classifier = Classifier(in_dim, num_classes)
        
        if i == -1: 
            net_glob = copy.deepcopy(model)
            clf_glob = copy.deepcopy(classifier)
            initial_net_dict = copy.deepcopy(net_glob.state_dict())
            initial_clf_dict = copy.deepcopy(clf_glob.state_dict())
        
        else:
            src_models.append(copy.deepcopy(model))
            src_clfs.append(copy.deepcopy(clf_glob))
            src_clfs[i].load_state_dict(initial_clf_dict)

    return net_glob, clf_glob, src_models, src_clfs, initial_net_dict, initial_clf_dict


