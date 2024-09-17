from utils import *
from client import *
from configs import args
from custom_transforms import *
from dataset import *
from torch.utils.data import DataLoader
import os
import gc

###############################Data########################################

train_dls = []
test_dls = []
U_clients = []
D_clients = []

for idx in range(args.n_src):
    root_dir = os.path.join(args.root, args.src[idx])
    train_data, test_data = get_dataset(root_dir, args.train_pct)

    train_ds  = CustomImageDataset(train_data, test_transform)
    U_temp = get_basis_vec(train_ds, args.n_basis)

    train_ds = CustomImageDataset(train_data, strong_transform, weak_transform)
    test_ds = CustomImageDataset(test_data, test_transform)

    train_dl = DataLoader(train_ds, batch_size = args.local_bs, shuffle = True)
    test_dl = DataLoader(test_ds, batch_size = args.local_bs, shuffle = False)

    train_dls.append(train_dl)
    test_dls.append(test_dl)

    U_clients.append(copy.deepcopy(np.hstack(U_temp)))
    D_clients.append(len(train_ds))

target_train_data, target_test_data = get_dataset(os.path.join(args.root, args.target), args.train_pct)
target_test_ds = CustomImageDataset(target_test_data, test_transform)
target_test_dl = DataLoader(target_test_ds, batch_size = args.local_bs, shuffle = False)

############################Init Models#######################################

net_glob, clf_glob, clients_models, clients_clfs, initial_net_dict, initial_clf_dict = init_models(args.n_src, args.in_dim, args.n_classes )


###############################Clients#########################################

clients = []
for idx in range(args.n_src):
    clients.append(Client(args.src[idx], copy.deepcopy(clients_models[idx]), copy.deepcopy(clients_clfs[idx]),
     args.local_bs, args.local_ep, args.lr, args.momentum, args.weight_decay, args.device, train_dls[idx], test_dls[idx]))


####################################Adj##########################################
client_idxs = np.arange(len(U_clients))
adj_mat = calculating_adjacency(client_idxs, U_clients)
adj_mat /= args.n_classes * args.n_basis

print(adj_mat)


#########################Train#############################
#########################Train#############################
#########################Train#############################
loss_train = []

init_tracc_pr = []  # initial train accuracy for each round 
final_tracc_pr = [] # final train accuracy for each round 

init_tacc_pr = []  # initial test accuarcy for each round 
final_tacc_pr = [] # final test accuracy for each round

init_tloss_pr = []  # initial test loss for each round 
final_tloss_pr = [] # final test loss for each round 

clients_best_acc = [0 for _ in range(args.n_src)]
w_locals, loss_locals = [], []

init_local_tacc = []       # initial local test accuracy at each round 
final_local_tacc = []      # final local test accuracy at each round 

init_local_tloss = []      # initial local test loss at each round 
final_local_tloss = []     # final local test loss at each round 

ckp_avg_tacc = []
ckp_avg_best_tacc = []

best_glob_acc = [0 for _ in range(args.n_src)]
current_glob_acc = [0 for _ in range(args.n_src)]

target_best_acc = 0
target_acc = []

w_glob_net = copy.deepcopy(initial_net_dict)
w_glob_clf = copy.deepcopy(initial_clf_dict)
print_flag = False
for iteration in range(args.rounds):
    #idxs_users = comm_users[iteration]
    
    print(f'###### ROUND {iteration+1} ######')
        
    for idx in range(args.n_src):
        
        clients[idx].set_state_dict(copy.deepcopy(w_glob_net), copy.deepcopy(w_glob_clf)) 
            
        #loss, acc = clients[idx].eval_test()        

        # init_local_tacc.append(acc)
        #init_local_tloss.append(loss)
        
        clfs = []
        sims = []
        
        for i in range(args.n_src):
            if i != idx:
                clfs.append(clients[i].get_clf())
                sims.append(adj_mat[idx][i])
        
        loss = clients[idx].train(clfs, sims, args.mu)
                        
        loss_locals.append(copy.deepcopy(loss))
                       
        #loss, acc = clients[idx].eval_test()

        #if acc > clients_best_acc[idx]:
        #    clients_best_acc[idx] = acc

        #final_local_tacc.append(acc)
        #final_local_tloss.append(loss)  
        
    
    total_data_points = sum(np.array(D_clients))
    fed_avg_freqs = [D_clients[i] / total_data_points for i in range(args.n_src)]

    domain_weight = []

    for i in range(args.n_src):
        domain_weight.append(1.0/args.n_src)
    
    if iteration == 0:
        print(f'Training sample: {total_data_points}')
        print(f'Samples per domain: {D_clients}')
        print(f'Weight: {fed_avg_freqs}')
        
    
    w_locals = []
    for idx in range(args.n_src):
        w_locals.append(copy.deepcopy(clients[idx].get_state_dict('net')))

    ww = FedAvg(w_locals, weight_avg = domain_weight)
    w_glob_net = copy.deepcopy(ww)
    net_glob.load_state_dict(copy.deepcopy(ww))

    w_locals = []
    for idx in range(args.n_src):
        w_locals.append(copy.deepcopy(clients[idx].get_state_dict('clf')))

    ww = FedAvg(w_locals, weight_avg = domain_weight)
    w_glob_clf = copy.deepcopy(ww)
    clf_glob.load_state_dict(copy.deepcopy(ww))
    
    for i in range(args.n_src):
        _, acc = clients[i].eval_test_glob(net_glob, clf_glob)
        current_glob_acc[i] = acc.item()
        if acc > best_glob_acc[i]:
            best_glob_acc[i] = acc .item()

    # print loss
    loss_avg = sum(loss_locals) / len(loss_locals)
    #avg_init_tloss = sum(init_local_tloss) / len(init_local_tloss)
    #avg_init_tacc = sum(init_local_tacc) / len(init_local_tacc)
    #avg_final_tloss = sum(final_local_tloss) / len(final_local_tloss)
    #avg_final_tacc = sum(final_local_tacc) / len(final_local_tacc)
         
    print('## END OF ROUND ##')
    template = 'Average Train loss: {:.3f}'
    print(template.format(loss_avg))
    
#     template = "AVG Init Test Loss: {:.3f}, AVG Init Test Acc: {:.3f}"
#     print(template.format(avg_init_tloss, avg_init_tacc))
    
#     template = "AVG Final Test Loss: {:.3f}, AVG Final Test Acc: {:.3f}"
#     print(template.format(avg_final_tloss, avg_final_tacc))
    
    print("\nGlobal Model Test Acc (Src):")
    
    for k in range(args.n_src):
        print('{} | current : {:3.3f} | best: {:3.3f}'.format(args.src[k], current_glob_acc[k], best_glob_acc[k]))

    _, acc = eval_test_glob(net_glob, clf_glob, target_test_dl, args.device)
    if acc > target_best_acc:
        target_best_acc = acc
    print('\nGlobal Model Test Acc (Target): ')
    print('{} | current : {:3.3f} | best: {:3.3f}'.format(args.target, acc, target_best_acc))
    
    print_flag = True
#     if iteration < 60:
#         print_flag = True
#     if iteration%args.print_freq == 0: 
#         print_flag = True
        
    if print_flag:
        print('\n--- PRINTING ALL CLIENTS STATUS ---')
        current_acc = []
        for k in range(args.n_src):
            loss, acc = clients[k].eval_test() 
            current_acc.append(acc)
            
            if acc > clients_best_acc[k]:
                clients_best_acc[k] = acc
                
           
            print("{} | current acc: {:3.3f} | best acc: {:3.3f}".format(args.src[k], current_acc[-1], clients_best_acc[k]))
            
        template = ("Round {:1d} | Avg current_acc {:3.3f} | Avg best_acc {:3.3f}")
        print(template.format(iteration+1, np.mean(current_acc), np.mean(clients_best_acc)))
        
        ckp_avg_tacc.append(np.mean(current_acc))
        ckp_avg_best_tacc.append(np.mean(clients_best_acc))
    
    print('----- Analysis End of Round -------\n')
           
    loss_train.append(loss_avg)
    
    #init_tacc_pr.append(avg_init_tacc)
    #init_tloss_pr.append(avg_init_tloss)
    
    #final_tacc_pr.append(avg_final_tacc)
    #final_tloss_pr.append(avg_final_tloss)
    
    #break;
    ## clear the placeholders for the next round 
    loss_locals.clear()
    init_local_tacc.clear()
    init_local_tloss.clear()
    final_local_tacc.clear()
    final_local_tloss.clear()
    
    ## calling garbage collector 
    gc.collect()