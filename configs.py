class Args:
    pass

args = Args()

args.root = '/teamspace/studios/this_studio/pacs/pacs_data/pacs_data'
args.src = ['cartoon', 'photo', 'sketch']
args.target = 'art_painting'
args.n_src = 3
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.n_basis = 3
args.n_classes = 7
args.rounds = 100
args.frac = 0.1 
args.local_ep = 1
args.local_bs = 128
args.lr=0.001
args.momentum = 0.9
args.mu = 1
args.out_dim = 128
args.in_dim = 512
args.train_pct = 0.9
args.weight_decay = 5e-4