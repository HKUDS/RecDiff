import numpy as np
from torch.utils.data import DataLoader
import torch.utils.data as data
import torch as t
from utils import load_data, load_model, save_model, fix_random_seed_as
import scipy.sparse as sp
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
from param import args
import  pickle
import torch, dgl

class DataHandler:
    def __init__(self):
        predir = ''
        if args.dataset == 'yelp':
            predir = './datasets/yelp/'
            self.datapath = predir + 'dataset.pkl'
        elif args.dataset == 'ciao':
            predir = './datasets/ciao/'
            self.datapath = predir + 'dataset.pkl'
        elif args.dataset == 'epinions':
            predir = './datasets/epinions/'
            self.datapath = predir + 'dataset.pkl'
        self.predir = predir
        


    def loadOneFile(self,data_path):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        return data

    def LoadData(self):
        self.dataset = self.loadOneFile(self.datapath)
        trnMat = self.dataset['train']
        tstMat = self.dataset['test']
        valMat = self.dataset['val']
        trainset = TrnData(trnMat)
        testset = TstData(tstMat, trnMat)
        valset = TstData(valMat,trnMat)
        self.n_user, self.n_item = self.dataset['userCount'], self.dataset['itemCount']
        args.user, args.item = self.n_user, self.n_item

        self.trainloader = DataLoader(
            dataset=trainset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )
        self.valloader = DataLoader(
            dataset=valset,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
        self.testloader = DataLoader(
            dataset=testset,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )

        
        self.uu_graph = dgl.from_scipy(self.dataset['trust'])
        uimat = self.dataset['train'].tocsr()
        self.ui_graph = self.makeBiAdj(uimat,self.n_user,self.n_item)

    def makeBiAdj(self, mat,n_user,n_item):
        a = sp.csr_matrix((n_user, n_user))
        b = sp.csr_matrix((n_item, n_item))
        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0
        mat = mat.tocoo()
        edge_src,edge_dst = mat.nonzero()
        ui_graph = dgl.graph(data=(edge_src, edge_dst),
                            idtype=torch.int32,
                             num_nodes=mat.shape[0]
                             )

        return ui_graph

    # def normalizeAdj(self, mat):
    #     degree = np.array(mat.sum(axis=-1))
    #     dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
    #     dInvSqrt[np.isinf(dInvSqrt)] = 0.0
    #     dInvSqrtMat = sp.diags(dInvSqrt)
    #     return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

class TrnData(data.Dataset):
    def __init__(self, coomat):
        self.rows = coomat.row
        self.cols = coomat.col
        self.dokmat = coomat.todok()
        self.negs = np.zeros(len(self.rows)).astype(np.int32)

    def negSampling(self):
        for i in range(len(self.rows)):
            u = self.rows[i]
            while True:
                iNeg = np.random.randint(args.item)
                if (u, iNeg) not in self.dokmat:
                    break
            self.negs[i] = iNeg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx], self.negs[idx]

class TstData(data.Dataset):
    def __init__(self, coomat, trnMat):
        self.csrmat = (trnMat.tocsr() != 0) * 1.0

        tstLocs = [None] * coomat.shape[0]
        tstUsrs = set()
        for i in range(len(coomat.data)):
            row = coomat.row[i]
            col = coomat.col[i]
            if tstLocs[row] is None:
                tstLocs[row] = list()
            tstLocs[row].append(col)
            tstUsrs.add(row)
        tstUsrs = np.array(list(tstUsrs))
        self.tstUsrs = tstUsrs
        self.tstLocs = tstLocs

    def __len__(self):
        return len(self.tstUsrs)

    def __getitem__(self, idx):
        return self.tstUsrs[idx], np.reshape(self.csrmat[self.tstUsrs[idx]].toarray(), [-1])



