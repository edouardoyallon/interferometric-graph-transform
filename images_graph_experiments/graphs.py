import numpy as np
import os
import pickle
import glob as glob
import argparse
class Data:





    def get_SBU(self,args):
        fold=args.split
        folds = sorted(glob.glob('./SBU/compact_representation/*.pkl'))

        if fold == 5:
            folds_train = folds[:4]
            folds_test = folds[-1]  # 100% rw conv6
        if fold == 2:
            folds_train = [folds[0], folds[2], folds[3], folds[4]]
            folds_test = folds[1]  # 100% rw conv6
        if fold == 3:
            folds_train = [folds[0], folds[1], folds[3], folds[4]]
            folds_test = folds[2]  # 100% rw conv6
        if fold == 4:
            folds_train = [folds[0], folds[1], folds[2], folds[4]]
            folds_test = folds[3]  # 100% rw conv6
        if  fold == 1:
            folds_train = [folds[3], folds[1], folds[2], folds[4]]
            folds_test = folds[0]  # 93% rw conv6

        with open(folds_test, 'rb') as handle:
            data_test = pickle.load(handle)  # cylindrical

        features = []
        adjacency = []
        label = []
        for fold in folds_train:
            with open(fold, 'rb') as handle:
                data = pickle.load(handle)
            features.append(data[0])
            adjacency.append(data[1])
            label.extend(data[3])
        features = np.vstack(features)
        adjacency = np.vstack(adjacency)
        label = np.asarray(label)

        labels_train,features_train,labels_test,features_test = label,features,data_test[3], data_test[0]

        # build the mask, which the binary adjacency matrix of the skeleton
        adj=adjacency[0]

        adj[adj > 0]=1.0
        mask=adj

        SBU_data=[labels_train,features_train,labels_test,features_test,mask]
        return SBU_data

    def get_NTU_xview_values(self,view='xview'):

        path_data='./NTU'

        path_train=path_data+'/'+view+'/train/'
        path_test = path_data + '/'+view+'/val/'

        with open(path_train+view+'_train.pkl', 'rb') as handle:
            data_train = pickle.load(handle)


        features_train = data_train[0]

        zer = np.argwhere(np.isnan(features_train))
        train_row_delete = np.unique(zer[:, 0])
        features_train = np.delete(features_train, (train_row_delete), axis=0)

        labels_train = data_train[2]
        labels_train = np.delete(labels_train, (train_row_delete), axis=0)

        binary_adj_matrix = data_train[5]
        binary_adj_matrix = np.asarray(binary_adj_matrix)
        adj =  binary_adj_matrix

        adj[adj > 0] = 1.0
        mask = adj

        with open(path_test+view+'_val.pkl', 'rb') as handle:
            data_test = pickle.load(handle)

        features_test = data_test[0]
        zer = np.argwhere(np.isnan(features_test))
        test_row_delete = np.unique(zer[:, 0])
        features_test = np.delete(features_test, (test_row_delete), axis=0)

        labels_test = data_test[2]
        labels_test = np.delete(labels_test, (test_row_delete), axis=0)

        NTU_data = [labels_train, features_train, labels_test, features_test, mask]

        return NTU_data
