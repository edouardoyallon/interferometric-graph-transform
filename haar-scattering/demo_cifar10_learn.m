% demo_cifar10_learn.m
% 
% ----------------------------------------------------------------------------%
% Classify CIFAR-10 by learnt Haar scattering with PLS dimension reduction
%  - Learning multiple permutation vectors.
%  - Apply Haar scattering.
%  - Bagging + PLS feature selection
%  - Classification with RBF-SVM
% ----------------------------------------------------------------------------%

clear; close all;

%% Load MNIST data set

path_to_cifar = '/users/data/oyallon/haarscat-0.1/cifar-10-batches-mat/'
[X_tr, X_te] = read_cifar10(path_to_cifar);

%% Set parameters 

NTree = 1;
scat_opt.J = 7;
scat_opt.M = 2;
scat_opt.pnorm = 2;

dim_perclass = 75;

%% Organize training and testing data

NClass = 10;

TrainData = [];
TestData = [];
TestLabel = [];
TrainLabel = [];

for iclass = 1:NClass
    TrainData = [TrainData X_tr{iclass}];
    TrainLabel = [TrainLabel; iclass*ones(size(X_tr{iclass},2),1)];
    TestData = [TestData X_te{iclass}];
    TestLabel = [TestLabel; iclass*ones(size(X_te{iclass},2),1)];
end

NTr = length(TrainLabel);
NTe = length(TestLabel);

%% Divide training data ( in order to learn multiple trees)

Data = cell(NTree,1);

idx_pt = randperm(length(TrainLabel));
NPtree = min(500,floor(length(TrainLabel)/NTree));
for itree = 1:NTree
    Data{itree} = TrainData(:,idx_pt((itree-1)*NPtree+1:itree*NPtree));
end

%% Learn multiple trees

nch = 3;
perm_vec = cell(NTree,1);

for itree = 1:NTree
    perm_vec{itree} = cell(nch, 1);
    for ich = 1:nch
        ind = ((ich-1)*1024+1:ich*1024)';
        data = Data{itree}(ind, :);
        perm_vec{itree}{ich} = haar_tree_learn(data, scat_opt);
    end
end

    
%% Compute Haar scattering coefficients

Data_all = [TrainData, TestData];

s = [];
for itree = 1:NTree
    for ich = 1: nch  
        ind = ((ich-1)*1024+1:ich*1024)';
        data = Data_all(ind, :);
        [ss,  ~] = haar_scat(data, perm_vec{itree}{ich}, scat_opt);
        if length(size(ss)) > 2
            ss = reshape(ss, [size(ss,1)*size(ss,2),NTr+NTe]);
        end
        s = [s; ss];
    end
end

Scat_Tr = s(:, 1:NTr);
Scat_Te = s(:, NTr+1 : NTr+NTe);

%% PLS feature selection

% [Feat_Tr, Feat_Te, meta_pls] = pls_multiclass( Scat_Tr, Scat_Te,  TrainLabel, NClass, dim_perclass);

Feat_Tr = Scat_Tr
Feat_Te = Scat_Te

%% RBF-SVM Classification

% Using prefixed parameters
params.C = 4;
params.gamma = 1;

Feat_all = [Feat_Tr Feat_Te];
[~,Feat_all] = preproc(Feat_all);
Feat_Tr = Feat_all(:, 1:NTr);
Feat_Te = Feat_all(:, NTr+1 : NTr+NTe);

% acc = svm_rbf(Feat_Tr', TrainLabel, Feat_Te', TestLabel, params);

% Uncomment the following line to perform exponential-grid cross validation
% Be aware of the computational time
[acc, opt_C, opt_gamma] = svm_rbf(Feat_Tr', TrainLabel, Feat_Te', TestLabel);





