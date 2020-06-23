% demo_mnist_learn.m
% 
% ----------------------------------------------------------------------------%
% Classify MNIST by learnt Haar scattering with PLS dimension reduction
%  - Learning multiple permutation vectors.
%  - Apply Haar scattering.
%  - Bagging + PLS feature selection
%  - Classification with RBF-SVM
% ----------------------------------------------------------------------------%

clear; close all;

%% Load MNIST data set

path_to_mnist = '/users/data/oyallon/haarscat-0.1/';
[X_tr, X_te] = read_mnist(path_to_mnist);

%% Set parameters 

NTree = 1; %40;
scat_opt.J = 6;
scat_opt.M = 4;
scat_opt.pnorm = 1;

dim_perclass = 50;

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

perm_vec = cell(NTree,1);

parfor itree = 1:NTree
    perm_vec{itree} = haar_tree_learn(Data{itree}, scat_opt);
end
    
%% Compute Haar scattering coefficients

Data_all = [TrainData, TestData];

s = []; 
for itree = 1:NTree
    [ss,  ~] = haar_scat(Data_all, perm_vec{itree}, scat_opt);
    if length(size(ss)) > 2
        ss = reshape(ss, [size(ss,1)*size(ss,2),NTr+NTe]);
    end
    s = [s; ss];
end

Scat_Tr = s(:, 1:NTr);
Scat_Te = s(:, NTr+1 : NTr+NTe);

%% PLS feature selection

% [Feat_Tr, Feat_Te, meta_pls] = pls_multiclass( Scat_Tr, Scat_Te,  TrainLabel, NClass, dim_perclass);

%% RBF-SVM Classification

% Using prefixed parameters
params.C = 4;
params.gamma = 1;

Feat_Tr=Scat_Tr
Feat_Te=Scat_Te

Feat_all = [Feat_Tr Feat_Te];
[~,Feat_all] = preproc(Feat_all);
Feat_Tr = Feat_all(:, 1:NTr);
Feat_Te = Feat_all(:, NTr+1 : NTr+NTe);

% acc = svm_rbf(Feat_Tr', TrainLabel, Feat_Te', TestLabel, params);

% Uncomment the following line to perform exponential-grid cross validation
% Be aware of the computational time
[acc, opt_C, opt_gamma] = svm_rbf(Feat_Tr', TrainLabel, Feat_Te', TestLabel);

