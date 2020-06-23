% demo_cifar10_2d_image_grid.m
% 
% ----------------------------------------------------------------------------%
% Classify CIFAR-10 by Haar scattering with PLS dimension reduction
%  - Form multiple permutation vectors corresponding to different shifts in
%  2d image grid.
%  - Apply Haar scattering.
%  - Bagging + PLS feature selection
%  - Classification with RBF-SVM
% ----------------------------------------------------------------------------%

clear; close all;

%% Load MNIST data set

path_to_cifar = '/users/data/oyallon/haarscat-0.1/cifar-10-batches-mat/'
[X_tr, X_te] = read_cifar10(path_to_cifar);

%% Set parameters 

N_shift = 1 % 64;
N_rot = 1 % 6; % number of rotations

scat_opt.J = 6;
scat_opt.M = 4;
scat_opt.direction = 'h';

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

%% Compute Haar scattering coefficients

Data_all = [TrainData, TestData];
dim1 = 32;
dim2 = 32;
nch = 3;

max_ang = 180;
s = [];
for ich = 1: nch
    
    ind = ((ich-1)*1024+1:ich*1024)';
    data = Data_all(ind, :);
    data = reshape( data, dim1, dim2, NTr + NTe);
    
    for irot = 1:N_rot
        temp_ang = (irot-1)*max_ang/N_rot;
        
        % rotate and crop
        temp_data = rotate_crop(data, temp_ang);
        
        % compute averaged haar scattering over multiple shifted image grid
        [ss, meta]  = haar_scat_2d_image_grid_multishift(temp_data, N_shift, scat_opt);
        if length(size(ss)) > 2
            ss = reshape(ss, [size(ss,1)*size(ss,2),NTr+NTe]);
        end
        
        s = [s; ss];
    end
end

Scat_Tr = s(:, 1:NTr);
Scat_Te = s(:, NTr+1 : NTr+NTe);

%% PLS feature selection

Feat_Tr=Scat_Tr
Feat_Te=Scat_Te
% [Feat_Tr, Feat_Te, meta_pls] = pls_multiclass( Scat_Tr, Scat_Te,  TrainLabel, NClass, dim_perclass);

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



