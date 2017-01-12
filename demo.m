function demo
%% For Simulate Data Sets
%% Add Path
clear; clc;

addpath('./util');
addpath('./Data');
addpath('./MySemi');

%% Load Data
load('sample data.mat'); 

%% MySemi
tic;
svm.type = 'RBF';
svm.para = 0.05;
[HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,Outputs,Pre_Labels] = JointSemi(train_data, train_target', test_data, test_target',train_data,svm);
CPUTime = toc;

clc;
%% Results
HammingLoss
RankingLoss
OneError
Coverage
Average_Precision
CPUTime