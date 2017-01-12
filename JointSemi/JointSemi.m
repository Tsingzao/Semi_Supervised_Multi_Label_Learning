function [HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,Outputs,Pre_Labels] =...
    JointSemi(feature_lab_train, label_lab_train, feature_all_test, label_all_test, feature_all_train, svm)

if nargin < 5
    feature_all_train = feature_lab_train;
    svm.type='Linear';
    svm.para=[];
end

if(nargin<6)
    svm.type='Linear';
    svm.para=[];
end

switch svm.type
    case 'RBF'
        gamma=num2str(svm.para);
        str=['-t 2 -g ',gamma,' -b 1'];
    case 'Poly'
        gamma=num2str(svm.para(1));
        coef=num2str(svm.para(2));
        degree=num2str(svm.para(3));
        str=['-t 1 ','-g ',gamma,' -r ', coef,' -d ',degree,' -b 1'];
    case 'Linear'
        str='-b 1';
    otherwise
        error('SVM types not supported, please type "help LIFT" for more information');
end

nClass = size(label_lab_train, 2);
nFeature = size(feature_lab_train,2);
if nFeature < 100
    nDimRed = nFeature;
else
    nDimRed = 100;
end
n_lab = size(feature_lab_train, 1);
nTest = size(feature_all_test, 1);

old_error = 100;
new_error = 0;
tol = abs(old_error - new_error);
Q = ones(nFeature, nDimRed);
alpha = zeros(n_lab, nClass);
D = cell(1,nClass);

for i = 1 : nClass
    D{1,i} = diag(label_lab_train(:,i));
end
iter = 1;
while iter < 100
    Models=cell(nClass,1);
    accuracy = [];
    UniqueMatrix = [];
    Pre_Labels=[];
    Outputs=[];
    S = zeros(n_lab, n_lab);
    new_feature = feature_lab_train * Q;
    new_featrue_test = feature_all_test * Q;
    
    for i = 1 : nClass
        label = label_lab_train(:,i);
        Models{i,1} = svmtrain(label, new_feature, str);
    end
    
    for i = 1 : nClass
        alpha(Models{i,1}.sv_indices,i) = Models{i,1}.sv_coef;
    end
    for i = 1 : nClass
        S = S + D{1,i} * alpha(:,i) * alpha(:,i)' * D{1,i}';
    end
    
    for i = 1 : nClass
        label_lab_test_i = label_all_test(:,i);
        
        [predicted_label,acc,prob_estimates]=svmpredict(label_lab_test_i,new_featrue_test,Models{i,1},'-b 1');
        
        if(isempty(predicted_label))
            predicted_label=label_lab_train(i,1)*ones(nTest,1);
            if(label_lab_train(i,1)==1)
                Prob_pos=ones(nTest,1);
            else
                Prob_pos=zeros(nTest,1);
            end
            Outputs=[Outputs;Prob_pos'];
            Pre_Labels=[Pre_Labels;predicted_label'];
        else
            pos_index=find(Models{i,1}.Label==1);
            Prob_pos=prob_estimates(:,pos_index);
            Outputs=[Outputs;Prob_pos'];
            Pre_Labels=[Pre_Labels;predicted_label'];
        end
    end
    clc;
    
    Average_Precision=Average_precision(Outputs,label_all_test')
    old_error = new_error;
    new_error = Average_Precision;
    tol = abs(old_error - new_error);
    if tol < 0.0001
        break;
    end
    iter=iter+1
    xtsx = sparse(feature_lab_train') * sparse(S) * sparse(feature_lab_train);
    Sh = cal(feature_all_train);
    Sh(isnan(Sh))=0;
    xtshx = sparse(feature_all_train') * sparse(Sh) * sparse(feature_all_train);
    UniqueMatrix = chol(xtsx+xtshx+eye(nFeature));
    Q = maxtrace(UniqueMatrix, nDimRed);
end
iter
HammingLoss=Hamming_loss(Pre_Labels,label_all_test');
RankingLoss=Ranking_loss(Outputs,label_all_test');
OneError=One_error(Outputs,label_all_test');
Coverage=coverage(Outputs,label_all_test');
Average_Precision=Average_precision(Outputs,label_all_test');