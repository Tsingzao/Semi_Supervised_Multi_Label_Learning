%%
function A_LS = cal(X)

scale = 0.04;              %% Scale to use for standard spectral clsutering
neighbor_num = 15;         %% Number of neighbors to consider in local scaling
X = X - repmat(mean(X),size(X,1),1);
X = X/max(max(abs(X)));

%%%%%%%%%%%%%%%%% Build affinity matrices
D = dist2(X,X);              %% Euclidean distance 
clear X;
% A = exp(-D/(scale^2));       %% Standard affinity matrix (single scale)
[D_LS,A_LS,LS] = scale_dist(D,floor(neighbor_num/2)); %% Locally scaled affinity matrix
clear D_LS; clear LS; clear D;

end