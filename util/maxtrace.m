%% Maximum Matrix Trace
function retQ = maxtrace(UniqueMatrix, nDimRed)
[u, s, v] = svd(UniqueMatrix);
retQ = u(:,1:nDimRed);
end