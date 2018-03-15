function seg = label2seg(L, splabels)

nsp = length(unique(splabels));
assert(nsp == length(L));

seg = zeros(size(splabels));
for j=1:nsp
    inds = (splabels==j-1);
    
    seg(inds) = L(j);
end

