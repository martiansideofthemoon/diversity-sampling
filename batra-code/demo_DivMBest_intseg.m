%% This is a demo script that computes multiple diverse binary segmentations.

%% Create a structure called 'params' that stores all the arguments, which will be passed to the DivMBest function.

% Paths
addpath(genpath('./utils'));
datadir = './voctest50data'; params.datadir = datadir;
gtdir = fullfile(params.datadir, 'gtdir'); params.gtdir = gtdir;
params.savedir = './savedir';

% DivMBest-related arguments
params.type = 'divMbest_boundary_';
params.nummodes = 5;
params.nlabels = 2;
params.lambda = 0.2;

% Image-related arguments
flist = dir(fullfile(params.datadir,'*.mat'));
for i = 25:50
    fname = flist(i).name(1:end-4); params.fname = fname;
    params.gt = imread(sprintf('%s/%s.png',gtdir,fname));
    
    % Model-related arguments
    % Load data and construct the energies
    load_struct = load(sprintf('%s/%s.mat',datadir,fname));
    data_term = load_struct.data_term;
    labels = load_struct.labels; params.labels = labels;
    sparse_term = load_struct.sparse_term;
    
    %% ne
    nnodes = size(data_term,2);
    assert(nnodes==length(unique(labels)));
    ne = data_term;
    
    % DB:  swap 1 and 0 terms because something funny seems to be going on. Maybe Payman was maximizing. Or maybe these are outputs of classifiers (so scores, not energies)
    ne([1 2],:) = ne([2 1],:); params.ne = ne;
    
    % el
    [node1 node2 wt] = find(triu(sparse_term));
    nedges = length(wt);
    el = [node1 node2]'; params.el = el;
    
    % ee
    ee = zeros(4,nedges);
    ee(2,:) = wt;
    ee(3,:) = wt;
    params.ee = ee;
    
    %% =====================================
    
    %% Perform DivMBest
    output = DivMBest_intseg(params);
    
    %% =====================================
    
    %% Visualize the diverse solutions
    handles.H = figure,
    subplot(2,5,3), imshow(params.gt);
    title('Ground Truth');
    
    for ii = 1:5
        subplot(2,5,5+ii), imshow(output.seg{ii});
        title(['Divsol #' num2str(ii)]);
    end
    axis on, waitforbuttonpress 
    close(handles.H)

end
