function output = DivMBest_intseg(params)

%% Collect Parameters

datadir = params.datadir;
gtdir = params.gtdir;
savedir = params.savedir;

type = params.type;

nummodes = params.nummodes;
nlabels = params.nlabels;

lambda = params.lambda;

fname = params.fname;

gt = params.gt;

% Collect Energies

labels = params.labels;
ne = params.ne;
el = params.el;
ee = params.ee;

%% Get the DivMBest solutions

nnodes = size(ne, 2);

% find Modes
divne = ne;
allL = [];

for ps = 1:nummodes
    [L en lb] = perform_inference(divne,el,ee,'trw');
    allL(ps,:) = L';
    seg{ps} = label2seg(L, labels);
    
    % compute accuracies
    [acc, precision, recall, sol_iou(ps), fmeasure] = computeStats(seg{ps}, gt);
    
    [~,~,sol_en(ps)] = get_state_energy(L+1,ne,el,ee);
    
    % suppress mode
    if(strcmp(type, 'perturb'))
	divne = ne;
	U = rand([size(ne,1) size(ne,2)]);
	gumbel = log(-log(U));
	divne = divne + lambda * gumbel;
    elseif(strcmp(type, 'divMbest'))
	inds = sub2ind([nlabels nnodes],L'+1,1:nnodes);
	divne(inds) = divne(inds) + lambda;
    elseif(strcmp(type, 'divMbest_boundary_'))
	%not everywhere. only penalize inds at boundaries
	bedges = find(L(el(1,:)) ~= L(el(2,:)));
	bnodes = unique(el(:,bedges));
	inds = sub2ind([nlabels nnodes],L(bnodes)+1,bnodes);
	
	divne(inds) = divne(inds) + lambda;
    end
    hamdist(ps) = sum(allL(ps,:)~=allL(1,:));
%                     figure, imshow(seg{ps,pf});
end

output.seg = seg;
output.sol_iou = sol_iou;
output.sol_en = sol_en;
end
