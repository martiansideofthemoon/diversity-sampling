function [L energy lower_bound] = perform_inference(node_energy, edge_list, edge_energy, inference_opt, ...
                                                  max_iter, planes, sp_data, en_type)

%
% function [L energy lower_bound] = perform_inference(node_energy, edge_list, edge_energy,
%                                   inference_opt, max_iter, planes, sp_data)
%
% Function to perform inference on a pairwise MRF. 
%
% Inputs:
% 1. node_energy     n_labels x n_nodes matrix of node energies. Each column corresponds to a node.
%
% 2. edge_list       {2,3} x n_edges matrix of adjacency list. Each column corresponds to an edge
%                    containing the following information: [start_node end_node 
%                    [pairwise_energy_table_index]]. If there are only 2 rows then 
%                    pairwise_energy_table_index is assumed to be the column number, therefore you 
%                    must have n_edges == P. 1-based numbering for nodes.
%
% 3. edge_energy     (n_labels^2) x P matrix holding pairwise energy terms. Each column holds a 
%                    pairwise/edge energy term strung-up into a vector by concatenating COLUMNS (like
%                    the matlab : operator). Thus for a 2-label edge term: [E00; E10; E01; E11].
% 
% 4. inference_opt   Options for inference algorithms are:
%
%                    {'qpbo', 'qpbo+node', 'qpbo+grad', 'qpbo+trw', 'qpbo+bp', 'trw', 'bp'
%                    'multi2bool+qpbo', 'multi2bool+qpbo+node', 'multi2bool+qpbo+grad',
%                    'multi2bool+qpbo+trw', 'multi2bool+qpbo+bp', 'multi2bool+trw', 'multi2bool+bp'}.
%                    
%                    All the multi2bool options first convert the multilabel problem into an equivalent
%                    boolean label problem with the Ramalingam Battleship transformation, and then
%                    apply the remaining method on this boolean problem.
%
%
% Outputs:
% 1. L               n_nodes x 1 vector of MAP labels (0-based numbering). If algorithm is 'qpbo' and
%                    energy is non-submodular, L might have unlabelled nodes (denoted by -ve
%                    labels). In this case energy cannot be computed and will be set to -inf.
%
% 2. energy          scalar holding energy of labelling L. Only computed if nargout > 1. Cannot be
%                    computed for partial labellings (set to +inf).
%
% 3. lower_bound     scalar holding lower bound on MAP energy. Only meaningful in the case of TRW. For
%                    others, set to -inf.
%
%
% Dhruv Batra (batradhruv -at- cmu.edu)
% Created: 04/21/2009
% Updated: 08/19/2009
% v0.9: Known bug in vgg_trw_bp (2-label, tabular energy) -- for now swapping rows of edge_energy to
%       work around bug. This should later be removed once that bug is fixed.
% Updated: 10/14/2009 -- Added max_iter and planes before sp_data in input arguments, thus breaking backward
%                        compatibility. 
% Updated: 11/5/2009: Added en_type so that truncated quadratic and linear models maybe used with trw-s.
% Updated: 03/15/2012: vgg_trw_bp bug fixes and workaround removed. 

error(nargchk(4,8,nargin));

if (~exist('max_iter','var') || isempty(max_iter))
  max_iter = 30; 
end

n_labels = size(node_energy,1);
n_nodes = size(node_energy,2);
verbose_level = 0;
lower_bound = -inf;

if (~exist('en_type','var') || isempty(en_type))
  if (size(edge_energy,1) == n_labels^2) % tabular input
    en_type = 0;
  else
    error('energy type must be specified if non-tabular input');
  end
end

first4 = inference_opt(1:min(4,length(inference_opt)));
first3 = inference_opt(1:min(3,length(inference_opt)));

if isequal(lower(first4),'qpbo') % check for algorithms involving qpbo
  if (n_labels~=2)
    error('QPBO can only be used for 2-label problems');
  end
  
  % Just qpbo for now (L might have unlabelled nodes. Energy cannot always be computed.)
  if verbose_level > 0
    fprintf('%s... ', inference_opt); t = clock;
  end
  
  edge_energy([2 3],:) = edge_energy([3 2],:); % vgg code expects row major edge_energy
  [L stats] = vgg_qpbo(node_energy, uint32(edge_list), edge_energy);
  edge_energy([3 2],:) = edge_energy([2 3],:); % Set back
  
  unlabelled = find(L<0);
  if verbose_level > 0
    fprintf('QPBO left %d (%f %%) nodes unlabelled\n',length(unlabelled),length(unlabelled)/size(node_energy,2));
  end
  
end
  
if isequal(lower(first3),'opd') % compute some things common to all opd algs
  if (n_labels~=2)
    error('OPD can only be used for 2-label problems (for now, we''re working on it..)');
  end
  primal_data.node_energy = node_energy;
  primal_data.edge_list = [edge_list(1:2,:) [1:n_nodes;(n_nodes+1)*ones(1,n_nodes)]];
  primal_data.edge_energy = edge_energy;
  edge_weights = energy_to_edge_weights(edge_list, node_energy, edge_energy);
  primal_data.edge_weights = edge_weights;
  
  [ue,pe,en0] = get_state_energy(ones(n_nodes,1),primal_data.node_energy,edge_list,primal_data.edge_energy);
  primal_data.en0 = en0;
  
  num_planes = length(planes);
  node_occ = planes(1).node_occ;
  edge_occ = planes(1).edge_occ;
  for i=1:num_planes
    planes(i).edge_weights = edge_weights(planes(i).edge_inds).*[edge_occ(planes(i).edge_inds(1:end-planes(i).num_nodes)) node_occ(planes(i).node_list(1:end-1))]';
    planes(i).node_energy = node_energy(:,planes(i).node_list(1:end-1)).*repmat(node_occ(planes(i).node_list(1:end-1)),2,1);
    if (size(edge_list,1) == 2)
      planes(i).edge_energy = edge_energy(:,planes(i).edge_inds(1:end-planes(i).num_nodes)).*repmat(edge_occ(planes(i).edge_inds(1:end-planes(i).num_nodes)),4,1);
    elseif (size(edge_list,1) == 3)
      planes(i).edge_energy = edge_energy(:,edge_list(3,planes(i).edge_inds(1:end-planes(i).num_nodes))).*repmat(edge_occ(planes(i).edge_inds(1:end-planes(i).num_nodes)),4,1);
    end
  end
%keyboard
  clear node_energy edge_energy edge_list edge_weights

end

switch lower(inference_opt)
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Just qpbo (L might have unlabelled nodes. Energy cannot always be computed.)
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 case 'qpbo'
  
  if isempty(unlabelled)
    [dummy,dummy,energy] = get_state_energy(L+1,node_energy,edge_list,edge_energy);  
  else
    energy = inf;
  end
  
  if verbose_level > 0
    fprintf('%f seconds\n',etime(clock,t));
  end

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % qpbo + set unlabelled nodes to maximize node energy
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 case 'qpbo+node'

  if ~isempty(unlabelled)
    L(unlabelled) = (node_energy(1,unlabelled) > node_energy(2,unlabelled));
  end
  
  [dummy,dummy,energy] = get_state_energy(L+1,node_energy,edge_list,edge_energy);  

  if verbose_level > 0
    fprintf('%f seconds\n',etime(clock,t));
  end

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % qpbo + node initialized + gradient descent on all nodes/pixels
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 case 'qpbo+gradsp'

  if ~isempty(unlabelled)
    % perform gradient descent to label these nodes
    [L2 energy] = label_nodes(L, node_energy, edge_list, edge_energy, sp_data);
    L = L2;
  end
  
  [dummy,dummy,energy] = get_state_energy(L+1,node_energy,edge_list,edge_energy);
  
  if verbose_level > 0
    fprintf('%f seconds\n',etime(clock,t));
  end

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % qpbo + node initialized + gradient descent on supernodes/superpixels
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 case 'qpbo+gradsp'

  if ~isempty(unlabelled)
    % perform gradient descent to label these nodes
    [L2 energy] = label_nodes(L, node_energy, edge_list, edge_energy, sp_data);
    L = L2;
  end
  
  [dummy,dummy,energy] = get_state_energy(L+1,node_energy,edge_list,edge_energy);
  
  if verbose_level > 0
    fprintf('%f seconds\n',etime(clock,t));
  end
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % qpbo + TRW-S/BP
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 case {'qpbo+trw', 'qpbo+bp'}

  if ~isempty(unlabelled)
    % get new energy on these unlabelled nodes
    inv_mapping = zeros(size(L));
    inv_mapping(unlabelled) = 1:length(unlabelled);
    
    % reduced node energy
    ul_node_energy = node_energy(:,unlabelled);
    
    % reduced edge set and edge energy
    ul_edge = ismember(edge_list,unlabelled);
    ul_edge_ind = find(and(ul_edge(1,:),ul_edge(2,:))); % both endpoints of edge are unlabelled
    ul_edge_list = inv_mapping(edge_list(:,ul_edge_ind)); % renumber nodes in these edges

    ul_edge_energy = edge_energy(:,ul_edge_ind); % get energy of these edges
% $$$     ul_edge_energy = reshape(ul_edge_energy(:),n_labels,n_labels,[]); % prepare for trw/bp wrapper
% $$$     ul_edge_energy = permute(num2cell(ul_edge_energy,[1 2]),[2 3 1]);
    
    % reduce parts of edge energy into node components
    e_ind = find(xor(ul_edge(1,:),ul_edge(2,:)));

    % edges where node1 is unlabelled
    e_ind1 = e_ind(find(ul_edge(1,e_ind)));
    for e=e_ind1
      ul_node_energy(:,inv_mapping(edge_list(1,e))) = ul_node_energy(:,inv_mapping(edge_list(1,e))) + ... 
          edge_energy(2*L(edge_list(2,e))+int32([1 2]),e);
    end
    % edges where node2 is unlabelled
    e_ind2 = e_ind(find(ul_edge(2,e_ind)));
    for e=e_ind2
      ul_node_energy(:,inv_mapping(edge_list(2,e))) = ul_node_energy(:,inv_mapping(edge_list(2,e))) + ... 
          edge_energy(L(edge_list(1,e))+int32([1 3]),e);
    end
% $$$     e_ind1 = e_ind(find(ul_edge_ind(1,e_ind)));
% $$$     ul_node_energy(:,inv_mapping(edge_list(1,e_ind1))) = ul_node_energy(:,inv_mapping(edge_list(1,e_ind1))) + ... 
% $$$         edge_energy(2*L(edge_list(2,e_ind1))+[1 2],e_ind1);
% $$$ 
% $$$     % edges where node2 is unlabelled
% $$$     e_ind2 = e_ind(find(ul_edge_ind(2,e_ind)));
% $$$     ul_node_energy(:,inv_mapping(edge_list(2,e_ind2))) = ul_node_energy(:,inv_mapping(edge_list(2,e_ind2))) + ... 
% $$$         edge_energy(L(edge_list(1,e_ind2))+[1 3],e_ind2);
    
    
    % perform TRW-S/BP to label these nodes
% $$$     if isequal(inference_opt,'qpbo+bp')
% $$$       opts = int32([0 0 30]);
% $$$     else
% $$$       opts = int32([1 0 30]);
% $$$     end
% $$$     [L2 energy] = vgg_trw_bp(num2cell(ul_node_energy',2), uint32(ul_edge_list), ul_edge_energy, opts);
% $$$     L(unlabelled) = L2-1;
    
    % call perform_inference again, instead of copying code
    [L2] = perform_inference(ul_node_energy, ul_edge_list, ul_edge_energy, inference_opt(6:end));
    L(unlabelled) = L2;
  end
  
  [dummy,dummy,energy] = get_state_energy(L+1,node_energy,edge_list,edge_energy);
  if verbose_level > 0
    fprintf('%f seconds\n',etime(clock,t));
  end

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % via Ramalingam multi2bool transformation + {method}
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 case {'multi2bool+qpbo','multi2bool+qpbo+node','multi2bool+qpbo+grad','multi2bool+qpbo+trw', ...
       'multi2bool+qpbo+bp','multi2bool+trw','multi2bool+bp'}
  
  if (n_labels==2) 
    warning('multi2bool called for a 2-label problem: skipping multi2bool transformation');
    
    % solve as it is
    [L energy lower_bound] = perform_inference(node_energy, edge_list, edge_energy, inference_opt(12:end));
  else
    % convert to boolean problem via battleship formation
    [o2b_node_en, o2b_edge_en, o2b_edge_list offset] = M2B_battleship_mex(node_energy, edge_energy, uint32(edge_list-1));
%keyboard
    % solve boolean problem
    [L energy lower_bound] = perform_inference(o2b_node_en, double(o2b_edge_list+1), o2b_edge_en, inference_opt(12:end));
%keyboard  
    L = bl2ml_bs(L,n_nodes,n_labels);
    
    energy = energy+offset;
    lower_bound = lower_bound+offset;
  end
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % via TRW-S or BP
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 case {'trw', 'bp'}
  if verbose_level > 0
    fprintf('%s... ', inference_opt); t = clock;
  end
  
  if (en_type == 0) % tabular input
      % v2.4 of imrender fixed the bug. 
%     if (n_labels == 2)
%       edge_energy([2 3],:) = edge_energy([3 2],:); % vgg trw/bp code has a known bug for 2-label case.
%     end
    edge_energy = reshape(edge_energy(:),n_labels,n_labels,[]);
    edge_energy = permute(num2cell(edge_energy,[1 2]),[2 3 1]);
  else
    if (size(edge_energy,1)~=2)
      error('incorrect size of edge_energy');
    end
    edge_energy = num2cell(edge_energy,1)';
  end
    
  if isequal(inference_opt,'bp')
    opts = int32([0 en_type max_iter]);
  else
    opts = int32([1 en_type max_iter]);
  end

  [L energy lower_bound] = vgg_trw_bp(num2cell(node_energy',2), uint32(edge_list), edge_energy, opts);
  L = L-1;

  if isequal(inference_opt,'bp')
    lower_bound = -inf;
  end
  
  if verbose_level > 0
    fprintf('%f seconds\n',etime(clock,t));
  end
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % via exhaustive enumeration
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 case 'exhaustive'
  energy = inf;
  for i=0:(n_labels^(n_nodes)-1)
    p = dec2base(i,n_labels,n_nodes);
    l = str2num([fliplr(p);blanks(length(p))]')';
    [ue,pe,en] = get_state_energy(l+1,node_energy,edge_list,edge_energy);
    if en < energy
      energy = en;
      L = l;
    end
  end
  lower_bound = energy;
  L = L';
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % via OPD
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 case 'opd-pmp'
  [L dummy energy] = opd_mp_ns(primal_data,planes,[],n_nodes,0,[],max_iter);
  L = L-1;
 case 'opd-smp'
  [L dummy energy] = opd_mp_ns(primal_data,planes,[],n_nodes,1,[],max_iter);
  L = L-1;  
 case 'opd-rmp'
  [L dummy energy] = opd_rmp_ns(primal_data,planes,[],n_nodes,[],max_iter);
  L = L-1;
 case 'opd-dd'
  [L dummy energy lower_bound] = opd_dd_ns(primal_data,planes,[],max_iter);
  L = L-1;
 case 'opd-dd-map'
  [L dummy energy lower_bound] = opd_dd_map_ns(primal_data,planes,[],max_iter);
  L = L-1;
  
 otherwise
  error(sprintf('Could not find an inference method with name -- %s',inference_opt));

end


L = double(L);