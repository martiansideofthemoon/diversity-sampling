function [ue,pe,total] = get_state_energy(state,node_energy,edge_list,edge_energy,tri_list,tri_energy)

%
% function [ue,pe,total] = get_state_energy(state,node_energy,edge_list,edge_energy,tri_list,tri_energy)
% 
% Function to compute the unary, pairwise and total energies of a state.
%
% Inputs
% 1. state         -- <num_nodes> x 1 vector of states. 1-based indexing.
% 2. node_energy   -- 2 x <num_nodes> node energy.
% 3. edge_list     -- 2 x <num_edges> adjacency list. Node numbering in edge_list is 1-based.
% 4. edge_energy   -- 4 x <num_edges> edge energies. Each column is [E00 E10 E01 E11].
%
% Outputs: 
% 1. ue     -- Unary energy (scalar)
% 2. pe     -- Pairwise energy (scalar)
% 3. total  -- ue + pe
%
% Dhruv Batra (batradhruv -at- cmu.edu)
% Created: 04/19/2009
% Modified: 05/03/2010 (added triplet cliques)

error(nargchk(4,6,nargin));

num_states = size(node_energy,1);
%state = state+1; % 0/1 to 1/2

% node energy
%lin_inds = sub2ind(size(node_energy),double(state)+1,[1:length(state)]');
%ue = sum(node_energy(lin_inds));
ue = 0;
for n=1:length(state)
  ue = ue + node_energy(state(n),n);
end

% edge energy
%state_inds = sub2ind([num_states num_states],state(edge_list(1,:))+1,state(edge_list(2,:))+1);
%lin_inds = sub2ind(size(edge_energy),double(state_inds),[1:size(edge_list,2)]');
%pe = sum(edge_energy(lin_inds));
%ind_table = [1 3; 2 4];
ind_table = reshape([1:num_states^2],num_states,num_states);
pe = 0;
for e=1:size(edge_list,2)
  %n1 = edge_list(1,e);
  %n2 = edge_list(2,e);
  %ind = ind_table(state(n1),state(n2));
  if size(edge_list,1) == 2
    pe = pe + edge_energy(ind_table(state(edge_list(1,e)),state(edge_list(2,e))),e);
  elseif size(edge_list,1)==3
    pe = pe + edge_energy(ind_table(state(edge_list(1,e)),state(edge_list(2,e))),edge_list(3,e));
  end
end

total = ue+pe;

if nargin>4
  if nargin~=6
    error('need more inputs');
  end
  ind_table = reshape([1:num_states^3],num_states,num_states,num_states);
  te = 0;
  for e=1:size(tri_list,2)
    if size(tri_list,1) == 3
      te = te + tri_energy(ind_table(state(tri_list(1,e)),state(tri_list(2,e)),state(tri_list(3,e))),e);
    elseif size(tri_list,1)==4
      te = te + tri_energy(ind_table(state(tri_list(1,e)),state(tri_list(2,e)),state(tri_list(3,e))),tri_list(4,e));
    end
  end
  
  total = total + te;
end

