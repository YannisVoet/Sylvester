function[M]=kron2mat(varargin)

% KRON2MAT: Explicitly computes the Kronecker product matrix from the
% factor matrices.
%
% M = KRON2MAT(M1,M2,...,Md) computes the Kronecker product matrix from the
% factor matrices stored in Mk; i.e. M = ∑ M1j ⨂ M2j ⨂ ... ⨂ Mdj.
% The arrays Mk may be either third order tensors or cell arrays but must 
% contain the same number of factor matrices. 

d=length(varargin);
V=cell(1,d);
blocksize=cell(1,d);

for k=1:d
    if isa(varargin{k}, 'double')
        [m,n,r]=size(varargin{k});
        V{k}=reshape(varargin{k}, [m*n r]);
        blocksize{k}=[m n];
    else
        V{k}=cell2vect(varargin{k});
        blocksize{k}=size(varargin{k}{1});
    end
end

switch d % To do: convert ktensor to sptensor and code up Rinvop to support sptensors
    case 2
        M=Rinvop(V{1}*V{2}', blocksize);
    case 3
        V=cellfun(@(x) full(x), V, 'UniformOutput', false);
        R=ktensor(V);
        R=double(tensor(R)); 
        M=Rinvop(R, blocksize);
end
end