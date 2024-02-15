function[S]=sparinv(A,varargin)

% SPARINV: Computes sparity patterns based on powers. This function is
% typically used for computing initial guesses for a Kronecker rank q
% approximate inverse.
%
% S = SPARINV(A) computes a boolean matrix S retaining the sparsity
% pattern of ∑ Ak. The factor matrices Ak may be stored along the
% pages of a third order tensor A or as elements of a cell array. This
% function call is useful for obtaining an initial guess for computing a
% Kronecker rank 1 approximate inverse.
%
% S = SPARINV(A,p) computes matrices Ss for s = 1,...,q retaining the
% sparsity pattern of (∑ Ak)^ps for a vector p = [p1,...,pq]. The sparsity 
% patterns are either stored along the pages of a third order tensor S or 
% as elements of a cell array or in vectorized format. By default, the
% sparsity patterns are stored in the same format as the input.
%
% S = SPARINV(A,p,format) also specifies the output format. Available
% formats are 'tensor', 'cell' or 'vect'. Default: same format as input.
%
% S = SPARINV(A,p,format,func) computes powers of the matrix func(∑ Ak),
% where the function func is provided as an anonymous function.
% Default: func = id.
%
% Reference:
% [1] T. Huckle. Approximate sparsity patterns for the inverse of a matrix
% and preconditioning. Applied numerical mathematics, 1999.
%
% See also KRONINV, KRONINVQ.

%% Set algorithm parameters

% Set default parameters
Default{1}=1;
Default{2}='auto';
Default{3}=@(x) x;

% Replace empty inputs with default parameters
def=cell2mat(cellfun(@isempty, varargin, 'UniformOutput', false));
[varargin{def}]=Default{def};

Param = inputParser;
Param.addRequired('A');
Param.addOptional('power',  Default{1}, @(x) isa(x, 'double'));
Param.addOptional('format', Default{2}, @(x) ismember(x,{'auto','tensor','cell','vect'}));
Param.addOptional('func',   Default{3}, @(x) isa(x, 'function_handle'));
Param.parse(A,varargin{:});

%% Retrieve parameters and check format
p=Param.Results.power;
fout=Param.Results.format;
func=Param.Results.func;

% Check input format
[A,sA,fA]=convert(A, {'vect'});

if sA(1)~=sA(2)
    error('Factor matrices must be square')
else
    n=sA(1);
end

if isequal(fout, 'auto')
    fout=fA;
end

% Initialization
S0=func(reshape(sum(A,2), [n n]));
q=length(p);

S=cell(1,q);

for k=1:q
    [I,J]=find(S0^p(k));
    S{k}=sparse(I,J,ones(length(I),1),n,n);
end

S=convert(S, fout);

end