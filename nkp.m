function[varargout]=nkp(M, d, varargin)

% NKP: Computes the nearest Kronecker product of a matrix M.
%
% Mh = NKP(M,d) computes the nearest Kronecker product approximation
% Mh = M1 ⨂ M2 ⨂ ... ⨂ Md of a matrix M.
%
% Mh = NKP(M,d,q) computes the nearest Kronecker rank q approximation.
% Default: Kronecker rank 1 approximation.
%
% Mh = NKP(M,d,q,tol) attempts to compute an approximation of Kronecker rank
% smaller or equal to q satisfying a tolerance tol. If the tolerance cannot
% be met, the approximation has Kronecker rank q. Default: 1e-14.
%
% Mh = NKP(M,d,q,tol,name,value) specifies name/value pairs for optional
% parameters.
%   'blocksize' - specifies the blocksize for the block
%                 partitioning of the matrix M, defined as
%                 blocksize = {[m1 n1], [m2 n2], ..., [md nd]}.
%                 If this parameter is not specified, the algorithm
%                 automatically tries to find it.
%   'algo'      - specifies the algorithm for the low rank approximation.
%                 Possible choices are 'svd' (default) and 'aca' for d = 2 
%                 and 'cp_als' for d = 3.
%   'format'    - output format for the factor matrices. Available formats
%                 are 'tensor' and 'cell'. Default: 'tensor'.
%   'singv'     - plots the singular values of the reordered matrix for
%                 d = 2. Default: true.
%
% [M1,M2,...,Md] = NKP(M,d,...) returns instead the factor matrices of
% the approximation, stored along the pages (or elements) of Mk.
%
% [Mh,M1,M2,...,Md] = NKP(M,d,...) returns the approximation as well as
% the factor matrices, stored along the pages (or elements) of Mk.
%
% References:
% [1] C. F. Van Loan and N. Pitsianis. Approximation with Kronecker products.
% In Linear Algebra for Large Scale and Real-Time Applications. Springer, 1993.
% [2] G. H. Golub and C. F. Van Loan. Matrix computations. JHU press, 2013.

%% Set algorithm parameters

Param = inputParser;
Param.addRequired('matrix', @(x) size(x,1)==size(x,2));
Param.addRequired('dimension', @(x) isscalar(x) & x > 0);
Param.addOptional('rank', 1, @(x) isscalar(x) & x > 0);
Param.addOptional('tolerance', 1e-14, @(x) isscalar(x) & x > 0);
Param.addParameter('blocksize', 'auto', @iscell);
Param.addParameter('algo', 'svd', @(x) ismember(x,{'svd','aca','cp_als'}));
Param.addParameter('format', 'tensor', @(x) ismember(x,{'tensor','cell'}));
Param.addParameter('singv', true, @islogical);
Param.parse(M, d, varargin{:});

%% Retrieve parameters
q=Param.Results.rank;
tol=Param.Results.tolerance;
blocksize=Param.Results.blocksize;
algo=Param.Results.algo;
fout=Param.Results.format;
singv=Param.Results.singv;

%% Nearest Kronecker product approximation

if ~iscell(blocksize)
    n=blksize(M);

    if length(n)~=d
        error('Blocksize partitioning failed. Consider providing it explicitly.')
    else
        blocksize=mat2cell([n n], ones(d,1), 2);
    end
end

[R]=Rop(M, blocksize);
U=cell(1,d);

switch d
    case 2

        [I,J]=find(R);

        I=unique(I);
        J=unique(J);

        Rs=full(R(I,J));

        I={I,J};

        switch algo
            case 'svd' % Truncated SVD
                [U{1},S,U{2}]=svds(Rs, q);
                S=diag(S);

                ind=S>tol;
                S=S(ind);

                U{1}=sqrt(S)'.*U{1}(:,ind);
                U{2}=sqrt(S)'.*U{2}(:,ind);

                q=sum(ind);

            case 'aca' % Adaptive cross approximation
                [U{1},U{2},~,~,q]=aca(Rs, tol, q);
        end


        if singv
            figure
            sigma=svd(Rs);
            sR=size(Rs);
            l=min([sR 50]);
            semilogy(1:l, sigma(1:l), '.-b', 'MarkerSize', 15)
            grid on;
            title('Singular values of reordered matrix')
        end

    case 3

        R=tensor(R);

        S=find(R);

        I=unique(S(:,1));
        J=unique(S(:,2));
        K=unique(S(:,3));

        Rs=full(R(I,J,K));

        I={I,J,K};

        % CP decomposition
        [T]=cp_als(Rs, q, 'tol', tol);

        lambda=double(T.lambda);
        U{1}=lambda'.*T.U{1};
        U{2}=T.U{2};
        U{3}=T.U{3};
        q=length(lambda);

end

s=prod(cat(1,blocksize{:}),2);

% Factor matrices
F=cell(1,d);

switch fout
    case 'tensor'
        for k=1:d
            F{k}=zeros(s(k), q);
            F{k}(I{k},:)=U{k};
            F{k}=reshape(F{k}, [blocksize{k} q]);
        end

    case 'cell'
        for k=1:d
            F{k}=sparse(s(k), q);
            F{k}(I{k},:)=U{k};
            F{k}=mat2cell(F{k}, s(k), ones(1,q));
            F{k}=cellfun(@(x) reshape(x, blocksize{k}), F{k}, 'UniformOutput', false);
        end
end

if nargout==1 || nargout==d+1
    Mh=kron2mat(F{:});
end

switch nargout
    case 1
        varargout{1}=Mh;

    case d
        [varargout{1:d}]=F{:};

    case d+1
        varargout{1}=Mh;
        [varargout{2:d+1}]=F{:};

    otherwise
        error('Invalid number of output arguments')
end
end