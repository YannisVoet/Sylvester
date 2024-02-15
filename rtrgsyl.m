function[X]=rtrgsyl(A,B,C,D,E,varargin)

% RTRGSYL: Recursive block triangular solver for the quasi-triangular
% continuous time generalized Sylvester equation AXB' + CXD' = E, where
% A, D are upper (quasi-)triangular and B, C are upper triangular.
% A,C are m x m, B,D are n x n, E,X are m x n.
%
% X = RTRGSYL(A,B,C,D,E) solves the generalized Sylvester equation
% AXB' + CXD' = E in upper (quasi-)triangular form.
%
% X = RTRGSYL(A,B,C,D,E,blks) specifies the block size for solving small
% sized Sylvester equations. Default: 10.
%
% References:
% [1] I. Jonsson and B. Kagstrom. Recursive blocked algorithms for solving
% triangular systems — Part II: Two-sided and generalized Sylvester and
% Lyapunov matrix equations. ACM Transactions on Mathematical Software
% (TOMS), 2002.
% [2] I. Jonsson and B. Kagstrom. Recursive blocked algorithms for solving
% triangular systems — Part I: One-sided and coupled Sylvester-type matrix
% equations. ACM Transactions on Mathematical Software (TOMS), 2002.

%% Set algorithm parameters

m=size(A,1);
n=size(B,1);

% Cheap version
if nargin>5
    blks=varargin{1};
else
    blks=10;
end

% Cleaner but more expensive version
% Param = inputParser;
% Param.addRequired('A', @(x) isbanded(x,1,m) && all(size(A)==[m m]));
% Param.addRequired('B', @(x) isbanded(x,0,n) && all(size(B)==[n n]));
% Param.addRequired('C', @(x) isbanded(x,0,m) && all(size(C)==[m m]));
% Param.addRequired('D', @(x) isbanded(x,1,n) && all(size(D)==[n n]));
% Param.addRequired('E', @(x) all(size(E)==[m n]));
% Param.addOptional('blks', 10, @(x) isscalar(x) & x > 0);
% Param.parse(A,B,C,D,E,varargin{:});

%% Recursive blocked algorithm
% blks=Param.Results.blks;

if min([n, m])>=1 && min([n, m])<=blks
    X=trgsyl(A, B, C, D, E);
else
    if n>=1 && n<=m/2 % Case 1: Split (A,C) by rows and columns and E by rows only.
        mt=floor(m/2);
        if A(mt+1,mt)~=0 % We are in a 2x2 block: split below this block.
            mt=mt+1;
        end
        X2=rtrgsyl(A(mt+1:end, mt+1:end), B, C(mt+1:end,mt+1:end), D, E(mt+1:end,:), blks);
        E1=axb(-A(1:mt, mt+1:end), B', E(1:mt, :), X2);
        E1=axb(-C(1:mt, mt+1:end), D', E1, X2);
        X1=rtrgsyl(A(1:mt, 1:mt), B, C(1:mt, 1:mt), D, E1, blks);
        X=[X1; X2];

    elseif m>=1 && m<= n/2 % Case 2: Split (B,D) by rows and columns and E by columns only.
        nt=floor(n/2);
        if D(nt+1, nt)~=0 % We are in a 2x2 block: split below this block.
            nt=nt+1;
        end
        X2=rtrgsyl(A, B(nt+1:end, nt+1:end), C, D(nt+1:end, nt+1:end), E(:,nt+1:end), blks);
        E1=axb(-A, B(1:nt, nt+1:end)', E(:,1:nt), X2);
        E1=axb(-C, D(1:nt, nt+1:end)', E1, X2);
        X1=rtrgsyl(A, B(1:nt, 1:nt), C, D(1:nt, 1:nt), E1, blks);
        X=[X1, X2];

    else % Case 3: Split (A,C), (B,D) and E by rows and columns.
        mt=floor(m/2);
        nt=floor(n/2);
        if A(mt+1,mt)~=0 % We are in a 2x2 block: split below this block.
            mt=mt+1;
        end
        if D(nt+1, nt)~=0 % We are in a 2x2 block: split below this block.
            nt=nt+1;
        end
        X22=rtrgsyl(A(mt+1:end, mt+1:end), B(nt+1:end, nt+1:end), C(mt+1:end, mt+1:end), D(nt+1:end, nt+1:end), E(mt+1:end, nt+1:end), blks);
        E12=axb(-A(1:mt,mt+1:end), B(nt+1:end, nt+1:end)', E(1:mt,nt+1:end), X22);
        E12=axb(-C(1:mt,mt+1:end), D(nt+1:end, nt+1:end)', E12, X22);
        E21=axb(-A(mt+1:end,mt+1:end), B(1:nt, nt+1:end)', E(mt+1:end,1:nt), X22);
        E21=axb(-C(mt+1:end,mt+1:end), D(1:nt, nt+1:end)', E21, X22);
        X12=rtrgsyl(A(1:mt,1:mt), B(nt+1:end, nt+1:end), C(1:mt,1:mt), D(nt+1:end, nt+1:end), E12, blks);
        X21=rtrgsyl(A(mt+1:end, mt+1:end), B(1:nt, 1:nt), C(mt+1:end, mt+1:end), D(1:nt, 1:nt), E21, blks);
        E11=axb(-A(1:mt,mt+1:end), B(1:nt, 1:nt)', E(1:mt,1:nt), X21);
        E11=axb(-C(1:mt,mt+1:end), D(1:nt, 1:nt)', E11, X21);
        E11=gemm(-(A(1:mt,1:mt)*X12+A(1:mt,mt+1:end)*X22), B(1:nt, nt+1:end)', E11);
        E11=gemm(-(C(1:mt,1:mt)*X12+C(1:mt,mt+1:end)*X22), D(1:nt, nt+1:end)', E11);
        X11=rtrgsyl(A(1:mt, 1:mt), B(1:nt, 1:nt), C(1:mt, 1:mt), D(1:nt, 1:nt), E11, blks);
        X=[X11, X12; X21, X22];
    end

end
end