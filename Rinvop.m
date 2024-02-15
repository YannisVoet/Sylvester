function[M]=Rinvop(R, blocksize)

% RINVOP: Computes the inverse rearrangement of the tensor R.
%
% M = RINVOP(R, blocksize) rearranges the tensor R of size
% m1n1 x m2n2 x ... x mdnd into a matrix M of size m1m2...md x n1n2...nd.
% The blocksize is specified as a cell array
% blocksize = {[m1 n1], [m2 n2], ..., [md nd]}. For the definition of the
% rearrangement operator, see [1] for d = 2 and [2] for d = 3.
%
% References:
% [1] C. F. Van Loan and N. Pitsianis. Approximation with Kronecker products. 
% In Linear Algebra for Large Scale and Real-Time Applications. Springer, 1993.
% [2] A. N. Langville and W. J. Stewart. A Kronecker product approximate 
% preconditioner for SANs. Numerical Linear Algebra with Applications, 2004.
%
% See also ROP.


switch length(blocksize)

    case 2
        [b1,b2]=blocksize{:};

        m1=b1(1); m2=b2(1);
        n1=b1(2); n2=b2(2);

        M=reshape(R', [m2 n2*m1*n1]);
        M=mat2cell(M, m2, n2*ones(1, m1*n1));
        M=reshape(M, [m1 n1]);
        M=cell2mat(M);

    case 3
        [b1,b2,b3]=blocksize{:};

        s1=prod(b1);

        M=cell(1,s1);

        for k=1:s1
            M{k}=Rinvop(squeeze(R(k,:,:)), {b2, b3});
        end

        M=reshape(M, b1);
        M=cell2mat(M);

    otherwise
        error('This block partitioning is not supported')

end
end