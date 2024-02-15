function[R]=Rop(M, blocksize)

% ROP: Computes the rearrangement of the matrix M.
%
% R = ROP(M, blocksize) rearranges the matrix M of size m1m2...md x
% n1n2...nd into a tensor R of size m1n1 x m2n2 x ... x mdnd. For d = 2, 
% see [1], for d = 3, see [2] for their definition. The blocksize is 
% specified as a cell array blocksize = {[m1 n1], [m2 n2], ..., [md nd]}.
%
% References:
% [1] C. F. Van Loan and N. Pitsianis. Approximation with Kronecker products. 
% In Linear Algebra for Large Scale and Real-Time Applications. Springer, 1993.
% [2] A. N. Langville and W. J. Stewart. A Kronecker product approximate 
% preconditioner for SANs. Numerical Linear Algebra with Applications, 2004.
%
% See also RINVOP.

switch length(blocksize)

    case 2
        [b1,b2]=blocksize{:};

        m1=b1(1); m2=b2(1);
        n1=b1(2); n2=b2(2);

        R=mat2cell(M, m2*ones(1, m1), n2*ones(1, n1));
        R=[R{:}];
        R=reshape(R, [m2*n2 m1*n1])';

    case 3
        [b1,b2,b3]=blocksize{:};

        m1=b1(1); m2=b2(1); m3=b3(1);
        n1=b1(2); n2=b2(2); n3=b3(2);

        P=mat2cell(M, (m2*m3)*ones(1, m1), (n2*n3)*ones(1, n1));

        R=zeros(m1*n1, m2*n2, m3*n3);

        for k=1:m1*n1
            R(k,:,:)=Rop(P{k}, {[m2 n2], [m3 n3]});
        end

    otherwise
        error('This block partitioning is not supported')

end
end