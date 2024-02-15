function[S]=solvesyst(S,M,RHS,Options)

% SOLVESYST: solves the linear systems in the ALS iterations for computing 
% low Kronecker rank approximate inverses.
% 
% S = SOLVESYST(S,M,RHS) computes the solution of M*S = RHS (or a sparse
% approximation thereof).
%
% S = SOLVESYST(S,M,RHS,Options) specifies the solver options as a struct 
% array. For sparse approximate solutions, the systems may be solved in 
% parallel (if available). Options may contain the following fields:
%   'sp'    - computes a sparse approximate inverse (true/false).
%   'par'   - solves linear systems in parallel (for computing sparse 
%             approximate inverses and if the parallel computing toolbox is 
%             available) (true/false).


if Options.sp

    [s1,s2]=size(S);

    if Options.par

        [I,J]=find(S);
        V=cell(s2,1);
        Ms=cell(s2,1);
        RHSs=cell(s2,1);

        % Preliminary serial slicing
        for j=1:s2
            Jl=find(S(:,j));
            Ms{j}=M(Jl,Jl);
            RHSs{j}=RHS(Jl,j);
        end

        % Heavy parallel computations
        parfor j=1:s2
            V{j}=Ms{j}\RHSs{j};
        end

        V=cat(1,V{:});
        S=sparse(I,J,V,s1,s2);
    else
        for j=1:s2
            J=find(S(:,j));
            S(J,j)=M(J,J)\RHS(J,j);
        end
    end
else
    S=M\RHS;
end

end