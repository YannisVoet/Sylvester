% Experiment 2: Isogeometric analysis
% The expriment requires GeoPDEs [1]. 
%
% [1] R. Vázquez. A new design for the implementation of isogeometric 
% analysis in Octave and Matlab: Geopdes 3.0. CAMWA, 2016.

clc
clear variables
close all

% EX_PLANE_STRAIN_SQUARE: solve the plane-strain problem on a square.

% 1) PHYSICAL DATA OF THE PROBLEM
clear problem_data
% Physical domain, defined as NURBS map given in a text file
problem_data.geo_name = 'geo_plate_with_hole.txt'; % Plate with a hole

% Type of boundary conditions for each side of the domain
problem_data.nmnn_sides   = [];
problem_data.drchlt_sides = [1 2 3 4];

% Physical parameters
problem_data.rho  = @(x, y) ones(size(x));
% problem_data.rho  = @(x, y) abs(sin(x.*y)).*(x>0.5)+(y.^3+cosh(x)).*(x<=0.5);

problem_data.c_diff  = @(x, y) ones(size(x));

% Source and boundary terms
problem_data.f = @(x, y) zeros (size (x));
problem_data.g = @test_square_g_nmnn;
problem_data.h = @(x, y, ind) exp (x) .* sin(y);

% Exact solution (optional)
problem_data.uex     = @(x, y) exp (x) .* sin (y);
problem_data.graduex = @(x, y) cat (1, ...
    reshape (exp(x).*sin(y), [1, size(x)]), ...
    reshape (exp(x).*cos(y), [1, size(x)]));

% Parameters
% Spline order
order=3;
% Number of subdivisions
Ns=200;

deg=[order order];
reg=[order-1 order-1];
ns=[Ns Ns];
nq=[order+1 order+1];

% 2) CHOICE OF THE DISCRETIZATION PARAMETERS
clear method_data
method_data.degree     = deg;    % Degree of the bsplines
method_data.regularity = reg;     % Regularity of the splines
method_data.nsub       = ns;     % Number of subdivisions
method_data.nquad      = nq;     % Points for the Gaussian quadrature rule

% Extract the fields from the data structures into local variables
data_names = fieldnames (problem_data);
for iopt  = 1:numel (data_names)
    eval ([data_names{iopt} '= problem_data.(data_names{iopt});']);
end
data_names = fieldnames (method_data);
for iopt  = 1:numel (data_names)
    eval ([data_names{iopt} '= method_data.(data_names{iopt});']);
end

% Construct geometry structure
geometry  = geo_load (geo_name);

[knots, zeta] = kntrefine (geometry.nurbs.knots, nsub-1, degree, regularity);

% Check for periodic conditions, and consistency with other boundary conditions
if (exist('periodic_directions', 'var'))
    knots = kntunclamp (knots, degree, regularity, periodic_directions);
    per_sides = union (periodic_directions*2 - 1, periodic_directions*2);
    if (~isempty (intersect(per_sides, [nmnn_sides, drchlt_sides])))
        error ('Neumann or Dirichlet conditions cannot be imposed on periodic sides')
    end
else
    periodic_directions = [];
end

% Construct msh structure
rule     = msh_gauss_nodes (nquad);
[qn, qw] = msh_set_quad_nodes (zeta, rule);
msh      = msh_cartesian (zeta, qn, qw, geometry);

% Construct space structure
space    = sp_bspline (knots, degree, msh, [], periodic_directions);

% Assemble the matrices
K = op_gradu_gradv_tp (space, space, msh, c_diff);
M = op_u_v_tp(space, space, msh, rho);
P = op_u_v_tp_tilde (space, space, msh);

% Apply Dirichlet boundary conditions
u = zeros (space.ndof, 1);
[u_drchlt, drchlt_dofs] = sp_drchlt_l2_proj (space, msh, h, drchlt_sides);
u(drchlt_dofs) = u_drchlt;

int_dofs = setdiff (1:space.ndof, drchlt_dofs);

% Reduced matrices
Kr=K(int_dofs,int_dofs);
Mr=M(int_dofs,int_dofs);
Pr=P(int_dofs,int_dofs);

%% Preconditioner of Loli for the mass matrix
Dm=diag(diag(Mr));
Dh=diag(diag(Pr));

Ds=sqrt(Dm);
Dsh=sqrt(Dh);

s=blksize(Mr);
n=s(1); m=s(2);
[M1, M2]=nkp(Pr,2,1,'singv',false);
Dm=Ds/Dsh;

Pr=Dm*Pr*Dm;
Dm=reshape(diag(Dm), m, n);

tic
dM1=decomposition(sparse(M1));
dM2=decomposition(sparse(M2));
time_loli.setup=toc;

prec_loli= @(X) 1./Dm.*(dM2\(1./Dm.*X)/dM1);

%% Nearest Kronecker product approximation

[A, B]=nkp(Mr, 2, 100, 1e-14, 'algo', 'aca', 'format', 'cell');

RHS=speye(m, n);
X0=sparse(m, n);

%% Define linear operator

linearOp = @(X) linOp(A,B,X,n,m);

%% Nearest Kronecker product approximation
% q-values for NKP preconditioners
q_nkp=[1 2];
b_nkp=length(q_nkp);

prec_nkp=cell(1,b_nkp);
setup_fact=cell(1,b_nkp);

tic
[Y,Z] = kronsvd(A,B,min([2 max(q_nkp)]));
setup_nkp=toc;

for k=1:b_nkp
    switch q_nkp(k)
        case 1
            tic
            dY=decomposition(Y{1}');
            dZ=decomposition(Z{1});
            setup_fact{k}=toc;

            prec_nkp{k} = @(X) dZ\(X/dY);
        otherwise
            tic
            [P,S,Q1,Z1]=qz(Z{1},Z{2},'real');
            [T,R,Q2,Z2]=qz(Y{2},Y{1},'real');
            setup_fact{k}=toc;

            prec_nkp{k} = @(X) Z1*rtrgsyl(P,R,S,T,Q1*X*Q2')*Z2';
    end
end

time_nkp=struct('setup_nkp', setup_nkp, 'setup_fact', setup_fact);

%% Low Kronecker rank approximation of the inverse
% q-values for KINV preconditioners
q_kinv=[1 2];
b_kinv=length(q_kinv);

prec_kinv=cell(1,b_kinv);
res_kinv=cell(1,b_kinv);
setup_kinv=cell(1,b_kinv);

for k=1:b_kinv
    tic
    [Cin]=sparinv(A,2+(1:q_kinv(k)));
    [Din]=sparinv(B,2+(1:q_kinv(k)));
    [C, D, res_kinv{k}] = kroninvq(A, B, q_kinv(k), 'C0', Cin, 'D0', Din, 'nitermax', 10, 'sparse', true);
    setup_kinv{k}=toc;
    prec_kinv{k} = @(X) linOp(C,D,X,n,m);
end

time_kinv=struct('setup_kinv', setup_kinv);

%% GMRES Solver
norm_RHS=norm(RHS, 'fro');
maxiter=100;
tol=1e-8*norm_RHS; % Relative tolerance

res_prec_nkp=cell(1,b_nkp);
res_prec_kinv=cell(1,b_kinv);

iter_prec_nkp=cell(1,b_nkp);
iter_prec_kinv=cell(1,b_kinv);

tic
[X_unprec, res_unprec, iter_unprec] = glgmresk(linearOp, RHS, [], tol, maxiter, @(X) X, X0);
time_unprec.gmres=toc;
tic
[X_prec_loli, res_prec_loli, iter_loli] = glgmresk(linearOp, RHS, [], tol, maxiter, prec_loli, X0);
time_loli.prec_gmres=toc;

for k=1:b_nkp
    tic
    [X_prec_nkp, res_prec_nkp{k}, iter_prec_nkp{k}] = glgmresk(linearOp, RHS, [], tol, maxiter, prec_nkp{k}, X0);
    time_nkp(k).prec_gmres=toc;
end

for k=1:b_kinv
    tic
    [X_prec_kinv, res_prec_kinv{k}, iter_prec_kinv{k}] = glgmresk(linearOp, RHS, [], tol, maxiter, prec_kinv{k}, X0);
    time_kinv(k).prec_gmres=toc;
end

%% Results

colors_nkp={[0 0.8 1], [0.9 0.5 0.1]};
colors_kinv={[0 0.2 0.8],[0.9 0.1 0.1],[0.4 0.7 0.1],[0.8 0.8 0.2],[0.8 0.4 0.8]};

linespec_nkp={'v', '^'};
linespec_kinv={'s', '*', 'o', 'square', 'diamond'};
iter=min(50,maxiter);

figure
semilogy(res_unprec(1:iter)/norm_RHS, '-xk','DisplayName', 'GMRES')
hold on; grid on;
semilogy(res_prec_loli/norm_RHS, 'Marker', 'pentagram', 'Color', [0.4660 0.6740 0.1880], 'DisplayName', 'Loli et al. 2022')

for k=1:b_nkp
    semilogy(res_prec_nkp{k}/norm_RHS, 'Marker', linespec_nkp{q_nkp(k)}, 'Color', colors_nkp{q_nkp(k)}, 'DisplayName', ['NKP(' num2str(min([q_nkp(k) 2])) ')'])
end

for k=1:b_kinv
    semilogy(res_prec_kinv{k}/norm_RHS, 'Marker', linespec_kinv{q_kinv(k)}, 'Color', colors_kinv{q_kinv(k)}, 'DisplayName', ['KINV(' num2str(q_kinv(k)) ')'])
end


legend show
legend('Location','southeast')
xlabel('Iteration number')
ylabel('Relative residual')
xlim([1 50])

%% Bi-CGSTAB Solver
norm_RHS=norm(RHS, 'fro');
maxiter=100;
tol=1e-8*norm_RHS; % Relative tolerance

res_prec_nkp=cell(1,b_nkp);
res_prec_kinv=cell(1,b_kinv);

iter_prec_nkp=cell(1,b_nkp);
iter_prec_kinv=cell(1,b_kinv);

tic
[X_unprec, res_unprec, iter_unprec] = glbicgstb(linearOp, RHS, tol, maxiter, @(X) X, X0);
time_unprec.bicgstab=toc;
tic
[X_prec_loli, res_prec_loli, iter_loli] = glbicgstb(linearOp, RHS, tol, maxiter, prec_loli, X0);
time_loli.prec_bicgstab=toc;

for k=1:b_nkp
    tic
    [X_prec_nkp, res_prec_nkp{k}, iter_prec_nkp{k}] = glbicgstb(linearOp, RHS, tol, maxiter, prec_nkp{k}, X0);
    time_nkp(k).prec_bicgstab=toc;
end

for k=1:b_kinv
    tic
    [X_prec_kinv, res_prec_kinv{k}, iter_prec_kinv{k}] = glbicgstb(linearOp, RHS, tol, maxiter, prec_kinv{k}, X0);
    time_kinv(k).prec_bicgstab=toc;
end

%% Results

colors_nkp={[0 0.8 1], [0.9 0.5 0.1]};
colors_kinv={[0 0.2 0.8],[0.9 0.1 0.1],[0.4 0.7 0.1],[0.8 0.8 0.2],[0.8 0.4 0.8]};

linespec_nkp={'v', '^'};
linespec_kinv={'s', '*', 'o', 'square', 'diamond'};

figure
semilogy(res_unprec(1:iter)/norm_RHS, '-xk','DisplayName', 'Bi-CGSTAB')
hold on; grid on;
semilogy(res_prec_loli/norm_RHS, 'Marker', 'pentagram', 'Color', [0.4660 0.6740 0.1880], 'DisplayName', 'Loli et al. 2022')

for k=1:b_nkp
    semilogy(res_prec_nkp{k}/norm_RHS, 'Marker', linespec_nkp{q_nkp(k)}, 'Color', colors_nkp{q_nkp(k)}, 'DisplayName', ['NKP(' num2str(min([q_nkp(k) 2])) ')'])
end

for k=1:b_kinv
    semilogy(res_prec_kinv{k}/norm_RHS, 'Marker', linespec_kinv{q_kinv(k)}, 'Color', colors_kinv{q_kinv(k)}, 'DisplayName', ['KINV(' num2str(q_kinv(k)) ')'])
end


legend show
legend('Location','southeast')
xlabel('Iteration number')
ylabel('Relative residual')
xlim([1 50])

%% Sparsity verification

bE=bandwidth(RHS);
bM=opwidth(A,B);
bP=opwidth(C,D);

(2*iter_prec_kinv{k}-1)*(bM+bP)+bP+bE
bandwidth(X_prec_kinv)