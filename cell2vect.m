function[V]=cell2vect(A,varargin)

% CELL2VECT: Transforms a cell array into a matrix containing the
% vectorization of its elements.
%
% V = CELL2VECT(A) returns a matrix V such that V = [vect(A1),...,vect(Ar)].
% All matrices Ak must have the same size for k = 1,...,r.
%
% V = CELL2VECT(A, 'transpose') returns instead the vectorization of the
% transpose such that V = [vect(A1'),...,vect(Ar')].

%% Set algorithm parameters

Param = inputParser;
Param.addRequired('A');
Param.addOptional('format', 'none', @(x) ismember(x,{'none','transpose'}));
Param.parse(A,varargin{:});

%% Retrieve parameters and check format
format=Param.Results.format;

r=length(A);
A=reshape(A, [1 r]);

if strcmp(format, 'transpose')
    A=cellfun(@(x) x', A, 'UniformOutput', false);
end

V=cellfun(@(x) x(:), A, 'UniformOutput', false);
V=[V{:}];
end