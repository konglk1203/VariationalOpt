% Example for how to implement a Lie group
% SO(n): special orthogonal group, the set of all n-by-n matrices with X^T X=I and det(X)=1

classdef SOn < MatrixLieGroup
   
   methods
      function obj = SOn(n)
         obj = obj@MatrixLieGroup(n);
      end
      function manifold_grad=project_grad(obj, g, grad)
         manifold_grad=grad-g*grad'*g;
      end
      function noise=get_noise(obj)
         noise=randn(obj.n, obj.n);
         noise=triu(noise,1);
         noise=(noise-noise');
      end
   end
 end