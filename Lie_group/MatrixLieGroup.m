% This is a abstract class for matrix Lie groups

classdef MatrixLieGroup < LieGroup
   properties (SetAccess = public)
      n
   end
   methods
      function obj = MatrixLieGroup(n)
         obj = obj@LieGroup();
         obj.n=n;
      end
   end
   methods
      function e=group_identity(obj)
         e=eye(obj.n, obj.n);
      end
      function xi=lie_algebra_zero(obj)
         xi=zeros(obj.n);
      end
      function gl_gr=multiplication(obj, g_left, g_right)
         gl_gr= g_left*g_right;
      end
      function trivialized_vector=trivialize(obj, g, tangent_vector)
         trivialized_vector=inv(g)*tangent_vector;
      end
      function exp_xi=exp(obj, xi)
         exp_xi=expm(xi);
      end
      function norm_xi=norm(obj, xi)
         norm_xi=norm(xi,"fro");
      end
   end
   
 end