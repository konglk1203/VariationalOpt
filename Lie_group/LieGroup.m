% This is a base class for Lie groups. To use the Lie group optimizer and sampler, please build a subclass and implement all the abstract methods.
% For more details, e.g., requirements for the Lie groups, see Sec.2 in https://proceedings.mlr.press/v247/kong24a/kong24a.pdf
classdef (Abstract)LieGroup
    methods
        % constructor method
        function obj=LieGroup()
        end
    end
    methods(Abstract)
        % return the group identity
        e=group_identity(obj)

        % return the zero element of a Lie algebra
        xi=lie_algebra_zero(obj)

        % group multiplication
        gl_gr=multiplication(obj, g_left, g_right)

        % left trivialization. move [tangent_vector] at [g] to a vector in the Lie algebra. 
        trivialized_vector=trivialize(obj, g, tangent_vector)

        % group exponential operator. input a vector in the Lie algebra [xi] and output a element [exp_xi] in the Lie group
        exp_xi=exp(obj, xi)

        % calculate the norm of a vector [xi] in the Lie algebra
        norm_xi=norm(obj, xi)

        % given a Euclidean vector [grad] at [g], project it to the tangent space of [g]
        manifold_grad=project_grad(obj, g, grad)

        % return a random vector in the Lie Lie algebra following the normal distribution
        noise=get_noise(obj)
     end
 end