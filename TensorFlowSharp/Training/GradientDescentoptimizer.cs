using System;
using TensorFlow;

namespace TensorFlowSharp.Training
{
    public class GradientDescentOptimizer : Optimizer
    {
        double m_learningRate;

        public GradientDescentOptimizer(double learningRate, bool useLocking = false, string name = "GradientDescent")
            : base(useLocking, name)
        {
            m_learningRate = learningRate;
        }

        protected override TFOutput ApplyDense(TFOutput grad, TFOutput var)
        {
            throw new NotImplementedException();
            //return training_ops.apply_gradient_descent(
            //    var,
            //    math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
            //    grad,
            //    use_locking = self._use_locking).op;
        }

        protected override TFOutput ResourceApplyDense(TFOutput grad, IntPtr handle)
        {
            throw new NotImplementedException();
            //return training_ops.resource_apply_gradient_descent(
            //    handle.handle, math_ops.cast(self._learning_rate_tensor,
            //                                 grad.dtype.base_dtype),
            //    grad, use_locking = self._use_locking)
        }

        protected override TFOutput ResourceApplySparseDuplicateIndices(TFOutput grad, IntPtr handle, int[] indices)
        {
            throw new NotImplementedException();
            //return resource_variable_ops.resource_scatter_add(
            //    handle.handle, indices, -grad * self._learning_rate);
        }

        protected override TFOutput ApplySparseDuplicateIndices(TFOutput grad, TFOutput var)
        {
            throw new NotImplementedException();
            //delta = ops.IndexedSlices(
            //    grad.values *
            //    math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
            //    grad.indices, grad.dense_shape)
            //return var.scatter_sub(delta, use_locking = self._use_locking)
        }

        protected override TFOutput Prepare()
        {
            throw new NotImplementedException();
            //self._learning_rate_tensor = ops.convert_to_tensor(self._learning_rate,
            //                                                   name = "learning_rate")
        }
    }
}
