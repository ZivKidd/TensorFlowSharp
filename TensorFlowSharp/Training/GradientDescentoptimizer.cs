using System;
using TensorFlow;

namespace TensorFlowSharp.Training
{
    /// <summary>
    /// Optimizer that implements the gradient descent algorithm.
    /// </summary>
    public class GradientDescentOptimizer : Optimizer
    {
        readonly TFTensor m_learningRate;

        /// <summary>
        /// Construct a new gradient descent optimizer.
        /// </summary>
        /// <param name="learningRate">A Tensor or a floating point value.The learning
        /// rate to use.</param>
        /// <param name="useLocking">If True use locks for update operations.</param>
        /// <param name="name">Optional name prefix for the operations created when applying
        /// gradients.Defaults to "GradientDescent".</param>
        public GradientDescentOptimizer(TFTensor learningRate, bool useLocking = false, string name = "GradientDescent")
            : base(useLocking, name)
        {
            m_learningRate = learningRate;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="grad"></param>
        /// <param name="var"></param>
        /// <returns></returns>
        protected override TFOutput ApplyDense(TFOutput grad, TFOutput var)
        {
            throw new NotImplementedException();
            //return training_ops.apply_gradient_descent(
            //    var,
            //    math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
            //    grad,
            //    use_locking = self._use_locking).op;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="grad"></param>
        /// <param name="handle"></param>
        /// <returns></returns>
        protected override TFOutput ResourceApplyDense(TFOutput grad, IntPtr handle)
        {
            throw new NotImplementedException();
            //return training_ops.resource_apply_gradient_descent(
            //    handle.handle, math_ops.cast(self._learning_rate_tensor,
            //                                 grad.dtype.base_dtype),
            //    grad, use_locking = self._use_locking)
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="grad"></param>
        /// <param name="handle"></param>
        /// <param name="indices"></param>
        /// <returns></returns>
        protected override TFOutput ResourceApplySparseDuplicateIndices(TFOutput grad, IntPtr handle, int[] indices)
        {
            throw new NotImplementedException();
            //return resource_variable_ops.resource_scatter_add(
            //    handle.handle, indices, -grad * self._learning_rate);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="grad"></param>
        /// <param name="var"></param>
        /// <returns></returns>
        protected override TFOutput ApplySparseDuplicateIndices(TFOutput grad, TFOutput var)
        {
            throw new NotImplementedException();
            //delta = ops.IndexedSlices(
            //    grad.values *
            //    math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
            //    grad.indices, grad.dense_shape)
            //return var.scatter_sub(delta, use_locking = self._use_locking)
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        protected override TFOutput Prepare()
        {
            throw new NotImplementedException();
            //self._learning_rate_tensor = ops.convert_to_tensor(self._learning_rate,
            //                                                   name = "learning_rate")
        }
    }
}
