using System;
using System.Collections.Generic;
using TensorFlow;

namespace TensorFlowSharp.Training
{
    /// <summary>
    /// Base class for optimizers.
    /// 
    /// This class defines the API to add Ops to train a model.You never use this
    /// class directly, but instead instantiate one of its subclasses such as
    /// `GradientDescentOptimizer`, `AdagradOptimizer`, or `MomentumOptimizer`.
    /// 
    /// ### Usage
    /// 
    /// ```python
    /// # Create an optimizer with the desired parameters.
    /// opt = GradientDescentOptimizer(learning_rate = 0.1)
    /// # Add Ops to the graph to minimize a cost by updating a list of variables.
    /// # "cost" is a Tensor, and the list of variables contains tf.Variable
    /// # objects.
    /// opt_op = opt.minimize(cost, var_list=<list of variables>)
    /// ```
    /// 
    /// In the training program you will just have to run the returned Op.
    /// 
    /// ```python
    /// # Execute opt_op to do one step of training:
    /// opt_op.run()
    /// ```
    /// 
    /// ### Processing gradients before applying them.
    /// 
    /// Calling `minimize()` takes care of both computing the gradients and
    /// applying them to the variables.If you want to process the gradients
    /// before applying them you can instead use the optimizer in three steps:
    /// 
    /// 1.  Compute the gradients with `compute_gradients()`.
    /// 2.  Process the gradients as you wish.
    /// 3.  Apply the processed gradients with `apply_gradients()`.
    /// 
    /// Example:
    /// 
    /// ```python
    /// # Create an optimizer.
    /// opt = GradientDescentOptimizer(learning_rate = 0.1)
    /// 
    /// # Compute the gradients for a list of variables.
    /// grads_and_vars = opt.compute_gradients(loss, < list of variables >)
    /// 
    /// # grads_and_vars is a list of tuples (gradient, variable).  Do whatever you
    /// # need to the 'gradient' part, for example cap them, etc.
    /// capped_grads_and_vars = [(MyCapper(gv[0]), gv[1]) for gv in grads_and_vars]
    /// 
    /// # Ask the optimizer to apply the capped gradients.
    /// opt.apply_gradients(capped_grads_and_vars)
    /// ```
    /// 
    /// ### Gating Gradients
    /// 
    /// Both `minimize()` and `compute_gradients()` accept a `gate_gradients`
    /// argument that controls the degree of parallelism during the application of
    /// the gradients.
    /// 
    /// The possible values are: `GATE_NONE`, `GATE_OP`, and `GATE_GRAPH`.
    /// 
    /// <b>`GATE_NONE`</b>: Compute and apply gradients in parallel.This provides
    /// the maximum parallelism in execution, at the cost of some non-reproducibility
    /// in the results.  For example the two gradients of `matmul` depend on the input
    /// values: With `GATE_NONE` one of the gradients could be applied to one of the
    /// inputs _before_ the other gradient is computed resulting in non-reproducible
    /// results.
    /// 
    /// <b>`GATE_OP`</b>: For each Op, make sure all gradients are computed before
    /// they are used.  This prevents race conditions for Ops that generate gradients
    /// for multiple inputs where the gradients depend on the inputs.
    /// 
    /// <b>`GATE_GRAPH`</b>: Make sure all gradients for all variables are computed
    /// before any one of them is used.This provides the least parallelism but can
    /// be useful if you want to process all gradients before applying any of them.
    /// 
    /// ### Slots
    /// 
    /// Some optimizer subclasses, such as `MomentumOptimizer` and `AdagradOptimizer`
    /// allocate and manage additional variables associated with the variables to
    /// train.These are called<i> Slots</i>.  Slots have names and you can ask the
    /// optimizer for the names of the slots that it uses.  Once you have a slot name
    /// you can ask the optimizer for the variable it created to hold the slot value.
    /// 
    /// This can be useful if you want to log debug a training algorithm, report stats
    /// about the slots, etc.
    /// </summary>
    public abstract class Optimizer
    {
        bool m_useLocking;
        string m_name;
        HashSet<TFDataType> m_validDTypes = new HashSet<TFDataType> { TFDataType.BFloat16, TFDataType.Float, TFDataType.Double };

        public Optimizer(bool useLocking, string name)
        {
            if (name == string.Empty) { throw new ArgumentException("Must specify the optimizer name"); }

            m_useLocking = useLocking;
            m_name = name;
        }

        public string GetName()
        {
            return m_name;
        }

        /// <summary>
        /// Add operations to minimize `loss` by updating `var_list`.
        /// This method simply combines calls `compute_gradients()` and
        /// `apply_gradients()`. If you want to process the gradient before applying
        /// them call `compute_gradients()` and `apply_gradients()` explicitly instead
        /// of using this function.
        /// </summary>
        /// <param name="loss">A `Tensor` containing the value to minimize.</param>
        /// <param name="global_step">Optional `Variable` to increment by one after the
        /// variables have been updated.</param>
        /// <param name="var_list">Optional list or tuple of `Variable` objects to update to
        /// minimize `loss`.  Defaults to the list of variables collected in
        /// the graph under the key `GraphKeys.TRAINABLE_VARIABLES`.</param>
        /// <param name="gate_gradients">How to gate the computation of gradients.Can be
        /// `GATE_NONE`, `GATE_OP`, or  `GATE_GRAPH`.</param>
        /// <param name="aggregation_method"> Specifies the method used to combine gradient terms.
        /// Valid values are defined in the class `AggregationMethod`.</param>
        /// <param name="colocate_gradients_with_ops">If True, try colocating gradients with
        /// the corresponding op.</param>
        /// <param name="name">Optional name for the returned operation.</param>
        /// <param name="grad_loss">Optional.A `Tensor` holding the gradient computed for `loss`.</param>
        public TFOutput Minimize(TFOutput loss, 
            TFOutput global_step = default(TFOutput), List<TFOutput> var_list = null,
            GateGradients gate_gradients = GateGradients.GATE_OP,
            AggregationMethod aggregation_method= AggregationMethod.ADD_N, // in TensorFlow this is set to None, which corresponds to 0 and thereby AggregationMethod.Add_N
            bool colocate_gradients_with_ops = false, 
            string name = null,
            TFOutput grad_loss = default(TFOutput)) // TODO: Should grad_loss and global_step be TFTensor type?
        {

            /*grads_and_vars = */ComputeGradients(loss, var_list, 
                gate_gradients,
                aggregation_method,
                colocate_gradients_with_ops,
                grad_loss);

            //vars_with_grad = [v for g, v in grads_and_vars if g is not None]
            //if not vars_with_grad:
            //  raise ValueError(
            //      "No gradients provided for any variable, check your graph for ops"
            //      " that do not support gradients, between variables %s and loss %s." %
            //      ([str(v) for _, v in grads_and_vars], loss))

            //return self.apply_gradients(grads_and_vars, global_step=global_step,
            //                            name=name)

            throw new NotImplementedException();
        }

        /// <summary>
        /// Compute gradients of `loss` for the variables in `var_list`.
        /// This is the first part of `minimize()`.  It returns a list
        /// of(gradient, variable) pairs where "gradient" is the gradient
        /// for "variable".  Note that "gradient" can be a `Tensor`, an
        /// `IndexedSlices`, or `None` if there is no gradient for the
        /// given variable.
        /// Args:
        ///   loss: A Tensor containing the value to minimize.
        ///   var_list: Optional list or tuple of `tf.Variable` to update to minimize
        ///     `loss`.  Defaults to the list of variables collected in the graph
        ///     under the key `GraphKey.TRAINABLE_VARIABLES`.
        ///   gate_gradients: How to gate the computation of gradients.Can be
        ///     `GATE_NONE`, `GATE_OP`, or `GATE_GRAPH`.
        ///   aggregation_method: Specifies the method used to combine gradient terms.
        ///     Valid values are defined in the class `AggregationMethod`.
        ///   colocate_gradients_with_ops: If True, try colocating gradients with
        ///     the corresponding op.
        ///   grad_loss: Optional.A `Tensor` holding the gradient computed for `loss`.
        /// Returns:
        ///   A list of (gradient, variable) pairs. Variable is always present, but
        ///   gradient can be `None`.
        /// Raises:
        ///   TypeError: If `var_list` contains anything else than `Variable` objects.
        ///   ValueError: If some arguments are invalid.
        /// </summary>
        /// <param name=""></param>
        /// <param name=""></param>
        /// <param name=""></param>
        /// <param name=""></param>
        /// <param name=""></param>
        /// <param name=""></param>
        /// <param name=""></param>
        public List<Tuple<TFOutput, TFOutput>> ComputeGradients(TFOutput loss, List<TFOutput> var_list,
                        GateGradients gate_gradients = GateGradients.GATE_OP,
                        AggregationMethod aggregation_method = AggregationMethod.ADD_N, // in TensorFlow this is set to None, which corresponds to 0 and thereby AggregationMethod.Add_N
                        bool colocate_gradients_with_ops = false,
                        TFOutput grad_loss = default(TFOutput))
        {
            if (gate_gradients != GateGradients.GATE_NONE &&
                gate_gradients != GateGradients.GATE_OP &&
                gate_gradients != GateGradients.GATE_GRAPH)
            {
                throw new ArgumentException("Gate operation must be contained in GateGradients enum. Got value: " + gate_gradients);
            }

            AssertValidDType(loss);

            // TODO: Figure out how to check if grad_loss is provided.
            //if(grad_loss != null)
            //{
            AssertValidDType(grad_loss);
            //}

            if(var_list == null)
            {
                // TODO: Figure out how to access find default trainable variables from graph.
                throw new ArgumentException("default trainable variables is not supported.");

                //var_list = (
                //    variables.trainable_variables() +
                //    ops.get_collection(ops.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))
            }

            // TODO: Figure out if flatten is needed. Currently var_list is aways an array.
            // var_list = nest.flatten(var_list)

            // TODO: Figure out how to get the necesarry collection from the graph
            // # pylint: disable=protected-access
            // var_list += ops.get_collection(ops.GraphKeys._STREAMING_MODEL_PORTS)
            // # pylint: enable=protected-access

            throw new NotImplementedException();
        }

        public TFOperation ApplyGradients(List<Tuple<TFOutput, TFOutput>> grads_and_vars, 
            TFOutput global_step = default(TFOutput), 
            string name = null)
        {
            throw new NotImplementedException();
        }

        protected abstract TFOutput ApplyDense(TFOutput grad, TFOutput var);

        protected abstract TFOutput ResourceApplyDense(TFOutput grad, IntPtr handle);

        protected virtual TFOutput ResourceApplySparseDuplicateIndices(TFOutput grad, IntPtr handle, int[] indices)
        {
            throw new NotImplementedException();
            //return resource_variable_ops.resource_scatter_add(
            //    handle.handle, indices, -grad * self._learning_rate);
        }

        protected abstract TFOutput ApplySparseDuplicateIndices(TFOutput grad, TFOutput var);

        protected abstract TFOutput Prepare();

        //public static def _get_processor(v)
        //{
        //    """The processor of v."""
        //  if v.op.type == "VarHandleOp":
        //    return _DenseResourceVariableProcessor(v)
        //  if isinstance(v, variables.Variable):
        //    return _RefVariableProcessor(v)
        //  if v.op.type == "SubmodelPort":
        //    return _StreamingModelPortProcessor(v)

        //  raise NotImplementedError("Trying to optimize unsupported type ", v)
        //}

        /// <summary>
        /// Asserts tensor has valid type (see `_valid_dtypes`).
        /// </summary>
        /// <param name="tensor"></param>
        void AssertValidDType(TFOutput tensor)
        {
            if(!m_validDTypes.Contains(tensor.OutputType))
            {
                throw new ArgumentException($"Invalid type {tensor.OutputType} expected: {string.Join(",", m_validDTypes)}");
            }
        }
    }
}
