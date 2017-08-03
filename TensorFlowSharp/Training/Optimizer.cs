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
    /// opt_op = opt.minimize(cost, var_list=list of variables)
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
    /// grads_and_vars = opt.compute_gradients(loss, list of variables)
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
        readonly bool m_useLocking;
        readonly string m_name;
        readonly HashSet<TFDataType> m_validDTypes = new HashSet<TFDataType> { TFDataType.BFloat16, TFDataType.Float, TFDataType.Double };

        /// <summary>
        /// Create a new Optimizer.
        /// 
        /// This must be called by the constructors of subclasses.
        /// 
        /// s</summary>
        /// <param name="useLocking">Bool.If True apply use locks to prevent concurrent updates
        /// to variables.</param>
        /// <param name="name">A non-empty string.  The name to use for accumulators created
        /// for the optimizer.</param>
        public Optimizer(bool useLocking, string name)
        {
            if (name == string.Empty) { throw new ArgumentException("Must specify the optimizer name"); }

            m_useLocking = useLocking;
            m_name = name;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
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

            var grads_and_vars = ComputeGradients(loss, var_list, 
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
        /// </summary>
        /// <param name="loss">A `Tensor` containing the value to minimize.</param>
        /// <param name="var_list">Optional list or tuple of `Variable` objects to update to
        /// minimize `loss`.  Defaults to the list of variables collected in
        /// the graph under the key `GraphKeys.TRAINABLE_VARIABLES`.</param>
        /// <param name="gate_gradients">How to gate the computation of gradients.Can be
        /// `GATE_NONE`, `GATE_OP`, or  `GATE_GRAPH`.</param>
        /// <param name="aggregation_method"> Specifies the method used to combine gradient terms.
        /// Valid values are defined in the class `AggregationMethod`.</param>
        /// <param name="colocate_gradients_with_ops">If True, try colocating gradients with
        /// the corresponding op.</param>
        /// <param name="grad_loss">Optional.A `Tensor` holding the gradient computed for `loss`.</param>
        /// <returns> A list of (gradient, variable) pairs. Variable is always present, but
        /// gradient can be `None`.</returns>
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

            // TODO: Figure out if flatten is needed. Currently var_list is aways a list.
            // var_list = nest.flatten(var_list)

            // TODO: Figure out how to get the necesarry collection from the graph
            // # pylint: disable=protected-access
            // var_list += ops.get_collection(ops.GraphKeys._STREAMING_MODEL_PORTS)
            // # pylint: enable=protected-access

            throw new NotImplementedException();
        }

        /// <summary>
        /// Apply gradients to variables.
        /// This is the second part of `minimize()`. It returns an `Operation` that
        /// applies gradients.

        /// This is a default implementation of apply_gradients() that can be shared
        /// by most optimizers.  It relies on the subclass implementing the following
        /// methods: _create_slots(), _prepare(), _apply_dense(), and _apply_sparse().        
        /// 
        /// </summary>
        /// <param name="grads_and_vars">List of(gradient, variable) pairs as returned by
        ///     `compute_gradients()`.</param>
        /// <param name="global_step">Optional `Variable` to increment by one after the
        ///     variables have been updated.</param>
        /// <param name="name">Optional name for the returned operation.Default to the
        ///     name passed to the `Optimizer` constructor.</param>
        /// <returns> An `Operation` that applies the specified gradients.If `global_step`
        ///   was not None, that operation also increments `global_step`.</returns>
        public virtual TFOperation ApplyGradients(List<Tuple<TFOutput, TFOutput>> grads_and_vars, 
            TFOutput global_step = default(TFOutput), 
            string name = null)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Return a slot named `name` created for `var` by the Optimizer.
        /// 
        /// Some `Optimizer` subclasses use additional variables.For example
        /// `Momentum` and `Adagrad` use variables to accumulate updates.This method
        /// gives access to these `Variable` objects if for some reason you need them.
        /// 
        /// Use `get_slot_names()` to get the list of slot names created by the
        /// `Optimizer`.
        /// 
        /// </summary>
        /// <param name="var">A variable passed to `minimize()` or `apply_gradients()`.</param>
        /// <param name="name">A string.</param>
        /// <returns>The `Variable` for the slot if it was created, `None` otherwise.</returns>
        public TFOutput GetSlot(TFOutput var, string name)
        {
            throw new NotImplementedException();

            //named_slots = self._slots.get(name, None)
            //if not named_slots:
            //            return None
            //return named_slots.get(_var_key(var), None)
        }

        /// <summary>
        /// Return a list of the names of slots created by the `Optimizer`.
        /// See `get_slot()`.
        /// </summary>
        /// <returns> A list of strings.</returns>
        public TFOutput GetSlotNames()
        {
            throw new NotImplementedException();

            // return sorted(self._slots.keys())
        }

        /// <summary>
        /// Asserts tensor has valid type (see `_valid_dtypes`).
        /// </summary>
        /// <param name="tensor"></param>
        void AssertValidDType(TFOutput tensor)
        {
            if (!m_validDTypes.Contains(tensor.OutputType))
            {
                throw new ArgumentException($"Invalid type {tensor.OutputType} expected: {string.Join(",", m_validDTypes)}");
            }
        }

        /// <summary>
        /// Create all slots needed by the variables.
        /// 
        /// No slots needed by default
        /// 
        /// </summary>
        /// <param name="var_list">A list of `Variable` objects.</param>
        protected virtual void CreateSlots(List<TFOutput> var_list)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Create all needed tensors before applying gradients.
        /// This is called with the name_scope using the "name" that
        /// users have chosen for the application of gradients.
        /// </summary>
        /// <returns></returns>
        protected virtual TFOutput Prepare()
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Add ops to apply dense gradients to `var`.
        /// </summary>
        /// <param name="grad">A `Tensor`</param>
        /// <param name="var">A `Variable` object.</param>
        /// <returns>An `Operation`.</returns>
        protected virtual TFOutput ApplyDense(TFOutput grad, TFOutput var)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        ///  Add ops to apply dense gradients to the variable `handle`.
        /// </summary>
        /// <param name="grad">a `Tensor` representing the gradient.</param>
        /// <param name="handle">a `Tensor` of dtype `resource` which points to the variable
        /// to be updated.</param>
        /// <returns>An `Operation` which updates the value of the variable.</returns>
        protected virtual TFOutput ResourceApplyDense(TFOutput grad, IntPtr handle)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Add ops to apply sparse gradients to `handle`, with repeated indices.
        /// 
        /// Optimizers which override this method must deal with repeated indices. See
        /// the docstring of `_apply_sparse_duplicate_indices` for details.By default
        /// the correct behavior, to sum non-unique indices and their associated
        /// gradients, is enforced by first pre-processing `grad` and `indices` and
        /// passing them on to `_resource_apply_sparse`. Optimizers which deal correctly
        /// with duplicate indices may instead override this method to avoid the
        /// overhead of summing.        
        /// </summary>
        /// <param name="grad"> a `Tensor` representing the gradient for the affected indices.</param>
        /// <param name="handle">a `Tensor` of dtype `resource` which points to the variable
        /// to be updated.</param>
        /// <param name="indices"> a `Tensor` of integral type representing the indices for
        /// which the gradient is nonzero.Indices may be repeated.</param>
        /// <returns>An `Operation` which updates the value of the variable.</returns>
        protected virtual TFOutput ResourceApplySparseDuplicateIndices(TFOutput grad, IntPtr handle, int[] indices)
        {
            throw new NotImplementedException();
            //summed_grad, unique_indices = _deduplicate_indexed_slices(
            //    values = grad, indices = indices)
            //return self._resource_apply_sparse(summed_grad, handle, unique_indices)        
        }

        /// <summary>
        /// Add ops to apply sparse gradients to the variable `handle`.
        /// 
        /// Similar to `_apply_sparse`, the `indices` argument to this method has been
        /// de-duplicated.Optimizers which deal correctly with non-unique indices may
        /// instead override `_resource_apply_sparse_duplicate_indices` to avoid this
        /// overhead.
        ///
        /// </summary>
        /// <param name="grad"> a `Tensor` representing the gradient for the affected indices.</param>
        /// <param name="handle">a `Tensor` of dtype `resource` which points to the variable
        /// to be updated.</param>
        /// <param name="indices">a `Tensor` of integral type representing the indices for
        /// which the gradient is nonzero.Indices are unique.</param>
        /// <returns>An `Operation` which updates the value of the variable.</returns>
        protected virtual TFOutput ResourceApplySparse(TFOutput grad, IntPtr handle, int[] indices)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Add ops to apply sparse gradients to `var`, with repeated sparse indices.
        /// 
        /// Optimizers which override this method must deal with IndexedSlices objects
        /// 
        /// such as the following:
        ///  
        ///     IndexedSlicesValue(values=[1, 1], indices=[0, 0], dense_shape=[1])
        /// 
        /// The correct interpretation is:
        ///   
        ///     IndexedSlicesValue(values=[2], indices=[0], dense_shape=[1])
        /// 
        /// Many optimizers deal incorrectly with repeated indices when updating based
        /// on sparse gradients(e.g.summing squares rather than squaring the sum, or
        /// applying momentum terms multiple times). Adding first is always the correct
        /// behavior, so this is enforced here by reconstructing the IndexedSlices to
        /// have only unique indices, then calling _apply_sparse.
        /// 
        /// Optimizers which deal correctly with repeated indices may instead override
        /// this method to avoid the overhead of summing indices.
        /// 
        /// </summary>
        /// <param name="grad">IndexedSlices</param>
        /// <param name="var"> A `Variable` object.</param>
        /// <returns></returns>
        protected virtual TFOutput ApplySparseDuplicateIndices(TFOutput grad, TFOutput var)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Add ops to apply sparse gradients to `var`.
        ///
        /// The IndexedSlices object passed to `grad` in this function is by default
        /// pre-processed in `_apply_sparse_duplicate_indices` to remove duplicate
        /// indices(see its docstring for details). Optimizers which can tolerate or
        /// have correct special cases for duplicate sparse indices may override
        /// `_apply_sparse_duplicate_indices` instead of this function, avoiding that
        /// overhead.        
        ///
        /// </summary>
        /// <param name="grad"> `IndexedSlices`, with no repeated indices.</param>
        /// <param name="var">A `Variable` object.</param>
        /// <returns>An `Operation`.</returns>
        protected virtual TFOutput ApplySparse(TFOutput grad, TFOutput var)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Do what is needed to finish the update.
        /// This is called with the `name_scope` using the "name" that
        /// users have chosen for the application of gradients.
        /// </summary>
        /// <param name="update_ops">List of `Operation` objects to update variables.  This list
        /// contains the values returned by the `_apply_dense()` and
        /// `_apply_sparse()` calls.</param>
        /// <param name="name_scope">String.Name to use for the returned operation.</param>
        /// <returns>The operation to apply updates.</returns>
        protected virtual TFOutput Finish(List<TFOutput> update_ops, string name_scope)
        {
            throw new NotImplementedException();
            //    return control_flow_ops.group(* update_ops, name= name_scope)
        }

        #region Utility methods for subclasses

        // TODO: Add utility methods as they become necesarry for the sub-class optimizes.

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

        # endregion
    }
}
