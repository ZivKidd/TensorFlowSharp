namespace TensorFlowSharp.Training
{
    /// <summary>
    /// Both `minimize()` and `compute_gradients()` accept a `gate_gradients`
    /// argument that controls the degree of parallelism during the application of
    /// the gradients.
    /// </summary>
    public enum GateGradients
    {
        /// <summary>
        /// Compute and apply gradients in parallel.This provides
        /// the maximum parallelism in execution, at the cost of some non-reproducibility
        /// in the results.  For example the two gradients of `matmul` depend on the input
        /// values: With `GATE_NONE` one of the gradients could be applied to one of the
        /// inputs _before_ the other gradient is computed resulting in non-reproducible
        /// results.
        /// </summary>
        GATE_NONE,

        /// <summary>
        /// For each Op, make sure all gradients are computed before
        /// they are used.  This prevents race conditions for Ops that generate gradients
        /// for multiple inputs where the gradients depend on the inputs.
        /// </summary>
        GATE_OP,

        /// <summary>
        /// Make sure all gradients for all variables are computed
        /// before any one of them is used.This provides the least parallelism but can
        /// be useful if you want to process all gradients before applying any of them.
        /// </summary>
        GATE_GRAPH,
    }
}
