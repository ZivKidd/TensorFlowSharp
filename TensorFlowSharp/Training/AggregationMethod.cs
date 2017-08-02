namespace TensorFlowSharp.Training
{
    /// <summary>
    /// Computing partial derivatives can require aggregating gradient
    /// contributions.This class lists the various methods that can
    /// be used to combine gradients in the graph:
    /// </summary>
    public enum AggregationMethod
    {
        /// <summary>
        /// `ADD_N`: All of the gradient terms are summed as part of one
        /// operation using the "AddN" op.It has the property that all
        /// gradients must be ready before any aggregation is performed.
        /// </summary>
        ADD_N = 0,

        /// <summary>
        /// `DEFAULT`: The system-chosen default aggregation method.
        /// </summary>
        DEFAULT = ADD_N,

        /// <summary>
        /// The following are experimental and may not be supported in future releases.
        /// </summary>
        EXPERIMENTAL_TREE = 1,

        /// <summary>
        /// The following are experimental and may not be supported in future releases.
        /// </summary>
        EXPERIMENTAL_ACCUMULATE_N = 2
    }
}
