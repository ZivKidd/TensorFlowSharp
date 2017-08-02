namespace TensorFlowSharp.Training
{
    /// <summary>
    /// Standard names to use for graph collections.
    /// 
    /// The standard library uses various well-known names to collect and
    /// retrieve values associated with a graph.For example, the
    /// `tf.Optimizer` subclasses default to optimizing the variables
    /// collected under `tf.GraphKeys.TRAINABLE_VARIABLES` if none is
    /// specified, but it is also possible to pass an explicit list of
    /// variables.
    /// 
    /// The following standard keys are defined:
    /// 
    /// * `GLOBAL_VARIABLES`: the default collection of `Variable` objects, shared
    ///   across distributed environment (model variables are subset of these). See
    ///   @{tf.global_variables }
    ///   for more details.
    ///   Commonly, all `TRAINABLE_VARIABLES` variables will be in `MODEL_VARIABLES`,
    ///   and all `MODEL_VARIABLES` variables will be in `GLOBAL_VARIABLES`.
    /// * `LOCAL_VARIABLES`: the subset of `Variable` objects that are local to each
    ///   machine.Usually used for temporarily variables, like counters.
    ///   Note: use `tf.contrib.framework.local_variable` to add to this collection.
    /// * `MODEL_VARIABLES`: the subset of `Variable` objects that are used in the
    ///   model for inference (feed forward). Note: use
    ///   `tf.contrib.framework.model_variable` to add to this collection.
    /// * `TRAINABLE_VARIABLES`: the subset of `Variable` objects that will
    ///   be trained by an optimizer. See
    ///   @{ tf.trainable_variables}
    ///   for more details.
    /// * `SUMMARIES`: the summary `Tensor` objects that have been created in the
    ///   graph. See
    ///   @{ tf.summary.merge_all}
    ///   for more details.
    /// * `QUEUE_RUNNERS`: the `QueueRunner` objects that are used to
    ///   produce input for a computation. See
    ///   @{ tf.train.start_queue_runners}
    ///   for more details.
    /// * `MOVING_AVERAGE_VARIABLES`: the subset of `Variable` objects that will also
    ///   keep moving averages.See
    ///   @{ tf.moving_average_variables}
    ///   for more details.
    /// * `REGULARIZATION_LOSSES`: regularization losses collected during graph
    ///   construction.
    /// 
    /// The following standard keys are _defined_, but their collections are **not**
    /// automatically populated as many of the others are:
    /// * `WEIGHTS`
    /// * `BIASES`
    /// * `ACTIVATIONS`
    /// </summary>
    public static class GraphKey
    {
        // Key to collect Variable objects that are global (shared across machines).
        // Default collection for all variables, except local ones.
        public static string GLOBAL_VARIABLES = "variables";
        // Key to collect local variables that are local to the machine and are not
        // saved/restored.
        public static string LOCAL_VARIABLES = "local_variables";
        // Key to collect model variables defined by layers.
        public static string MODEL_VARIABLES = "model_variables";
        // Key to collect Variable objects that will be trained by the
        // optimizers.
        public static string TRAINABLE_VARIABLES = "trainable_variables";
        // Key to collect summaries.
        public static string SUMMARIES = "summaries";
        // Key to collect QueueRunners.
        public static string QUEUE_RUNNERS = "queue_runners";
        // Key to collect table initializers.
        public static string TABLE_INITIALIZERS = "table_initializer";
        // Key to collect asset filepaths. An asset represents an external resource
        // like a vocabulary file.
        public static string ASSET_FILEPATHS = "asset_filepaths";
        // Key to collect Variable objects that keep moving averages.
        public static string MOVING_AVERAGE_VARIABLES = "moving_average_variables";
        // Key to collect regularization losses at graph construction.
        public static string REGULARIZATION_LOSSES = "regularization_losses";
        // Key to collect concatenated sharded variables.
        public static string CONCATENATED_VARIABLES = "concatenated_variables";
        // Key to collect savers.
        public static string SAVERS = "savers";
        // Key to collect weights
        public static string WEIGHTS = "weights";
        // Key to collect biases
        public static string BIASES = "biases";
        // Key to collect activations
        public static string ACTIVATIONS = "activations";
        // Key to collect update_ops
        public static string UPDATE_OPS = "update_ops";
        // Key to collect losses
        public static string LOSSES = "losses";
        // Key to collect BaseSaverBuilder.SaveableObject instances for checkpointing.
        public static string SAVEABLE_OBJECTS = "saveable_objects";
        // Key to collect all shared resources used by the graph which need to be
        // initialized once per cluster.
        public static string RESOURCES = "resources";
        // Key to collect all shared resources used in this graph which need to be
        // initialized once per session.
        public static string LOCAL_RESOURCES = "local_resources";
        // Trainable resource-style variables.
        public static string TRAINABLE_RESOURCE_VARIABLES = "trainable_resource_variables";

        // Key to indicate various ops.
        public static string INIT_OP = "init_op";
        public static string LOCAL_INIT_OP = "local_init_op";
        public static string READY_OP = "ready_op";
        public static string READY_FOR_LOCAL_INIT_OP = "ready_for_local_init_op";
        public static string SUMMARY_OP = "summary_op";
        public static string GLOBAL_STEP = "global_step";

        // Used to count the number of evaluations performed during a single evaluation
        // run.
        public static string EVAL_STEP = "eval_step";
        public static string TRAIN_OP = "train_op";

        // Key for control flow context.
        public static string COND_CONTEXT = "cond_context";
        public static string WHILE_CONTEXT = "while_context";

        // Key for streaming model ports.
        // NOTE(yuanbyu): internal and experimental.
        public static string _STREAMING_MODEL_PORTS = "streaming_model_ports";

        //@decorator_utils.classproperty
        //def VARIABLES(cls):  // pylint: disable=no-self-argument
        //  logging.warning("VARIABLES collection name is deprecated, "
        //                  "please use GLOBAL_VARIABLES instead; "
        //                  "VARIABLES will be removed after 2017-03-02.")
        //  return cls.GLOBAL_VARIABLES
    }
}
