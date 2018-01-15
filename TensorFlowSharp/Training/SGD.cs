using System;
using System.Collections.Generic;
using System.Linq;
using TensorFlow;

namespace TensorFlowSharp.Training
{
    /// <summary>
    /// Simplified stochastic gradient descent class
    /// </summary>
    public class SGD
    {
        readonly double m_learningRate;
        readonly bool m_useLocking;

        /// <summary>
        /// Simplified stochastic gradient descent class
        /// </summary>
        /// <param name="learningRate"></param>
        /// <param name="useLocking"></param>
        public SGD(double learningRate = 0.001, bool useLocking = false)
        {
            m_learningRate = learningRate;
            m_useLocking = useLocking;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="loss"></param>
        /// <param name="graph"></param>
        /// <returns></returns>
        public TFOutput[] Minimize(TFOutput loss, TFGraph graph) // extend with necesarry inputs
        {
            // loss
            var y = new TFOutput[] { loss };

			// get trainable parameters
			var x = graph.GetTrainableVariables().Select(v => v.VariableOp).ToArray();

            // get gradients
            var delta = graph.AddGradients(y, x);

            if(x.Length != delta.Length)
            { throw new InvalidOperationException($"variable length: {x.Length} differs from gradient length: {delta.Length}"); }

			// update trainable parameters
			var ops = new TFOutput[y.Length];
            for (int i = 0; i < y.Length; i++)
            {
				ops[i] = new TFOutput(graph.ResourceApplyGradientDescent(x[i], graph.Const(m_learningRate), delta[i], m_useLocking));
            }

			return ops;
        }
    }
}
