//! MorphoDiff: SciRust's backend-neutral differential program transformer.
//!
//! MorphoDiff operates on canonical Tensor IR rather than LLVM IR. It builds
//! explicit JVP/VJP/gradient graphs which can subsequently be optimized and
//! lowered by the normal SciRust compilation pipeline.

use alloc::{vec, vec::Vec};

use crate::{
    AutodiffError, GradGraph, Graph, JvpGraph, NodeId, VjpGraph, grad, jvp, validate_semantics,
    value_and_grad, vjp,
};

/// Stable engine name used by diagnostics and benchmark reports.
pub const MORPHODIFF_ENGINE: &str = "MorphoDiff";

/// Differential transform requested from MorphoDiff.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MorphoDiffMode {
    /// Forward-mode Jacobian-vector product with explicit tangent inputs.
    ForwardJvp,
    /// Reverse-mode vector-Jacobian product with an explicit cotangent input.
    ReverseVjp,
    /// Reverse-mode gradient seeded with an all-one cotangent.
    ReverseGrad,
    /// Return the primal output together with reverse-mode gradients.
    ReverseValueAndGrad,
}

/// Structural accounting for one MorphoDiff transform.
///
/// These counters deliberately describe graph structure only. Runtime latency,
/// peak memory and generated-code size are measured by the benchmark layer after
/// lowering, where comparisons against compiler-level AD systems such as Enzyme
/// are meaningful.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MorphoDiffReport {
    pub source_nodes: usize,
    pub transformed_nodes: usize,
    pub generated_nodes: usize,
    pub wrt_count: usize,
    pub seed_input_count: usize,
    pub derivative_output_count: usize,
}

/// A backend-neutral differentiated program produced by MorphoDiff.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MorphoDiffProgram {
    pub mode: MorphoDiffMode,
    pub graph: Graph,
    pub primal_output: NodeId,
    pub derivative_outputs: Vec<NodeId>,
    /// Explicit derivative seeds required at execution time.
    ///
    /// JVP programs contain tangent inputs; VJP programs contain one cotangent
    /// input; seeded gradient transforms contain none.
    pub seed_inputs: Vec<NodeId>,
    pub report: MorphoDiffReport,
}

/// Stateless Tensor-IR differentiation engine.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MorphoDiff;

impl MorphoDiff {
    /// Transform a canonical Tensor IR graph into an explicit differential graph.
    pub fn transform(
        source: &Graph,
        output: NodeId,
        wrt: &[NodeId],
        mode: MorphoDiffMode,
    ) -> Result<MorphoDiffProgram, AutodiffError> {
        let source_nodes = source.nodes().len();

        let (graph, primal_output, derivative_outputs, seed_inputs) = match mode
        {
            MorphoDiffMode::ForwardJvp =>
            {
                let JvpGraph {
                    graph,
                    primal_output,
                    tangent_output,
                    tangent_inputs,
                } = jvp(source, output, wrt)?;
                (graph, primal_output, vec![tangent_output], tangent_inputs)
            },
            MorphoDiffMode::ReverseVjp =>
            {
                let VjpGraph {
                    graph,
                    primal_output,
                    cotangent_input,
                    gradients,
                } = vjp(source, output, wrt)?;
                (graph, primal_output, gradients, vec![cotangent_input])
            },
            MorphoDiffMode::ReverseGrad =>
            {
                let GradGraph {
                    graph,
                    primal_output,
                    gradients,
                } = grad(source, output, wrt)?;
                (graph, primal_output, gradients, Vec::new())
            },
            MorphoDiffMode::ReverseValueAndGrad =>
            {
                let GradGraph {
                    graph,
                    primal_output,
                    gradients,
                } = value_and_grad(source, output, wrt)?;
                (graph, primal_output, gradients, Vec::new())
            },
        };

        graph.validate()?;
        validate_semantics(&graph).map_err(AutodiffError::InvalidSemantics)?;

        let transformed_nodes = graph.nodes().len();
        let report = MorphoDiffReport {
            source_nodes,
            transformed_nodes,
            generated_nodes: transformed_nodes.saturating_sub(source_nodes),
            wrt_count: wrt.len(),
            seed_input_count: seed_inputs.len(),
            derivative_output_count: derivative_outputs.len(),
        };

        Ok(MorphoDiffProgram {
            mode,
            graph,
            primal_output,
            derivative_outputs,
            seed_inputs,
            report,
        })
    }

    /// Build a forward-mode Jacobian-vector-product program.
    ///
    /// The returned program exposes tangent seed inputs in `seed_inputs` and the
    /// resulting JVP node in `derivative_outputs`.
    pub fn jvp(
        source: &Graph,
        output: NodeId,
        wrt: &[NodeId],
    ) -> Result<MorphoDiffProgram, AutodiffError> {
        Self::transform(source, output, wrt, MorphoDiffMode::ForwardJvp)
    }

    /// Build a reverse-mode vector-Jacobian-product program.
    ///
    /// The returned program exposes one cotangent seed input and one derivative
    /// output for every requested `wrt` node.
    pub fn vjp(
        source: &Graph,
        output: NodeId,
        wrt: &[NodeId],
    ) -> Result<MorphoDiffProgram, AutodiffError> {
        Self::transform(source, output, wrt, MorphoDiffMode::ReverseVjp)
    }

    /// Build a reverse-mode gradient program with an all-one cotangent seed.
    ///
    /// This convenience transform is self-seeded, so `seed_inputs` is empty and
    /// `derivative_outputs` contains the gradients of `output` with respect to
    /// the requested nodes.
    pub fn grad(
        source: &Graph,
        output: NodeId,
        wrt: &[NodeId],
    ) -> Result<MorphoDiffProgram, AutodiffError> {
        Self::transform(source, output, wrt, MorphoDiffMode::ReverseGrad)
    }

    /// Build a reverse-mode program that retains the primal and its gradients.
    ///
    /// The transformed graph keeps the primal output first and appends one
    /// derivative output for every requested `wrt` node.
    pub fn value_and_grad(
        source: &Graph,
        output: NodeId,
        wrt: &[NodeId],
    ) -> Result<MorphoDiffProgram, AutodiffError> {
        Self::transform(source, output, wrt, MorphoDiffMode::ReverseValueAndGrad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, Operation, Scalar, Shape, TensorType};

    fn vector_type() -> TensorType {
        TensorType::new(DType::F32, Shape::new(vec![4]))
    }

    fn quadratic_graph() -> (Graph, NodeId, NodeId) {
        let mut graph = Graph::new();
        let x = graph.add_input("x", vector_type()).unwrap();
        let square = graph
            .add_node(Operation::Mul, vec![x, x], vector_type())
            .unwrap();
        let scaled = graph
            .add_node(
                Operation::Scale {
                    factor: Scalar::f32(0.5),
                },
                vec![square],
                vector_type(),
            )
            .unwrap();
        graph.set_outputs(vec![scaled]).unwrap();
        (graph, x, scaled)
    }

    #[test]
    fn morphodiff_forward_exposes_tangent_seed_and_report() {
        let (graph, x, output) = quadratic_graph();
        let program = MorphoDiff::jvp(&graph, output, &[x]).unwrap();

        assert_eq!(program.mode, MorphoDiffMode::ForwardJvp);
        assert_eq!(program.seed_inputs.len(), 1);
        assert_eq!(program.derivative_outputs.len(), 1);
        assert_eq!(program.report.source_nodes, graph.nodes().len());
        assert_eq!(program.report.wrt_count, 1);
        assert!(program.report.generated_nodes > 0);
        assert_eq!(program.graph.validate(), Ok(()));
    }

    #[test]
    fn morphodiff_vjp_exposes_single_cotangent_seed() {
        let (graph, x, output) = quadratic_graph();
        let program = MorphoDiff::vjp(&graph, output, &[x]).unwrap();

        assert_eq!(program.mode, MorphoDiffMode::ReverseVjp);
        assert_eq!(program.seed_inputs.len(), 1);
        assert_eq!(program.derivative_outputs.len(), 1);
        assert_eq!(program.report.seed_input_count, 1);
    }

    #[test]
    fn morphodiff_grad_is_self_seeded() {
        let (graph, x, output) = quadratic_graph();
        let program = MorphoDiff::grad(&graph, output, &[x]).unwrap();

        assert_eq!(program.mode, MorphoDiffMode::ReverseGrad);
        assert!(program.seed_inputs.is_empty());
        assert_eq!(program.derivative_outputs.len(), 1);
        assert_eq!(
            program.graph.outputs(),
            program.derivative_outputs.as_slice()
        );
    }

    #[test]
    fn morphodiff_value_and_grad_keeps_primal_in_outputs() {
        let (graph, x, output) = quadratic_graph();
        let program = MorphoDiff::value_and_grad(&graph, output, &[x]).unwrap();

        assert_eq!(program.mode, MorphoDiffMode::ReverseValueAndGrad);
        assert_eq!(program.graph.outputs()[0], program.primal_output);
        assert_eq!(
            &program.graph.outputs()[1..],
            program.derivative_outputs.as_slice()
        );
    }
}
