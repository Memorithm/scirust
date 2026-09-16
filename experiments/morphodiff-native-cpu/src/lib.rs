use std::collections::BTreeMap;
use std::fmt;
use std::mem;

use cranelift::codegen::ir::UserFuncName;
use cranelift::jit::{JITBuilder, JITModule};
use cranelift::module::{Linkage, Module, default_libcall_names};
use cranelift::prelude::*;
use scirust_tensor_ir::{
    ConstantId, DType, Graph, NodeId, Operation, Scalar, TensorType, validate_semantics,
};

#[derive(Debug)]
pub struct NativeCompileError(String);

impl NativeCompileError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for NativeCompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for NativeCompileError {}

/// Owns one Cranelift JIT module and a native two-input scalar derivative.
///
/// The module is retained for the whole lifetime of the function pointer because
/// `JITModule` owns the executable memory backing `entry`.
pub struct NativeScalarFn2 {
    entry: extern "C" fn(f32, f32) -> f32,
    _module: JITModule,
}

impl NativeScalarFn2 {
    #[inline(never)]
    pub fn call(&self, x: f32, y: f32) -> f32 {
        (self.entry)(x, y)
    }

    pub fn entry(&self) -> extern "C" fn(f32, f32) -> f32 {
        self.entry
    }
}

/// Compile a canonical scalar-F32 Tensor IR graph to native host machine code.
///
/// This first MorphoDiff native lane intentionally supports exactly two graph
/// inputs and one graph output, each containing one F32 element. It covers the
/// arithmetic needed by the Rosenbrock gradient benchmark without pulling a JIT
/// dependency into SciRust's production/MSRV workspace.
pub fn compile_scalar_f32_2_to_1(
    graph: &Graph,
    constants: &BTreeMap<ConstantId, f32>,
) -> Result<NativeScalarFn2, NativeCompileError> {
    graph
        .validate()
        .map_err(|error| NativeCompileError::new(format!("invalid graph: {error}")))?;
    validate_semantics(graph)
        .map_err(|error| NativeCompileError::new(format!("invalid semantics: {error}")))?;

    if graph.outputs().len() != 1 {
        return Err(NativeCompileError::new(format!(
            "native scalar lane requires exactly one graph output, found {}",
            graph.outputs().len()
        )));
    }

    let input_count = graph
        .nodes()
        .iter()
        .filter(|node| matches!(node.operation, Operation::Input { .. }))
        .count();
    if input_count != 2 {
        return Err(NativeCompileError::new(format!(
            "native scalar lane requires exactly two graph inputs, found {input_count}"
        )));
    }

    for (index, node) in graph.nodes().iter().enumerate() {
        require_scalar_f32(NodeId::new(index as u32), &node.output)?;
    }

    let mut flag_builder = settings::builder();
    flag_builder
        .set("use_colocated_libcalls", "false")
        .map_err(|error| NativeCompileError::new(error.to_string()))?;
    flag_builder
        .set("is_pic", "false")
        .map_err(|error| NativeCompileError::new(error.to_string()))?;
    let isa_builder = cranelift::native::builder()
        .map_err(|error| NativeCompileError::new(format!("unsupported host ISA: {error}")))?;
    let isa = isa_builder
        .finish(settings::Flags::new(flag_builder))
        .map_err(|error| NativeCompileError::new(format!("cannot configure host ISA: {error}")))?;
    let mut module = JITModule::new(JITBuilder::with_isa(isa, default_libcall_names()));

    let mut signature = module.make_signature();
    signature.params.push(AbiParam::new(types::F32));
    signature.params.push(AbiParam::new(types::F32));
    signature.returns.push(AbiParam::new(types::F32));

    let function_id = module
        .declare_function("morphodiff_native_scalar_2_1", Linkage::Local, &signature)
        .map_err(|error| NativeCompileError::new(format!("declare JIT function: {error}")))?;
    let frontend_config = module.target_config();
    let mut context = module.make_context();
    context.func.signature = signature;
    context.func.name = UserFuncName::user(0, function_id.as_u32());
    let mut builder_context = FunctionBuilderContext::new();

    {
        let mut builder = FunctionBuilder::new(&mut context.func, &mut builder_context);
        let block = builder.create_block();
        builder.switch_to_block(block);
        builder.append_block_params_for_function_params(block);
        let parameters = builder.block_params(block).to_vec();

        let mut input_index = 0usize;
        let mut values: Vec<Option<Value>> = vec![None; graph.nodes().len()];

        for (index, node) in graph.nodes().iter().enumerate() {
            let id = NodeId::new(index as u32);
            let value = match &node.operation {
                Operation::Input { .. } => {
                    let value = parameters[input_index];
                    input_index += 1;
                    value
                }
                Operation::Constant { id } => {
                    let value = constants.get(id).copied().ok_or_else(|| {
                        NativeCompileError::new(format!(
                            "missing F32 value for constant {}",
                            id.get()
                        ))
                    })?;
                    builder.ins().f32const(Ieee32::with_float(value))
                }
                Operation::Add => {
                    let (lhs, rhs) = binary_inputs(&values, id, &node.inputs)?;
                    builder.ins().fadd(lhs, rhs)
                }
                Operation::Sub => {
                    let (lhs, rhs) = binary_inputs(&values, id, &node.inputs)?;
                    builder.ins().fsub(lhs, rhs)
                }
                Operation::Mul => {
                    let (lhs, rhs) = binary_inputs(&values, id, &node.inputs)?;
                    builder.ins().fmul(lhs, rhs)
                }
                Operation::Div => {
                    let (lhs, rhs) = binary_inputs(&values, id, &node.inputs)?;
                    builder.ins().fdiv(lhs, rhs)
                }
                Operation::Scale { factor } => {
                    let input = unary_input(&values, id, &node.inputs)?;
                    let factor = scalar_f32(*factor, id)?;
                    let factor = builder.ins().f32const(Ieee32::with_float(factor));
                    builder.ins().fmul(input, factor)
                }
                Operation::Relu => {
                    let input = unary_input(&values, id, &node.inputs)?;
                    let zero = builder.ins().f32const(Ieee32::with_float(0.0));
                    let positive = builder.ins().fcmp(FloatCC::GreaterThan, input, zero);
                    builder.ins().select(positive, input, zero)
                }
                Operation::ReluGrad => {
                    let (primal, cotangent) = binary_inputs(&values, id, &node.inputs)?;
                    let zero = builder.ins().f32const(Ieee32::with_float(0.0));
                    let positive = builder.ins().fcmp(FloatCC::GreaterThan, primal, zero);
                    builder.ins().select(positive, cotangent, zero)
                }
                Operation::ZerosLike => builder.ins().f32const(Ieee32::with_float(0.0)),
                Operation::OnesLike => builder.ins().f32const(Ieee32::with_float(1.0)),
                Operation::StopGradient
                | Operation::Checkpoint
                | Operation::Reshape { .. }
                | Operation::Transpose { .. }
                | Operation::BroadcastTo { .. }
                | Operation::ReduceSumTo { .. } => unary_input(&values, id, &node.inputs)?,
                Operation::Exp
                | Operation::Log
                | Operation::MatMul
                | Operation::BatchMatMul => {
                    return Err(NativeCompileError::new(format!(
                        "native scalar lane does not yet lower node {} operation {:?}",
                        id.get(), node.operation
                    )));
                }
                _ => {
                    return Err(NativeCompileError::new(format!(
                        "native scalar lane does not yet lower node {} operation {:?}",
                        id.get(), node.operation
                    )));
                }
            };
            values[index] = Some(value);
        }

        let output = value_for(&values, graph.outputs()[0])?;
        builder.ins().return_(&[output]);
        builder.seal_all_blocks();
        builder.finalize(frontend_config);
    }

    module
        .define_function(function_id, &mut context)
        .map_err(|error| NativeCompileError::new(format!("define JIT function: {error}")))?;
    module
        .finalize_definitions()
        .map_err(|error| NativeCompileError::new(format!("finalize JIT function: {error}")))?;

    let code = module.get_finalized_function(function_id);
    let entry = unsafe {
        // SAFETY: the Cranelift signature above is exactly `(F32, F32) -> F32`
        // using the host calling convention selected by JITModule. `module` is
        // retained inside `NativeScalarFn2`, so the executable memory outlives
        // every call through this pointer.
        mem::transmute::<*const u8, extern "C" fn(f32, f32) -> f32>(code)
    };

    Ok(NativeScalarFn2 {
        entry,
        _module: module,
    })
}

fn require_scalar_f32(node: NodeId, ty: &TensorType) -> Result<(), NativeCompileError> {
    if ty.dtype != DType::F32 {
        return Err(NativeCompileError::new(format!(
            "node {} has unsupported dtype {:?}; native prototype is F32-only",
            node.get(), ty.dtype
        )));
    }
    let elements = ty
        .shape
        .dims()
        .iter()
        .try_fold(1usize, |count, &dimension| count.checked_mul(dimension))
        .ok_or_else(|| NativeCompileError::new(format!("node {} shape overflows", node.get())))?;
    if elements != 1 {
        return Err(NativeCompileError::new(format!(
            "node {} has {} elements; native prototype is scalar-only",
            node.get(), elements
        )));
    }
    Ok(())
}

fn scalar_f32(value: Scalar, node: NodeId) -> Result<f32, NativeCompileError> {
    value.as_f32().ok_or_else(|| {
        NativeCompileError::new(format!(
            "node {} has a non-F32 scalar attribute",
            node.get()
        ))
    })
}

fn unary_input(
    values: &[Option<Value>],
    node: NodeId,
    inputs: &[NodeId],
) -> Result<Value, NativeCompileError> {
    inputs
        .first()
        .copied()
        .ok_or_else(|| NativeCompileError::new(format!("node {} lacks unary input", node.get())))
        .and_then(|input| value_for(values, input))
}

fn binary_inputs(
    values: &[Option<Value>],
    node: NodeId,
    inputs: &[NodeId],
) -> Result<(Value, Value), NativeCompileError> {
    if inputs.len() != 2 {
        return Err(NativeCompileError::new(format!(
            "node {} expected two inputs, found {}",
            node.get(),
            inputs.len()
        )));
    }
    Ok((value_for(values, inputs[0])?, value_for(values, inputs[1])?))
}

fn value_for(values: &[Option<Value>], id: NodeId) -> Result<Value, NativeCompileError> {
    values
        .get(id.get() as usize)
        .and_then(|value| *value)
        .ok_or_else(|| {
            NativeCompileError::new(format!("node {} references unavailable value", id.get()))
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirust_tensor_ir::{MorphoDiff, Shape};

    fn scalar_type() -> TensorType {
        TensorType::new(DType::F32, Shape::new(vec![1]))
    }

    fn rosenbrock_gradient() -> (Graph, BTreeMap<ConstantId, f32>) {
        let ty = scalar_type();
        let mut graph = Graph::new();
        let x = graph.add_input("x", ty.clone()).unwrap();
        let y = graph.add_input("y", ty.clone()).unwrap();
        let one_id = ConstantId::new(0);
        let one = graph.add_constant(one_id, ty.clone()).unwrap();
        let one_minus_x = graph
            .add_node(Operation::Sub, vec![one, x], ty.clone())
            .unwrap();
        let first = graph
            .add_node(
                Operation::Mul,
                vec![one_minus_x, one_minus_x],
                ty.clone(),
            )
            .unwrap();
        let x_squared = graph
            .add_node(Operation::Mul, vec![x, x], ty.clone())
            .unwrap();
        let residual = graph
            .add_node(Operation::Sub, vec![y, x_squared], ty.clone())
            .unwrap();
        let residual_squared = graph
            .add_node(Operation::Mul, vec![residual, residual], ty.clone())
            .unwrap();
        let weighted = graph
            .add_node(
                Operation::Scale {
                    factor: Scalar::f32(100.0),
                },
                vec![residual_squared],
                ty.clone(),
            )
            .unwrap();
        let output = graph
            .add_node(Operation::Add, vec![first, weighted], ty)
            .unwrap();
        graph.set_outputs(vec![output]).unwrap();

        let differentiated = MorphoDiff::grad(&graph, output, &[x]).unwrap();
        let mut constants = BTreeMap::new();
        constants.insert(one_id, 1.0);
        (differentiated.graph, constants)
    }

    fn analytic_dx(x: f32, y: f32) -> f32 {
        -2.0 * (1.0 - x) - 400.0 * x * (y - x * x)
    }

    #[test]
    fn native_rosenbrock_gradient_matches_analytic_oracle() {
        let (graph, constants) = rosenbrock_gradient();
        let native = compile_scalar_f32_2_to_1(&graph, &constants).unwrap();

        for (x, y) in [
            (1.0f32, 1.0f32),
            (3.0, 1.0),
            (-1.25, 0.75),
            (0.5, -0.25),
            (2.25, 4.0),
            (-0.75, 0.1),
        ] {
            let expected = analytic_dx(x, y);
            let actual = native.call(x, y);
            let tolerance = 2.0e-4 * (1.0 + expected.abs());
            assert!(
                (actual - expected).abs() <= tolerance,
                "x={x}, y={y}: native={actual}, analytic={expected}"
            );
        }
    }
}
