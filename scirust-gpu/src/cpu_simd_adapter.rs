//! SIMD-accelerated CPU adapter for canonical Reference plans.
//!
//! This is intentionally a separate backend from [`crate::CpuComputeAdapter`].
//! The latter remains SciRust's deterministic scalar oracle. This adapter keeps
//! the same Reference module/buffer contract but routes matrix products through
//! `scirust-simd::gemm::sgemm_tiled` (AVX-512, NEON, or its scalar fallback).

use alloc::{string::ToString, vec, vec::Vec};

use scirust_compute::{
    BufferBinding, ComputeBackend, ComputeError, ComputeResult, DType, DeviceCapabilities,
    DeviceId, KernelFormat, KernelModule, LaunchConfig, MemorySpace,
};
use scirust_simd::gemm::sgemm_tiled;
use scirust_simd::matrix::view::{MatrixView, MatrixViewMut};
use scirust_tensor_reference::PreparedReferenceKernel;

use crate::compute_adapter::{CpuBuffer, CpuComputeAdapter};
use crate::cpu_reference;
use crate::cpu_simd_reference::{self, CpuSimdMatrixSpec};
use crate::{BackendError, BackendResult, RawComputeBackend};

/// Optimized host adapter for MorphoDiff and canonical Tensor-IR CPU execution.
#[derive(Debug)]
pub struct CpuSimdComputeAdapter {
    reference: CpuComputeAdapter,
    capabilities: DeviceCapabilities,
}

impl CpuSimdComputeAdapter {
    pub fn new() -> Self {
        Self {
            reference: CpuComputeAdapter::new(),
            capabilities: DeviceCapabilities {
                device: DeviceId::cpu(),
                name: "scirust-gpu-cpu-simd".to_string(),
                supported_dtypes: vec![DType::F32],
                max_buffer_bytes: None,
                max_workgroup_size: [1, 1, 1],
                supports_async_execution: false,
            },
        }
    }

    pub const fn capabilities(&self) -> &DeviceCapabilities {
        &self.capabilities
    }
}

impl Default for CpuSimdComputeAdapter {
    fn default() -> Self {
        Self::new()
    }
}

/// Reference kernel compiled for the optimized CPU adapter.
///
/// `matrix` is populated only for MatMul/BatchMatMul and is derived once at
/// compilation. Non-matrix kernels reuse the exact scalar Reference launcher.
#[derive(Debug, Clone)]
pub struct CpuSimdKernel {
    module: KernelModule,
    prepared: PreparedReferenceKernel,
    matrix: Option<CpuSimdMatrixSpec>,
}

impl PartialEq for CpuSimdKernel {
    fn eq(&self, other: &Self) -> bool {
        self.module == other.module
    }
}

impl Eq for CpuSimdKernel {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuSimdStream(());

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuSimdEvent(());

impl RawComputeBackend for CpuSimdComputeAdapter {
    fn device_name(&self) -> &'static str {
        "cpu-simd"
    }

    fn gemm_f32(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> BackendResult<Vec<f32>> {
        let a_expected = m
            .checked_mul(k)
            .ok_or_else(|| BackendError::ShapeMismatch("A shape overflows usize".to_string()))?;
        let b_expected = k
            .checked_mul(n)
            .ok_or_else(|| BackendError::ShapeMismatch("B shape overflows usize".to_string()))?;
        let output_len = m
            .checked_mul(n)
            .ok_or_else(|| BackendError::ShapeMismatch("C shape overflows usize".to_string()))?;
        if a.len() != a_expected || b.len() != b_expected {
            return Err(BackendError::ShapeMismatch(format!(
                "SIMD GEMM needs A={a_expected} and B={b_expected} elements, got A={} B={}",
                a.len(),
                b.len()
            )));
        }

        let mut output = vec![0.0f32; output_len];
        sgemm_tiled(
            1.0,
            MatrixView::new(a, m, k),
            MatrixView::new(b, k, n),
            0.0,
            MatrixViewMut::new(&mut output, m, n),
        );
        Ok(output)
    }
}

impl ComputeBackend for CpuSimdComputeAdapter {
    type Buffer = CpuBuffer;
    type Kernel = CpuSimdKernel;
    type Stream = CpuSimdStream;
    type Event = CpuSimdEvent;

    fn capabilities(&self) -> &DeviceCapabilities {
        &self.capabilities
    }

    fn hardware_capabilities(&self) -> scirust_compute::HardwareCapabilities {
        self.reference.hardware_capabilities()
    }

    fn allocate(
        &self,
        bytes: usize,
        alignment: usize,
        memory_space: MemorySpace,
    ) -> ComputeResult<Self::Buffer> {
        self.reference.allocate(bytes, alignment, memory_space)
    }

    fn write(
        &self,
        destination: &Self::Buffer,
        offset_bytes: usize,
        data: &[u8],
    ) -> ComputeResult<()> {
        self.reference.write(destination, offset_bytes, data)
    }

    fn read(
        &self,
        source: &Self::Buffer,
        offset_bytes: usize,
        destination: &mut [u8],
    ) -> ComputeResult<()> {
        self.reference.read(source, offset_bytes, destination)
    }

    fn compile(&self, module: &KernelModule) -> ComputeResult<Self::Kernel> {
        if module.format != KernelFormat::Reference {
            return Err(ComputeError::Unsupported(
                "SIMD CPU adapter accepts reference kernels only",
            ));
        }
        let prepared = PreparedReferenceKernel::from_kernel_module(module)
            .map_err(|error| ComputeError::Compilation(error.to_string()))?;
        let matrix = cpu_simd_reference::matrix_spec_from_module(module)
            .map_err(|error| ComputeError::Compilation(error.to_string()))?;
        Ok(CpuSimdKernel {
            module: module.clone(),
            prepared,
            matrix,
        })
    }

    fn create_stream(&self) -> ComputeResult<Self::Stream> {
        Ok(CpuSimdStream(()))
    }

    fn launch(
        &self,
        kernel: &Self::Kernel,
        _stream: &Self::Stream,
        config: LaunchConfig,
        bindings: &[BufferBinding<'_, Self::Buffer>],
    ) -> ComputeResult<Self::Event> {
        match kernel.matrix {
            Some(spec) => cpu_simd_reference::launch_matrix_kernel(
                &kernel.prepared,
                spec,
                config,
                bindings,
            )
            .map_err(|error| ComputeError::Launch(error.to_string()))?,
            None => cpu_reference::launch_reference_kernel(&kernel.prepared, config, bindings)
                .map_err(|error| ComputeError::Launch(cpu_reference::describe(&error)))?,
        }
        Ok(CpuSimdEvent(()))
    }

    fn wait(&self, _event: &Self::Event) -> ComputeResult<()> {
        Ok(())
    }

    fn synchronize(&self, _stream: &Self::Stream) -> ComputeResult<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simd_adapter_keeps_a_distinct_public_identity() {
        let adapter = CpuSimdComputeAdapter::new();
        assert_eq!(adapter.device_name(), "cpu-simd");
        assert_eq!(adapter.capabilities().name, "scirust-gpu-cpu-simd");
        assert!(adapter.capabilities().supports_dtype(DType::F32));
    }

    #[test]
    fn raw_simd_gemm_matches_known_product() {
        let adapter = CpuSimdComputeAdapter::new();
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
        assert_eq!(
            adapter.gemm_f32(&a, &b, 2, 3, 2).unwrap(),
            vec![58.0, 64.0, 139.0, 154.0]
        );
    }
}
