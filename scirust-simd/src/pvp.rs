//! Pascal Vector Projection (PVP) address-major bitplane primitives.
//!
//! PVP stores the same Boolean address across many independent gates next to
//! each other:
//!
//! ```text
//! address 0: [gate bits 0..G)
//! address 1: [gate bits 0..G)
//! ...
//! ```
//!
//! The packed CPU representation is `[K, ceil(G / 64)] u64`, where `K` is
//! the number of Boolean addresses and `G` is the number of independent
//! gates/functions. This makes the gate-word dimension contiguous and turns
//! subset-zeta/Pascal execution into regular XOR butterflies without graph
//! nodes, adjacency lists, or pointer chasing.
//!
//! PVP is representation/execution infrastructure only. It does not attach
//! model, attention, or ANF policy semantics to the bits.

use core::fmt;

/// Schema version of the address-major PVP layout.
pub const PVP_LAYOUT_SCHEMA_VERSION: u32 = 1;

/// Physical word width of the portable CPU reference representation.
pub const PVP_WORD_BITS: usize = u64::BITS as usize;

/// Stable CPU execution candidates for the PVP XOR butterfly.
///
/// PVP-1 keeps the representation fixed and changes only how one contiguous
/// destination gate-word row is XORed with its source row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PvpBackendV1 {
    /// Portable scalar-u64 reference.
    Scalar,
    /// x86_64 AVX2, four u64 gate words per vector.
    Avx2,
    /// x86_64 AVX-512F, eight u64 gate words per vector.
    Avx512,
    /// aarch64 NEON, two u64 gate words per vector.
    Neon,
    /// aarch64 SVE, scalable u64 vector length selected at runtime.
    #[cfg(feature = "nightly-simd")]
    Sve,
}

impl PvpBackendV1 {
    /// Stable backend label used in evidence records.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self
        {
            Self::Scalar => "scalar-u64",
            Self::Avx2 => "x86_64-avx2-u64x4",
            Self::Avx512 => "x86_64-avx512-u64x8",
            Self::Neon => "aarch64-neon-u64x2",
            #[cfg(feature = "nightly-simd")]
            Self::Sve => "aarch64-sve-scalable-u64",
        }
    }

    /// Fixed vector width in bits for the candidate.
    ///
    /// Returns 0 for scalable-vector candidates whose width is a runtime
    /// hardware property rather than a compile-time constant.
    #[must_use]
    pub const fn register_bits(self) -> usize {
        match self
        {
            Self::Scalar => 64,
            Self::Avx2 => 256,
            Self::Avx512 => 512,
            Self::Neon => 128,
            #[cfg(feature = "nightly-simd")]
            Self::Sve => 0,
        }
    }

    /// Whether this backend is executable on the current process/CPU.
    #[must_use]
    pub fn available(self) -> bool {
        match self
        {
            Self::Scalar => true,
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 => std::arch::is_x86_feature_detected!("avx2"),
            #[cfg(not(target_arch = "x86_64"))]
            Self::Avx2 => false,
            #[cfg(target_arch = "x86_64")]
            Self::Avx512 => std::arch::is_x86_feature_detected!("avx512f"),
            #[cfg(not(target_arch = "x86_64"))]
            Self::Avx512 => false,
            #[cfg(target_arch = "aarch64")]
            Self::Neon => std::arch::is_aarch64_feature_detected!("neon"),
            #[cfg(not(target_arch = "aarch64"))]
            Self::Neon => false,
            #[cfg(all(feature = "nightly-simd", target_arch = "aarch64"))]
            Self::Sve => std::arch::is_aarch64_feature_detected!("sve"),
            #[cfg(all(feature = "nightly-simd", not(target_arch = "aarch64")))]
            Self::Sve => false,
        }
    }
}

/// Detect the widest stable PVP CPU backend available on this process.
#[must_use]
pub fn detect_best_pvp_backend_v1() -> PvpBackendV1 {
    #[cfg(target_arch = "x86_64")]
    {
        if PvpBackendV1::Avx512.available()
        {
            return PvpBackendV1::Avx512;
        }
        if PvpBackendV1::Avx2.available()
        {
            return PvpBackendV1::Avx2;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        #[cfg(feature = "nightly-simd")]
        if PvpBackendV1::Sve.available()
        {
            return PvpBackendV1::Sve;
        }
        if PvpBackendV1::Neon.available()
        {
            return PvpBackendV1::Neon;
        }
    }

    PvpBackendV1::Scalar
}

/// Errors raised by checked PVP layout/storage operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PvpError {
    /// A PVP domain must contain at least one address.
    ZeroAddresses,
    /// Subset-zeta/Pascal stages require a power-of-two address domain.
    AddressesNotPowerOfTwo {
        /// Rejected address count.
        addresses: usize,
    },
    /// A PVP bank must contain at least one gate.
    ZeroGates,
    /// Checked layout/storage arithmetic overflowed.
    SizeOverflow,
    /// A packed buffer does not match the declared layout.
    StorageLengthMismatch {
        /// Expected number of u64 words.
        expected_words: usize,
        /// Observed number of u64 words.
        actual_words: usize,
    },
    /// A logical address was outside the declared domain.
    AddressOutOfRange {
        /// Requested address.
        address: usize,
        /// Declared address count.
        addresses: usize,
    },
    /// A logical gate was outside the declared bank.
    GateOutOfRange {
        /// Requested gate.
        gate: usize,
        /// Declared gate count.
        gates: usize,
    },
    /// Canonical padding bits outside the declared logical domain were nonzero.
    NonZeroPadding {
        /// Logical row/gate whose packed tail contained nonzero padding.
        major_index: usize,
    },
    /// A requested architecture candidate is unavailable on this process/CPU.
    BackendUnavailable {
        /// Rejected candidate.
        backend: PvpBackendV1,
    },
}

impl fmt::Display for PvpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::ZeroAddresses => write!(f, "PVP address count must be nonzero"),
            Self::AddressesNotPowerOfTwo { addresses } =>
            {
                write!(f, "PVP address count {addresses} is not a power of two")
            },
            Self::ZeroGates => write!(f, "PVP gate count must be nonzero"),
            Self::SizeOverflow => write!(f, "PVP layout/storage size overflow"),
            Self::StorageLengthMismatch {
                expected_words,
                actual_words,
            } => write!(
                f,
                "PVP storage length mismatch: expected {expected_words} u64 words, got {actual_words}"
            ),
            Self::AddressOutOfRange { address, addresses } =>
            {
                write!(f, "PVP address {address} is outside 0..{addresses}")
            },
            Self::GateOutOfRange { gate, gates } =>
            {
                write!(f, "PVP gate {gate} is outside 0..{gates}")
            },
            Self::NonZeroPadding { major_index } => write!(
                f,
                "PVP canonical padding is nonzero at major index {major_index}"
            ),
            Self::BackendUnavailable { backend } =>
            {
                write!(f, "PVP backend {} is unavailable", backend.label())
            },
        }
    }
}

impl std::error::Error for PvpError {}

/// Versioned address-major PVP layout.
///
/// Logical shape is `[addresses, gates]`. Physical CPU storage is
/// `[addresses, gate_words_per_address]` in row-major order with each word
/// carrying up to 64 independent gates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PvpLayoutV1 {
    addresses: usize,
    gates: usize,
    gate_words_per_address: usize,
    address_words_per_gate: usize,
    storage_words: usize,
    logical_bits: usize,
    storage_bits: usize,
}

impl PvpLayoutV1 {
    /// Construct a checked PVP layout.
    pub fn new(addresses: usize, gates: usize) -> Result<Self, PvpError> {
        if addresses == 0
        {
            return Err(PvpError::ZeroAddresses);
        }
        if !addresses.is_power_of_two()
        {
            return Err(PvpError::AddressesNotPowerOfTwo { addresses });
        }
        if gates == 0
        {
            return Err(PvpError::ZeroGates);
        }

        let gate_words_per_address = gates.div_ceil(PVP_WORD_BITS);
        let address_words_per_gate = addresses.div_ceil(PVP_WORD_BITS);
        let storage_words = addresses
            .checked_mul(gate_words_per_address)
            .ok_or(PvpError::SizeOverflow)?;
        let logical_bits = addresses.checked_mul(gates).ok_or(PvpError::SizeOverflow)?;
        let storage_bits = storage_words
            .checked_mul(PVP_WORD_BITS)
            .ok_or(PvpError::SizeOverflow)?;

        Ok(Self {
            addresses,
            gates,
            gate_words_per_address,
            address_words_per_gate,
            storage_words,
            logical_bits,
            storage_bits,
        })
    }

    /// Number of Boolean addresses (`K`).
    #[must_use]
    pub const fn addresses(self) -> usize {
        self.addresses
    }

    /// Number of independent gates/functions (`G`).
    #[must_use]
    pub const fn gates(self) -> usize {
        self.gates
    }

    /// Packed u64 words in one address-major row.
    #[must_use]
    pub const fn gate_words_per_address(self) -> usize {
        self.gate_words_per_address
    }

    /// Packed address words in one canonical gate-major input row.
    #[must_use]
    pub const fn address_words_per_gate(self) -> usize {
        self.address_words_per_gate
    }

    /// Total u64 words in the address-major PVP buffer.
    #[must_use]
    pub const fn storage_words(self) -> usize {
        self.storage_words
    }

    /// Total u64 words in a canonical gate-major packed buffer.
    pub fn gate_major_storage_words(self) -> Result<usize, PvpError> {
        self.gates
            .checked_mul(self.address_words_per_gate)
            .ok_or(PvpError::SizeOverflow)
    }

    /// Exact logical payload bits (`K * G`).
    #[must_use]
    pub const fn logical_bits(self) -> usize {
        self.logical_bits
    }

    /// Exact physical storage bits including tail padding.
    #[must_use]
    pub const fn storage_bits(self) -> usize {
        self.storage_bits
    }

    /// Tail padding bits introduced by 64-bit gate words.
    #[must_use]
    pub const fn padding_bits(self) -> usize {
        self.storage_bits - self.logical_bits
    }

    /// Number of subset-zeta butterfly stages.
    #[must_use]
    pub const fn stages(self) -> u32 {
        self.addresses.ilog2()
    }

    /// Stable text identity of the layout contract.
    #[must_use]
    pub fn canonical_record(self) -> String {
        format!(
            "pvp-bitplanes/v{};addresses={};gates={};word_bits={};order=address-major-gate-word;words_per_address={};storage_bits={};padding_bits={}",
            PVP_LAYOUT_SCHEMA_VERSION,
            self.addresses,
            self.gates,
            PVP_WORD_BITS,
            self.gate_words_per_address,
            self.storage_bits,
            self.padding_bits()
        )
    }

    fn word_index(self, address: usize, gate_word: usize) -> Result<usize, PvpError> {
        if address >= self.addresses
        {
            return Err(PvpError::AddressOutOfRange {
                address,
                addresses: self.addresses,
            });
        }
        address
            .checked_mul(self.gate_words_per_address)
            .and_then(|base| base.checked_add(gate_word))
            .ok_or(PvpError::SizeOverflow)
    }
}

/// Owned canonical address-major PVP bitplane buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PvpBitplanesV1 {
    layout: PvpLayoutV1,
    words: Vec<u64>,
}

impl PvpBitplanesV1 {
    /// Allocate a zeroed canonical buffer.
    pub fn zeroed(layout: PvpLayoutV1) -> Self {
        Self {
            layout,
            words: vec![0; layout.storage_words()],
        }
    }

    /// Construct from an already address-major packed buffer.
    ///
    /// Canonical tail padding must be zero.
    pub fn from_words(layout: PvpLayoutV1, words: Vec<u64>) -> Result<Self, PvpError> {
        if words.len() != layout.storage_words()
        {
            return Err(PvpError::StorageLengthMismatch {
                expected_words: layout.storage_words(),
                actual_words: words.len(),
            });
        }
        let value = Self { layout, words };
        value.validate_padding_zero()?;
        Ok(value)
    }

    /// Transpose canonical gate-major packed bits into address-major PVP words.
    ///
    /// Gate-major input shape is `[G, ceil(K/64)] u64`; address-major output
    /// shape is `[K, ceil(G/64)] u64`.
    pub fn from_gate_major_words(
        layout: PvpLayoutV1,
        gate_major: &[u64],
    ) -> Result<Self, PvpError> {
        validate_gate_major_storage(layout, gate_major)?;

        let mut output = Self::zeroed(layout);
        for gate in 0..layout.gates()
        {
            let source_base = gate
                .checked_mul(layout.address_words_per_gate())
                .ok_or(PvpError::SizeOverflow)?;
            for address in 0..layout.addresses()
            {
                let source_word = gate_major[source_base + address / PVP_WORD_BITS];
                if source_word & (1_u64 << (address % PVP_WORD_BITS)) != 0
                {
                    let target_word = layout.word_index(address, gate / PVP_WORD_BITS)?;
                    output.words[target_word] |= 1_u64 << (gate % PVP_WORD_BITS);
                }
            }
        }
        Ok(output)
    }

    /// Transpose this address-major PVP buffer back to canonical gate-major
    /// packed storage.
    pub fn to_gate_major_words(&self) -> Result<Vec<u64>, PvpError> {
        self.validate_padding_zero()?;
        let mut output = vec![0_u64; self.layout.gate_major_storage_words()?];

        for address in 0..self.layout.addresses()
        {
            let row_base = address
                .checked_mul(self.layout.gate_words_per_address())
                .ok_or(PvpError::SizeOverflow)?;
            for gate in 0..self.layout.gates()
            {
                if self.words[row_base + gate / PVP_WORD_BITS] & (1_u64 << (gate % PVP_WORD_BITS))
                    != 0
                {
                    let target = gate
                        .checked_mul(self.layout.address_words_per_gate())
                        .and_then(|base| base.checked_add(address / PVP_WORD_BITS))
                        .ok_or(PvpError::SizeOverflow)?;
                    output[target] |= 1_u64 << (address % PVP_WORD_BITS);
                }
            }
        }

        Ok(output)
    }

    /// Layout carried by this buffer.
    #[must_use]
    pub const fn layout(&self) -> PvpLayoutV1 {
        self.layout
    }

    /// Read one logical bit.
    pub fn get(&self, address: usize, gate: usize) -> Result<bool, PvpError> {
        if gate >= self.layout.gates()
        {
            return Err(PvpError::GateOutOfRange {
                gate,
                gates: self.layout.gates(),
            });
        }
        let word = self.layout.word_index(address, gate / PVP_WORD_BITS)?;
        Ok(self.words[word] & (1_u64 << (gate % PVP_WORD_BITS)) != 0)
    }

    /// Set one logical bit.
    pub fn set(&mut self, address: usize, gate: usize, value: bool) -> Result<(), PvpError> {
        if gate >= self.layout.gates()
        {
            return Err(PvpError::GateOutOfRange {
                gate,
                gates: self.layout.gates(),
            });
        }
        let word = self.layout.word_index(address, gate / PVP_WORD_BITS)?;
        let mask = 1_u64 << (gate % PVP_WORD_BITS);
        if value
        {
            self.words[word] |= mask;
        }
        else
        {
            self.words[word] &= !mask;
        }
        Ok(())
    }

    /// Read-only packed storage.
    #[must_use]
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    fn validate_padding_zero(&self) -> Result<(), PvpError> {
        let tail = self.layout.gates() % PVP_WORD_BITS;
        if tail == 0
        {
            return Ok(());
        }
        let padding_mask = !((1_u64 << tail) - 1);
        let last = self.layout.gate_words_per_address() - 1;
        for address in 0..self.layout.addresses()
        {
            let index = self.layout.word_index(address, last)?;
            if self.words[index] & padding_mask != 0
            {
                return Err(PvpError::NonZeroPadding {
                    major_index: address,
                });
            }
        }
        Ok(())
    }
}

/// Exact accounting returned by the scalar Pascal/subset-zeta transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PvpTransformStatsV1 {
    /// CPU realization used for this transform.
    pub backend: PvpBackendV1,
    /// Address count.
    pub addresses: usize,
    /// Gate count.
    pub gates: usize,
    /// Butterfly stages.
    pub stages: u32,
    /// Logical single-gate XOR updates.
    pub logical_gate_xor_ops: u128,
    /// Physical packed-u64 destination updates.
    pub packed_word_updates: u128,
    /// Physical state words.
    pub storage_words: usize,
    /// Physical state bits including padding.
    pub storage_bits: usize,
    /// Padding bits.
    pub padding_bits: usize,
    /// Algorithmic scratch words.
    pub scratch_words: usize,
}

/// Apply the subset-zeta/Pascal transform in place to every gate bitplane.
///
/// The transform is self-inverse over GF(2). The implementation performs no
/// heap allocation and uses no algorithmic scratch after the buffer already
/// exists.
pub fn pascal_subset_zeta_scalar_in_place(
    bitplanes: &mut PvpBitplanesV1,
) -> Result<PvpTransformStatsV1, PvpError> {
    pascal_subset_zeta_with_backend_in_place(bitplanes, PvpBackendV1::Scalar)
}

/// Apply the PVP subset-zeta transform with the best stable CPU candidate
/// available on the current process.
pub fn pascal_subset_zeta_auto_in_place(
    bitplanes: &mut PvpBitplanesV1,
) -> Result<PvpTransformStatsV1, PvpError> {
    pascal_subset_zeta_with_backend_in_place(bitplanes, detect_best_pvp_backend_v1())
}

/// Apply the exact same PVP transform with an explicitly selected CPU backend.
///
/// Candidate availability is checked before any state mutation. SIMD candidates
/// only vectorize the contiguous inner gate-word XOR; stage order, address
/// pairing, storage, padding and operation accounting remain identical to the
/// scalar PVP-0 reference.
pub fn pascal_subset_zeta_with_backend_in_place(
    bitplanes: &mut PvpBitplanesV1,
    backend: PvpBackendV1,
) -> Result<PvpTransformStatsV1, PvpError> {
    if !backend.available()
    {
        return Err(PvpError::BackendUnavailable { backend });
    }

    match backend
    {
        PvpBackendV1::Scalar => transform_with_row_xor(bitplanes, backend, xor_rows_scalar),
        #[cfg(target_arch = "x86_64")]
        PvpBackendV1::Avx2 =>
        {
            transform_with_row_xor(bitplanes, backend, |words, src, dst, len| unsafe {
                xor_rows_avx2(words, src, dst, len)
            })
        },
        #[cfg(not(target_arch = "x86_64"))]
        PvpBackendV1::Avx2 => unreachable!("AVX2 availability is false on this target"),
        #[cfg(target_arch = "x86_64")]
        PvpBackendV1::Avx512 =>
        {
            transform_with_row_xor(bitplanes, backend, |words, src, dst, len| unsafe {
                xor_rows_avx512(words, src, dst, len)
            })
        },
        #[cfg(not(target_arch = "x86_64"))]
        PvpBackendV1::Avx512 => unreachable!("AVX-512 availability is false on this target"),
        #[cfg(target_arch = "aarch64")]
        PvpBackendV1::Neon =>
        {
            transform_with_row_xor(bitplanes, backend, |words, src, dst, len| unsafe {
                xor_rows_neon(words, src, dst, len)
            })
        },
        #[cfg(not(target_arch = "aarch64"))]
        PvpBackendV1::Neon => unreachable!("NEON availability is false on this target"),
        #[cfg(all(feature = "nightly-simd", target_arch = "aarch64"))]
        PvpBackendV1::Sve =>
        {
            transform_with_row_xor(bitplanes, backend, |words, src, dst, len| unsafe {
                xor_rows_sve(words, src, dst, len)
            })
        },
        #[cfg(all(feature = "nightly-simd", not(target_arch = "aarch64")))]
        PvpBackendV1::Sve => unreachable!("SVE availability is false on this target"),
    }
}

fn transform_with_row_xor<F>(
    bitplanes: &mut PvpBitplanesV1,
    backend: PvpBackendV1,
    mut xor_rows: F,
) -> Result<PvpTransformStatsV1, PvpError>
where
    F: FnMut(&mut [u64], usize, usize, usize),
{
    bitplanes.validate_padding_zero()?;
    let layout = bitplanes.layout;
    let words_per_address = layout.gate_words_per_address();

    let mut stride = 1_usize;
    while stride < layout.addresses()
    {
        let block = stride.checked_mul(2).ok_or(PvpError::SizeOverflow)?;
        let mut block_start = 0_usize;
        while block_start < layout.addresses()
        {
            for offset in 0..stride
            {
                let source_address = block_start + offset;
                let target_address = source_address + stride;
                let source_base = source_address
                    .checked_mul(words_per_address)
                    .ok_or(PvpError::SizeOverflow)?;
                let target_base = target_address
                    .checked_mul(words_per_address)
                    .ok_or(PvpError::SizeOverflow)?;
                xor_rows(
                    &mut bitplanes.words,
                    source_base,
                    target_base,
                    words_per_address,
                );
            }
            block_start = block_start
                .checked_add(block)
                .ok_or(PvpError::SizeOverflow)?;
        }
        stride = block;
    }

    bitplanes.validate_padding_zero()?;

    let butterfly_pairs = (layout.addresses() / 2) as u128 * u128::from(layout.stages());
    Ok(PvpTransformStatsV1 {
        backend,
        addresses: layout.addresses(),
        gates: layout.gates(),
        stages: layout.stages(),
        logical_gate_xor_ops: butterfly_pairs * layout.gates() as u128,
        packed_word_updates: butterfly_pairs * words_per_address as u128,
        storage_words: layout.storage_words(),
        storage_bits: layout.storage_bits(),
        padding_bits: layout.padding_bits(),
        scratch_words: 0,
    })
}

fn xor_rows_scalar(words: &mut [u64], source_base: usize, target_base: usize, len: usize) {
    debug_assert!(source_base + len <= target_base);
    debug_assert!(target_base + len <= words.len());
    let (before_target, target_and_after) = words.split_at_mut(target_base);
    let source = &before_target[source_base..source_base + len];
    let target = &mut target_and_after[..len];
    for (dst, &src) in target.iter_mut().zip(source)
    {
        *dst ^= src;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn xor_rows_avx2(words: &mut [u64], source_base: usize, target_base: usize, len: usize) {
    use core::arch::x86_64::*;

    debug_assert!(source_base + len <= target_base);
    debug_assert!(target_base + len <= words.len());

    let base = words.as_mut_ptr();
    let mut i = 0_usize;
    while i + 4 <= len
    {
        let source = _mm256_loadu_si256(base.add(source_base + i).cast::<__m256i>());
        let target = _mm256_loadu_si256(base.add(target_base + i).cast::<__m256i>());
        _mm256_storeu_si256(
            base.add(target_base + i).cast::<__m256i>(),
            _mm256_xor_si256(target, source),
        );
        i += 4;
    }
    while i < len
    {
        *base.add(target_base + i) ^= *base.add(source_base + i);
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xor_rows_avx512(words: &mut [u64], source_base: usize, target_base: usize, len: usize) {
    use core::arch::x86_64::*;

    debug_assert!(source_base + len <= target_base);
    debug_assert!(target_base + len <= words.len());

    let base = words.as_mut_ptr();
    let mut i = 0_usize;
    while i + 8 <= len
    {
        let source = _mm512_loadu_si512(base.add(source_base + i).cast::<__m512i>());
        let target = _mm512_loadu_si512(base.add(target_base + i).cast::<__m512i>());
        _mm512_storeu_si512(
            base.add(target_base + i).cast::<__m512i>(),
            _mm512_xor_si512(target, source),
        );
        i += 8;
    }
    while i < len
    {
        *base.add(target_base + i) ^= *base.add(source_base + i);
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn xor_rows_neon(words: &mut [u64], source_base: usize, target_base: usize, len: usize) {
    use core::arch::aarch64::*;

    debug_assert!(source_base + len <= target_base);
    debug_assert!(target_base + len <= words.len());

    let base = words.as_mut_ptr();
    let mut i = 0_usize;
    while i + 2 <= len
    {
        let source = vld1q_u64(base.add(source_base + i));
        let target = vld1q_u64(base.add(target_base + i));
        vst1q_u64(base.add(target_base + i), veorq_u64(target, source));
        i += 2;
    }
    while i < len
    {
        *base.add(target_base + i) ^= *base.add(source_base + i);
        i += 1;
    }
}

#[cfg(all(feature = "nightly-simd", target_arch = "aarch64"))]
#[target_feature(enable = "sve")]
unsafe fn xor_rows_sve(words: &mut [u64], source_base: usize, target_base: usize, len: usize) {
    use core::arch::aarch64::*;

    debug_assert!(source_base + len <= target_base);
    debug_assert!(target_base + len <= words.len());

    let base = words.as_mut_ptr();
    let step = svcntd() as usize;
    let mut i = 0_usize;
    while i < len
    {
        let predicate = svwhilelt_b64_u64(i as u64, len as u64);
        let source = svld1_u64(predicate, base.add(source_base + i));
        let target = svld1_u64(predicate, base.add(target_base + i));
        let value = sveor_u64_x(predicate, target, source);
        svst1_u64(predicate, base.add(target_base + i), value);
        i += step;
    }
}

/// Independent direct subset-enumeration oracle.
///
/// This intentionally does not use the butterfly implementation. Its aggregate
/// work grows as the full submask relation and it is therefore intended for
/// small/medium correctness fixtures, not large production domains.
pub fn direct_subset_zeta_oracle_v1(
    layout: PvpLayoutV1,
    gate_major: &[u64],
) -> Result<PvpBitplanesV1, PvpError> {
    validate_gate_major_storage(layout, gate_major)?;
    let mut output = PvpBitplanesV1::zeroed(layout);

    for gate in 0..layout.gates()
    {
        for address in 0..layout.addresses()
        {
            let mut parity = false;
            let mut submask = address;
            loop
            {
                parity ^= gate_major_bit(layout, gate_major, gate, submask)?;
                if submask == 0
                {
                    break;
                }
                submask = (submask - 1) & address;
            }
            if parity
            {
                output.set(address, gate, true)?;
            }
        }
    }

    Ok(output)
}

fn validate_gate_major_storage(layout: PvpLayoutV1, gate_major: &[u64]) -> Result<(), PvpError> {
    let expected_words = layout.gate_major_storage_words()?;
    if gate_major.len() != expected_words
    {
        return Err(PvpError::StorageLengthMismatch {
            expected_words,
            actual_words: gate_major.len(),
        });
    }

    let tail = layout.addresses() % PVP_WORD_BITS;
    if tail == 0
    {
        return Ok(());
    }

    let padding_mask = !((1_u64 << tail) - 1);
    let last = layout.address_words_per_gate() - 1;
    for gate in 0..layout.gates()
    {
        let index = gate
            .checked_mul(layout.address_words_per_gate())
            .and_then(|base| base.checked_add(last))
            .ok_or(PvpError::SizeOverflow)?;
        if gate_major[index] & padding_mask != 0
        {
            return Err(PvpError::NonZeroPadding { major_index: gate });
        }
    }
    Ok(())
}

fn gate_major_bit(
    layout: PvpLayoutV1,
    gate_major: &[u64],
    gate: usize,
    address: usize,
) -> Result<bool, PvpError> {
    if gate >= layout.gates()
    {
        return Err(PvpError::GateOutOfRange {
            gate,
            gates: layout.gates(),
        });
    }
    if address >= layout.addresses()
    {
        return Err(PvpError::AddressOutOfRange {
            address,
            addresses: layout.addresses(),
        });
    }

    let index = gate
        .checked_mul(layout.address_words_per_gate())
        .and_then(|base| base.checked_add(address / PVP_WORD_BITS))
        .ok_or(PvpError::SizeOverflow)?;
    Ok(gate_major[index] & (1_u64 << (address % PVP_WORD_BITS)) != 0)
}

#[cfg(test)]
#[path = "pvp_anf_bank_tests.rs"]
mod anf_bank_tests;

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_gate_major(layout: PvpLayoutV1) -> Vec<u64> {
        let mut words = vec![0_u64; layout.gate_major_storage_words().unwrap()];
        for gate in 0..layout.gates()
        {
            for address in 0..layout.addresses()
            {
                let bit = ((gate * 17 + address * 13 + (gate ^ address)) % 11) < 5;
                if bit
                {
                    let index = gate * layout.address_words_per_gate() + address / PVP_WORD_BITS;
                    words[index] |= 1_u64 << (address % PVP_WORD_BITS);
                }
            }
        }
        words
    }

    #[test]
    fn layout_accounting_is_exact_and_versioned() {
        let layout = PvpLayoutV1::new(128, 65).unwrap();
        assert_eq!(layout.addresses(), 128);
        assert_eq!(layout.gates(), 65);
        assert_eq!(layout.gate_words_per_address(), 2);
        assert_eq!(layout.address_words_per_gate(), 2);
        assert_eq!(layout.storage_words(), 256);
        assert_eq!(layout.logical_bits(), 8_320);
        assert_eq!(layout.storage_bits(), 16_384);
        assert_eq!(layout.padding_bits(), 8_064);
        assert_eq!(layout.stages(), 7);
        assert_eq!(
            layout.canonical_record(),
            "pvp-bitplanes/v1;addresses=128;gates=65;word_bits=64;order=address-major-gate-word;words_per_address=2;storage_bits=16384;padding_bits=8064"
        );
    }

    #[test]
    fn invalid_layouts_fail_closed() {
        assert_eq!(PvpLayoutV1::new(0, 1), Err(PvpError::ZeroAddresses));
        assert_eq!(
            PvpLayoutV1::new(3, 1),
            Err(PvpError::AddressesNotPowerOfTwo { addresses: 3 })
        );
        assert_eq!(PvpLayoutV1::new(8, 0), Err(PvpError::ZeroGates));
    }

    #[test]
    fn gate_major_transpose_round_trips_with_tail_bits() {
        let layout = PvpLayoutV1::new(32, 70).unwrap();
        let source = fixture_gate_major(layout);
        let pvp = PvpBitplanesV1::from_gate_major_words(layout, &source).unwrap();
        assert_eq!(pvp.to_gate_major_words().unwrap(), source);

        for gate in 0..layout.gates()
        {
            for address in 0..layout.addresses()
            {
                assert_eq!(
                    pvp.get(address, gate).unwrap(),
                    gate_major_bit(layout, &source, gate, address).unwrap()
                );
            }
        }
    }

    #[test]
    fn nonzero_canonical_padding_is_rejected() {
        let layout = PvpLayoutV1::new(8, 65).unwrap();
        let mut address_major = vec![0_u64; layout.storage_words()];
        address_major[1] = 1_u64 << 63;
        assert_eq!(
            PvpBitplanesV1::from_words(layout, address_major),
            Err(PvpError::NonZeroPadding { major_index: 0 })
        );

        let mut gate_major = fixture_gate_major(layout);
        gate_major[layout.address_words_per_gate() - 1] |= 1_u64 << 63;
        assert_eq!(
            PvpBitplanesV1::from_gate_major_words(layout, &gate_major),
            Err(PvpError::NonZeroPadding { major_index: 0 })
        );
    }

    #[test]
    fn scalar_butterfly_matches_independent_direct_oracle() {
        let layout = PvpLayoutV1::new(16, 70).unwrap();
        let source = fixture_gate_major(layout);
        let expected = direct_subset_zeta_oracle_v1(layout, &source).unwrap();
        let mut actual = PvpBitplanesV1::from_gate_major_words(layout, &source).unwrap();

        let stats = pascal_subset_zeta_scalar_in_place(&mut actual).unwrap();

        assert_eq!(actual, expected);
        assert_eq!(stats.backend, PvpBackendV1::Scalar);
        assert_eq!(stats.addresses, 16);
        assert_eq!(stats.gates, 70);
        assert_eq!(stats.stages, 4);
        assert_eq!(stats.logical_gate_xor_ops, 2_240);
        assert_eq!(stats.packed_word_updates, 64);
        assert_eq!(stats.storage_words, 32);
        assert_eq!(stats.storage_bits, 2_048);
        assert_eq!(stats.padding_bits, 928);
        assert_eq!(stats.scratch_words, 0);
    }

    #[test]
    fn scalar_butterfly_is_self_inverse() {
        let layout = PvpLayoutV1::new(64, 129).unwrap();
        let source = fixture_gate_major(layout);
        let original = PvpBitplanesV1::from_gate_major_words(layout, &source).unwrap();
        let mut transformed = original.clone();

        pascal_subset_zeta_scalar_in_place(&mut transformed).unwrap();
        pascal_subset_zeta_scalar_in_place(&mut transformed).unwrap();

        assert_eq!(transformed, original);
    }

    #[test]
    fn stable_simd_candidates_match_scalar_reference() {
        let backends = [PvpBackendV1::Avx512, PvpBackendV1::Avx2, PvpBackendV1::Neon];
        let geometries = [(8, 1), (16, 63), (32, 64), (64, 65), (64, 257)];

        for (addresses, gates) in geometries
        {
            let layout = PvpLayoutV1::new(addresses, gates).unwrap();
            let source = fixture_gate_major(layout);
            let mut reference = PvpBitplanesV1::from_gate_major_words(layout, &source).unwrap();
            let scalar_stats = pascal_subset_zeta_scalar_in_place(&mut reference).unwrap();

            for backend in backends
            {
                if !backend.available()
                {
                    continue;
                }
                let mut candidate = PvpBitplanesV1::from_gate_major_words(layout, &source).unwrap();
                let stats =
                    pascal_subset_zeta_with_backend_in_place(&mut candidate, backend).unwrap();
                assert_eq!(candidate, reference, "backend={}", backend.label());
                assert_eq!(stats.backend, backend);
                assert_eq!(
                    stats.logical_gate_xor_ops,
                    scalar_stats.logical_gate_xor_ops
                );
                assert_eq!(stats.packed_word_updates, scalar_stats.packed_word_updates);
                assert_eq!(stats.storage_bits, scalar_stats.storage_bits);
                assert_eq!(stats.padding_bits, scalar_stats.padding_bits);
                assert_eq!(stats.scratch_words, 0);
            }
        }
    }

    #[cfg(all(feature = "nightly-simd", target_arch = "aarch64"))]
    #[test]
    #[ignore = "requires an SVE runtime; executed by the dedicated QEMU/native PVP gate"]
    fn sve_candidate_matches_scalar_reference_on_sve_runtime() {
        assert!(
            PvpBackendV1::Sve.available(),
            "dedicated SVE execution gate requires runtime SVE support"
        );

        let geometries = [(8, 1), (16, 63), (32, 65), (64, 257), (128, 513)];
        for (addresses, gates) in geometries
        {
            let layout = PvpLayoutV1::new(addresses, gates).unwrap();
            let source = fixture_gate_major(layout);
            let mut scalar = PvpBitplanesV1::from_gate_major_words(layout, &source).unwrap();
            let mut sve = scalar.clone();

            let scalar_stats = pascal_subset_zeta_scalar_in_place(&mut scalar).unwrap();
            let sve_stats =
                pascal_subset_zeta_with_backend_in_place(&mut sve, PvpBackendV1::Sve).unwrap();

            assert_eq!(sve, scalar);
            assert_eq!(sve_stats.backend, PvpBackendV1::Sve);
            assert_eq!(
                sve_stats.logical_gate_xor_ops,
                scalar_stats.logical_gate_xor_ops
            );
            assert_eq!(
                sve_stats.packed_word_updates,
                scalar_stats.packed_word_updates
            );
            assert_eq!(sve_stats.storage_bits, scalar_stats.storage_bits);
            assert_eq!(sve_stats.padding_bits, scalar_stats.padding_bits);
            assert_eq!(sve_stats.scratch_words, 0);
        }
    }

    #[cfg(all(feature = "nightly-simd", target_arch = "aarch64"))]
    #[test]
    #[ignore = "requires an SVE runtime; executed by the dedicated QEMU/native PVP gate"]
    fn auto_dispatch_prefers_sve_on_sve_runtime() {
        assert!(
            PvpBackendV1::Sve.available(),
            "dedicated SVE execution gate requires runtime SVE support"
        );
        assert_eq!(detect_best_pvp_backend_v1(), PvpBackendV1::Sve);
    }

    #[test]
    fn auto_dispatch_matches_scalar_reference() {
        let layout = PvpLayoutV1::new(64, 513).unwrap();
        let source = fixture_gate_major(layout);
        let mut scalar = PvpBitplanesV1::from_gate_major_words(layout, &source).unwrap();
        let mut automatic = scalar.clone();

        pascal_subset_zeta_scalar_in_place(&mut scalar).unwrap();
        let stats = pascal_subset_zeta_auto_in_place(&mut automatic).unwrap();

        assert_eq!(automatic, scalar);
        assert_eq!(stats.backend, detect_best_pvp_backend_v1());
        assert!(stats.backend.available());
    }

    #[test]
    fn unavailable_backend_fails_before_mutation() {
        #[cfg(not(feature = "nightly-simd"))]
        let backends = [PvpBackendV1::Avx512, PvpBackendV1::Avx2, PvpBackendV1::Neon];
        #[cfg(feature = "nightly-simd")]
        let backends = [
            PvpBackendV1::Avx512,
            PvpBackendV1::Avx2,
            PvpBackendV1::Neon,
            PvpBackendV1::Sve,
        ];
        let unavailable = backends
            .into_iter()
            .find(|backend| !backend.available())
            .expect("at least one architecture backend is unavailable");

        let layout = PvpLayoutV1::new(8, 65).unwrap();
        let source = fixture_gate_major(layout);
        let mut value = PvpBitplanesV1::from_gate_major_words(layout, &source).unwrap();
        let before = value.clone();

        assert_eq!(
            pascal_subset_zeta_with_backend_in_place(&mut value, unavailable),
            Err(PvpError::BackendUnavailable {
                backend: unavailable
            })
        );
        assert_eq!(value, before);
    }

    #[test]
    fn logical_bits_can_be_read_and_updated_safely() {
        let layout = PvpLayoutV1::new(8, 65).unwrap();
        let mut pvp = PvpBitplanesV1::zeroed(layout);
        pvp.set(7, 64, true).unwrap();
        assert!(pvp.get(7, 64).unwrap());
        pvp.set(7, 64, false).unwrap();
        assert!(!pvp.get(7, 64).unwrap());

        assert_eq!(
            pvp.get(8, 0),
            Err(PvpError::AddressOutOfRange {
                address: 8,
                addresses: 8
            })
        );
        assert_eq!(
            pvp.get(0, 65),
            Err(PvpError::GateOutOfRange {
                gate: 65,
                gates: 65
            })
        );
    }
}
