//! Frozen synthetic corpus; see docs/PVP_ANF_BANK_QUALIFICATION_PROTOCOL.md.

use super::*;

const GEOMETRIES: [(usize, usize); 8] = [
    (1, 1),
    (2, 31),
    (4, 65),
    (8, 129),
    (64, 257),
    (256, 513),
    (2048, 257),
    (256, 2048),
];
const BANKS: [&str; 3] = ["boundary", "sparse4", "dense32"];

struct AnfBank {
    k: usize,
    terms: Vec<Vec<usize>>,
}

impl AnfBank {
    fn frozen(k: usize, gates: usize, kind: &str) -> Self {
        let terms = (0..gates)
            .map(|g| match kind
            {
                "boundary" => match g % 6
                {
                    0 => vec![],
                    1 => vec![0],
                    2 => vec![usize::from(k > 1)],
                    3 => vec![k - 1],
                    4 => vec![0, usize::from(k > 1)],
                    _ => (0..k.ilog2()).map(|bit| 1_usize << bit).collect(),
                },
                "sparse4" | "dense32" =>
                {
                    let count = k.min(if kind == "sparse4" { 4 } else { 32 });
                    let masks: Vec<_> = (0..count).map(|t| (257 * g + 73 * t) % k).collect();
                    let mut seen = vec![false; k];
                    for &mask in &masks
                    {
                        assert!(!seen[mask], "duplicate frozen ANF term");
                        seen[mask] = true;
                    }
                    masks
                },
                _ => panic!("unknown frozen bank"),
            })
            .collect();
        Self { k, terms }
    }

    fn coefficients_gate_major(&self) -> Vec<u64> {
        let words_per_gate = self.k.div_ceil(64);
        let mut words = vec![0_u64; words_per_gate * self.terms.len()];
        for (gate, terms) in self.terms.iter().enumerate()
        {
            for &mask in terms
            {
                words[gate * words_per_gate + mask / 64] ^= 1_u64 << (mask % 64);
            }
        }
        words
    }

    fn truth_words(&self) -> Vec<u64> {
        // Direct monomial evaluation; no layout adapter, butterfly or submask
        // enumeration constructs the expected truth values or their packing.
        let words_per_address = self.terms.len().div_ceil(64);
        let mut words = vec![0_u64; self.k * words_per_address];
        for address in 0..self.k
        {
            for (gate, terms) in self.terms.iter().enumerate()
            {
                let truth = terms
                    .iter()
                    .fold(false, |value, &mask| value ^ ((address & mask) == mask));
                if truth
                {
                    words[address * words_per_address + gate / 64] |= 1_u64 << (gate % 64);
                }
            }
        }
        words
    }
}

fn candidates() -> Vec<PvpBackendV1> {
    let backends = vec![
        PvpBackendV1::Scalar,
        PvpBackendV1::Avx2,
        PvpBackendV1::Avx512,
        PvpBackendV1::Neon,
    ];
    #[cfg(feature = "nightly-simd")]
    let backends = {
        let mut backends = backends;
        backends.push(PvpBackendV1::Sve);
        backends
    };
    backends
}

fn qualify(backend: PvpBackendV1) {
    assert!(
        backend.available(),
        "required backend {} unavailable",
        backend.label()
    );
    let mut cases = 0;
    for (k, g) in GEOMETRIES
    {
        for kind in BANKS
        {
            let bank = AnfBank::frozen(k, g, kind);
            let coefficients = bank.coefficients_gate_major();
            let layout = PvpLayoutV1::new(k, g).unwrap();
            let mut state = PvpBitplanesV1::from_gate_major_words(layout, &coefficients).unwrap();
            let initial = state.words().to_vec();
            let stats = pascal_subset_zeta_with_backend_in_place(&mut state, backend).unwrap();
            assert_eq!(
                state.words(),
                bank.truth_words(),
                "K={k} G={g} bank={kind} backend={}",
                backend.label()
            );
            assert_eq!(stats.backend, backend);
            assert_eq!(stats.scratch_words, 0);
            assert_eq!(stats.storage_bits, layout.storage_bits());
            assert_eq!(stats.padding_bits, layout.padding_bits());
            assert_eq!(
                stats.logical_gate_xor_ops,
                (k / 2) as u128 * u128::from(layout.stages()) * g as u128
            );
            assert_eq!(
                stats.packed_word_updates,
                (k / 2) as u128
                    * u128::from(layout.stages())
                    * layout.gate_words_per_address() as u128
            );
            pascal_subset_zeta_with_backend_in_place(&mut state, backend).unwrap();
            assert_eq!(
                state.words(),
                initial,
                "coefficient recovery backend={}",
                backend.label()
            );
            assert_eq!(state.to_gate_major_words().unwrap(), coefficients);
            cases += 1;
        }
    }
    assert_eq!(cases, 24);
    println!(
        "PVP_ANF_BANK_COMPLETE,backend={},cases=24,truth=exact,round_trip=exact,performance_claim=none",
        backend.label()
    );
}

#[test]
fn direct_anf_oracle_matches_known_truth_values() {
    assert_eq!(
        AnfBank::frozen(8, 6, "boundary").truth_words(),
        vec![18, 38, 50, 6, 50, 6, 18, 46]
    );
    let constants = AnfBank::frozen(1, 6, "boundary");
    assert_eq!(constants.truth_words(), vec![14]);
    assert_eq!(constants.coefficients_gate_major(), vec![0, 1, 1, 1, 0, 0]);
}

#[test]
fn every_available_backend_matches_direct_anf_and_recovers_coefficients() {
    for backend in candidates()
    {
        if backend.available()
        {
            qualify(backend);
        }
        else
        {
            let bank = AnfBank::frozen(8, 65, "sparse4");
            let mut state = PvpBitplanesV1::from_gate_major_words(
                PvpLayoutV1::new(8, 65).unwrap(),
                &bank.coefficients_gate_major(),
            )
            .unwrap();
            let initial = state.clone();
            assert_eq!(
                pascal_subset_zeta_with_backend_in_place(&mut state, backend),
                Err(PvpError::BackendUnavailable { backend })
            );
            assert_eq!(state, initial);
            println!(
                "PVP_ANF_BANK_UNAVAILABLE,backend={},state_unchanged=true,performance_claim=none",
                backend.label()
            );
        }
    }
}

#[test]
fn auto_dispatch_preserves_direct_anf_truth_and_coefficients() {
    for (k, g) in GEOMETRIES
    {
        for kind in BANKS
        {
            let bank = AnfBank::frozen(k, g, kind);
            let mut state = PvpBitplanesV1::from_gate_major_words(
                PvpLayoutV1::new(k, g).unwrap(),
                &bank.coefficients_gate_major(),
            )
            .unwrap();
            let initial = state.clone();
            let stats = pascal_subset_zeta_auto_in_place(&mut state).unwrap();
            assert_eq!(stats.backend, detect_best_pvp_backend_v1());
            assert_eq!(state.words(), bank.truth_words());
            pascal_subset_zeta_auto_in_place(&mut state).unwrap();
            assert_eq!(state, initial);
        }
    }
    println!(
        "PVP_ANF_BANK_COMPLETE,backend=auto,cases=24,truth=exact,round_trip=exact,performance_claim=none"
    );
}

#[cfg(target_arch = "aarch64")]
#[test]
#[ignore = "requires ARM execution; mandatory in the dedicated QEMU gate"]
fn required_neon_executes_the_complete_anf_bank_corpus() {
    qualify(PvpBackendV1::Neon);
}

#[cfg(all(feature = "nightly-simd", target_arch = "aarch64"))]
#[test]
#[ignore = "requires SVE execution; mandatory in the dedicated QEMU gate"]
fn required_sve_executes_the_complete_anf_bank_corpus() {
    qualify(PvpBackendV1::Sve);
}
