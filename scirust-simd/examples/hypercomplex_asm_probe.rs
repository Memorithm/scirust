use scirust_simd::hypercomplex::{OctonionSimd, SedenionSimd};

#[unsafe(no_mangle)]
#[inline(never)]
pub fn scirust_octonion_mul_probe(lhs: OctonionSimd, rhs: OctonionSimd) -> OctonionSimd {
    lhs * rhs
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn scirust_sedenion_mul_probe(lhs: SedenionSimd, rhs: SedenionSimd) -> SedenionSimd {
    lhs * rhs
}

fn main() {
    let o = scirust_octonion_mul_probe(OctonionSimd::ONE, OctonionSimd::ONE);
    let s = scirust_sedenion_mul_probe(SedenionSimd::ONE, SedenionSimd::ONE);

    std::hint::black_box((o, s));
}
