//! Link the optional Intel MKL BLAS backend only from an operator-installed
//! local library. No build-time OCI downloads or implicit network fetches.
use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=SCIRUST_MKL_LIB_DIR");
    if env::var_os("CARGO_FEATURE_BLAS_MKL").is_none() {
        return;
    }

    if env::var_os("CARGO_FEATURE_BLAS_OPENBLAS").is_some() {
        panic!("blas-mkl and blas-openblas are mutually exclusive BLAS backends");
    }
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    assert!(
        target_os == "linux" && target_arch == "x86_64",
        "SCIRUST blas-mkl system backend currently requires Linux x86_64"
    );

    let dir: PathBuf = env::var_os("SCIRUST_MKL_LIB_DIR")
        .map(PathBuf::from)
        .expect("SCIRUST_MKL_LIB_DIR must point to a trusted local MKL library directory");
    assert!(
        dir.is_absolute() && dir.is_dir() && dir.join("libmkl_rt.so").is_file(),
        "SCIRUST_MKL_LIB_DIR must be an absolute directory containing libmkl_rt.so"
    );
    let canonical = dir.canonicalize().expect("cannot resolve MKL library directory");
    let printed = canonical.to_string_lossy();
    assert!(
        !printed.contains(['\n', '\r']),
        "MKL library path must not contain line breaks"
    );

    println!("cargo:rustc-link-search=native={printed}");
    println!("cargo:rustc-link-lib=dylib=mkl_rt");
}
