//! Link an explicitly installed MKL runtime without invoking vulnerable OCI installers.
//!
//! The previous blas-src/intel-mkl provider transitively used ocipkg 0.2.9,
//! affected by RUSTSEC-2026-0450. No archive downloads occur in this build script.
use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=MKLROOT");

    if env::var_os("CARGO_FEATURE_BLAS_MKL").is_none() {
        return;
    }
    assert!(
        env::var_os("CARGO_FEATURE_BLAS_OPENBLAS").is_none(),
        "blas-mkl and blas-openblas are mutually exclusive providers"
    );

    let root = PathBuf::from(
        env::var_os("MKLROOT")
            .expect("blas-mkl requires an existing, trusted system oneMKL installation (MKLROOT)"),
    );
    assert!(root.is_absolute(), "MKLROOT must be an absolute path");

    let os = env::var("CARGO_CFG_TARGET_OS").expect("Cargo target OS must be present");
    let library_name = match os.as_str() {
        "linux" => "libmkl_rt.so",
        "macos" => "libmkl_rt.dylib",
        "windows" => "mkl_rt.lib",
        _ => panic!("blas-mkl is unsupported on target OS {os}"),
    };
    let directories = ["lib", "lib/intel64", "lib/intel64_lin"];
    let library_dir = directories
        .iter()
        .map(|part| root.join(part))
        .find(|dir| dir.join(library_name).is_file())
        .unwrap_or_else(|| {
            panic!(
                "system oneMKL library {library_name} not found under MKLROOT {}; refusing download or fallback",
                root.display()
            )
        });

    println!("cargo:rustc-link-search=native={}", library_dir.display());
    println!("cargo:rustc-link-lib=dylib=mkl_rt");
}
