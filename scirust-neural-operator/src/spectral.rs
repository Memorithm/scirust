use crate::{NeuralOperatorError, PeriodicGrid1d, Result};
use scirust_signal::{Complex, fft::fft_portable, fft::ifft_portable};

fn validate_real_field(what: &'static str, field: &[f64]) -> Result<()> {
    if field.is_empty()
    {
        return Err(NeuralOperatorError::Empty { what });
    }
    if let Some((index, _)) = field.iter().enumerate().find(|(_, x)| !x.is_finite())
    {
        return Err(NeuralOperatorError::NonFinite { what, index });
    }
    Ok(())
}

fn complex_pow_i_k(k: f64, order: usize) -> Complex {
    let base = Complex::new(0.0, k);
    let mut out = Complex::new(1.0, 0.0);
    for _ in 0..order
    {
        out *= base;
    }
    out
}

/// Periodic spectral derivative using SciRust's deterministic radix-2 FFT.
///
/// For a sampled field `u`, this applies `(i k)^order` to each Fourier mode.
/// On an even grid the Nyquist mode is zeroed for odd derivatives so the
/// derivative of a real input remains represented by a real-valued grid field.
pub fn spectral_derivative_1d(field: &[f64], length: f64, order: usize) -> Result<Vec<f64>> {
    validate_real_field("spectral field", field)?;
    if order == 0
    {
        return Err(NeuralOperatorError::ZeroDerivativeOrder);
    }
    let grid = PeriodicGrid1d::new(field.len(), length)?;
    let mut spectrum: Vec<Complex> = field.iter().map(|&x| Complex::new(x, 0.0)).collect();
    fft_portable(&mut spectrum);
    let nyquist = field.len() / 2;
    for (index, bin) in spectrum.iter_mut().enumerate()
    {
        if order % 2 == 1 && index == nyquist
        {
            *bin = Complex::zero();
            continue;
        }
        let k = grid.angular_wavenumber(index);
        *bin *= complex_pow_i_k(k, order);
    }
    ifft_portable(&mut spectrum);
    Ok(spectrum.into_iter().map(|z| z.re).collect())
}

/// Evaluate the one-dimensional periodic Laplacian as the second spectral derivative.
pub fn spectral_laplacian_1d(field: &[f64], length: f64) -> Result<Vec<f64>> {
    spectral_derivative_1d(field, length, 2)
}

/// Spectral derivative of a row-major 2-D periodic scalar field.
pub fn spectral_derivative_2d(
    field: &[f64],
    rows: usize,
    cols: usize,
    length_axis: f64,
    axis: usize,
    order: usize,
) -> Result<Vec<f64>> {
    validate_real_field("spectral 2-D field", field)?;
    if rows.saturating_mul(cols) != field.len()
    {
        return Err(NeuralOperatorError::ShapeMismatch {
            what: "spectral 2-D field",
            expected: rows.saturating_mul(cols),
            got: field.len(),
        });
    }
    if axis > 1
    {
        return Err(NeuralOperatorError::InvalidAxis { axis });
    }
    let line_len = if axis == 0 { rows } else { cols };
    PeriodicGrid1d::new(line_len, length_axis)?;
    let mut out = vec![0.0; field.len()];
    if axis == 1
    {
        for r in 0..rows
        {
            let start = r * cols;
            let d = spectral_derivative_1d(&field[start..start + cols], length_axis, order)?;
            out[start..start + cols].copy_from_slice(&d);
        }
    }
    else
    {
        let mut line = vec![0.0; rows];
        for c in 0..cols
        {
            for r in 0..rows
            {
                line[r] = field[r * cols + c];
            }
            let d = spectral_derivative_1d(&line, length_axis, order)?;
            for r in 0..rows
            {
                out[r * cols + c] = d[r];
            }
        }
    }
    Ok(out)
}

/// Evaluate the two-dimensional periodic Laplacian as `d²/dx² + d²/dy²`.
pub fn spectral_laplacian_2d(
    field: &[f64],
    rows: usize,
    cols: usize,
    length_y: f64,
    length_x: f64,
) -> Result<Vec<f64>> {
    let dyy = spectral_derivative_2d(field, rows, cols, length_y, 0, 2)?;
    let dxx = spectral_derivative_2d(field, rows, cols, length_x, 1, 2)?;
    Ok(dxx.into_iter().zip(dyy).map(|(a, b)| a + b).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    #[test]
    fn first_derivative_is_spectral_on_bandlimited_signal() {
        let n = 64usize;
        let x: Vec<f64> = (0..n).map(|i| TAU * i as f64 / n as f64).collect();
        let u: Vec<f64> = x
            .iter()
            .map(|&x| (3.0 * x).sin() + 0.2 * (5.0 * x).cos())
            .collect();
        let want: Vec<f64> = x
            .iter()
            .map(|&x| 3.0 * (3.0 * x).cos() - (5.0 * x).sin())
            .collect();
        let got = spectral_derivative_1d(&u, TAU, 1).unwrap();
        let max = got
            .iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(max < 1e-10, "max derivative error {max}");
    }

    #[test]
    fn laplacian_2d_matches_periodic_trigonometric_oracle() {
        let (rows, cols) = (16usize, 32usize);
        let mut field = vec![0.0; rows * cols];
        let mut want = vec![0.0; rows * cols];
        for r in 0..rows
        {
            let y = TAU * r as f64 / rows as f64;
            for c in 0..cols
            {
                let x = TAU * c as f64 / cols as f64;
                let u = (2.0 * x).sin() + 0.5 * (3.0 * y).cos();
                field[r * cols + c] = u;
                want[r * cols + c] = -4.0 * (2.0 * x).sin() - 4.5 * (3.0 * y).cos();
            }
        }
        let got = spectral_laplacian_2d(&field, rows, cols, TAU, TAU).unwrap();
        let max = got
            .iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(max < 1e-9, "max laplacian error {max}");
    }
}
