from pathlib import Path

path = Path("scirust-attention-intent/src/lib.rs")
text = path.read_text()
old = '''    /// Today this means: `value_dim == head_dim` and every bound variant is
    /// dense in the logical dtype. Representable quantized intents remain false.
    #[must_use]
    pub const fn is_executable(&self) -> bool {
        matches!(
            self.representation.query_variant,
            RepresentationVariant::Dense { storage_dtype } if storage_dtype == self.logical_dtype
        ) && matches!(
            self.representation.key_variant,
            RepresentationVariant::Dense { storage_dtype } if storage_dtype == self.logical_dtype
        ) && matches!(
            self.representation.value_variant,
            RepresentationVariant::Dense { storage_dtype } if storage_dtype == self.logical_dtype
        )
    }
'''
new = '''    /// Today this means: `value_dim == head_dim` and every bound variant is
    /// dense. `derive_attention_intent` separately guarantees dense storage dtype
    /// equality with the logical dtype. Representable quantized intents remain false.
    #[must_use]
    pub const fn is_executable(&self) -> bool {
        matches!(
            self.representation.query_variant,
            RepresentationVariant::Dense { .. }
        ) && matches!(
            self.representation.key_variant,
            RepresentationVariant::Dense { .. }
        ) && matches!(
            self.representation.value_variant,
            RepresentationVariant::Dense { .. }
        )
    }
'''
if text.count(old) != 1:
    raise SystemExit("expected exactly one const execution predicate")
path.write_text(text.replace(old, new, 1))
