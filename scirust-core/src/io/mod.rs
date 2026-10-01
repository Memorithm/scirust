// scirust-core/src/io/mod.rs
//
// Module io — sérialisation / désérialisation des tenseurs et modèles.

pub mod safetensors;

pub use safetensors::{
    deserialize, deserialize_state_dict, deserialize_state_dict_nd, deserialize_with_metadata,
    load_safetensors, load_safetensors_with_limits, load_state_dict,
    load_state_dict_nd, load_state_dict_nd_with_limits, load_state_dict_with_limits,
    save_safetensors, save_state_dict, save_state_dict_nd, serialize, serialize_state_dict,
    serialize_state_dict_nd, serialize_with_metadata, SafetensorsReadLimits,
    DEFAULT_MAX_SAFETENSORS_BYTES,
};
