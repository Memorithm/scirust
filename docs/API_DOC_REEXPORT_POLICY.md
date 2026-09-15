# Re-export documentation policy

A public item can be reachable through a re-export even when its defining module
is not the path users normally consume. Conversely, a syntactically public item
can be unreachable behind a private module.

Rustdoc/compiler-derived reachability is therefore authoritative. The lexicon
should record canonical/user-facing paths where practical and avoid duplicate
semantic entries for the same callable merely because it has multiple re-export
paths.

Examples should prefer the stable public path advertised to downstream users.
Internal defining paths may appear in source links but should not become the only
usage example when a façade/re-export is the intended API.
