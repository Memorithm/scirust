//! V888-BOOL-0.2a research staging. No public graph API or neural dynamics.
//! Reads the separately qualified V8CSR001 contract. All control arms are
//! unit-edge topologies; measured contact multiplicities are retained separately.
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::str::FromStr;

type Pair = (usize, usize);
type Edges = BTreeSet<Pair>;
fn bad(s: &str) -> io::Error { io::Error::new(io::ErrorKind::InvalidData, s) }
fn parse<T: FromStr>(s: &str) -> io::Result<T> { s.parse().map_err(|_| bad("invalid integer")) }
fn add(x: &mut u64, y: u64) -> io::Result<()> {
    *x = x.checked_add(y).ok_or_else(|| bad("integer counter overflow"))?;
    Ok(())
}
fn writer(p: &Path) -> io::Result<BufWriter<File>> {
    Ok(BufWriter::new(OpenOptions::new().write(true).create_new(true).open(p)?))
}
fn take<const N: usize>(bytes: &[u8], cursor: &mut usize) -> io::Result<[u8; N]> {
    let end = cursor.checked_add(N).ok_or_else(|| bad("offset overflow"))?;
    let x = bytes.get(*cursor..end).ok_or_else(|| bad("truncated CSR"))?;
    *cursor = end;
    x.try_into().map_err(|_| bad("invalid field"))
}
#[derive(Debug)]
struct Source {
    ids: Vec<u64>, offsets: Vec<usize>, targets: Vec<u32>, weights: Vec<u64>,
}
impl Source {
    fn decode(bytes: &[u8]) -> io::Result<Self> {
        let mut cursor = 0;
        if take::<8>(bytes, &mut cursor)? != *b"V8CSR001" { return Err(bad("wrong CSR magic")); }
        let n = u64::from_le_bytes(take(bytes, &mut cursor)?);
        let m = u64::from_le_bytes(take(bytes, &mut cursor)?);
        if n == 0 || n > 1_000_000 || m > 20_000_000 { return Err(bad("CSR dimensions exceed staging bounds")); }
        let length = 32 + 16 * n + 12 * m;
        if bytes.len() as u64 != length { return Err(bad("CSR exact byte length mismatch")); }
        let (n, m) = (n as usize, m as usize);
        let mut ids = Vec::with_capacity(n);
        for _ in 0..n { ids.push(u64::from_le_bytes(take(bytes, &mut cursor)?)); }
        if ids.windows(2).any(|w| w[0] >= w[1]) { return Err(bad("node IDs not unique/sorted")); }
        let mut offsets = Vec::with_capacity(n + 1);
        for _ in 0..=n {
            let x = u64::from_le_bytes(take(bytes, &mut cursor)?);
            if x > m as u64 { return Err(bad("row offset out of bounds")); }
            offsets.push(x as usize);
        }
        if offsets[0] != 0 || offsets[n] != m || offsets.windows(2).any(|w| w[0] > w[1]) {
            return Err(bad("invalid row offsets"));
        }
        let mut targets = Vec::with_capacity(m);
        for _ in 0..m { targets.push(u32::from_le_bytes(take(bytes, &mut cursor)?)); }
        for u in 0..n {
            let row = &targets[offsets[u]..offsets[u + 1]];
            if row.iter().any(|&v| v as usize >= n || v as usize == u)
                || row.windows(2).any(|w| w[0] >= w[1]) {
                return Err(bad("invalid/duplicate/self target"));
            }
        }
        let mut weights = Vec::with_capacity(m);
        let mut total = 0;
        for _ in 0..m {
            let w = u64::from_le_bytes(take(bytes, &mut cursor)?);
            if w == 0 { return Err(bad("zero contact multiplicity")); }
            add(&mut total, w)?;
            weights.push(w);
        }
        Ok(Self { ids, offsets, targets, weights })
    }
    fn load(path: &Path) -> io::Result<Self> {
        if fs::metadata(path)?.len() > 300_000_000 { return Err(bad("CSR file too large")); }
        Self::decode(&fs::read(path)?)
    }
    fn reverse(&self) -> Vec<Vec<u32>> {
        let mut result = vec![Vec::new(); self.ids.len()];
        for u in 0..self.ids.len() {
            for &v in &self.targets[self.offsets[u]..self.offsets[u + 1]] {
                result[v as usize].push(u as u32);
            }
        }
        result
    }
    fn select(&self, count: usize, seed: u64, policy: &str) -> io::Result<(Vec<usize>, usize)> {
        if count < 3 || count > self.ids.len() || count > 4096 { return Err(bad("subset size outside bounds")); }
        let mut ranked: Vec<_> = (0..self.ids.len()).collect();
        ranked.sort_unstable_by_key(|&i| (mix(self.ids[i] ^ seed), self.ids[i]));
        if policy == "ranked" {
            ranked.truncate(count); ranked.sort_unstable();
            return Ok((ranked, 0));
        }
        if policy != "weak_bfs" { return Err(bad("unknown selection policy")); }
        let reverse = self.reverse();
        let mut seen = vec![false; self.ids.len()];
        let mut queue = VecDeque::new();
        let mut selected = Vec::with_capacity(count);
        let (mut anchor, mut starts) = (0, 0);
        while selected.len() < count {
            if queue.is_empty() {
                while seen[ranked[anchor]] { anchor += 1; }
                let root = ranked[anchor]; seen[root] = true; queue.push_back(root); starts += 1;
            }
            let u = queue.pop_front().ok_or_else(|| bad("empty BFS queue"))?;
            selected.push(u);
            if selected.len() == count { break; }
            let mut neighbors: Vec<usize> = self.targets[self.offsets[u]..self.offsets[u + 1]]
                .iter().chain(reverse[u].iter()).map(|&v| v as usize).collect();
            neighbors.sort_unstable_by_key(|&i| (mix(self.ids[i] ^ seed), self.ids[i]));
            neighbors.dedup();
            for v in neighbors { if !seen[v] { seen[v] = true; queue.push_back(v); } }
        }
        selected.sort_unstable();
        Ok((selected, starts))
    }
}
fn read_blocks(path: &Path, ids: &[u64]) -> io::Result<Vec<u32>> {
    if fs::metadata(path)?.len() > 64_000_000 { return Err(bad("block table too large")); }
    let text = fs::read_to_string(path)?;
    let lines: Vec<_> = text.lines().collect();
    if lines.len() != ids.len() { return Err(bad("block table node count mismatch")); }
    let mut result = Vec::with_capacity(ids.len());
    for (line, &id) in lines.iter().zip(ids) {
        let fields: Vec<_> = line.split('\t').collect();
        if fields.len() != 2 || parse::<u64>(fields[0])? != id { return Err(bad("block table ID/order mismatch")); }
        result.push(parse(fields[1])?);
    }
    Ok(result)
}
fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let value = mix(self.0);
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        value
    }
    fn below(&mut self, upper: usize) -> usize {
        assert!(upper > 0);
        let bound = upper as u64;
        let limit = u64::MAX - u64::MAX % bound;
        loop { let x = self.next(); if x < limit { return (x % bound) as usize; } }
    }
}
// Floyd sampling of exactly M distinct non-self directed pairs; no dense matrix.
fn edge_count_control(n: usize, m: usize, seed: u64) -> io::Result<Edges> {
    let universe = n.checked_mul(n.saturating_sub(1)).ok_or_else(|| bad("pair universe overflow"))?;
    if n < 2 || m > universe { return Err(bad("impossible edge count")); }
    let mut rng = Rng(seed);
    let mut indices = BTreeSet::new();
    for j in (universe - m)..universe {
        let t = rng.below(j + 1);
        let chosen = if indices.contains(&t) { j } else { t };
        indices.insert(chosen);
    }
    Ok(indices.into_iter().map(|i| {
        let u = i / (n - 1); let r = i % (n - 1);
        (u, if r >= u { r + 1 } else { r })
    }).collect())
}
#[derive(Clone, Copy, Debug)]
enum Match { Degree, Reciprocal, Block }
// Four distinct endpoints make the reciprocal delta local and unambiguous.
fn switch_allowed(edges: &Edges, old: [Pair; 2], groups: &[u32], mode: Match) -> bool {
    let [(a, b), (c, d)] = old;
    if a == b || c == d || a == c || a == d || b == c || b == d { return false; }
    if edges.contains(&(a, d)) || edges.contains(&(c, b)) { return false; }
    if matches!(mode, Match::Block) && groups[a] != groups[c] && groups[b] != groups[d] { return false; }
    if !matches!(mode, Match::Degree) {
        let ab = edges.contains(&(b, a)); let cd = edges.contains(&(d, c));
        let ad = edges.contains(&(d, a)); let cb = edges.contains(&(b, c));
        if [ab, ab, cd, cd] != [ad, cb, cb, ad] { return false; }
    }
    true
}
fn rewire(reference: &Edges, groups: &[u32], seed: u64, mode: Match, attempts: usize) -> (Edges, usize) {
    let mut edges = reference.clone();
    let mut order: Vec<Pair> = reference.iter().copied().collect();
    let mut rng = Rng(seed);
    let mut accepted = 0;
    if order.len() < 2 { return (edges, accepted); }
    for _ in 0..attempts {
        let i = rng.below(order.len()); let j = rng.below(order.len());
        let [left, right] = [order[i], order[j]];
        if !switch_allowed(&edges, [left, right], groups, mode) { continue; }
        let new_left = (left.0, right.1); let new_right = (right.0, left.1);
        edges.remove(&left); edges.remove(&right);
        edges.insert(new_left); edges.insert(new_right);
        order[i] = new_left; order[j] = new_right;
        accepted += 1;
    }
    (edges, accepted)
}
#[derive(Debug, PartialEq, Eq)]
struct Stats {
    incoming: Vec<usize>, outgoing: Vec<usize>, reciprocal: Vec<usize>, blocks: BTreeMap<(u32, u32), usize>,
}
fn stats(n: usize, edges: &Edges, groups: &[u32]) -> Stats {
    let mut x = Stats { incoming: vec![0; n], outgoing: vec![0; n], reciprocal: vec![0; n], blocks: BTreeMap::new() };
    for &(u, v) in edges {
        x.outgoing[u] += 1; x.incoming[v] += 1;
        x.reciprocal[u] += usize::from(edges.contains(&(v, u)));
        *x.blocks.entry((groups[u], groups[v])).or_insert(0) += 1;
    }
    x
}
fn write_edges(path: &Path, edges: &Edges) -> io::Result<()> {
    let mut out = writer(path)?;
    for &(u, v) in edges { writeln!(out, "{u}\t{v}")?; }
    out.flush()
}
fn execute(source: &Source, groups: &[u32], out: &Path, count: usize, seed: u64, policy: &str) -> io::Result<()> {
    let (selected, starts) = source.select(count, seed, policy)?;
    fs::create_dir(out)?;
    let mut mapping = vec![usize::MAX; source.ids.len()];
    let local_groups: Vec<_> = selected.iter().map(|&u| groups[u]).collect();
    let mut nodes = writer(&out.join("nodes.tsv"))?;
    for (local, &global) in selected.iter().enumerate() {
        mapping[global] = local;
        writeln!(nodes, "{local}\t{}\t{}", source.ids[global], groups[global])?;
    }
    nodes.flush()?;
    let mut ports: Vec<_> = (0..count).collect();
    ports.sort_unstable_by_key(|&i| (mix(source.ids[selected[i]] ^ seed ^ 0x1234abcd), i));
    let mut port_file = writer(&out.join("ports.tsv"))?;
    for (name, &port) in ["input_a", "input_b", "readout"].iter().zip(&ports) {
        writeln!(port_file, "{name}\t{port}")?;
    }
    port_file.flush()?;
    let mut measured = writer(&out.join("source_pairs.tsv"))?;
    let mut boundary = writer(&out.join("boundary.tsv"))?;
    let mut partitions = [[0_u64; 2]; 4];
    let mut reference = Edges::new();
    for u in 0..source.ids.len() {
        for e in source.offsets[u]..source.offsets[u + 1] {
            let v = source.targets[e] as usize; let w = source.weights[e];
            let inside_u = mapping[u] != usize::MAX; let inside_v = mapping[v] != usize::MAX;
            let category = 2 * usize::from(inside_u) + usize::from(inside_v);
            add(&mut partitions[category][0], 1)?; add(&mut partitions[category][1], w)?;
            if category == 3 {
                reference.insert((mapping[u], mapping[v]));
                writeln!(measured, "{}\t{}\t{w}", mapping[u], mapping[v])?;
            } else if category != 0 {
                writeln!(boundary, "{}\t{}\t{w}\t{category}", source.ids[u], source.ids[v])?;
            }
        }
    }
    measured.flush()?; boundary.flush()?;
    write_edges(&out.join("reference.edges.tsv"), &reference)?;
    let expected = stats(count, &reference, &local_groups);
    let mut summaries = Vec::new();
    let attempts = reference.len().saturating_mul(32).min(1_000_000);
    for (index, name) in ["edge_count", "degree", "degree_reciprocal", "degree_reciprocal_block"].iter().enumerate() {
        let arm_seed = seed ^ (0x6a09e667f3bcc909_u64.wrapping_mul(index as u64 + 1));
        let (edges, accepted, trials) = if index == 0 {
            (edge_count_control(count, reference.len(), arm_seed)?, 0, 0)
        } else {
            let mode = [Match::Degree, Match::Reciprocal, Match::Block][index - 1];
            let (x, a) = rewire(&reference, &local_groups, arm_seed, mode, attempts);
            (x, a, attempts)
        };
        let observed = stats(count, &edges, &local_groups);
        let degree_mismatches = (0..count).filter(|&i| expected.incoming[i] != observed.incoming[i] || expected.outgoing[i] != observed.outgoing[i]).count();
        let reciprocal_mismatches = (0..count).filter(|&i| expected.reciprocal[i] != observed.reciprocal[i]).count();
        let cells: BTreeSet<_> = expected.blocks.keys().chain(observed.blocks.keys()).copied().collect();
        let block_mismatches = cells.iter().filter(|k| expected.blocks.get(k).unwrap_or(&0) != observed.blocks.get(k).unwrap_or(&0)).count();
        if edges.len() != reference.len() || (index >= 1 && degree_mismatches != 0)
            || (index >= 2 && reciprocal_mismatches != 0) || (index == 3 && block_mismatches != 0) {
            return Err(bad("generated control violated its declared invariant"));
        }
        let replaced_edges = reference.difference(&edges).count();
        write_edges(&out.join(format!("{name}.edges.tsv")), &edges)?;
        summaries.push(format!("{{\"arm\":\"{name}\",\"seed\":{arm_seed},\"edge_count\":{},\"attempts\":{trials},\"accepted_swaps\":{accepted},\"replaced_edges\":{replaced_edges},\"degree_mismatch_nodes\":{degree_mismatches},\"reciprocal_mismatch_nodes\":{reciprocal_mismatches},\"block_mismatch_cells\":{block_mismatches}}}", edges.len()));
    }
    let categories = partitions.iter().map(|p| format!("{{\"pairs\":{},\"contacts\":{}}}", p[0], p[1])).collect::<Vec<_>>().join(",");
    let distinct_blocks = local_groups.iter().collect::<BTreeSet<_>>().len();
    let mut summary = writer(&out.join("summary.json"))?;
    writeln!(summary, "{{\"schema_version\":1,\"source_nodes\":{},\"source_pairs\":{},\"selection\":\"{policy}\",\"seed\":{seed},\"nodes\":{count},\"pairs\":{},\"bfs_component_starts\":{starts},\"block_labels\":{distinct_blocks},\"partitions_outside_incoming_outgoing_internal\":[{categories}],\"arms\":[{}]}}", source.ids.len(), source.targets.len(), reference.len(), summaries.join(","))?;
    summary.flush()
}
fn run() -> io::Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 7 { return Err(bad("usage: subset-controls source.csr blocks.tsv output_dir size seed ranked|weak_bfs")); }
    let source = Source::load(Path::new(&args[1]))?;
    let groups = read_blocks(Path::new(&args[2]), &source.ids)?;
    execute(&source, &groups, Path::new(&args[3]), parse(&args[4])?, parse(&args[5])?, &args[6])
}
fn main() { if let Err(e) = run() { eprintln!("subset-control qualification failed: {e}"); std::process::exit(2); } }

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture(n: usize, edges: &Edges) -> Source {
        let mut offsets = vec![0; n + 1]; let mut targets = Vec::new();
        for &(u, v) in edges { offsets[u + 1] += 1; targets.push(v as u32); }
        for i in 0..n { offsets[i + 1] += offsets[i]; }
        Source { ids: (0..n).map(|i| (1_u64 << 54) + i as u64 * 7).collect(), offsets, weights: vec![1; targets.len()], targets }
    }
    #[test]
    fn exact_edge_count_including_empty_and_complete() {
        for n in 3..12 { for m in [0, 1, n, n * (n - 1)] {
            let e = edge_count_control(n, m, 11).unwrap();
            assert_eq!(e.len(), m); assert!(e.iter().all(|&(u, v)| u < n && v < n && u != v));
            assert_eq!(e, edge_count_control(n, m, 11).unwrap());
        } }
    }
    #[test]
    fn all_declared_rewiring_invariants() {
        for seed in 0..32 {
            let e = edge_count_control(18, 70, seed).unwrap(); let g: Vec<u32> = (0..18).map(|i| (i / 6) as u32).collect();
            let before = stats(18, &e, &g);
            for mode in [Match::Degree, Match::Reciprocal, Match::Block] {
                let (result, _) = rewire(&e, &g, seed, mode, 4000); let after = stats(18, &result, &g);
                assert_eq!(e.len(), result.len()); assert_eq!(before.incoming, after.incoming); assert_eq!(before.outgoing, after.outgoing);
                if !matches!(mode, Match::Degree) { assert_eq!(before.reciprocal, after.reciprocal); }
                if matches!(mode, Match::Block) { assert_eq!(before.blocks, after.blocks); }
                assert_eq!(result, rewire(&e, &g, seed, mode, 4000).0);
            }
        }
    }
    #[test]
    fn reciprocal_constraint_rejects_degree_only_swap() {
        let e: Edges = [(0, 1), (1, 0), (2, 3)].into_iter().collect();
        assert!(switch_allowed(&e, [(0, 1), (2, 3)], &[0; 4], Match::Degree));
        assert!(!switch_allowed(&e, [(0, 1), (2, 3)], &[0; 4], Match::Reciprocal));
    }
    #[test]
    fn block_constraint_rejects_cross_block_change() {
        let e: Edges = [(0, 1), (2, 3)].into_iter().collect();
        assert!(switch_allowed(&e, [(0, 1), (2, 3)], &[0, 0, 1, 1], Match::Reciprocal));
        assert!(!switch_allowed(&e, [(0, 1), (2, 3)], &[0, 0, 1, 1], Match::Block));
    }
    #[test]
    fn dense_and_tiny_graphs_report_no_swaps() {
        let e = edge_count_control(8, 56, 0).unwrap();
        assert_eq!(rewire(&e, &[0; 8], 1, Match::Degree, 100).1, 0);
        assert_eq!(rewire(&Edges::new(), &[0; 8], 1, Match::Block, 100).1, 0);
    }
    #[test]
    fn subset_is_deterministic_and_retains_isolated_nodes() {
        let source = fixture(12, &[(0, 1), (1, 2)].into_iter().collect());
        for policy in ["ranked", "weak_bfs"] {
            assert_eq!(source.select(6, 7, policy).unwrap(), source.select(6, 7, policy).unwrap());
            assert_eq!(source.select(12, 7, policy).unwrap().0, (0..12).collect::<Vec<_>>());
        }
        assert!(source.select(13, 0, "ranked").is_err());
    }
    #[test]
    fn malformed_header_fails_before_large_allocation() {
        assert!(Source::decode(b"bad").is_err());
        let mut bytes = b"V8CSR001".to_vec(); bytes.extend(u64::MAX.to_le_bytes()); bytes.extend(0_u64.to_le_bytes());
        assert!(Source::decode(&bytes).is_err());
    }
    #[test]
    fn impossible_edge_count_is_rejected() {
        assert!(edge_count_control(4, 13, 0).is_err());
    }
}
