//! PORTS-DYN1: bounded source-graph binding to the existing DYN-REF1 library.
//! Administrative impulse probes, never a game policy or an inference of learning.
#![forbid(unsafe_code)]
use recurrent::{Dynamics, Edge, Limits, Network};
use std::collections::VecDeque;
use std::fs;
use std::io::{self, Write};
use std::path::Path;

fn bad(s: &str) -> String { s.to_owned() }
fn number<T: std::str::FromStr>(s: &str) -> Result<T, String> {
    if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) { return Err(bad("noncanonical integer")); }
    s.parse().map_err(|_| bad("integer out of range"))
}
fn text(path: &Path, maximum: u64) -> Result<String, String> {
    let size = fs::metadata(path).map_err(|e| e.to_string())?.len();
    if size > maximum { return Err(bad("input file exceeds bound")); }
    fs::read_to_string(path).map_err(|e| e.to_string())
}
fn nodes(s: &str) -> Result<Vec<u64>, String> {
    let mut ids = Vec::new();
    for line in s.lines() {
        if ids.len() >= 4096 { return Err(bad("node count exceeds bound")); }
        let f: Vec<_> = line.split('\t').collect();
        if f.len() != 3 || number::<usize>(f[0])? != ids.len() { return Err(bad("invalid node table")); }
        let id = number::<u64>(f[1])?;
        let _group = number::<u32>(f[2])?;
        if ids.last().is_some_and(|previous| *previous >= id) { return Err(bad("unordered node identity")); }
        ids.push(id);
    }
    if ids.len() < 3 { return Err(bad("at least three distinct nodes required")); }
    Ok(ids)
}
fn edges(s: &str, n: usize) -> Result<Vec<Edge>, String> {
    let mut result: Vec<Edge> = Vec::new();
    for line in s.lines() {
        if result.len() >= 1_000_000 { return Err(bad("edge count exceeds bound")); }
        let f: Vec<_> = line.split('\t').collect();
        if f.len() != 2 { return Err(bad("invalid edge fields")); }
        let source = number::<usize>(f[0])?;
        let target = number::<usize>(f[1])?;
        if source >= n || target >= n || source == target { return Err(bad("invalid edge endpoint")); }
        if result.last().is_some_and(|e| (e.source, e.target) >= (source, target)) { return Err(bad("duplicate or unordered edge")); }
        result.push(Edge { source, target, weight: 1, delay: 1 });
    }
    Ok(result)
}
fn ports(s: &str, n: usize) -> Result<Vec<[usize; 3]>, String> {
    let mut result = Vec::new();
    for line in s.lines() {
        if result.len() >= 32 { return Err(bad("port panel exceeds bound")); }
        let f: Vec<_> = line.split('\t').collect();
        if f.len() != 3 { return Err(bad("invalid port fields")); }
        let p = [number(f[0])?, number(f[1])?, number(f[2])?];
        if p.iter().any(|&v| v >= n) || p[0] == p[1] || p[0] == p[2] || p[1] == p[2] { return Err(bad("ports must be distinct valid indices")); }
        result.push(p);
    }
    if result.is_empty() { return Err(bad("missing original ports")); }
    Ok(result)
}
fn distances(adjacency: &[Vec<usize>], start: usize) -> Vec<i32> {
    let mut d = vec![-1; adjacency.len()];
    let mut queue = VecDeque::from([start]);
    d[start] = 0;
    while let Some(u) = queue.pop_front() {
        for &v in &adjacency[u] {
            if d[v] == -1 { d[v] = d[u] + 1; queue.push_back(v); }
        }
    }
    d
}
fn hex(bits: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    bits.chunks(4).map(|chunk| {
        let mut v = 0usize;
        for (i, &b) in chunk.iter().enumerate() { v |= usize::from(b) << i; }
        DIGITS[v] as char
    }).collect()
}
fn execute(ids: &[u64], graph: &[Edge], panel: &[[usize; 3]], ticks: usize, out: &mut impl Write) -> Result<(), String> {
    let n = ids.len();
    let mut adjacency = vec![Vec::new(); n];
    let mut reverse = vec![Vec::new(); n];
    for e in graph { adjacency[e.source].push(e.target); reverse[e.target].push(e.source); }
    let emit = |out: &mut dyn Write, s: String| -> Result<(), String> { writeln!(out, "{s}").map_err(|e| e.to_string()) };
    emit(out, format!("{{\"kind\":\"graph\",\"nodes\":{n},\"edges\":{},\"ticks_per_trial\":{ticks}}}", graph.len()))?;
    for (index, &[a, b, readout]) in panel.iter().enumerate() {
        let da = distances(&adjacency, a);
        let db = distances(&adjacency, b);
        let dr = distances(&reverse, readout);
        let common = (0..n).filter(|&i| da[i] >= 0 && db[i] >= 0).count();
        let joint = (0..n).filter(|&i| da[i] >= 0 && db[i] >= 0 && dr[i] >= 0).count();
        emit(out, format!("{{\"kind\":\"ports\",\"index\":{index},\"a\":{a},\"b\":{b},\"readout\":{readout},\"a_distance\":{},\"b_distance\":{},\"a_reachable\":{},\"b_reachable\":{},\"common_reachable\":{common},\"joint_route_nodes\":{joint}}}",
            da[readout], db[readout], da.iter().filter(|&&d| d >= 0).count(), db.iter().filter(|&&d| d >= 0).count()))?;
    }
    let [a, b, _readout] = panel[0];
    for nonlinear in [false, true] {
        let limits = Limits { max_ticks: ticks as u64, max_edge_visits: (graph.len() as u64) * ticks as u64, max_payload_bytes: 64 * 1024 * 1024 };
        let dynamics = Dynamics::Boolean { nonlinear };
        let mut network = Network::new(n, graph, dynamics, limits).map_err(bad)?;
        let mut drive = vec![0; n];
        for mask in 0..4 {
            network.reset_activity();
            for t in 0..ticks {
                drive.fill(0);
                if t == 0 { drive[a] = mask & 1; drive[b] = (mask >> 1) & 1; }
                network.step(&drive).map_err(bad)?;
                emit(out, format!("{{\"kind\":\"state\",\"nonlinear\":{nonlinear},\"mask\":{mask},\"t\":{t},\"hex\":\"{}\"}}", hex(network.spikes())))?;
            }
            let c = network.counters();
            let payload = network.payload_bytes();
            network.reset_activity(); drive.fill(0);
            let zero = network.step(&drive).map_err(bad)?.iter().all(|&x| x == 0);
            if !zero || network.parameters() != dynamics || network.edges() != graph { return Err(bad("reset or parameter preservation failed")); }
            emit(out, format!("{{\"kind\":\"trial\",\"nonlinear\":{nonlinear},\"mask\":{mask},\"ticks\":{},\"edge_visits\":{},\"active_edge_events\":{},\"spikes\":{},\"direct_vector_payload_bytes\":{payload},\"reset_zero\":true,\"reset_probe_ticks\":1,\"learning_performed\":false}}", c.ticks, c.edge_visits, c.active_edge_events, c.spikes))?;
        }
    }
    Ok(())
}
fn run() -> Result<(), String> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 5 { return Err(bad("usage: ports-dyn1 nodes.tsv graph.edges.tsv panel.tsv ticks")); }
    let ticks = number::<usize>(&a[4])?;
    if !(1..=257).contains(&ticks) { return Err(bad("tick horizon outside bound")); }
    let ids = nodes(&text(Path::new(&a[1]), 300_000)?)?;
    let graph = edges(&text(Path::new(&a[2]), 32_000_000)?, ids.len())?;
    let panel = ports(&text(Path::new(&a[3]), 4096)?, ids.len())?;
    let stdout = io::stdout(); let mut out = io::BufWriter::new(stdout.lock());
    execute(&ids, &graph, &panel, ticks, &mut out)?;
    out.flush().map_err(|e| e.to_string())
}
fn main() { if let Err(e) = run() { eprintln!("ports-dyn1: {e}"); std::process::exit(2); } }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn large_integer_identity_is_exact() {
        assert_eq!(nodes("0\t9007199254740993\t0\n1\t9007199254740995\t0\n2\t18446744073709551615\t1\n").unwrap()[0], 9007199254740993);
    }
    #[test]
    fn malformed_nodes_fail() {
        for s in ["", "0\t1.0\t0", "0\t1\t0\n1\t1\t0\n2\t3\t0", "1\t1\t0\n2\t2\t0\n3\t3\t3"] { assert!(nodes(s).is_err()); }
    }
    #[test]
    fn invalid_pairs_fail() {
        for s in ["0\t0", "0\t3", "0\t1\n0\t1", "1\t2\n0\t1", "-1\t2", "0\t1\t2"] { assert!(edges(s, 3).is_err()); }
        assert!(edges("", 3).unwrap().is_empty());
    }
    #[test]
    fn invalid_ports_fail() {
        for s in ["", "0\t0\t1", "0\t1\t3", "0\t1"] { assert!(ports(s, 3).is_err()); }
    }
    #[test]
    fn directed_paths_do_not_imply_reverse_paths() {
        let a = vec![vec![1], vec![2], vec![], vec![1]];
        assert_eq!(distances(&a, 0), [0, 1, 2, -1]);
        assert_eq!(distances(&a, 2), [-1, -1, 0, -1]);
    }
    #[test]
    fn state_hex_has_explicit_little_bit_order_and_padding() { assert_eq!(hex(&[1,0,1,0,1]), "51"); }
}
