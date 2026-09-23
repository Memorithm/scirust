//! V888-BOOL-0.1 research staging: exact pair audit, not a public graph API.
//! Source multiplicity remains an integer contact count, never a conductance.
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const MAX_NODES: u64 = 1_000_000;
const MAX_EDGES: u64 = 20_000_000;
const MAX_RAW: u64 = 250_000_000;

fn u64_read(r: &mut impl Read) -> Result<u64> {
    let mut b = [0; 8]; r.read_exact(&mut b)?; Ok(u64::from_le_bytes(b))
}
fn header(r: &mut impl Read, magic: &[u8; 8], limit: u64) -> Result<u64> {
    let mut b = [0; 8]; r.read_exact(&mut b)?;
    if &b != magic { return Err("invalid format magic".into()); }
    let n = u64_read(r)?;
    if n > limit { return Err("declared count exceeds resource limit".into()); }
    Ok(n)
}
fn eof(r: &mut impl Read) -> Result<()> {
    let mut b = [0; 1];
    if r.read(&mut b)? != 0 { return Err("trailing bytes".into()); }
    Ok(())
}
fn add(x: &mut u64, n: u64) -> Result<()> {
    *x = x.checked_add(n).ok_or("integer count overflow")?; Ok(())
}
#[derive(Debug)]
struct Edge { pre: u32, post: u32, weight: u64, observed: u64 }
struct Audit {
    ids: Vec<u64>, index: HashMap<u64, u32>,
    edges: Vec<Edge>, pairs: HashMap<(u32, u32), usize>,
    membership: [u64; 4], unexpected: u64, autapses: u64,
}
impl Audit {
    fn new(ids: Vec<u64>, input: Vec<[u64; 3]>) -> Result<Self> {
        if ids.is_empty() || ids.len() as u64 > MAX_NODES {
            return Err("invalid node count".into());
        }
        if ids.windows(2).any(|w| w[0] >= w[1]) {
            return Err("node IDs must be strictly increasing".into());
        }
        if input.len() as u64 > MAX_EDGES { return Err("too many edges".into()); }
        let index: HashMap<u64, u32> = ids.iter().enumerate()
            .map(|(i, &id)| (id, i as u32)).collect();
        let mut edges = Vec::with_capacity(input.len());
        for [pre, post, weight] in input {
            if pre == post || weight == 0 { return Err("autapse or zero reference weight".into()); }
            edges.push(Edge {
                pre: *index.get(&pre).ok_or("unknown reference source")?,
                post: *index.get(&post).ok_or("unknown reference target")?, weight, observed: 0,
            });
        }
        edges.sort_unstable_by_key(|e| (e.pre, e.post));
        let mut pairs = HashMap::with_capacity(edges.len());
        for (i, e) in edges.iter().enumerate() {
            if pairs.insert((e.pre, e.post), i).is_some() {
                return Err("duplicate directed reference pair".into());
            }
        }
        Ok(Self { ids, index, edges, pairs, membership: [0;4], unexpected: 0, autapses: 0 })
    }
    fn observe(&mut self, pre: u64, post: u64) -> Result<()> {
        if pre == post { add(&mut self.autapses, 1)?; }
        match (self.index.get(&pre), self.index.get(&post)) {
            (Some(&p), Some(&q)) => {
                add(&mut self.membership[0], 1)?;
                if let Some(&i) = self.pairs.get(&(p,q)) {
                    add(&mut self.edges[i].observed, 1)?;
                } else { add(&mut self.unexpected, 1)?; }
            }
            (Some(_), None) => add(&mut self.membership[1], 1)?,
            (None, Some(_)) => add(&mut self.membership[2], 1)?,
            (None, None) => add(&mut self.membership[3], 1)?,
        }
        Ok(())
    }
    fn mismatches(&self) -> usize {
        self.edges.iter().filter(|e| e.weight != e.observed).count()
    }
    fn passed(&self) -> bool { self.mismatches() == 0 && self.unexpected == 0 && self.autapses == 0 }
    fn write_csr(&self, path: &Path) -> Result<()> {
        if !self.passed() { return Err("refuse graph publication before pairwise agreement".into()); }
        let f = OpenOptions::new().write(true).create_new(true).open(path)?;
        let mut w = BufWriter::new(f);
        w.write_all(b"V8CSR001")?;
        w.write_all(&(self.ids.len() as u64).to_le_bytes())?;
        w.write_all(&(self.edges.len() as u64).to_le_bytes())?;
        for id in &self.ids { w.write_all(&id.to_le_bytes())?; }
        let mut position = 0usize;
        for row in 0..=self.ids.len() {
            while position < self.edges.len() && (self.edges[position].pre as usize) < row { position += 1; }
            w.write_all(&(position as u64).to_le_bytes())?;
        }
        for edge in &self.edges { w.write_all(&edge.post.to_le_bytes())?; }
        for edge in &self.edges { w.write_all(&edge.weight.to_le_bytes())?; }
        w.flush()?; w.get_ref().sync_all()?;
        Ok(())
    }
}
fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 { return Err("usage: pair_graph nodes.bin edges.bin output-directory".into()); }
    let out = Path::new(&args[3]);
    let mut nr = BufReader::new(File::open(&args[1])?);
    let n = header(&mut nr, b"V8NODE01", MAX_NODES)?;
    let mut ids = Vec::with_capacity(n as usize);
    for _ in 0..n { ids.push(u64_read(&mut nr)?); } eof(&mut nr)?;
    let mut er = BufReader::new(File::open(&args[2])?);
    let m = header(&mut er, b"V8EDGE01", MAX_EDGES)?;
    let mut input = Vec::with_capacity(m as usize);
    for _ in 0..m { input.push([u64_read(&mut er)?, u64_read(&mut er)?, u64_read(&mut er)?]); }
    eof(&mut er)?;
    let mut audit = Audit::new(ids, input)?;
    let mut r = BufReader::with_capacity(1024*1024, io::stdin().lock());
    let count = header(&mut r, b"V8RAW001", MAX_RAW)?;
    for i in 0..count {
        let pre = u64_read(&mut r)?; let post = u64_read(&mut r)?;
        audit.observe(pre,post)?;
        if (i+1) % 25_000_000 == 0 { eprintln!("raw records reconciled: {}", i+1); }
    }
    eof(&mut r)?;
    let mut weight = 0;
    for e in &audit.edges { add(&mut weight,e.weight)?; }
    let s = format!(concat!("{{\"nodes\":{},\"reference_pairs\":{},\"reference_weight\":{},",
        "\"raw_rows\":{},\"both_known\":{},\"pre_only\":{},\"post_only\":{},\"neither_known\":{},",
        "\"unexpected_known_pair_records\":{},\"mismatching_reference_pairs\":{},",
        "\"raw_autapses\":{},\"pairwise_equality\":{}}}\n"),
        n,m,weight,count,audit.membership[0],audit.membership[1],audit.membership[2],audit.membership[3],
        audit.unexpected,audit.mismatches(),audit.autapses,audit.passed());
    let mut summary = OpenOptions::new().write(true).create_new(true).open(out.join("pairwise.json"))?;
    summary.write_all(s.as_bytes())?; summary.sync_all()?;
    if !audit.passed() { return Err("pairwise disagreement; diagnostic retained, graph not published".into()); }
    audit.write_csr(&out.join("graph.csr"))?;
    print!("{s}"); Ok(())
}
fn main() { if let Err(e) = run() { eprintln!("{e}"); std::process::exit(1); } }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_multiplicity_and_boundary() {
        let mut a = Audit::new(vec![10,20,30],vec![[10,20,2],[30,10,1]]).unwrap();
        for (p,q) in [(10,20),(30,10),(10,20),(10,99),(99,20),(99,98)] { a.observe(p,q).unwrap(); }
        assert!(a.passed()); assert_eq!(a.membership,[3,1,1,1]);
    }
    #[test]
    fn same_node_totals_can_hide_different_pairs() {
        let mut a = Audit::new(vec![1,2,3,4],vec![[1,3,1],[2,4,1]]).unwrap();
        a.observe(1,4).unwrap(); a.observe(2,3).unwrap();
        assert!(!a.passed()); assert_eq!(a.mismatches(),2); assert_eq!(a.unexpected,2);
    }
    #[test]
    fn integer_identity_above_float_precision() {
        let big = (1u64 << 54)+1;
        let mut a=Audit::new(vec![big,big+1],vec![[big,big+1,1]]).unwrap();
        a.observe(big,big+1).unwrap(); assert!(a.passed());
    }
    #[test]
    fn malformed_reference_fails() {
        assert!(Audit::new(vec![2,1],vec![]).is_err());
        assert!(Audit::new(vec![1,1],vec![]).is_err());
        assert!(Audit::new(vec![1,2],vec![[1,2,0]]).is_err());
        assert!(Audit::new(vec![1,2],vec![[1,2,1],[1,2,1]]).is_err());
        assert!(Audit::new(vec![1,2],vec![[1,3,1]]).is_err());
        assert!(Audit::new(vec![1,2],vec![[1,1,1]]).is_err());
    }
    #[test]
    fn mismatch_and_autapse_fail() {
        let mut a=Audit::new(vec![1,2],vec![[1,2,1]]).unwrap();
        assert!(!a.passed()); a.observe(1,2).unwrap(); assert!(a.passed());
        a.observe(1,2).unwrap(); assert!(!a.passed());
        a.observe(1,1).unwrap(); assert_eq!(a.autapses,1);
    }
    #[test]
    fn truncated_and_extra_bytes_fail() {
        assert!(u64_read(&mut &b"123"[..]).is_err());
        assert!(eof(&mut &b"x"[..]).is_err());
        let mut b=b"V8RAW001".to_vec(); b.extend_from_slice(&(MAX_RAW+1).to_le_bytes());
        assert!(header(&mut &b[..],b"V8RAW001",MAX_RAW).is_err());
    }
    #[test]
    fn overflow_is_not_saturation() {
        let mut value=u64::MAX; assert!(add(&mut value,1).is_err());
    }
}
