//! Streaming weighted endpoint counts. Research executable; not a public API.
//! Input: V888CNT1 followed by little-endian (pre: u64, post: u64, weight: u64).
//! Node TSV: strictly increasing root ID and integer annotation-group ID (0 = missing).
//! O(nodes + observed group pairs) memory, independent of synapse row count.
#![forbid(unsafe_code)]
use std::collections::{BTreeMap, HashMap};
use std::error::Error;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};

#[derive(Default, Debug)]
struct Counts {
    records: u64,
    weight: u64,
    unknown_pre: u64,
    unknown_post: u64,
    autapses: u64,
    annotated_both: u64,
    same_group: u64,
    incoming: Vec<u64>,
    outgoing: Vec<u64>,
    mixing: BTreeMap<(u64, u64), u64>,
}
fn add(target: &mut u64, value: u64) -> io::Result<()> {
    *target = target.checked_add(value).ok_or_else(|| io::Error::other("counter overflow"))?;
    Ok(())
}
fn aggregate<R: Read>(mut reader: R, ids: &[u64], groups: &[u64]) -> io::Result<Counts> {
    if ids.len() != groups.len() || ids.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(io::Error::other("node IDs must be unique and strictly increasing"));
    }
    let index: HashMap<u64, usize> = ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    let mut header = [0_u8; 8];
    reader.read_exact(&mut header)?;
    if &header != b"V888CNT1" { return Err(io::Error::other("invalid stream header")); }
    let mut counts = Counts { incoming: vec![0; ids.len()], outgoing: vec![0; ids.len()], ..Counts::default() };
    loop {
        let mut record = [0_u8; 24];
        let mut used = 0;
        while used < record.len() {
            match reader.read(&mut record[used..]) {
                Ok(0) if used == 0 => return Ok(counts),
                Ok(0) => return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "partial endpoint record")),
                Ok(n) => used += n,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            }
        }
        let pre = u64::from_le_bytes(record[0..8].try_into().unwrap());
        let post = u64::from_le_bytes(record[8..16].try_into().unwrap());
        let weight = u64::from_le_bytes(record[16..24].try_into().unwrap());
        if weight == 0 { return Err(io::Error::other("zero edge weight")); }
        add(&mut counts.records, 1)?;
        add(&mut counts.weight, weight)?;
        if pre == post { add(&mut counts.autapses, weight)?; }
        let a = index.get(&pre).copied();
        let b = index.get(&post).copied();
        match a { Some(i) => add(&mut counts.outgoing[i], weight)?, None => add(&mut counts.unknown_pre, weight)? }
        match b { Some(i) => add(&mut counts.incoming[i], weight)?, None => add(&mut counts.unknown_post, weight)? }
        if let (Some(i), Some(j)) = (a, b) {
            let g = groups[i];
            let h = groups[j];
            if g != 0 && h != 0 {
                add(&mut counts.annotated_both, weight)?;
                if g == h { add(&mut counts.same_group, weight)?; }
                add(counts.mixing.entry((g, h)).or_default(), weight)?;
            }
        }
    }
}
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 { return Err("usage: endpoint_counts NODE_TSV OUTPUT_PREFIX".into()); }
    let mut ids = Vec::new();
    let mut groups = Vec::new();
    for line in BufReader::new(File::open(&args[1])?).lines() {
        let line = line?;
        let (id, group) = line.split_once('\t').ok_or("invalid node TSV")?;
        ids.push(id.parse::<u64>()?);
        groups.push(group.parse::<u64>()?);
    }
    let counts = aggregate(BufReader::with_capacity(1024 * 1024, io::stdin().lock()), &ids, &groups)?;
    let mut out = BufWriter::new(File::create(format!("{}.nodes.csv", args[2]))?);
    writeln!(out, "root_id,incoming_v3,outgoing_v3")?;
    for (i, id) in ids.iter().enumerate() { writeln!(out, "{id},{},{}", counts.incoming[i], counts.outgoing[i])?; }
    out.flush()?;
    let mut mix = BufWriter::new(File::create(format!("{}.mixing.csv", args[2]))?);
    writeln!(mix, "pre_group,post_group,weight")?;
    for ((a,b), weight) in &counts.mixing { writeln!(mix, "{a},{b},{weight}")?; }
    mix.flush()?;
    let text = format!("{{\"records\":{},\"weight\":{},\"unknown_pre_weight\":{},\"unknown_post_weight\":{},\"autapse_weight\":{},\"annotated_both_weight\":{},\"same_annotation_group_weight\":{}}}\n", counts.records, counts.weight, counts.unknown_pre, counts.unknown_post, counts.autapses, counts.annotated_both, counts.same_group);
    std::fs::write(format!("{}.summary.json", args[2]), text)?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    fn stream(records: &[(u64,u64,u64)]) -> Vec<u8> {
        let mut out = b"V888CNT1".to_vec();
        for &(a,b,w) in records { for x in [a,b,w] { out.extend(x.to_le_bytes()); } }
        out
    }
    #[test]
    fn exact_weighted_counts_and_missing_endpoints() {
        let c = aggregate(&stream(&[(10,20,2),(20,10,3),(10,10,1),(99,20,4)])[..], &[10,20], &[1,1]).unwrap();
        assert_eq!(c.incoming, [4,6]); assert_eq!(c.outgoing, [3,3]);
        assert_eq!((c.records,c.weight,c.unknown_pre,c.autapses,c.same_group), (4,10,4,1,6));
    }
    #[test]
    fn incomplete_records_rejected() {
        let mut bytes = stream(&[(1,2,1)]); bytes.pop();
        assert!(aggregate(&bytes[..], &[1,2], &[0,0]).is_err());
    }
    #[test]
    fn duplicate_nodes_and_zero_weights_rejected() {
        assert!(aggregate(&stream(&[])[..], &[1,1], &[0,0]).is_err());
        assert!(aggregate(&stream(&[(1,2,0)])[..], &[1,2], &[0,0]).is_err());
    }
    #[test]
    fn integer_overflow_rejected() {
        assert!(aggregate(&stream(&[(1,2,u64::MAX),(1,2,1)])[..], &[1,2], &[0,0]).is_err());
    }
    #[test]
    fn ids_above_f64_precision_preserved() {
        let a = (1_u64 << 54) + 1; let b = a + 1;
        let c = aggregate(&stream(&[(a,b,1)])[..], &[a,b], &[1,2]).unwrap();
        assert_eq!(c.incoming, [0,1]); assert_eq!(c.outgoing, [1,0]); assert_eq!(c.same_group,0);
    }
}
