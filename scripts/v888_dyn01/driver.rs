//! Administrative differential-test driver. Never a game candidate input protocol.
#![forbid(unsafe_code)]
use recurrent::{Dynamics, Edge, Limits, Network};
use std::io::{self, BufRead, Write};
use std::str::FromStr;

fn value<T: FromStr>(fields: &mut std::str::SplitWhitespace<'_>) -> Result<T, &'static str> {
    fields.next().ok_or("missing field")?.parse().map_err(|_| "invalid field")
}

fn execute(line: &str) -> Result<String, &'static str> {
    let mut f = line.split_whitespace();
    if f.next() != Some("DYN1") { return Err("wrong protocol version"); }
    let kind = f.next().ok_or("missing mode")?;
    let n: usize = value(&mut f)?;
    let e: usize = value(&mut f)?;
    let t: usize = value(&mut f)?;
    if n == 0 || n > 64 || e > 512 || t == 0 || t > 128 { return Err("administrative fixture bounds"); }
    let p = f.next().ok_or("missing p")?;
    let q = f.next().ok_or("missing q")?;
    let r = f.next().ok_or("missing r")?;
    let refractory: u16 = value(&mut f)?;
    let dynamics = match kind {
        "lif" => {
            if r != "0" { return Err("unused LIF parameter must be zero"); }
            Dynamics::Lif { leak:p.parse().map_err(|_| "bad leak")?, threshold:q.parse().map_err(|_| "bad threshold")?, refractory }
        }
        "integer" => Dynamics::Integer {
            divisor:p.parse().map_err(|_| "bad divisor")?, threshold:q.parse().map_err(|_| "bad threshold")?,
            cap:r.parse().map_err(|_| "bad cap")?, refractory,
        },
        "boolean" => {
            if q != "0" || r != "0" || refractory != 0 { return Err("unused Boolean parameters must be zero"); }
            Dynamics::Boolean { nonlinear: match p { "0"=>false,"1"=>true,_=>return Err("bad Boolean rule") } }
        }
        _ => return Err("unknown mode"),
    };
    let limits = Limits {max_ticks:value(&mut f)?,max_edge_visits:value(&mut f)?,max_payload_bytes:value(&mut f)?};
    let mut edges=Vec::with_capacity(e);
    for _ in 0..e { edges.push(Edge{source:value(&mut f)?,target:value(&mut f)?,weight:value(&mut f)?,delay:value(&mut f)?}); }
    let mut network=Network::new(n,&edges,dynamics,limits)?;
    let parameters_before=network.parameters();
    let mut traces=Vec::with_capacity(t);
    for _ in 0..t {
        let mut drive=Vec::with_capacity(n);
        for _ in 0..n { drive.push(value::<i32>(&mut f)?); }
        network.step(&drive)?;
        let c=network.counters();
        traces.push(format!("{{\"spikes\":{:?},\"integer\":{:?},\"voltage\":{:?},\"refractory\":{:?},\"counters\":{{\"ticks\":{},\"edge_visits\":{},\"active_edge_events\":{},\"spikes\":{},\"clipped_nodes\":{},\"refractory_nodes\":{}}}}",
            network.spikes(),network.integer_state(),network.voltage_state(),network.refractory_state(),
            c.ticks,c.edge_visits,c.active_edge_events,c.spikes,c.clipped_nodes,c.refractory_nodes));
    }
    if f.next().is_some(){return Err("trailing fixture fields");}
    let payload=network.payload_bytes();let edges_retained=network.edges().len();
    network.reset_activity();
    let reset_zero=network.counters()==Default::default() && network.spikes().iter().all(|&x|x==0)
        && network.integer_state().iter().all(|&x|x==0) && network.voltage_state().iter().all(|&x|x==0.0)
        && network.refractory_state().iter().all(|&x|x==0);
    Ok(format!("{{\"schema\":\"dyn-ref1-trace\",\"ticks\":[{}],\"direct_vector_payload_bytes\":{},\"edge_count\":{},\"reset_zero\":{},\"parameters_preserved\":{},\"learning_performed\":false}}",
        traces.join(","),payload,edges_retained,reset_zero,network.parameters()==parameters_before))
}

fn run() -> Result<(), String> {
    if std::env::args().skip(1).collect::<Vec<_>>() != ["--batch"] {return Err("usage: dyn-ref1 --batch".into());}
    let stdin=io::stdin();let mut input=stdin.lock();let stdout=io::stdout();let mut output=io::BufWriter::new(stdout.lock());
    loop {
        let mut bytes=Vec::new();
        loop {
            let available=input.fill_buf().map_err(|e|e.to_string())?;
            if available.is_empty(){break;}
            let count=available.iter().position(|&x|x==b'\n').map_or(available.len(),|i|i+1);
            if bytes.len()+count>131_072{return Err("fixture line exceeds 128 KiB".into());}
            bytes.extend_from_slice(&available[..count]);let done=available[count-1]==b'\n';input.consume(count);if done{break;}
        }
        if bytes.is_empty(){break;}
        let line=std::str::from_utf8(&bytes).map_err(|_|"invalid UTF8")?;
        let result=execute(line).map_err(str::to_owned)?;
        writeln!(output,"{result}").map_err(|e|e.to_string())?;
    }
    output.flush().map_err(|e|e.to_string())
}
fn main(){if let Err(e)=run(){eprintln!("protocol_error: {e}");std::process::exit(2);}}
