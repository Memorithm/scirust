pub mod autodiff;
pub mod constrained;
pub mod symbolic;

use scirust_symbolic::Expr;
use std::collections::HashMap;

/// One symbolic Euler-Lagrange residual associated with a generalized coordinate.
#[derive(Debug, Clone)]
pub struct ELEquation {
    /// Name of the generalized coordinate represented by this equation.
    pub coordinate: String,
    /// Symbolic residual whose equation of motion is `residual = 0`.
    pub residual: Expr,
    /// Acceleration variable names referenced by the residual.
    pub acceleration_deps: Vec<String>,
    /// Whether the equation is represented explicitly in acceleration form.
    pub is_explicit: bool,
}

impl std::fmt::Display for ELEquation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Euler–Lagrange for q[{}]: {} = 0",
            self.coordinate, self.residual
        )
    }
}

/// Symbolic Euler-Lagrange derivation for a Lagrangian and its coordinates.
#[derive(Debug, Clone)]
pub struct ELDerivation {
    /// Derived equation for each generalized coordinate.
    pub equations: Vec<ELEquation>,
    /// Symbolic Lagrangian used to derive the equations.
    pub lagrangian: Expr,
    /// Ordered generalized-coordinate names.
    pub coordinates: Vec<String>,
    /// Optional symbolic time-variable name.
    pub time_var: Option<String>,
}

impl ELDerivation {
    /// Returns the number of generalized coordinates in the derivation.
    pub fn num_coordinates(&self) -> usize {
        self.coordinates.len()
    }

    /// Returns whether every derived equation is marked explicit in acceleration.
    pub fn is_acceleration_explicit(&self) -> bool {
        self.equations.iter().all(|eq| eq.is_explicit)
    }

    /// Finds the Euler-Lagrange equation associated with `coord`.
    pub fn get_equation(&self, coord: &str) -> Option<&ELEquation> {
        self.equations.iter().find(|eq| eq.coordinate == coord)
    }
}

/// Recursively substitutes every variable named `from` with `to` in an expression tree.
pub fn substitute(expr: &Expr, from: &str, to: &Expr) -> Expr {
    match expr
    {
        Expr::Const(_) => expr.clone(),
        Expr::Var(v) =>
        {
            if v == from
            {
                to.clone()
            }
            else
            {
                expr.clone()
            }
        },
        Expr::Add(a, b) => substitute(a, from, to) + substitute(b, from, to),
        Expr::Sub(a, b) => substitute(a, from, to) - substitute(b, from, to),
        Expr::Mul(a, b) => substitute(a, from, to) * substitute(b, from, to),
        Expr::Div(a, b) => substitute(a, from, to) / substitute(b, from, to),
        Expr::Neg(a) => -substitute(a, from, to),
        Expr::Pow(a, b) => Expr::Pow(
            Box::new(substitute(a, from, to)),
            Box::new(substitute(b, from, to)),
        ),
        Expr::Sin(a) => Expr::Sin(Box::new(substitute(a, from, to))),
        Expr::Cos(a) => Expr::Cos(Box::new(substitute(a, from, to))),
        Expr::Exp(a) => Expr::Exp(Box::new(substitute(a, from, to))),
        Expr::Ln(a) => Expr::Ln(Box::new(substitute(a, from, to))),
        Expr::Sqrt(a) => Expr::Sqrt(Box::new(substitute(a, from, to))),
        Expr::Abs(a) => Expr::Abs(Box::new(substitute(a, from, to))),
    }
}

/// Collects the distinct variable names occurring in `expr` in sorted order.
pub fn collect_vars_set(expr: &Expr) -> Vec<String> {
    let mut vars = std::collections::BTreeSet::new();
    collect_vars_into(expr, &mut vars);
    vars.into_iter().collect()
}

fn collect_vars_into(expr: &Expr, out: &mut std::collections::BTreeSet<String>) {
    match expr
    {
        Expr::Var(v) =>
        {
            out.insert(v.clone());
        },
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) | Expr::Pow(a, b) =>
        {
            collect_vars_into(a, out);
            collect_vars_into(b, out);
        },
        Expr::Neg(a)
        | Expr::Sin(a)
        | Expr::Cos(a)
        | Expr::Exp(a)
        | Expr::Ln(a)
        | Expr::Sqrt(a)
        | Expr::Abs(a) =>
        {
            collect_vars_into(a, out);
        },
        Expr::Const(_) =>
        {},
    }
}

/// Builds symbolic coordinate, velocity, and acceleration variables for a Lagrangian.
///
/// For each coordinate `q`, the returned vectors contain `q`, `q_dot`, and `q_ddot`.
/// The fourth vector mirrors the acceleration symbols for callers that need a dedicated
/// acceleration dependency list. When `time_label` is supplied, the returned binding map
/// contains that time symbol.
pub fn make_lagrangian_symbolic(
    coords: &[&str],
    time_label: Option<&str>,
) -> (
    Vec<Expr>,
    Vec<Expr>,
    Vec<Expr>,
    Vec<Expr>,
    HashMap<String, Expr>,
) {
    use scirust_symbolic::Expr;

    let mut q_vars = Vec::new();
    let mut dq_vars = Vec::new();
    let mut ddq_vars = Vec::new();
    let mut accel_vars = Vec::new();
    let mut bindings = HashMap::new();

    for &c in coords
    {
        let q = Expr::Var(c.to_string());
        let dq = Expr::Var(format!("{}_dot", c));
        let ddq = Expr::Var(format!("{}_ddot", c));
        q_vars.push(q);
        dq_vars.push(dq.clone());
        ddq_vars.push(ddq.clone());
        accel_vars.push(ddq);
    }

    if let Some(t) = time_label
    {
        let t_var = Expr::Var(t.to_string());
        bindings.insert(t.to_string(), t_var);
    }

    (q_vars, dq_vars, ddq_vars, accel_vars, bindings)
}
