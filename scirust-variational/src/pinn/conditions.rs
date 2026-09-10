use crate::error::{Result, VariationalError};

#[derive(Debug, Clone)]
pub enum ConditionKind {
    Initial,
    Dirichlet,
    Neumann,
    Robin,
    Periodic,
}

impl std::fmt::Display for ConditionKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Initial => write!(f, "initial"),
            Self::Dirichlet => write!(f, "Dirichlet"),
            Self::Neumann => write!(f, "Neumann"),
            Self::Robin => write!(f, "Robin"),
            Self::Periodic => write!(f, "periodic"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Condition {
    pub kind: ConditionKind,
    pub points: Vec<Vec<f32>>,
    pub target_fn: fn(&[f32]) -> f32,
    pub weight: f32,
    pub name: String,
}

impl Condition {
    pub fn new(
        kind: ConditionKind,
        points: Vec<Vec<f32>>,
        target_fn: fn(&[f32]) -> f32,
        weight: f32,
        name: &str,
    ) -> Result<Self> {
        validate_weight(weight, name)?;
        validate_points(&points, name)?;
        Ok(Self {
            kind,
            points,
            target_fn,
            weight,
            name: name.to_string(),
        })
    }

    /// Evaluates scalar boundary targets and rejects non-finite callback results.
    pub fn try_evaluate_targets(&self) -> Result<Vec<f32>> {
        validate_weight(self.weight, &self.name)?;
        validate_points(&self.points, &self.name)?;
        let mut targets = Vec::with_capacity(self.points.len());
        for point in &self.points {
            let value = (self.target_fn)(point);
            if !value.is_finite() {
                return Err(VariationalError::NonFiniteValue {
                    component: "PINN boundary target",
                    value,
                });
            }
            targets.push(value);
        }
        Ok(targets)
    }

    pub fn evaluate_targets(&self) -> Vec<f32> {
        self.try_evaluate_targets()
            .expect("Condition::evaluate_targets requires valid finite boundary data")
    }
}

#[derive(Debug, Clone)]
pub struct ConditionConfig {
    pub conditions: Vec<Condition>,
}

impl ConditionConfig {
    pub fn new() -> Self {
        Self {
            conditions: Vec::new(),
        }
    }

    pub fn with_condition(mut self, condition: Condition) -> Self {
        self.conditions.push(condition);
        self
    }

    pub fn total_points(&self) -> usize {
        self.conditions.iter().map(|c| c.points.len()).sum()
    }
}

impl Default for ConditionConfig {
    fn default() -> Self {
        Self::new()
    }
}

fn validate_weight(weight: f32, name: &str) -> Result<()> {
    if !weight.is_finite() || weight < 0.0 {
        return Err(VariationalError::InvalidBoundaryCondition {
            details: format!("condition '{name}' has invalid weight {weight}"),
        });
    }
    Ok(())
}

fn validate_points(points: &[Vec<f32>], name: &str) -> Result<()> {
    if points.is_empty() {
        return Err(VariationalError::InvalidBoundaryCondition {
            details: format!("condition '{name}' has no points"),
        });
    }
    let ndim = points[0].len();
    if ndim == 0 {
        return Err(VariationalError::InvalidBoundaryCondition {
            details: format!("condition '{name}' contains zero-dimensional points"),
        });
    }
    for point in points {
        if point.len() != ndim {
            return Err(VariationalError::InvalidBoundaryCondition {
                details: format!(
                    "condition '{name}' has inconsistent point dimensions: expected {ndim}, got {}",
                    point.len()
                ),
            });
        }
        if let Some(&value) = point.iter().find(|value| !value.is_finite()) {
            return Err(VariationalError::NonFiniteValue {
                component: "PINN boundary point",
                value,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_condition_creation() {
        let pts = vec![vec![0.0], vec![1.0]];
        let c = Condition::new(
            ConditionKind::Dirichlet,
            pts.clone(),
            |x| x[0].sin(),
            1.0,
            "test",
        )
        .unwrap();
        assert_eq!(c.points.len(), 2);
        let targets = c.evaluate_targets();
        assert!((targets[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_negative_weight_error() {
        let result = Condition::new(
            ConditionKind::Dirichlet,
            vec![vec![0.0]],
            |_| 0.0,
            -1.0,
            "bad",
        );
        assert!(result.is_err());
    }

    #[test]
    fn condition_rejects_nan_weight() {
        assert!(
            Condition::new(
                ConditionKind::Dirichlet,
                vec![vec![0.0]],
                |_| 0.0,
                f32::NAN,
                "bad"
            )
            .is_err()
        );
    }

    #[test]
    fn condition_rejects_inconsistent_or_non_finite_points() {
        assert!(
            Condition::new(
                ConditionKind::Dirichlet,
                vec![vec![0.0], vec![1.0, 2.0]],
                |_| 0.0,
                1.0,
                "bad"
            )
            .is_err()
        );
        assert!(
            Condition::new(
                ConditionKind::Dirichlet,
                vec![vec![f32::NAN]],
                |_| 0.0,
                1.0,
                "bad"
            )
            .is_err()
        );
    }

    #[test]
    fn checked_target_evaluation_rejects_non_finite_result() {
        let condition = Condition::new(
            ConditionKind::Dirichlet,
            vec![vec![0.0]],
            |_| f32::NAN,
            1.0,
            "nan-target",
        )
        .unwrap();
        assert!(condition.try_evaluate_targets().is_err());
    }

    #[test]
    fn checked_target_evaluation_revalidates_mutated_state() {
        let mut condition = Condition::new(
            ConditionKind::Dirichlet,
            vec![vec![0.0]],
            |_| 0.0,
            1.0,
            "mutated",
        )
        .unwrap();
        condition.weight = f32::NAN;
        assert!(condition.try_evaluate_targets().is_err());
    }
}
