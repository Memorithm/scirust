//! Registre d'outils MCP.
//!
//! Chaque outil expose un nom, une description, un schéma JSON d'entrée
//! (pour que le client — LLM ou script — sache le construire sans deviner),
//! et un handler synchrone `Value -> Result<Value, String>`. C'est le point
//! d'extension unique du serveur : un nouveau domaine SciRust s'enregistre
//! ici et devient immédiatement appelable par n'importe quel agent MCP.

use serde_json::Value;

pub type ToolHandler = Box<dyn Fn(Value) -> Result<Value, String> + Send + Sync>;

pub struct McpTool {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
    pub handler: ToolHandler,
}

#[derive(Default)]
pub struct ToolRegistry {
    tools: Vec<McpTool>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Enregistre un outil. Panique si son nom est déjà pris — une
    /// collision de noms est une erreur de configuration du serveur, pas un
    /// cas d'exécution à gérer silencieusement.
    pub fn register(&mut self, tool: McpTool) {
        assert!(
            !self.tools.iter().any(|t| t.name == tool.name),
            "duplicate MCP tool name: {}",
            tool.name
        );
        self.tools.push(tool);
    }

    pub fn list_json(&self) -> Value {
        Value::Array(
            self.tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "name": t.name,
                        "description": t.description,
                        "inputSchema": t.input_schema,
                    })
                })
                .collect(),
        )
    }

    pub fn call(&self, name: &str, arguments: Value) -> Result<Value, String> {
        let tool = self
            .tools
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| format!("unknown tool: {name}"))?;

        // The published schema is part of the executable MCP contract. Reject
        // basic schema violations before any domain handler gets to apply its
        // own defaults or conversions. This validator intentionally implements
        // the JSON-Schema subset currently used by the registry: type, enum,
        // required, properties and items. Domain semantics remain in their
        // dedicated boundary/domain validators.
        validate_schema(&arguments, &tool.input_schema, "$")?;

        // Trading accepts untrusted time-series data and historically contained
        // permissive parser fallbacks. Keep those checks at the common MCP
        // dispatch boundary so all production calls receive the same policy.
        let arguments = crate::tools::trader_guard::prepare_arguments(name, arguments)?;
        (tool.handler)(arguments)
    }

    pub fn names(&self) -> Vec<&str> {
        self.tools.iter().map(|t| t.name.as_str()).collect()
    }

    pub fn len(&self) -> usize {
        self.tools.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

fn validate_schema(value: &Value, schema: &Value, path: &str) -> Result<(), String> {
    if let Some(allowed) = schema.get("enum").and_then(Value::as_array)
    {
        if !allowed.iter().any(|candidate| candidate == value)
        {
            return Err(schema_error(
                path,
                "value is not one of the schema enum choices",
            ));
        }
    }

    if let Some(kind) = schema.get("type")
    {
        let type_matches = match kind
        {
            Value::String(expected) => matches_type(value, expected),
            Value::Array(expected) => expected
                .iter()
                .filter_map(Value::as_str)
                .any(|expected| matches_type(value, expected)),
            _ => true,
        };
        if !type_matches
        {
            return Err(schema_error(path, "value has the wrong JSON type"));
        }
    }

    if let Some(object) = value.as_object()
    {
        if let Some(required) = schema.get("required").and_then(Value::as_array)
        {
            for key in required.iter().filter_map(Value::as_str)
            {
                if !object.contains_key(key)
                {
                    return Err(schema_error(
                        &format!("{path}.{key}"),
                        "required field is missing",
                    ));
                }
            }
        }
        if let Some(properties) = schema.get("properties").and_then(Value::as_object)
        {
            for (key, child_schema) in properties
            {
                if let Some(child) = object.get(key)
                {
                    validate_schema(child, child_schema, &format!("{path}.{key}"))?;
                }
            }
        }
    }

    if let Some(items) = value.as_array()
    {
        if let Some(item_schema) = schema.get("items")
        {
            for (index, item) in items.iter().enumerate()
            {
                validate_schema(item, item_schema, &format!("{path}[{index}]"))?;
            }
        }
    }

    Ok(())
}

fn matches_type(value: &Value, expected: &str) -> bool {
    match expected
    {
        "object" => value.is_object(),
        "array" => value.is_array(),
        "string" => value.is_string(),
        "number" => value.as_f64().is_some(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "boolean" => value.is_boolean(),
        "null" => value.is_null(),
        _ => true,
    }
}

fn schema_error(path: &str, message: &str) -> String {
    format!("invalid_arguments: {path}: {message}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn dummy_tool(name: &str) -> McpTool {
        McpTool {
            name: name.to_string(),
            description: "a dummy tool".to_string(),
            input_schema: json!({"type": "object"}),
            handler: Box::new(|_args| Ok(json!({"ok": true}))),
        }
    }

    #[test]
    fn register_and_call() {
        let mut reg = ToolRegistry::new();
        reg.register(dummy_tool("echo"));
        assert_eq!(reg.len(), 1);
        let result = reg.call("echo", json!({})).unwrap();
        assert_eq!(result, json!({"ok": true}));
    }

    #[test]
    fn call_unknown_tool_errors() {
        let reg = ToolRegistry::new();
        assert!(reg.call("nope", json!({})).is_err());
    }

    #[test]
    #[should_panic(expected = "duplicate MCP tool name")]
    fn duplicate_registration_panics() {
        let mut reg = ToolRegistry::new();
        reg.register(dummy_tool("echo"));
        reg.register(dummy_tool("echo"));
    }

    #[test]
    fn list_json_includes_schema() {
        let mut reg = ToolRegistry::new();
        reg.register(dummy_tool("echo"));
        let list = reg.list_json();
        assert_eq!(list[0]["name"], "echo");
        assert_eq!(list[0]["inputSchema"]["type"], "object");
    }

    #[test]
    fn call_enforces_required_type_and_enum_from_published_schema() {
        let mut reg = ToolRegistry::new();
        reg.register(McpTool {
            name: "schema_contract".to_string(),
            description: "schema contract test".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["paper", "replay"]},
                    "count": {"type": "integer"}
                },
                "required": ["mode", "count"]
            }),
            handler: Box::new(Ok),
        });

        assert!(reg.call("schema_contract", json!({"count": 1})).is_err());
        assert!(
            reg.call("schema_contract", json!({"mode": "live", "count": 1}))
                .is_err()
        );
        assert!(
            reg.call("schema_contract", json!({"mode": "paper", "count": 1.5}))
                .is_err()
        );
        assert_eq!(
            reg.call("schema_contract", json!({"mode": "replay", "count": 2}))
                .unwrap(),
            json!({"mode": "replay", "count": 2})
        );
    }
}
