use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    pub fn as_str(&self) -> &str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
        }
    }
}

impl From<(String, String)> for ChatMessage {
    fn from((role, content): (String, String)) -> Self {
        let parsed_role = match role.trim().to_lowercase().as_str() {
            "system" => Role::System,
            "assistant" | "model" => Role::Assistant,
            _ => Role::User,
        };
        Self {
            role: parsed_role,
            content,
        }
    }
}

impl From<(&str, &str)> for ChatMessage {
    fn from((role, content): (&str, &str)) -> Self {
        Self::from((role.to_string(), content.to_string()))
    }
}

impl From<(&String, &String)> for ChatMessage {
    fn from((role, content): (&String, &String)) -> Self {
        Self::from((role.clone(), content.clone()))
    }
}

impl From<(Role, String)> for ChatMessage {
    fn from((role, content): (Role, String)) -> Self {
        Self { role, content }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RateLimits {
    pub remaining_requests: Option<usize>,
    pub remaining_tokens: Option<usize>,
    pub limit_requests: Option<usize>,
    pub limit_tokens: Option<usize>,
    pub reset_requests: Option<f64>,
    pub reset_tokens: Option<f64>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub content: String,
    pub model: String,
    pub usage: Option<TokenUsage>,
    pub rate_limits: RateLimits,
    pub cost_usd: Option<f64>,
}

#[derive(Clone, Debug, Default)]
pub struct CompletionParams {
    pub prompt: String,
    pub system: Option<String>,
    pub history: Vec<ChatMessage>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f64>,
    pub reasoning_effort: Option<String>,
}

impl CompletionParams {
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            ..Default::default()
        }
    }

    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    /// Set conversation history from any collection/iterator of items convertible into `ChatMessage`
    /// (e.g. `Vec<ChatMessage>`, `Vec<(String, String)>`, `Vec<(&str, &str)>`).
    pub fn with_history<I, M>(mut self, history: I) -> Self
    where
        I: IntoIterator<Item = M>,
        M: Into<ChatMessage>,
    {
        self.history = history.into_iter().map(Into::into).collect();
        self
    }

    /// Convenience method to inject slices of tuple history `&[(String, String)]`
    /// (used extensively in agent engines like Uneven).
    pub fn with_history_tuples(mut self, history: &[(String, String)]) -> Self {
        self.history = history
            .iter()
            .map(|(r, c)| ChatMessage::from((r.as_str(), c.as_str())))
            .collect();
        self
    }

    pub fn with_max_tokens(mut self, max: usize) -> Self {
        self.max_tokens = Some(max);
        self
    }

    pub fn with_temperature(mut self, temp: f64) -> Self {
        self.temperature = Some(temp);
        self
    }

    pub fn with_reasoning_effort(mut self, effort: impl Into<String>) -> Self {
        self.reasoning_effort = Some(effort.into());
        self
    }
}
