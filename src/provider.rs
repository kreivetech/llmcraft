use crate::error::LlmError;
use serde::{Deserialize, Serialize};
use std::str::FromStr;

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Provider {
    OpenAi,
    Claude,
    Gemini,
    Ollama,
    DeepSeek,
    Mistral,
    Groq,
    Custom(String),
}

impl Provider {
    pub fn from_string(s: &str) -> Result<Self, LlmError> {
        Self::from_str(s)
    }

    pub fn as_str(&self) -> &str {
        match self {
            Provider::OpenAi => "openai",
            Provider::Claude => "claude",
            Provider::Gemini => "gemini",
            Provider::Ollama => "ollama",
            Provider::DeepSeek => "deepseek",
            Provider::Mistral => "mistral",
            Provider::Groq => "groq",
            Provider::Custom(name) => name.as_str(),
        }
    }

    pub fn default_env_var(&self) -> Option<&'static str> {
        match self {
            Provider::OpenAi => Some("OPENAI_API_KEY"),
            Provider::Claude => Some("ANTHROPIC_API_KEY"),
            Provider::Gemini => Some("GOOGLE_API_KEY"),
            Provider::DeepSeek => Some("DEEPSEEK_API_KEY"),
            Provider::Mistral => Some("MISTRAL_API_KEY"),
            Provider::Groq => Some("GROQ_API_KEY"),
            Provider::Ollama => Some("OLLAMA_API"), // optional host url
            Provider::Custom(_) => None,
        }
    }

    pub fn default_base_url(&self) -> &str {
        match self {
            Provider::OpenAi => "https://api.openai.com/v1",
            Provider::Claude => "https://api.anthropic.com/v1",
            Provider::Gemini => "https://generativelanguage.googleapis.com/v1beta",
            Provider::DeepSeek => "https://api.deepseek.com",
            Provider::Mistral => "https://api.mistral.ai/v1",
            Provider::Groq => "https://api.groq.com/openai/v1",
            Provider::Ollama => "http://localhost:11434",
            Provider::Custom(_) => "",
        }
    }

    pub fn is_openai_compatible(&self) -> bool {
        matches!(
            self,
            Provider::OpenAi
                | Provider::DeepSeek
                | Provider::Mistral
                | Provider::Groq
                | Provider::Ollama
                | Provider::Custom(_)
        )
    }

    pub fn default_max_output_tokens(&self, model: &str) -> usize {
        let m = model.to_lowercase();
        match self {
            Provider::Gemini => {
                if m.contains("2.5") || m.contains("2.0") || m.contains("3.5") {
                    65536
                } else {
                    8192
                }
            }
            Provider::Claude => {
                if m.contains("3-7") || m.contains("3.7") || m.contains("4") {
                    64000
                } else if m.contains("3-5") || m.contains("3.5") {
                    8192
                } else {
                    4096
                }
            }
            Provider::OpenAi => {
                if m.contains("o1") || m.contains("o3") {
                    100000
                } else if m.contains("gpt-4o") || m.contains("gpt-5") {
                    16384
                } else {
                    4096
                }
            }
            Provider::DeepSeek | Provider::Groq | Provider::Mistral | Provider::Ollama => 8192,
            Provider::Custom(_) => 8192,
        }
    }
}

impl FromStr for Provider {
    type Err = LlmError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_lowercase().as_str() {
            "openai" => Ok(Provider::OpenAi),
            "claude" | "anthropic" => Ok(Provider::Claude),
            "gemini" | "google" => Ok(Provider::Gemini),
            "ollama" => Ok(Provider::Ollama),
            "deepseek" => Ok(Provider::DeepSeek),
            "mistral" => Ok(Provider::Mistral),
            "groq" => Ok(Provider::Groq),
            other if !other.is_empty() => Ok(Provider::Custom(other.to_string())),
            _ => Err(LlmError::UnknownProvider(s.to_string())),
        }
    }
}

impl std::fmt::Display for Provider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}
