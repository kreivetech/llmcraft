//! # llmcraft
//!
//! A lightweight, universal multi-provider LLM gateway and token telemetry client for Rust.
//!
//! Supports:
//! - **OpenAI** (GPT-4o, GPT-5, o1, o3, o4)
//! - **Anthropic Claude** (Claude 3.5, Claude 3.7 Sonnet, Claude 4) with prompt caching and extended thinking
//! - **Google Gemini** (Gemini 2.0, 2.5, 3.1, 3.5) with thinking budget
//! - **Ollama** (Local offline inference)
//! - **DeepSeek** (DeepSeek Chat, DeepSeek Reasoner)
//! - **Groq** (Fast Llama and Qwen models)
//! - **Mistral AI** (Mistral Large, Codestral, Mixtral)
//! - **Custom OpenAI-compatible endpoints** (vLLM, LM Studio, etc.)
//!
//! Includes real-time token tracking and estimated USD cost calculation.
//!
//! ## Quick Start
//!
//! ```no_run
//! use llmcraft::{LlmClient, Provider, CompletionParams};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Explicitly inject API key, or let it fallback to ANTHROPIC_API_KEY
//!     let client = LlmClient::builder(Provider::Claude)
//!         .model("claude-3-7-sonnet-20250219")
//!         .api_key("sk-ant-api03-...")
//!         .build()?;
//!
//!     let response = client.complete("Explain Rust concurrency in one sentence.").await?;
//!     println!("Response: {}", response.content);
//!
//!     if let Some(cost) = response.cost_usd {
//!         println!("Estimated Cost: ${:.6}", cost);
//!     }
//!
//!     Ok(())
//! }
//! ```

pub mod client;
pub mod error;
pub mod pricing;
pub mod protocols;
pub mod provider;
pub mod types;

pub use client::{ClientBuilder, LlmClient};
pub use error::LlmError;
pub use pricing::{
    calculate_cost_and_saved, get_comparison_rates, get_model_rates,
    get_model_rates_by_provider_and_model, load_stats_from_path, record_call_stat, resolve_rate,
    save_stats_to_path, CallStat, ModelCostRule, SessionStats, COST_TABLE,
};
pub use provider::Provider;
pub use types::{
    ChatMessage, CompletionParams, CompletionResponse, RateLimits, Role, TokenUsage,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_parsing() {
        assert_eq!(Provider::from_string("openai").unwrap(), Provider::OpenAi);
        assert_eq!(Provider::from_string("claude").unwrap(), Provider::Claude);
        assert_eq!(Provider::from_string("anthropic").unwrap(), Provider::Claude);
        assert_eq!(Provider::from_string("gemini").unwrap(), Provider::Gemini);
        assert_eq!(Provider::from_string("google").unwrap(), Provider::Gemini);
        assert_eq!(Provider::from_string("ollama").unwrap(), Provider::Ollama);
        assert_eq!(Provider::from_string("deepseek").unwrap(), Provider::DeepSeek);
        assert_eq!(Provider::from_string("mistral").unwrap(), Provider::Mistral);
        assert_eq!(Provider::from_string("groq").unwrap(), Provider::Groq);
        assert!(matches!(
            Provider::from_string("vllm-local").unwrap(),
            Provider::Custom(name) if name == "vllm-local"
        ));
    }

    #[test]
    fn test_client_builder_with_explicit_key() {
        let client = LlmClient::builder(Provider::OpenAi)
            .model("gpt-4o")
            .api_key("test-key-12345")
            .build()
            .unwrap();

        assert_eq!(client.api_key.as_deref(), Some("test-key-12345"));
        assert_eq!(client.model, "gpt-4o");
        assert_eq!(client.base_url, "https://api.openai.com/v1");
    }

    #[test]
    fn test_client_builder_ollama_requires_no_key() {
        let client = LlmClient::builder(Provider::Ollama)
            .model("llama3")
            .build()
            .unwrap();

        assert_eq!(client.model, "llama3");
        assert!(client.base_url.contains("11434"));
    }

    #[test]
    fn test_openai_request_serialization() {
        let params = CompletionParams::new("Hello world")
            .with_system("You are a helpful assistant")
            .with_max_tokens(4096)
            .with_temperature(0.7);

        let req = protocols::openai::build_openai_request("gpt-4o", &params, 4096, false);
        let json = serde_json::to_string(&req).unwrap();

        assert!(json.contains("max_completion_tokens"));
        assert!(!json.contains("\"max_tokens\""));
        assert!(json.contains("Hello world"));
        assert!(json.contains("You are a helpful assistant"));
    }

    #[test]
    fn test_openai_compat_request_serialization() {
        let params = CompletionParams::new("Hello world")
            .with_system("System prompt")
            .with_max_tokens(2048)
            .with_temperature(0.5);

        let req = protocols::openai::build_openai_compat_request("deepseek-chat", &params, 2048, false);
        let json = serde_json::to_string(&req).unwrap();

        assert!(json.contains("max_tokens"));
        assert!(!json.contains("reasoning_effort"));
        assert!(json.contains("System prompt"));
    }

    #[test]
    fn test_claude_request_caching_and_budget() {
        let params = CompletionParams::new("Write a poem")
            .with_system("Be creative")
            .with_reasoning_effort("high");

        let req = protocols::claude::build_claude_request("claude-3-7-sonnet-20250219", &params, 8192, false);
        let json = serde_json::to_string(&req).unwrap();

        assert!(json.contains("cache_control"));
        assert!(json.contains("ephemeral"));
        assert!(json.contains("budget_tokens"));
        assert!(json.contains("4096"));
    }

    #[test]
    fn test_gemini_thinking_budget() {
        let params = CompletionParams::new("Explain physics")
            .with_reasoning_effort("medium");

        let req = protocols::gemini::build_gemini_request("gemini-2.5-flash", &params, 8192);
        let json = serde_json::to_string(&req).unwrap();

        assert!(json.contains("thinkingBudget"));
        assert!(json.contains("2048"));
        assert!(!json.contains("thinkingLevel"));
    }

    #[test]
    fn test_pricing_calculation() {
        let (cost, saved) = calculate_cost_and_saved("claude-3-5-sonnet", 1000, 500);
        assert!(cost > 0.0);
        assert!(cost < 1.0);
        assert!(saved >= 0.0);
    }

    #[test]
    fn test_history_conversions_and_injection() {
        // From (String, String)
        let tuple_hist = vec![
            ("system".to_string(), "sys msg".to_string()),
            ("assistant".to_string(), "bot msg".to_string()),
            ("model".to_string(), "gemini bot msg".to_string()),
            ("user".to_string(), "user msg".to_string()),
        ];

        let params = CompletionParams::new("next question")
            .with_history(tuple_hist);

        assert_eq!(params.history.len(), 4);
        assert_eq!(params.history[0].role, Role::System);
        assert_eq!(params.history[1].role, Role::Assistant);
        assert_eq!(params.history[2].role, Role::Assistant); // "model" mapped to Assistant
        assert_eq!(params.history[3].role, Role::User);

        // From slice of tuples (Uneven pattern)
        let slice = [("user".to_string(), "test".to_string())];
        let params2 = CompletionParams::new("prompt").with_history_tuples(&slice);
        assert_eq!(params2.history.len(), 1);
        assert_eq!(params2.history[0].role, Role::User);

        // From &str literal tuples
        let str_tuples = vec![("user", "hello")];
        let params3 = CompletionParams::new("prompt").with_history(str_tuples);
        assert_eq!(params3.history.len(), 1);
        assert_eq!(params3.history[0].role, Role::User);
    }
}
