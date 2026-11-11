# llmcraft

[![Crates.io](https://img.shields.io/crates/v/llmcraft.svg)](https://crates.io/crates/llmcraft)
[![Docs.rs](https://docs.rs/llmcraft/badge.svg)](https://docs.rs/llmcraft)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight, universal multi-provider LLM gateway and token telemetry client for Rust.

Built and maintained by **[Kreive Tech](https://kreive.com.br)**.

---

## Features

- **Universal Multi-Provider Gateway**: One unified API for OpenAI, Anthropic Claude, Google Gemini, Ollama, DeepSeek, Groq, Mistral, and custom OpenAI-compatible endpoints (vLLM, LM Studio).
- **Advanced Capabilities**:
  - Ephemeral prompt caching and extended thinking budgets for Claude.
  - Thinking budget configuration for Gemini models.
  - Automatic `max_completion_tokens` and reasoning effort for modern OpenAI models.
- **Flexible Credential Injection**: Explicit runtime injection (`.api_key(...)` / `.base_url(...)`) with automatic fallback to standard environment variables.
- **Real-Time Token Streaming**: Unified asynchronous token streaming across all providers.
- **Cost & Token Telemetry**: Built-in pricing tables with real-time USD cost calculation and savings benchmarks.
- **Lightweight**: Zero heavyweight machine learning runtimes. Fast compilation with minimal dependencies.

---

## Installation

Add `llmcraft` to your `Cargo.toml`:

```toml
[dependencies]
llmcraft = "0.1.0"
tokio = { version = "1", features = ["full"] }
```

---

## Quick Examples

### 1. Basic Completion with Explicit API Key

```rust
use llmcraft::{LlmClient, Provider};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = LlmClient::builder(Provider::Claude)
        .model("claude-3-7-sonnet-20250219")
        .api_key("sk-ant-api03-...")
        .build()?;

    let response = client.complete("Explain concurrency in one sentence.").await?;
    println!("Response: {}", response.content);

    if let Some(cost) = response.cost_usd {
        println!("Estimated Cost: ${:.6}", cost);
    }

    Ok(())
}
```

---

### 2. Multi-Turn Conversation & System Prompt

```rust
use llmcraft::{ChatMessage, CompletionParams, LlmClient, Provider};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = LlmClient::builder(Provider::OpenAi)
        .model("gpt-4o")
        .build()?; // Reads OPENAI_API_KEY from environment

    let params = CompletionParams::new("What was my favorite number?")
        .with_system("You are an assistant with perfect memory.")
        .with_history(vec![
            ChatMessage::user("My favorite number is 42."),
            ChatMessage::assistant("Got it! I will remember that your favorite number is 42."),
        ])
        .with_temperature(0.3);

    let response = client.complete_with_params(params).await?;
    println!("{}", response.content);

    Ok(())
}
```

---

### 3. Real-Time Token Streaming

```rust
use llmcraft::{LlmClient, Provider};
use std::io::{self, Write};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = LlmClient::builder(Provider::Gemini)
        .model("gemini-2.5-flash")
        .build()?; // Reads GOOGLE_API_KEY from environment

    client
        .stream("Write a short story about an astronaut.", |token| {
            print!("{}", token);
            io::stdout().flush().unwrap();
        })
        .await?;

    println!();
    Ok(())
}
```

---

### 4. Local Offline Inference with Ollama

```rust
use llmcraft::{LlmClient, Provider};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = LlmClient::builder(Provider::Ollama)
        .model("llama3")
        .base_url("http://localhost:11434")
        .build()?;

    let response = client.complete("Hello from Rust!").await?;
    println!("{}", response.content);

    Ok(())
}
```

---

### 5. Token Usage & Cost Estimation

```rust
use llmcraft::pricing::calculate_cost_and_saved;

fn main() {
    let (cost, saved) = calculate_cost_and_saved("claude-3-5-sonnet", 1200, 450);
    println!("Actual Cost: ${:.6}", cost);
    println!("Saved compared to flagship models: ${:.6}", saved);
}
```

---

## Supported Providers & Environment Variables

| Provider | Enum Variant | Default Env Var | Default Base URL |
| :--- | :--- | :--- | :--- |
| **OpenAI** | `Provider::OpenAi` | `OPENAI_API_KEY` | `https://api.openai.com/v1` |
| **Anthropic Claude** | `Provider::Claude` | `ANTHROPIC_API_KEY` | `https://api.anthropic.com/v1` |
| **Google Gemini** | `Provider::Gemini` | `GOOGLE_API_KEY` | `https://generativelanguage.googleapis.com/v1beta` |
| **Ollama** | `Provider::Ollama` | `OLLAMA_API` (optional) | `http://localhost:11434` |
| **DeepSeek** | `Provider::DeepSeek` | `DEEPSEEK_API_KEY` | `https://api.deepseek.com` |
| **Groq** | `Provider::Groq` | `GROQ_API_KEY` | `https://api.groq.com/openai/v1` |
| **Mistral AI** | `Provider::Mistral` | `MISTRAL_API_KEY` | `https://api.mistral.ai/v1` |
| **Custom / vLLM** | `Provider::Custom(name)` | Any | User specified |

---

## Sponsor & Support

Developed by **[Kreive Tech](https://kreive.com.br)**.

For questions, enterprise support, or contributions, reach out to `contact@kreive.com.br`.

---

## License

MIT © [Kreive Tech](https://kreive.com.br)
