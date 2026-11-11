use crate::error::LlmError;
use crate::protocols::ratelimits::extract_ratelimits;
use crate::types::{CompletionParams, CompletionResponse, Role, TokenUsage};
use futures::StreamExt;
use serde::Serialize;

#[derive(Serialize)]
pub struct ClaudeMessage {
    pub role: String,
    pub content: serde_json::Value,
}

#[derive(Serialize)]
pub struct ClaudeThinkingConfig {
    #[serde(rename = "type")]
    pub thinking_type: String,
    pub budget_tokens: usize,
}

#[derive(Serialize)]
pub struct ClaudeRequest {
    pub model: String,
    pub max_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<serde_json::Value>,
    pub messages: Vec<ClaudeMessage>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ClaudeThinkingConfig>,
}

pub fn build_claude_request(
    model: &str,
    params: &CompletionParams,
    default_max: usize,
    stream: bool,
) -> ClaudeRequest {
    let mut messages = Vec::new();
    for msg in &params.history {
        messages.push(ClaudeMessage {
            role: if msg.role == Role::Assistant {
                "assistant".to_string()
            } else {
                "user".to_string()
            },
            content: serde_json::Value::String(msg.content.clone()),
        });
    }

    messages.push(ClaudeMessage {
        role: "user".to_string(),
        content: serde_json::json!([
            {
                "type": "text",
                "text": params.prompt,
                "cache_control": { "type": "ephemeral" }
            }
        ]),
    });

    let system_value = params.system.as_ref().map(|sys| {
        serde_json::json!([
            {
                "type": "text",
                "text": sys,
                "cache_control": { "type": "ephemeral" }
            }
        ])
    });

    let user_max = params.max_tokens.unwrap_or(0);
    let mut final_max_tokens = if user_max == 0 || user_max <= 2048 {
        default_max
    } else {
        user_max.max(default_max)
    };

    let is_claude4 = model.contains("claude-opus-4")
        || model.contains("claude-sonnet-4")
        || model.contains("claude-haiku-4");

    let mut final_temperature = if is_claude4 { None } else { params.temperature };
    let mut thinking = None;

    if let Some(ref e) = params.reasoning_effort {
        let budget = match e.to_lowercase().as_str() {
            "low" => Some(1024),
            "medium" => Some(2048),
            "high" => Some(4096),
            other => other.parse::<usize>().ok().filter(|&val| val >= 1024),
        };
        if let Some(b) = budget {
            thinking = Some(ClaudeThinkingConfig {
                thinking_type: "enabled".to_string(),
                budget_tokens: b,
            });
            final_max_tokens = final_max_tokens.max(b + 8192);
            final_temperature = None;
        }
    }

    ClaudeRequest {
        model: model.to_string(),
        max_tokens: final_max_tokens,
        system: system_value,
        messages,
        stream,
        temperature: final_temperature,
        thinking,
    }
}

pub async fn execute_claude(
    client: &reqwest::Client,
    base_url: &str,
    api_key: &str,
    model: &str,
    params: &CompletionParams,
    default_max: usize,
) -> Result<CompletionResponse, LlmError> {
    let endpoint = format!("{}/messages", base_url.trim_end_matches('/'));
    let req_body = build_claude_request(model, params, default_max, false);

    let res = client
        .post(&endpoint)
        .header("Content-Type", "application/json")
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .json(&req_body)
        .send()
        .await?;

    if !res.status().is_success() {
        let status = res.status().as_u16();
        let err_text = res.text().await.unwrap_or_default();
        return Err(LlmError::Api {
            status,
            message: err_text,
        });
    }

    let rate_limits = extract_ratelimits(res.headers());
    let json: serde_json::Value = res.json().await?;

    let text = json["content"][0]["text"]
        .as_str()
        .ok_or_else(|| LlmError::InvalidResponse("Claude response missing text".to_string()))?
        .to_string();

    let prompt_tokens = json["usage"]["input_tokens"].as_u64().unwrap_or(0) as usize;
    let completion_tokens = json["usage"]["output_tokens"].as_u64().unwrap_or(0) as usize;

    let usage = if prompt_tokens > 0 || completion_tokens > 0 {
        Some(TokenUsage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        })
    } else {
        None
    };

    Ok(CompletionResponse {
        content: text,
        model: model.to_string(),
        usage,
        rate_limits,
        cost_usd: None,
    })
}

pub async fn execute_claude_streaming<F>(
    client: &reqwest::Client,
    base_url: &str,
    api_key: &str,
    model: &str,
    params: &CompletionParams,
    default_max: usize,
    mut on_token: F,
) -> Result<CompletionResponse, LlmError>
where
    F: FnMut(String) + Send + 'static,
{
    let endpoint = format!("{}/messages", base_url.trim_end_matches('/'));
    let req_body = build_claude_request(model, params, default_max, true);

    let res = client
        .post(&endpoint)
        .header("Content-Type", "application/json")
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .json(&req_body)
        .send()
        .await?;

    if !res.status().is_success() {
        let status = res.status().as_u16();
        let err_text = res.text().await.unwrap_or_default();
        return Err(LlmError::Api {
            status,
            message: err_text,
        });
    }

    let rate_limits = extract_ratelimits(res.headers());
    let mut stream = res.bytes_stream();
    let mut buffer = String::new();
    let mut full_response = String::new();

    while let Some(item) = stream.next().await {
        let bytes = item.map_err(|e| LlmError::Stream(e.to_string()))?;
        let chunk = String::from_utf8_lossy(&bytes);
        buffer.push_str(&chunk);

        while let Some(pos) = buffer.find('\n') {
            let line = buffer[..pos].trim().to_string();
            buffer = buffer[pos + 1..].to_string();

            if let Some(data_str) = line.strip_prefix("data: ") {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(data_str) {
                    if json["type"].as_str() == Some("content_block_delta") {
                        if let Some(token) = json["delta"]["text"].as_str() {
                            full_response.push_str(token);
                            on_token(token.to_string());
                        }
                    }
                }
            }
        }
    }

    Ok(CompletionResponse {
        content: full_response,
        model: model.to_string(),
        usage: None,
        rate_limits,
        cost_usd: None,
    })
}
