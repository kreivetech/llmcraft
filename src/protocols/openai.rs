use crate::error::LlmError;
use crate::protocols::ratelimits::extract_ratelimits;
use crate::provider::Provider;
use crate::types::{CompletionParams, CompletionResponse, TokenUsage};
use futures::StreamExt;
use serde::{Deserialize, Serialize};

#[derive(Serialize)]
pub struct OpenAiMessage {
    pub role: String,
    pub content: String,
}

#[derive(Serialize)]
pub struct OpenAiRequest {
    pub model: String,
    pub messages: Vec<OpenAiMessage>,
    #[serde(rename = "max_completion_tokens", skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    pub stream: bool,
    #[serde(rename = "reasoning_effort", skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
}

#[derive(Serialize)]
pub struct OpenAiCompatRequest {
    pub model: String,
    pub messages: Vec<OpenAiMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    pub stream: bool,
}

#[derive(Deserialize)]
struct OpenAiChoiceMessage {
    content: Option<String>,
}

#[derive(Deserialize)]
struct OpenAiDeltaMessage {
    content: Option<String>,
}

#[derive(Deserialize)]
struct OpenAiChoice {
    message: Option<OpenAiChoiceMessage>,
    delta: Option<OpenAiDeltaMessage>,
}

#[derive(Deserialize)]
struct OpenAiUsage {
    prompt_tokens: usize,
    completion_tokens: usize,
}

#[derive(Deserialize)]
struct OpenAiResponse {
    choices: Vec<OpenAiChoice>,
    usage: Option<OpenAiUsage>,
}

pub fn build_messages(params: &CompletionParams) -> Vec<OpenAiMessage> {
    let mut messages = Vec::new();
    if let Some(ref sys) = params.system {
        messages.push(OpenAiMessage {
            role: "system".to_string(),
            content: sys.clone(),
        });
    }
    for msg in &params.history {
        messages.push(OpenAiMessage {
            role: msg.role.as_str().to_string(),
            content: msg.content.clone(),
        });
    }
    messages.push(OpenAiMessage {
        role: "user".to_string(),
        content: params.prompt.clone(),
    });
    messages
}

pub fn build_openai_request(
    model: &str,
    params: &CompletionParams,
    default_max: usize,
    stream: bool,
) -> OpenAiRequest {
    let messages = build_messages(params);
    let user_max = params.max_tokens.unwrap_or(0);
    let final_max_tokens = if user_max == 0 || user_max <= 2048 {
        default_max
    } else {
        user_max.max(default_max)
    };

    let has_effort = params.reasoning_effort.is_some();
    OpenAiRequest {
        model: model.to_string(),
        messages,
        max_completion_tokens: Some(final_max_tokens),
        temperature: if has_effort { None } else { params.temperature },
        stream,
        reasoning_effort: params.reasoning_effort.clone(),
    }
}

pub fn build_openai_compat_request(
    model: &str,
    params: &CompletionParams,
    default_max: usize,
    stream: bool,
) -> OpenAiCompatRequest {
    let messages = build_messages(params);
    let user_max = params.max_tokens.unwrap_or(0);
    let final_max_tokens = if user_max == 0 || user_max <= 2048 {
        default_max
    } else {
        user_max.max(default_max)
    };

    OpenAiCompatRequest {
        model: model.to_string(),
        messages,
        max_tokens: Some(final_max_tokens),
        temperature: params.temperature,
        stream,
    }
}

pub async fn execute_openai_or_compat(
    client: &reqwest::Client,
    provider: &Provider,
    base_url: &str,
    api_key: Option<&str>,
    model: &str,
    params: &CompletionParams,
    default_max: usize,
) -> Result<CompletionResponse, LlmError> {
    let endpoint = if base_url.ends_with("/chat/completions") {
        base_url.to_string()
    } else {
        format!("{}/chat/completions", base_url.trim_end_matches('/'))
    };

    let mut req = client.post(&endpoint).header("Content-Type", "application/json");

    if let Some(key) = api_key {
        req = req.header("Authorization", format!("Bearer {}", key));
    }

    let req = if matches!(provider, Provider::OpenAi) {
        let body = build_openai_request(model, params, default_max, false);
        req.json(&body)
    } else {
        let body = build_openai_compat_request(model, params, default_max, false);
        req.json(&body)
    };

    let res = req.send().await?;

    if !res.status().is_success() {
        let status = res.status().as_u16();
        let err_text = res.text().await.unwrap_or_default();
        return Err(LlmError::Api {
            status,
            message: err_text,
        });
    }

    let rate_limits = extract_ratelimits(res.headers());
    let resp: OpenAiResponse = res.json().await?;

    let text = resp
        .choices
        .first()
        .and_then(|c| c.message.as_ref())
        .and_then(|m| m.content.as_ref())
        .ok_or_else(|| LlmError::InvalidResponse("Response choice contains no text content".to_string()))?
        .clone();

    let usage = resp.usage.map(|u| TokenUsage {
        prompt_tokens: u.prompt_tokens,
        completion_tokens: u.completion_tokens,
        total_tokens: u.prompt_tokens + u.completion_tokens,
    });

    Ok(CompletionResponse {
        content: text,
        model: model.to_string(),
        usage,
        rate_limits,
        cost_usd: None,
    })
}

pub async fn execute_openai_or_compat_streaming<F>(
    client: &reqwest::Client,
    provider: &Provider,
    base_url: &str,
    api_key: Option<&str>,
    model: &str,
    params: &CompletionParams,
    default_max: usize,
    mut on_token: F,
) -> Result<CompletionResponse, LlmError>
where
    F: FnMut(String) + Send + 'static,
{
    let endpoint = if base_url.ends_with("/chat/completions") {
        base_url.to_string()
    } else {
        format!("{}/chat/completions", base_url.trim_end_matches('/'))
    };

    let mut req = client.post(&endpoint).header("Content-Type", "application/json");

    if let Some(key) = api_key {
        req = req.header("Authorization", format!("Bearer {}", key));
    }

    let req = if matches!(provider, Provider::OpenAi) {
        let body = build_openai_request(model, params, default_max, true);
        req.json(&body)
    } else {
        let body = build_openai_compat_request(model, params, default_max, true);
        req.json(&body)
    };

    let res = req.send().await?;

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
                if data_str == "[DONE]" {
                    break;
                }
                if let Ok(resp) = serde_json::from_str::<OpenAiResponse>(data_str) {
                    if let Some(choice) = resp.choices.first() {
                        if let Some(delta) = &choice.delta {
                            if let Some(token) = &delta.content {
                                full_response.push_str(token);
                                on_token(token.clone());
                            }
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
