use crate::error::LlmError;
use crate::protocols::ratelimits::extract_ratelimits;
use crate::types::{CompletionParams, CompletionResponse, Role, TokenUsage};
use futures::StreamExt;
use serde::Serialize;

#[derive(Serialize)]
pub struct GeminiPart {
    pub text: String,
}

#[derive(Serialize)]
pub struct GeminiContent {
    pub role: String,
    pub parts: Vec<GeminiPart>,
}

#[derive(Serialize)]
pub struct GeminiSystemInstruction {
    pub parts: Vec<GeminiPart>,
}

#[derive(Serialize)]
pub struct GeminiThinkingConfig {
    #[serde(rename = "thinkingBudget", skip_serializing_if = "Option::is_none")]
    pub thinking_budget: Option<i32>,
}

#[derive(Serialize)]
pub struct GeminiGenerationConfig {
    #[serde(rename = "maxOutputTokens")]
    pub max_output_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(rename = "thinkingConfig", skip_serializing_if = "Option::is_none")]
    pub thinking_config: Option<GeminiThinkingConfig>,
}

#[derive(Serialize)]
pub struct GeminiRequest {
    #[serde(rename = "systemInstruction", skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<GeminiSystemInstruction>,
    pub contents: Vec<GeminiContent>,
    #[serde(rename = "generationConfig")]
    pub generation_config: GeminiGenerationConfig,
}

pub fn build_gemini_request(model: &str, params: &CompletionParams, default_max: usize) -> GeminiRequest {
    let system_instruction = params.system.as_ref().map(|sys| GeminiSystemInstruction {
        parts: vec![GeminiPart {
            text: sys.to_string(),
        }],
    });

    let mut contents = Vec::new();
    for msg in &params.history {
        contents.push(GeminiContent {
            role: if msg.role == Role::Assistant {
                "model".to_string()
            } else {
                "user".to_string()
            },
            parts: vec![GeminiPart {
                text: msg.content.clone(),
            }],
        });
    }

    contents.push(GeminiContent {
        role: "user".to_string(),
        parts: vec![GeminiPart {
            text: params.prompt.clone(),
        }],
    });

    let mut thinking_config: Option<GeminiThinkingConfig> = None;
    if let Some(ref e) = params.reasoning_effort {
        let is_gemini_2_5 = model.contains("2.5");
        if is_gemini_2_5 {
            let budget: Option<i32> = match e.to_lowercase().as_str() {
                "low" => Some(1024),
                "medium" => Some(2048),
                "high" => Some(4096),
                other => other.parse::<i32>().ok(),
            };
            if let Some(b) = budget {
                thinking_config = Some(GeminiThinkingConfig {
                    thinking_budget: Some(b),
                });
            }
        }
    }

    let final_temperature = if thinking_config.is_some() {
        None
    } else {
        params.temperature
    };

    let user_max = params.max_tokens.unwrap_or(0);
    let mut final_max_tokens = if user_max == 0 || user_max <= 2048 {
        default_max
    } else {
        user_max.max(default_max)
    };

    if let Some(b) = thinking_config.as_ref().and_then(|tc| tc.thinking_budget) {
        final_max_tokens = final_max_tokens.max((b as usize) + 8192);
    }

    GeminiRequest {
        system_instruction,
        contents,
        generation_config: GeminiGenerationConfig {
            max_output_tokens: final_max_tokens,
            temperature: final_temperature,
            thinking_config,
        },
    }
}

pub async fn execute_gemini(
    client: &reqwest::Client,
    base_url: &str,
    api_key: &str,
    model: &str,
    params: &CompletionParams,
    default_max: usize,
) -> Result<CompletionResponse, LlmError> {
    let endpoint = format!("{}/models/{}:generateContent?key={}", base_url, model, api_key);
    let req_body = build_gemini_request(model, params, default_max);

    let res = client
        .post(&endpoint)
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

    let text = json["candidates"][0]["content"]["parts"][0]["text"]
        .as_str()
        .ok_or_else(|| LlmError::InvalidResponse("Gemini response is missing text".to_string()))?
        .to_string();

    let prompt_tokens = json["usageMetadata"]["promptTokenCount"]
        .as_u64()
        .unwrap_or(0) as usize;
    let completion_tokens = json["usageMetadata"]["candidatesTokenCount"]
        .as_u64()
        .unwrap_or(0) as usize;

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

pub async fn execute_gemini_streaming<F>(
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
    let endpoint = format!("{}/models/{}:streamGenerateContent?key={}", base_url, model, api_key);
    let req_body = build_gemini_request(model, params, default_max);

    let res = client
        .post(&endpoint)
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
    let mut bracket_count = 0;
    let mut start_idx = None;
    let mut in_string = false;
    let mut escaped = false;
    let mut full_response = String::new();

    while let Some(item) = stream.next().await {
        let bytes = item.map_err(|e| LlmError::Stream(e.to_string()))?;
        let chunk = String::from_utf8_lossy(&bytes);

        for c in chunk.chars() {
            if escaped {
                escaped = false;
                buffer.push(c);
                continue;
            }
            if c == '\\' {
                escaped = true;
                buffer.push(c);
                continue;
            }
            if c == '"' {
                in_string = !in_string;
            }
            buffer.push(c);

            if !in_string {
                if c == '{' {
                    if bracket_count == 0 {
                        start_idx = Some(buffer.len() - 1);
                    }
                    bracket_count += 1;
                } else if c == '}' {
                    bracket_count -= 1;
                    if bracket_count == 0 {
                        if let Some(start) = start_idx {
                            let json_str = &buffer[start..];
                            if let Ok(json) = serde_json::from_str::<serde_json::Value>(json_str) {
                                if let Some(token) = json["candidates"][0]["content"]["parts"][0]["text"].as_str() {
                                    full_response.push_str(token);
                                    on_token(token.to_string());
                                }
                            }
                        }
                        start_idx = None;
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
