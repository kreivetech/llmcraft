use crate::error::LlmError;
use crate::pricing::calculate_cost_and_saved;
use crate::protocols::{claude, gemini, openai};
use crate::provider::Provider;
use crate::types::{CompletionParams, CompletionResponse};
use std::time::Duration;

#[derive(Clone, Debug)]
pub struct LlmClient {
    pub provider: Provider,
    pub model: String,
    pub api_key: Option<String>,
    pub base_url: String,
    pub http_client: reqwest::Client,
    pub default_max_tokens: Option<usize>,
    pub default_temperature: Option<f64>,
}

pub struct ClientBuilder {
    provider: Provider,
    model: Option<String>,
    api_key: Option<String>,
    base_url: Option<String>,
    timeout: Option<Duration>,
    default_max_tokens: Option<usize>,
    default_temperature: Option<f64>,
}

impl ClientBuilder {
    pub fn new(provider: Provider) -> Self {
        Self {
            provider,
            model: None,
            api_key: None,
            base_url: None,
            timeout: Some(Duration::from_secs(60)),
            default_max_tokens: None,
            default_temperature: None,
        }
    }

    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    pub fn default_max_tokens(mut self, max: usize) -> Self {
        self.default_max_tokens = Some(max);
        self
    }

    pub fn default_temperature(mut self, temp: f64) -> Self {
        self.default_temperature = Some(temp);
        self
    }

    pub fn build(self) -> Result<LlmClient, LlmError> {
        let model = self.model.unwrap_or_else(|| match self.provider {
            Provider::OpenAi => "gpt-4o".to_string(),
            Provider::Claude => "claude-3-7-sonnet-20250219".to_string(),
            Provider::Gemini => "gemini-2.5-flash".to_string(),
            Provider::DeepSeek => "deepseek-chat".to_string(),
            Provider::Mistral => "mistral-large-latest".to_string(),
            Provider::Groq => "llama-3.3-70b-versatile".to_string(),
            Provider::Ollama => "llama3".to_string(),
            Provider::Custom(_) => "default".to_string(),
        });

        // Resolve API key with environment fallback
        let api_key = match self.api_key {
            Some(k) if !k.trim().is_empty() => Some(k),
            _ => {
                if let Some(env_name) = self.provider.default_env_var() {
                    std::env::var(env_name).ok()
                } else {
                    None
                }
            }
        };

        // Enforce key presence for cloud providers
        if api_key.is_none() && !matches!(self.provider, Provider::Ollama | Provider::Custom(_)) {
            return Err(LlmError::MissingApiKey(format!(
                "{}. Set the {} environment variable or provide it via builder.",
                self.provider,
                self.provider.default_env_var().unwrap_or("API_KEY")
            )));
        }

        // Resolve base URL
        let base_url = match self.base_url {
            Some(u) => u,
            None => {
                if self.provider == Provider::Ollama {
                    std::env::var("OLLAMA_API")
                        .unwrap_or_else(|_| self.provider.default_base_url().to_string())
                        .trim_end_matches('/')
                        .trim_end_matches("/v1")
                        .to_string()
                } else {
                    self.provider.default_base_url().to_string()
                }
            }
        };

        let mut http_builder = reqwest::Client::builder();
        if let Some(t) = self.timeout {
            http_builder = http_builder.timeout(t);
        }
        let http_client = http_builder.build()?;

        Ok(LlmClient {
            provider: self.provider,
            model,
            api_key,
            base_url,
            http_client,
            default_max_tokens: self.default_max_tokens,
            default_temperature: self.default_temperature,
        })
    }
}

impl LlmClient {
    pub fn builder(provider: Provider) -> ClientBuilder {
        ClientBuilder::new(provider)
    }

    pub fn default_max_output_tokens(&self) -> usize {
        self.default_max_tokens
            .unwrap_or_else(|| self.provider.default_max_output_tokens(&self.model))
    }

    pub async fn complete(&self, prompt: impl Into<String>) -> Result<CompletionResponse, LlmError> {
        let params = CompletionParams::new(prompt);
        self.complete_with_params(params).await
    }

    pub async fn complete_with_params(
        &self,
        mut params: CompletionParams,
    ) -> Result<CompletionResponse, LlmError> {
        if params.temperature.is_none() {
            params.temperature = self.default_temperature;
        }

        let default_max = self.default_max_output_tokens();
        let mut response = match self.provider {
            Provider::Gemini => {
                let key = self
                    .api_key
                    .as_deref()
                    .ok_or_else(|| LlmError::MissingApiKey("gemini".to_string()))?;
                gemini::execute_gemini(
                    &self.http_client,
                    &self.base_url,
                    key,
                    &self.model,
                    &params,
                    default_max,
                )
                .await?
            }
            Provider::Claude => {
                let key = self
                    .api_key
                    .as_deref()
                    .ok_or_else(|| LlmError::MissingApiKey("claude".to_string()))?;
                claude::execute_claude(
                    &self.http_client,
                    &self.base_url,
                    key,
                    &self.model,
                    &params,
                    default_max,
                )
                .await?
            }
            ref provider => {
                openai::execute_openai_or_compat(
                    &self.http_client,
                    provider,
                    &self.base_url,
                    self.api_key.as_deref(),
                    &self.model,
                    &params,
                    default_max,
                )
                .await?
            }
        };

        if let Some(ref usage) = response.usage {
            let (cost, _) = calculate_cost_and_saved(&self.model, usage.prompt_tokens, usage.completion_tokens);
            response.cost_usd = Some(cost);
        }

        Ok(response)
    }

    pub async fn stream<F>(
        &self,
        prompt: impl Into<String>,
        on_token: F,
    ) -> Result<CompletionResponse, LlmError>
    where
        F: FnMut(String) + Send + 'static,
    {
        let params = CompletionParams::new(prompt);
        self.stream_with_params(params, on_token).await
    }

    pub async fn stream_with_params<F>(
        &self,
        mut params: CompletionParams,
        on_token: F,
    ) -> Result<CompletionResponse, LlmError>
    where
        F: FnMut(String) + Send + 'static,
    {
        if params.temperature.is_none() {
            params.temperature = self.default_temperature;
        }

        let default_max = self.default_max_output_tokens();
        match self.provider {
            Provider::Gemini => {
                let key = self
                    .api_key
                    .as_deref()
                    .ok_or_else(|| LlmError::MissingApiKey("gemini".to_string()))?;
                gemini::execute_gemini_streaming(
                    &self.http_client,
                    &self.base_url,
                    key,
                    &self.model,
                    &params,
                    default_max,
                    on_token,
                )
                .await
            }
            Provider::Claude => {
                let key = self
                    .api_key
                    .as_deref()
                    .ok_or_else(|| LlmError::MissingApiKey("claude".to_string()))?;
                claude::execute_claude_streaming(
                    &self.http_client,
                    &self.base_url,
                    key,
                    &self.model,
                    &params,
                    default_max,
                    on_token,
                )
                .await
            }
            ref provider => {
                openai::execute_openai_or_compat_streaming(
                    &self.http_client,
                    provider,
                    &self.base_url,
                    self.api_key.as_deref(),
                    &self.model,
                    &params,
                    default_max,
                    on_token,
                )
                .await
            }
        }
    }
}
