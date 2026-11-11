use thiserror::Error;

#[derive(Error, Debug)]
pub enum LlmError {
    #[error("Missing API key for provider: {0}")]
    MissingApiKey(String),

    #[error("Unknown provider: {0}")]
    UnknownProvider(String),

    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),

    #[error("API error (status {status}): {message}")]
    Api { status: u16, message: String },

    #[error("Failed to parse JSON: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Streaming error: {0}")]
    Stream(String),

    #[error("Invalid response: {0}")]
    InvalidResponse(String),

    #[error("{0}")]
    Custom(String),
}
