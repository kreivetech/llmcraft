use chrono::Local;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct CallStat {
    pub timestamp: String,
    pub model: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub cost: f64,
    pub saved: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct SessionStats {
    pub total_cost: f64,
    pub total_saved: f64,
    pub monthly_limit: usize,
    #[serde(default = "default_limit_usd")]
    pub monthly_limit_usd: f64,
    pub history: Vec<CallStat>,
}

fn default_limit_usd() -> f64 {
    10.0
}

impl Default for SessionStats {
    fn default() -> Self {
        Self {
            total_cost: 0.0,
            total_saved: 0.0,
            monthly_limit: 1_000_000,
            monthly_limit_usd: default_limit_usd(),
            history: Vec::new(),
        }
    }
}

pub struct ModelCostRule {
    pub provider: &'static str,
    pub model_prefix: &'static str,
    pub input_usd_per_1k: f64,
    pub output_usd_per_1k: f64,
}

pub const COST_TABLE: &[ModelCostRule] = &[
    // Embeddings
    ModelCostRule { provider: "openai", model_prefix: "text-embedding-3-small", input_usd_per_1k: 0.000020, output_usd_per_1k: 0.0 },
    ModelCostRule { provider: "openai", model_prefix: "text-embedding-3-large", input_usd_per_1k: 0.000130, output_usd_per_1k: 0.0 },
    ModelCostRule { provider: "openai", model_prefix: "text-embedding-ada-002", input_usd_per_1k: 0.000100, output_usd_per_1k: 0.0 },
    ModelCostRule { provider: "gemini", model_prefix: "gemini-embedding-2", input_usd_per_1k: 0.000200, output_usd_per_1k: 0.0 },
    ModelCostRule { provider: "gemini", model_prefix: "gemini-embedding-001", input_usd_per_1k: 0.000150, output_usd_per_1k: 0.0 },
    ModelCostRule { provider: "gemini", model_prefix: "text-embedding-004", input_usd_per_1k: 0.000025, output_usd_per_1k: 0.0 },

    // OpenAI Models
    ModelCostRule { provider: "openai", model_prefix: "gpt-5.5-pro", input_usd_per_1k: 0.030000, output_usd_per_1k: 0.180000 },
    ModelCostRule { provider: "openai", model_prefix: "gpt-5.5", input_usd_per_1k: 0.005000, output_usd_per_1k: 0.030000 },
    ModelCostRule { provider: "openai", model_prefix: "gpt-5.4-pro", input_usd_per_1k: 0.030000, output_usd_per_1k: 0.180000 },
    ModelCostRule { provider: "openai", model_prefix: "gpt-5.4-mini", input_usd_per_1k: 0.000750, output_usd_per_1k: 0.004500 },
    ModelCostRule { provider: "openai", model_prefix: "gpt-5.4-nano", input_usd_per_1k: 0.000200, output_usd_per_1k: 0.001250 },
    ModelCostRule { provider: "openai", model_prefix: "gpt-5.4", input_usd_per_1k: 0.002500, output_usd_per_1k: 0.015000 },
    ModelCostRule { provider: "openai", model_prefix: "gpt-4o-mini", input_usd_per_1k: 0.000150, output_usd_per_1k: 0.000600 },
    ModelCostRule { provider: "openai", model_prefix: "gpt-4o", input_usd_per_1k: 0.002500, output_usd_per_1k: 0.010000 },
    ModelCostRule { provider: "openai", model_prefix: "gpt-4.1-mini", input_usd_per_1k: 0.000400, output_usd_per_1k: 0.001600 },
    ModelCostRule { provider: "openai", model_prefix: "gpt-4.1-nano", input_usd_per_1k: 0.000100, output_usd_per_1k: 0.000400 },
    ModelCostRule { provider: "openai", model_prefix: "gpt-4.1", input_usd_per_1k: 0.002000, output_usd_per_1k: 0.008000 },
    ModelCostRule { provider: "openai", model_prefix: "o4-mini", input_usd_per_1k: 0.001100, output_usd_per_1k: 0.004400 },
    ModelCostRule { provider: "openai", model_prefix: "o3-mini", input_usd_per_1k: 0.001100, output_usd_per_1k: 0.004400 },
    ModelCostRule { provider: "openai", model_prefix: "o3", input_usd_per_1k: 0.002000, output_usd_per_1k: 0.008000 },
    ModelCostRule { provider: "openai", model_prefix: "o1", input_usd_per_1k: 0.015000, output_usd_per_1k: 0.060000 },
    ModelCostRule { provider: "openai", model_prefix: "*", input_usd_per_1k: 0.002500, output_usd_per_1k: 0.010000 },

    // Claude Models
    ModelCostRule { provider: "claude", model_prefix: "claude-opus-4", input_usd_per_1k: 0.005000, output_usd_per_1k: 0.025000 },
    ModelCostRule { provider: "claude", model_prefix: "claude-sonnet-4", input_usd_per_1k: 0.003000, output_usd_per_1k: 0.015000 },
    ModelCostRule { provider: "claude", model_prefix: "claude-haiku-4-5", input_usd_per_1k: 0.001000, output_usd_per_1k: 0.005000 },
    ModelCostRule { provider: "claude", model_prefix: "claude-haiku-4", input_usd_per_1k: 0.001000, output_usd_per_1k: 0.005000 },
    ModelCostRule { provider: "claude", model_prefix: "claude-3-7-sonnet", input_usd_per_1k: 0.003000, output_usd_per_1k: 0.015000 },
    ModelCostRule { provider: "claude", model_prefix: "claude-3-5-sonnet", input_usd_per_1k: 0.003000, output_usd_per_1k: 0.015000 },
    ModelCostRule { provider: "claude", model_prefix: "claude-3-5-haiku", input_usd_per_1k: 0.000800, output_usd_per_1k: 0.004000 },
    ModelCostRule { provider: "claude", model_prefix: "claude-3-opus", input_usd_per_1k: 0.015000, output_usd_per_1k: 0.075000 },
    ModelCostRule { provider: "claude", model_prefix: "*", input_usd_per_1k: 0.003000, output_usd_per_1k: 0.015000 },

    // Gemini Models
    ModelCostRule { provider: "gemini", model_prefix: "gemini-3.5-flash", input_usd_per_1k: 0.001500, output_usd_per_1k: 0.009000 },
    ModelCostRule { provider: "gemini", model_prefix: "gemini-3.1-flash-lite", input_usd_per_1k: 0.000250, output_usd_per_1k: 0.001500 },
    ModelCostRule { provider: "gemini", model_prefix: "gemini-3.1-flash", input_usd_per_1k: 0.000750, output_usd_per_1k: 0.003000 },
    ModelCostRule { provider: "gemini", model_prefix: "gemini-3.1-pro", input_usd_per_1k: 0.002000, output_usd_per_1k: 0.012000 },
    ModelCostRule { provider: "gemini", model_prefix: "gemini-3-flash", input_usd_per_1k: 0.000500, output_usd_per_1k: 0.003000 },
    ModelCostRule { provider: "gemini", model_prefix: "gemini-2.5-flash-lite", input_usd_per_1k: 0.000100, output_usd_per_1k: 0.000400 },
    ModelCostRule { provider: "gemini", model_prefix: "gemini-2.5-flash", input_usd_per_1k: 0.000075, output_usd_per_1k: 0.000300 },
    ModelCostRule { provider: "gemini", model_prefix: "gemini-2.5-pro", input_usd_per_1k: 0.001250, output_usd_per_1k: 0.005000 },
    ModelCostRule { provider: "gemini", model_prefix: "*", input_usd_per_1k: 0.000300, output_usd_per_1k: 0.001200 },

    // DeepSeek Models
    ModelCostRule { provider: "deepseek", model_prefix: "deepseek-v4-pro", input_usd_per_1k: 0.000435, output_usd_per_1k: 0.000870 },
    ModelCostRule { provider: "deepseek", model_prefix: "deepseek-v4-flash", input_usd_per_1k: 0.000140, output_usd_per_1k: 0.000280 },
    ModelCostRule { provider: "deepseek", model_prefix: "deepseek-chat", input_usd_per_1k: 0.000140, output_usd_per_1k: 0.000280 },
    ModelCostRule { provider: "deepseek", model_prefix: "deepseek-reasoner", input_usd_per_1k: 0.000550, output_usd_per_1k: 0.002190 },
    ModelCostRule { provider: "deepseek", model_prefix: "*", input_usd_per_1k: 0.000140, output_usd_per_1k: 0.000280 },

    // Mistral Models
    ModelCostRule { provider: "mistral", model_prefix: "mistral-large-latest", input_usd_per_1k: 0.002000, output_usd_per_1k: 0.006000 },
    ModelCostRule { provider: "mistral", model_prefix: "open-mixtral-8x22b", input_usd_per_1k: 0.002000, output_usd_per_1k: 0.006000 },
    ModelCostRule { provider: "mistral", model_prefix: "*", input_usd_per_1k: 0.002000, output_usd_per_1k: 0.006000 },

    // Groq Models
    ModelCostRule { provider: "groq", model_prefix: "llama-3.3-70b-versatile", input_usd_per_1k: 0.000590, output_usd_per_1k: 0.000790 },
    ModelCostRule { provider: "groq", model_prefix: "llama-3.3-70b", input_usd_per_1k: 0.000590, output_usd_per_1k: 0.000790 },
    ModelCostRule { provider: "groq", model_prefix: "llama-3.1-8b", input_usd_per_1k: 0.000050, output_usd_per_1k: 0.000080 },
    ModelCostRule { provider: "groq", model_prefix: "llama-4-scout", input_usd_per_1k: 0.000110, output_usd_per_1k: 0.000340 },
    ModelCostRule { provider: "groq", model_prefix: "qwen3-32b", input_usd_per_1k: 0.000290, output_usd_per_1k: 0.000590 },
    ModelCostRule { provider: "groq", model_prefix: "*", input_usd_per_1k: 0.000590, output_usd_per_1k: 0.000790 },

    // Local / Ollama (Free)
    ModelCostRule { provider: "ollama", model_prefix: "*", input_usd_per_1k: 0.0, output_usd_per_1k: 0.0 },
    ModelCostRule { provider: "local", model_prefix: "*", input_usd_per_1k: 0.0, output_usd_per_1k: 0.0 },
];

pub fn get_model_rates_by_provider_and_model(provider: &str, model: &str) -> (f64, f64) {
    let model_lower = model.to_lowercase();
    let candidates: Vec<&ModelCostRule> = COST_TABLE.iter().filter(|r| r.provider == provider).collect();
    if candidates.is_empty() {
        return (0.0, 0.0);
    }
    for row in &candidates {
        if row.model_prefix == "*" {
            continue;
        }
        if model_lower.starts_with(row.model_prefix) {
            return (row.input_usd_per_1k / 1000.0, row.output_usd_per_1k / 1000.0);
        }
    }
    if let Some(fallback) = candidates.iter().find(|r| r.model_prefix == "*") {
        return (fallback.input_usd_per_1k / 1000.0, fallback.output_usd_per_1k / 1000.0);
    }
    (0.0, 0.0)
}

pub fn resolve_rate(provider: &str, model: &str) -> f64 {
    let (in_rate, _) = get_model_rates_by_provider_and_model(provider, model);
    in_rate * 1000.0
}

pub fn get_model_rates(model_id: &str) -> (f64, f64) {
    let model_lower = model_id.to_lowercase();
    let provider = if model_lower == "local" || model_lower == "ollama" || model_lower.contains("-q4") || model_lower.contains("-q8") || model_lower.contains("q4_") || model_lower.contains("q8_") || model_lower.contains("phi") || model_lower.contains("qwen-2.5") {
        "local"
    } else if model_lower.contains("gpt") || model_lower.starts_with("o1") || model_lower.starts_with("o3") || model_lower.starts_with("o4") || model_lower.contains("text-embedding-3") {
        "openai"
    } else if model_lower.contains("claude") {
        "claude"
    } else if model_lower.contains("gemini") || model_lower.contains("text-embedding-004") {
        "gemini"
    } else if model_lower.contains("deepseek") {
        "deepseek"
    } else if model_lower.contains("mistral") {
        "mistral"
    } else if model_lower.contains("llama") || model_lower.contains("mixtral") {
        "groq"
    } else {
        "local"
    };

    get_model_rates_by_provider_and_model(provider, model_id)
}

pub fn get_comparison_rates() -> (f64, f64) {
    // GPT-5.5 Flagship pricing benchmark
    (15.00 / 1_000_000.0, 75.00 / 1_000_000.0)
}

pub fn load_stats_from_path(stats_path: &Path) -> SessionStats {
    if stats_path.exists()
        && let Ok(content) = fs::read_to_string(stats_path)
        && let Ok(stats) = serde_json::from_str::<SessionStats>(&content) {
        return stats;
    }
    SessionStats::default()
}

pub fn save_stats_to_path(stats_path: &Path, stats: &SessionStats) -> Result<(), std::io::Error> {
    if let Some(parent) = stats_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let content = serde_json::to_string_pretty(stats)?;
    fs::write(stats_path, content)?;
    Ok(())
}

pub fn calculate_cost_and_saved(model: &str, prompt_tokens: usize, completion_tokens: usize) -> (f64, f64) {
    let (in_rate, out_rate) = get_model_rates(model);
    let (comp_in, comp_out) = get_comparison_rates();
    let cost = (prompt_tokens as f64 * in_rate) + (completion_tokens as f64 * out_rate);
    let comp_cost = (prompt_tokens as f64 * comp_in) + (completion_tokens as f64 * comp_out);
    let saved = comp_cost - cost;
    (cost, if saved > 0.0 { saved } else { 0.0 })
}

pub fn record_call_stat(
    stats: &mut SessionStats,
    model: &str,
    prompt_tokens: usize,
    completion_tokens: usize,
) -> (f64, f64) {
    let (cost, saved) = calculate_cost_and_saved(model, prompt_tokens, completion_tokens);
    let new_stat = CallStat {
        timestamp: Local::now().to_rfc3339(),
        model: model.to_string(),
        prompt_tokens,
        completion_tokens,
        cost,
        saved,
    };
    stats.total_cost += cost;
    stats.total_saved += saved;
    stats.history.push(new_stat);
    (cost, saved)
}
