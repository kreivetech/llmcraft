use crate::types::RateLimits;

pub fn extract_ratelimits(headers: &reqwest::header::HeaderMap) -> RateLimits {
    let parse_header_usize = |names: &[&str]| -> Option<usize> {
        for &name in names {
            if let Some(val) = headers.get(name).and_then(|v| v.to_str().ok()) {
                if let Ok(num) = val.parse::<usize>() {
                    return Some(num);
                }
            }
        }
        None
    };

    let parse_header_seconds = |names: &[&str]| -> Option<f64> {
        for &name in names {
            if let Some(val) = headers.get(name).and_then(|v| v.to_str().ok()) {
                if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(val) {
                    let now = chrono::Utc::now();
                    let diff = dt.with_timezone(&chrono::Utc).signed_duration_since(now);
                    return Some(diff.num_milliseconds() as f64 / 1000.0);
                } else if val.ends_with("ms") {
                    if let Ok(ms) = val.strip_suffix("ms").unwrap().parse::<f64>() {
                        return Some(ms / 1000.0);
                    }
                } else if val.ends_with('s') {
                    if let Ok(s) = val.strip_suffix('s').unwrap().parse::<f64>() {
                        return Some(s);
                    }
                } else if val.ends_with('m') {
                    if let Ok(m) = val.strip_suffix('m').unwrap().parse::<f64>() {
                        return Some(m * 60.0);
                    }
                } else if val.ends_with('h') {
                    if let Ok(h) = val.strip_suffix('h').unwrap().parse::<f64>() {
                        return Some(h * 3600.0);
                    }
                } else if let Ok(sec) = val.parse::<f64>() {
                    return Some(sec);
                }
            }
        }
        None
    };

    let remaining_requests = parse_header_usize(&[
        "x-ratelimit-remaining-requests",
        "anthropic-ratelimit-requests-remaining",
    ]);
    let remaining_tokens = parse_header_usize(&[
        "x-ratelimit-remaining-tokens",
        "anthropic-ratelimit-input-tokens-remaining",
    ]);
    let limit_requests = parse_header_usize(&[
        "x-ratelimit-limit-requests",
        "anthropic-ratelimit-requests-limit",
    ]);
    let limit_tokens = parse_header_usize(&[
        "x-ratelimit-limit-tokens",
        "anthropic-ratelimit-input-tokens-limit",
    ]);
    let reset_requests = parse_header_seconds(&[
        "x-ratelimit-reset-requests",
        "anthropic-ratelimit-requests-reset",
    ]);
    let reset_tokens = parse_header_seconds(&[
        "x-ratelimit-reset-tokens",
        "anthropic-ratelimit-input-tokens-reset",
    ]);

    RateLimits {
        remaining_requests,
        remaining_tokens,
        limit_requests,
        limit_tokens,
        reset_requests,
        reset_tokens,
    }
}
