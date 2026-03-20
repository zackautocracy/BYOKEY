//! Codex-specific quota fetcher — queries the Codex usage/rate-limit endpoint.

use crate::credentials::{QuotaFetcher, QuotaSnapshot};
use async_trait::async_trait;
use byokey_auth::AuthManager;
use byokey_types::ProviderId;

/// Codex usage endpoint.
const CODEX_USAGE_URL: &str = "https://chatgpt.com/backend-api/wham/usage";

/// Fetches quota from the Codex rate limit API.
pub struct CodexQuotaFetcher {
    http: rquest::Client,
}

impl CodexQuotaFetcher {
    /// Creates a new fetcher with the given HTTP client.
    #[must_use]
    pub fn new(http: rquest::Client) -> Self {
        Self { http }
    }
}

#[async_trait]
impl QuotaFetcher for CodexQuotaFetcher {
    async fn fetch_quota(
        &self,
        auth: &AuthManager,
        provider: &ProviderId,
        account_id: &str,
    ) -> Option<QuotaSnapshot> {
        let token = auth
            .get_token_for(provider, account_id)
            .await
            .ok()?
            .access_token;

        // Extract ChatGPT account ID from the JWT payload for the header.
        let chatgpt_account_id = extract_chatgpt_account_id(&token);

        let mut builder = self
            .http
            .get(CODEX_USAGE_URL)
            .header("authorization", format!("Bearer {token}"))
            .header("accept", "application/json")
            .header("user-agent", super::CODEX_USER_AGENT);

        if let Some(ref acct_id) = chatgpt_account_id {
            builder = builder.header("chatgpt-account-id", acct_id.as_str());
        }

        let resp = builder.send().await.ok()?;

        if !resp.status().is_success() {
            return None;
        }

        let json: serde_json::Value = resp.json().await.ok()?;
        let rate_limit = json.get("rate_limit")?;

        let primary_used = rate_limit
            .pointer("/primary_window/used_percent")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0);
        let secondary_used = rate_limit
            .pointer("/secondary_window/used_percent")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0);

        let primary_remaining = 100.0 - primary_used;
        let secondary_remaining = 100.0 - secondary_used;

        Some(QuotaSnapshot {
            percent_remaining: primary_remaining.min(secondary_remaining).max(0.0),
            unlimited: false,
        })
    }
}

/// Decode the `ChatGPT` account ID from a Codex JWT access token.
///
/// The JWT payload contains `https://api.openai.com/auth.chatgpt_account_id`.
fn extract_chatgpt_account_id(token: &str) -> Option<String> {
    use base64::Engine as _;

    let parts: Vec<&str> = token.split('.').collect();
    if parts.len() < 2 {
        return None;
    }
    // Base64url decode the payload.
    let payload = parts[1];
    let mut padded = payload.to_string();
    let pad_len = (4 - padded.len() % 4) % 4;
    padded.extend(std::iter::repeat_n('=', pad_len));

    let bytes = base64::engine::general_purpose::URL_SAFE
        .decode(padded.as_bytes())
        .ok()?;
    let json: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    json.pointer("/https:~1~1api.openai.com~1auth/chatgpt_account_id")
        .and_then(serde_json::Value::as_str)
        .map(String::from)
}
