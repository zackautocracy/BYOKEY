//! Copilot-specific quota fetcher — queries GitHub Copilot internal quota endpoint.

use crate::credentials::{QuotaFetcher, QuotaSnapshot};
use async_trait::async_trait;
use byokey_auth::AuthManager;
use byokey_types::ProviderId;

/// Copilot usage/quota endpoint.
const COPILOT_USER_URL: &str = "https://api.github.com/copilot_internal/user";

/// Fetches quota from the GitHub Copilot internal API.
pub struct CopilotQuotaFetcher {
    http: rquest::Client,
}

impl CopilotQuotaFetcher {
    /// Creates a new fetcher with the given HTTP client.
    #[must_use]
    pub fn new(http: rquest::Client) -> Self {
        Self { http }
    }
}

#[async_trait]
impl QuotaFetcher for CopilotQuotaFetcher {
    async fn fetch_quota(
        &self,
        auth: &AuthManager,
        provider: &ProviderId,
        account_id: &str,
    ) -> Option<QuotaSnapshot> {
        let github_token = auth
            .get_token_for(provider, account_id)
            .await
            .ok()?
            .access_token;

        let resp = self
            .http
            .get(COPILOT_USER_URL)
            .header("authorization", format!("token {github_token}"))
            .header("accept", "application/json")
            .header("user-agent", super::USER_AGENT)
            .send()
            .await
            .ok()?;

        if !resp.status().is_success() {
            return None;
        }

        let json: serde_json::Value = resp.json().await.ok()?;
        let pi = json.pointer("/quota_snapshots/premium_interactions")?;
        let unlimited = pi
            .get("unlimited")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        let percent = pi
            .get("percent_remaining")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0);

        Some(QuotaSnapshot {
            percent_remaining: percent,
            unlimited,
        })
    }
}
