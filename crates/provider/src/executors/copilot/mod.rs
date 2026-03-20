//! GitHub Copilot executor — OpenAI-compatible API.
//!
//! Auth: device code flow → GitHub token → exchange for short-lived Copilot API token.
//! Format: `OpenAI` passthrough (no translation needed).

pub mod quota;

use crate::http_util::ProviderHttp;
use crate::registry;
use async_trait::async_trait;
use byokey_auth::AuthManager;
use byokey_types::{
    ByokError, ChatRequest, ProviderId, RateLimitStore,
    traits::{ApiFormat, ProviderExecutor, ProviderResponse, Result},
};
use serde_json::Value;
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

/// GitHub Copilot Chat Completions API base URL.
const API_BASE_URL: &str = "https://api.githubcopilot.com";

/// Endpoint to exchange a GitHub OAuth token for a short-lived Copilot API token.
const COPILOT_TOKEN_URL: &str = "https://api.github.com/copilot_internal/v2/token";

// Header values matching the VS Code Copilot Chat extension.
pub const USER_AGENT: &str = "GitHubCopilotChat/0.35.0";
pub const EDITOR_VERSION: &str = "vscode/1.107.0";
pub const PLUGIN_VERSION: &str = "copilot-chat/0.35.0";
pub const INTEGRATION_ID: &str = "vscode-chat";
pub const OPENAI_INTENT: &str = "conversation-panel";
pub const GITHUB_API_VERSION: &str = "2025-04-01";

/// Anthropic API version header for Copilot's native Messages endpoint.
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Base anthropic-beta features to enable on all Copilot Messages requests.
const ANTHROPIC_BETA_BASE: &str = "claude-code-20250219,oauth-2025-04-20,interleaved-thinking-2025-05-14,fine-grained-tool-streaming-2025-05-14,prompt-caching-2024-07-31";

/// A cached Copilot API token with its expiry time.
struct CachedToken {
    token: String,
    api_endpoint: String,
    expires_at: Instant,
    /// `true` = Pro/Business/Enterprise, `false` = Free tier.
    is_pro: bool,
}

/// Executor for the GitHub Copilot API.
pub struct CopilotExecutor {
    ph: ProviderHttp,
    api_key: Option<String>,
    account_id: Option<String>,
    auth: Arc<AuthManager>,
    /// Cache: GitHub token → short-lived Copilot API token.
    cache: Mutex<HashMap<String, CachedToken>>,
}

impl CopilotExecutor {
    /// Creates a new Copilot executor with an optional API key and auth manager.
    pub fn new(
        http: rquest::Client,
        api_key: Option<String>,
        account_id: Option<String>,
        auth: Arc<AuthManager>,
        ratelimit: Option<Arc<RateLimitStore>>,
    ) -> Self {
        let mut ph = ProviderHttp::new(http);
        if let Some(store) = ratelimit {
            ph = ph.with_ratelimit(store, ProviderId::Copilot);
        }
        Self {
            ph,
            api_key,
            account_id,
            auth,
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// Exchange a GitHub token for a Copilot API token and cache the result.
    ///
    /// Returns `(copilot_api_token, api_endpoint)`.
    async fn exchange_and_cache(&self, github_token: &str) -> Result<(String, String)> {
        // Check cache first
        {
            let cache = self.cache.lock().unwrap();
            if let Some(cached) = cache.get(github_token)
                && cached.expires_at > Instant::now()
            {
                return Ok((cached.token.clone(), cached.api_endpoint.clone()));
            }
        }

        // Exchange GitHub token for Copilot API token
        let resp = self
            .ph
            .client()
            .get(COPILOT_TOKEN_URL)
            .header("authorization", format!("token {github_token}"))
            .header("accept", "application/json")
            .header("user-agent", USER_AGENT)
            .header("editor-version", EDITOR_VERSION)
            .header("editor-plugin-version", PLUGIN_VERSION)
            .send()
            .await?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(ByokError::Auth(format!(
                "Copilot token exchange {status}: {text}"
            )));
        }

        let json: Value = resp.json().await?;

        let api_token = json
            .get("token")
            .and_then(Value::as_str)
            .ok_or_else(|| ByokError::Auth("missing token in Copilot response".into()))?
            .to_string();

        let expires_at_unix = json.get("expires_at").and_then(Value::as_i64).unwrap_or(0);

        let ttl = if expires_at_unix > 0 {
            let now_unix = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
                .cast_signed();
            let secs = (expires_at_unix - now_unix).max(0).cast_unsigned();
            Duration::from_secs(secs)
        } else {
            Duration::from_secs(1500) // default ~25 min
        };

        let api_endpoint = json
            .pointer("/endpoints/api")
            .and_then(Value::as_str)
            .unwrap_or(API_BASE_URL)
            .trim_end_matches('/')
            .to_string();

        // If "copilot_plan" is absent or not "copilot_free", assume Pro+.
        let is_pro = json
            .get("copilot_plan")
            .and_then(Value::as_str)
            .is_none_or(|plan| plan != "copilot_free");

        // Cache the new token
        {
            let mut cache = self.cache.lock().unwrap();
            cache.insert(
                github_token.to_string(),
                CachedToken {
                    token: api_token.clone(),
                    api_endpoint: api_endpoint.clone(),
                    expires_at: Instant::now() + ttl,
                    is_pro,
                },
            );
        }

        Ok((api_token, api_endpoint))
    }

    /// Obtain a Copilot API token for a specific account.
    async fn copilot_token_for_account(&self, account_id: &str) -> Result<(String, String)> {
        let github_token = self
            .auth
            .get_token_for(&ProviderId::Copilot, account_id)
            .await?
            .access_token;
        self.exchange_and_cache(&github_token).await
    }

    /// Returns the Copilot API token and base endpoint URL (without path suffix).
    ///
    /// When `api_key` is set it is used directly (skip token exchange).
    /// With multiple accounts, selects the account with the most remaining quota.
    /// Otherwise falls back to the active account.
    ///
    /// # Errors
    ///
    /// Returns [`ByokError::Auth`] if the token exchange fails.
    ///
    /// # Panics
    ///
    /// Panics if the internal token cache mutex is poisoned.
    pub async fn copilot_token(&self) -> Result<(String, String)> {
        if let Some(key) = &self.api_key {
            return Ok((key.clone(), API_BASE_URL.to_string()));
        }

        // If a specific account was pinned (e.g. by credential rotation), use it directly.
        if let Some(id) = &self.account_id {
            return self.copilot_token_for_account(id).await;
        }

        // Single/no account: use active account.
        let github_token = self
            .auth
            .get_token(&ProviderId::Copilot)
            .await?
            .access_token;
        self.exchange_and_cache(&github_token).await
    }

    /// Obtains the Copilot API token and chat completions URL.
    async fn copilot_creds(&self) -> Result<(String, String)> {
        let (token, endpoint) = self.copilot_token().await?;
        Ok((token, format!("{endpoint}/chat/completions")))
    }

    /// Returns `true` if the active Copilot account belongs to a Pro/Business/Enterprise plan.
    ///
    /// Checks only the currently active OAuth token's cached entry.
    /// Defaults to `true` (Pro) if the plan cannot be determined (e.g. no cached token yet
    /// or the `copilot_plan` field was absent in the token exchange response).
    ///
    /// # Panics
    ///
    /// Panics if the internal token cache mutex is poisoned.
    pub async fn is_pro(&self) -> bool {
        if let Ok(github_token) = self
            .auth
            .get_token(&ProviderId::Copilot)
            .await
            .map(|t| t.access_token)
        {
            let cache = self.cache.lock().unwrap();
            if let Some(cached) = cache.get(&github_token)
                && cached.expires_at > Instant::now()
            {
                return cached.is_pro;
            }
        }
        true // conservative default
    }

    /// Returns the `X-Initiator` header value based on whether the request
    /// contains any assistant/tool messages (agent) or only user messages.
    fn initiator(request: &ChatRequest) -> &'static str {
        let is_agent = request.messages.iter().any(|m| {
            matches!(
                m.get("role").and_then(Value::as_str),
                Some("assistant" | "tool")
            )
        });
        if is_agent { "agent" } else { "user" }
    }

    /// Build the `anthropic-beta` header from base features + request betas.
    pub fn build_anthropic_beta(body: &Value) -> String {
        let mut betas = ANTHROPIC_BETA_BASE.to_string();
        if let Some(arr) = body.get("betas").and_then(Value::as_array) {
            for b in arr {
                if let Some(s) = b.as_str()
                    && !betas.split(',').any(|existing| existing == s)
                {
                    betas.push(',');
                    betas.push_str(s);
                }
            }
        }
        betas
    }

    /// Detect the `X-Initiator` value from Anthropic-format messages.
    fn detect_anthropic_initiator(body: &Value) -> &'static str {
        let is_agent = body
            .get("messages")
            .and_then(Value::as_array)
            .is_some_and(|msgs| {
                msgs.iter().any(|m| {
                    matches!(
                        m.get("role").and_then(Value::as_str),
                        Some("assistant" | "tool")
                    )
                })
            });
        if is_agent { "agent" } else { "user" }
    }
}

#[async_trait]
impl ProviderExecutor for CopilotExecutor {
    async fn chat_completion(&self, request: ChatRequest) -> Result<ProviderResponse> {
        let stream = request.stream;
        let initiator = Self::initiator(&request);
        let body = request.into_body();

        let (token, endpoint) = self.copilot_creds().await?;

        let builder = self
            .ph
            .client()
            .post(&endpoint)
            .header("authorization", format!("Bearer {token}"))
            .header("user-agent", USER_AGENT)
            .header("editor-version", EDITOR_VERSION)
            .header("editor-plugin-version", PLUGIN_VERSION)
            .header("openai-intent", OPENAI_INTENT)
            .header("copilot-integration-id", INTEGRATION_ID)
            .header("x-github-api-version", GITHUB_API_VERSION)
            .header("x-initiator", initiator)
            .header("content-type", "application/json")
            .json(&body);

        self.ph.send_passthrough(builder, stream).await
    }

    fn supported_models(&self) -> Vec<String> {
        registry::models_for_provider(&ProviderId::Copilot)
    }

    async fn forward_request(
        &self,
        format: ApiFormat,
        body: Value,
        stream: bool,
    ) -> Result<ProviderResponse> {
        if format != ApiFormat::Anthropic {
            return Err(ByokError::UnsupportedModel(
                "CopilotExecutor only supports Anthropic format forwarding".into(),
            ));
        }

        let (token, endpoint) = self.copilot_token().await?;
        let url = format!("{endpoint}/v1/messages");

        let accept = if stream {
            "text/event-stream"
        } else {
            "application/json"
        };
        let beta = Self::build_anthropic_beta(&body);
        let initiator = Self::detect_anthropic_initiator(&body);

        let builder = self
            .ph
            .client()
            .post(&url)
            .header("authorization", format!("Bearer {token}"))
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("anthropic-beta", &beta)
            .header("content-type", "application/json")
            .header("accept", accept)
            .header("user-agent", USER_AGENT)
            .header("editor-version", EDITOR_VERSION)
            .header("editor-plugin-version", PLUGIN_VERSION)
            .header("copilot-integration-id", INTEGRATION_ID)
            .header("openai-intent", OPENAI_INTENT)
            .header("x-github-api-version", GITHUB_API_VERSION)
            .header("x-initiator", initiator)
            .json(&body);

        self.ph.send_passthrough(builder, stream).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use byokey_store::InMemoryTokenStore;
    use rquest::Client;

    fn make_executor() -> CopilotExecutor {
        let store = Arc::new(InMemoryTokenStore::new());
        let auth = Arc::new(AuthManager::new(store, rquest::Client::new()));
        CopilotExecutor::new(Client::new(), None, None, auth, None)
    }

    #[test]
    fn test_supported_models_non_empty() {
        let ex = make_executor();
        assert!(!ex.supported_models().is_empty());
    }

    #[test]
    fn test_initiator_user() {
        let req: ChatRequest = serde_json::from_value(serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .unwrap();
        assert_eq!(CopilotExecutor::initiator(&req), "user");
    }

    #[test]
    fn test_initiator_agent() {
        let req: ChatRequest = serde_json::from_value(serde_json::json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"}
            ]
        }))
        .unwrap();
        assert_eq!(CopilotExecutor::initiator(&req), "agent");
    }
}
