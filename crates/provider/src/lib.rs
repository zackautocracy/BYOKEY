//! Provider executor implementations and model registry.
//!
//! Each provider module implements [`ProviderExecutor`] for a specific AI backend.
//! The [`make_executor`] and [`make_executor_for_model`] functions create boxed
//! executors based on provider or model identifiers.

pub mod credentials;
pub mod executors;
pub mod http_util;
pub mod registry;
pub mod routing;

pub use executors::{
    AntigravityExecutor, AnthropicExecutor, OpenAiExecutor, CopilotExecutor, GeminiExecutor,
    IFlowExecutor, KimiExecutor, KiroExecutor, QwenExecutor, make_executor,
};
pub use registry::{
    ModelEntry, all_models, api_name_for_provider, is_copilot_free_model, models_for_provider,
    parse_qualified_model, resolve_model_id, resolve_provider, resolve_provider_with,
};
pub use credentials::ApiKeySource;

pub use http_util::ProviderHttp;

use byokey_auth::AuthManager;
use byokey_config::ProviderConfig;
use byokey_types::{
    ByokError, ProviderId, RateLimitStore,
    traits::ProviderExecutor,
};
use rquest::Client;
use std::collections::HashSet;
use std::hash::BuildHasher;
use std::sync::Arc;

/// Build the appropriate OAuth selection strategy for a provider.
fn build_oauth_strategy(
    provider: &ProviderId,
    strategy: &byokey_config::BalancingStrategy,
    http: &Client,
) -> credentials::SelectionStrategy {
    use byokey_config::BalancingStrategy;
    match strategy {
        BalancingStrategy::Failover => credentials::SelectionStrategy::Failover,
        BalancingStrategy::RoundRobin => credentials::SelectionStrategy::RoundRobin,
        BalancingStrategy::QuotaAware => {
            let fetcher: Arc<dyn credentials::QuotaFetcher> = match provider {
                ProviderId::Copilot => {
                    Arc::new(crate::executors::copilot::quota::CopilotQuotaFetcher::new(
                        http.clone(),
                    ))
                }
                ProviderId::OpenAI => {
                    Arc::new(crate::executors::openai::quota::CodexQuotaFetcher::new(
                        http.clone(),
                    ))
                }
                other => {
                    tracing::warn!(
                        provider = %other,
                        "quota_aware strategy requested but no quota fetcher available, falling back to failover"
                    );
                    return credentials::SelectionStrategy::Failover;
                }
            };
            credentials::SelectionStrategy::QuotaAware {
                quota_fetcher: fetcher,
                rebalance_interval: std::time::Duration::from_secs(300),
            }
        }
    }
}

/// Create an executor by resolving the model string to its provider.
///
/// Respects `ProviderConfig::routing` (ordered list of credential sources
/// wrapped in a [`routing::RoutingExecutor`]). When no explicit routing is
/// configured, entries are auto-generated from available credentials.
///
/// # Errors
///
/// Returns [`ByokError::UnsupportedModel`] if the model string is not recognised
/// or if the resolved provider does not have an executor implemented yet.
pub fn make_executor_for_model<S: BuildHasher>(
    model: &str,
    config_fn: impl Fn(&ProviderId) -> Option<ProviderConfig>,
    oauth_providers: &HashSet<ProviderId, S>,
    provider_hint: Option<&ProviderId>,
    auth: Arc<AuthManager>,
    http: Client,
    ratelimit: Option<Arc<RateLimitStore>>,
) -> Result<Box<dyn ProviderExecutor>, ByokError> {
    let provider = if let Some(p) = provider_hint {
        p.clone()
    } else {
        registry::resolve_provider_with(model, |p| {
            config_fn(p)
                .as_ref()
                .is_some_and(|c| c.api_key.is_some() || !c.api_keys.is_empty())
                || oauth_providers.contains(p)
        })
        .or_else(|| registry::resolve_provider(model))
        .ok_or_else(|| ByokError::UnsupportedModel(model.to_string()))?
    };

    let config = config_fn(&provider).unwrap_or_default();

    // Resolve routing entries (explicit or auto-generated).
    let entries = if config.routing.is_empty() {
        routing::auto_generate_routing(&provider, &config, oauth_providers)
    } else {
        config.routing.clone()
    };

    // Single-credential optimization: skip RoutingExecutor wrapper.
    // Only applies when the single entry targets the primary provider.
    let all_keys = config.all_api_keys();
    if entries.len() == 1
        && entries[0].provider == provider
        && entries[0].source == byokey_config::CredentialSourceKind::ApiKeys
        && all_keys.len() == 1
    {
        let api_key = all_keys.into_iter().next().map(String::from);
        return make_executor(
            &provider,
            api_key,
            None,
            auth,
            http,
            ratelimit,
        )
        .ok_or_else(|| ByokError::UnsupportedModel(model.to_string()));
    }

    // Build routing slots.
    let cooldown = std::time::Duration::from_secs(config.cooldown_seconds);
    let slots = routing::build_routing_slots(
        &entries,
        &provider,
        &config,
        &config_fn,
        oauth_providers,
        &auth,
        &http,
        cooldown,
    );

    if slots.is_empty() {
        return make_executor(&provider, config.api_key, None, auth, http, ratelimit)
            .ok_or_else(|| ByokError::UnsupportedModel(model.to_string()));
    }

    // Get model list from a throwaway executor.
    let models = make_executor(&provider, None, None, Arc::clone(&auth), http.clone(), None)
        .map(|e| e.supported_models())
        .unwrap_or_default();

    Ok(Box::new(routing::RoutingExecutor::new(
        slots,
        models,
        auth,
        http,
        ratelimit,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use byokey_store::InMemoryTokenStore;

    fn make_auth() -> Arc<AuthManager> {
        Arc::new(AuthManager::new(
            Arc::new(InMemoryTokenStore::new()),
            rquest::Client::new(),
        ))
    }

    fn make_http() -> Client {
        Client::new()
    }

    fn empty_oauth() -> HashSet<ProviderId> {
        HashSet::new()
    }

    #[test]
    fn test_make_executor_claude() {
        let auth = make_auth();
        let ex = make_executor(&ProviderId::Anthropic, None, None, auth, make_http(), None);
        assert!(ex.is_some());
        assert!(
            ex.unwrap()
                .supported_models()
                .iter()
                .any(|m| m.starts_with("claude-"))
        );
    }

    #[test]
    fn test_make_executor_codex() {
        let auth = make_auth();
        let ex = make_executor(
            &ProviderId::OpenAI,
            Some("sk-test".into()),
            None,
            auth,
            make_http(),
            None,
        );
        assert!(ex.is_some());
    }

    #[test]
    fn test_make_executor_gemini() {
        let auth = make_auth();
        let ex = make_executor(&ProviderId::Gemini, None, None, auth, make_http(), None);
        assert!(ex.is_some());
    }

    #[test]
    fn test_make_executor_copilot() {
        let auth = make_auth();
        let ex = make_executor(&ProviderId::Copilot, None, None, auth, make_http(), None);
        assert!(ex.is_some());
    }

    #[test]
    fn test_make_executor_antigravity() {
        let auth = make_auth();
        let ex = make_executor(&ProviderId::Antigravity, None, None, auth, make_http(), None);
        assert!(ex.is_some());
        // Antigravity models now use canonical names (no ag- prefix).
        assert!(
            ex.unwrap()
                .supported_models()
                .iter()
                .any(|m| m.contains("gemini"))
        );
    }

    #[test]
    fn test_make_executor_kimi() {
        let auth = make_auth();
        let ex = make_executor(&ProviderId::Kimi, None, None, auth, make_http(), None);
        assert!(ex.is_some());
        assert!(
            ex.unwrap()
                .supported_models()
                .iter()
                .any(|m| m.starts_with("kimi-"))
        );
    }

    #[test]
    fn test_make_executor_for_model_claude() {
        let auth = make_auth();
        let ex = make_executor_for_model(
            "claude-opus-4-6",
            |_| None,
            &empty_oauth(),
            None,
            auth,
            make_http(),
            None,
        );
        assert!(ex.is_ok());
    }

    #[test]
    fn test_make_executor_for_model_unknown() {
        let auth = make_auth();
        let result = make_executor_for_model(
            "nonexistent-model",
            |_| None,
            &empty_oauth(),
            None,
            auth,
            make_http(),
            None,
        );
        assert!(matches!(result, Err(ByokError::UnsupportedModel(_))));
    }

    #[test]
    fn test_make_executor_for_model_passes_api_key() {
        let auth = make_auth();
        let ex = make_executor_for_model(
            "gpt-4o",
            |p| match p {
                ProviderId::Copilot => Some(ProviderConfig {
                    api_key: Some("sk-test".into()),
                    ..Default::default()
                }),
                _ => None,
            },
            &empty_oauth(),
            None,
            auth,
            make_http(),
            None,
        );
        assert!(ex.is_ok());
    }

    #[test]
    fn test_make_executor_for_model_cross_provider_via_routing() {
        use byokey_config::{BalancingStrategy, CredentialSourceKind, RoutingEntry};
        let auth = make_auth();
        // gemini model with routing through copilot
        let ex = make_executor_for_model(
            "gemini-2.0-flash",
            |p| match p {
                ProviderId::Gemini => Some(ProviderConfig {
                    routing: vec![RoutingEntry {
                        provider: ProviderId::Copilot,
                        source: CredentialSourceKind::ApiKeys,
                        strategy: BalancingStrategy::Failover,
                    }],
                    ..Default::default()
                }),
                ProviderId::Copilot => Some(ProviderConfig {
                    api_key: Some("sk-copilot".into()),
                    ..Default::default()
                }),
                _ => None,
            },
            &empty_oauth(),
            None,
            auth,
            make_http(),
            None,
        );
        assert!(ex.is_ok());
    }

    #[test]
    fn test_make_executor_for_model_multi_key_routing() {
        use byokey_config::ApiKeyEntry;

        let auth = make_auth();
        let ex = make_executor_for_model(
            "claude-opus-4-6",
            |p| match p {
                ProviderId::Anthropic => Some(ProviderConfig {
                    api_keys: vec![
                        ApiKeyEntry {
                            api_key: "sk-key-1".into(),
                            label: None,
                        },
                        ApiKeyEntry {
                            api_key: "sk-key-2".into(),
                            label: None,
                        },
                    ],
                    ..Default::default()
                }),
                _ => None,
            },
            &empty_oauth(),
            None,
            auth,
            make_http(),
            None,
        );
        assert!(ex.is_ok());
        let models = ex.unwrap().supported_models();
        assert!(models.iter().any(|m| m.starts_with("claude-")));
    }

    #[test]
    fn test_make_executor_for_model_single_api_key_no_retry() {
        let auth = make_auth();
        // Single api_key → direct executor, no RoutingExecutor wrapper
        let ex = make_executor_for_model(
            "claude-opus-4-6",
            |p| match p {
                ProviderId::Anthropic => Some(ProviderConfig {
                    api_key: Some("sk-single".into()),
                    ..Default::default()
                }),
                _ => None,
            },
            &empty_oauth(),
            None,
            auth,
            make_http(),
            None,
        );
        assert!(ex.is_ok());
    }

    #[test]
    fn test_make_executor_for_model_explicit_routing() {
        use byokey_config::{BalancingStrategy, CredentialSourceKind, RoutingEntry};
        let config_fn = |_: &ProviderId| {
            Some(ProviderConfig {
                api_key: Some("sk-test".into()),
                routing: vec![RoutingEntry {
                    provider: ProviderId::OpenAI,
                    source: CredentialSourceKind::ApiKeys,
                    strategy: BalancingStrategy::RoundRobin,
                }],
                ..Default::default()
            })
        };

        let ex = make_executor_for_model(
            "gpt-4o",
            config_fn,
            &empty_oauth(),
            Some(&ProviderId::OpenAI),
            make_auth(),
            make_http(),
            None,
        );
        assert!(ex.is_ok());
    }

    #[tokio::test]
    async fn test_make_executor_for_model_auto_generated_routing() {
        use byokey_types::OAuthToken;

        let store = Arc::new(InMemoryTokenStore::new());
        let auth = Arc::new(AuthManager::new(
            Arc::clone(&store) as _,
            rquest::Client::new(),
        ));
        auth.save_token_for(
            &ProviderId::OpenAI,
            "acct-a",
            None,
            OAuthToken::new("tok-a").with_expiry(3600),
        )
        .await
        .unwrap();

        let mut oauth_providers = HashSet::new();
        oauth_providers.insert(ProviderId::OpenAI);

        let config_fn = |_: &ProviderId| {
            Some(ProviderConfig {
                api_key: Some("sk-test".into()),
                ..Default::default()
            })
        };

        let ex = make_executor_for_model(
            "gpt-4o",
            config_fn,
            &oauth_providers,
            Some(&ProviderId::OpenAI),
            auth,
            make_http(),
            None,
        );
        assert!(ex.is_ok());
        let models = ex.unwrap().supported_models();
        assert!(!models.is_empty());
    }

    #[test]
    fn test_make_executor_for_model_single_key_no_routing_executor() {
        let config_fn = |_: &ProviderId| {
            Some(ProviderConfig {
                api_key: Some("sk-test".into()),
                ..Default::default()
            })
        };

        let ex = make_executor_for_model(
            "gpt-4o",
            config_fn,
            &empty_oauth(),
            Some(&ProviderId::OpenAI),
            make_auth(),
            make_http(),
            None,
        );
        assert!(ex.is_ok());
        let models = ex.unwrap().supported_models();
        assert!(!models.is_empty());
    }

    #[test]
    fn test_make_executor_for_model_cross_provider_routing() {
        use byokey_config::{BalancingStrategy, CredentialSourceKind, RoutingEntry};
        let config_fn = |p: &ProviderId| match p {
            ProviderId::OpenAI => Some(ProviderConfig {
                routing: vec![RoutingEntry {
                    provider: ProviderId::Copilot,
                    source: CredentialSourceKind::ApiKeys,
                    strategy: BalancingStrategy::RoundRobin,
                }],
                ..Default::default()
            }),
            ProviderId::Copilot => Some(ProviderConfig {
                api_key: Some("sk-copilot".into()),
                ..Default::default()
            }),
            _ => None,
        };

        let ex = make_executor_for_model(
            "gpt-4o",
            config_fn,
            &empty_oauth(),
            Some(&ProviderId::OpenAI),
            make_auth(),
            make_http(),
            None,
        );
        assert!(ex.is_ok());
    }

    #[tokio::test]
    async fn test_make_executor_for_model_routing_with_oauth() {
        use byokey_config::{BalancingStrategy, CredentialSourceKind, RoutingEntry};
        use byokey_types::OAuthToken;

        let store = Arc::new(InMemoryTokenStore::new());
        let auth = Arc::new(AuthManager::new(
            Arc::clone(&store) as _,
            rquest::Client::new(),
        ));

        auth.save_token_for(
            &ProviderId::OpenAI,
            "acct-a",
            None,
            OAuthToken::new("tok-a").with_expiry(3600),
        )
        .await
        .unwrap();

        let mut oauth_providers = HashSet::new();
        oauth_providers.insert(ProviderId::OpenAI);

        let config_fn = |_: &ProviderId| {
            Some(ProviderConfig {
                routing: vec![RoutingEntry {
                    provider: ProviderId::OpenAI,
                    source: CredentialSourceKind::OAuth,
                    strategy: BalancingStrategy::RoundRobin,
                }],
                cooldown_seconds: 300,
                ..Default::default()
            })
        };

        let ex = make_executor_for_model(
            "gpt-4o",
            config_fn,
            &oauth_providers,
            Some(&ProviderId::OpenAI),
            auth,
            make_http(),
            None,
        );
        assert!(ex.is_ok());
        let models = ex.unwrap().supported_models();
        assert!(!models.is_empty());
    }
}
