//! Tiered credential routing executor.
//!
//! Replaces `FallbackExecutor`, `RetryExecutor`, and `CredentialChain` with a
//! single executor that tries routing slots in order. Each slot holds a credential
//! source and a provider ID; credentials are rotated within a slot on retryable
//! errors, and execution cascades to the next slot when a slot is exhausted.

use crate::credentials::{Credential, CredentialSource};
use async_trait::async_trait;
use byokey_auth::AuthManager;
use byokey_config::{BalancingStrategy, CredentialSourceKind, ProviderConfig, RoutingEntry};
use byokey_types::{
    ChatRequest, ProviderId, RateLimitStore,
    traits::{ApiFormat, ProviderExecutor, ProviderResponse, Result as ProviderResult},
};
use rquest::Client;
use serde_json::Value;
use std::collections::HashSet;
use std::hash::BuildHasher;
use std::sync::Arc;
use std::time::Duration;

/// Whether a provider natively supports a given API format.
fn provider_supports_format(provider: &ProviderId, format: ApiFormat) -> bool {
    matches!(
        (provider, format),
        (_, ApiFormat::OpenAI)
            | (ProviderId::Copilot | ProviderId::Anthropic, ApiFormat::Anthropic)
    )
}

/// A single slot in the routing chain, holding a credential source for one provider.
pub struct RoutingSlot {
    pub provider: ProviderId,
    pub source: Box<dyn CredentialSource>,
    /// Models this slot's provider can serve. Empty = try unconditionally.
    pub models: HashSet<String>,
}

/// Executes chat-completion requests by trying routing slots in order.
///
/// Within a slot, credentials rotate on retryable errors. When a slot is exhausted
/// (all credentials tried or cooled down) or a non-retryable error occurs, execution
/// cascades to the next slot.
pub struct RoutingExecutor {
    slots: Vec<RoutingSlot>,
    models: Vec<String>,
    auth: Arc<AuthManager>,
    http: Client,
    ratelimit: Option<Arc<RateLimitStore>>,
}

impl RoutingExecutor {
    pub fn new(
        slots: Vec<RoutingSlot>,
        models: Vec<String>,
        auth: Arc<AuthManager>,
        http: Client,
        ratelimit: Option<Arc<RateLimitStore>>,
    ) -> Self {
        Self {
            slots,
            models,
            auth,
            http,
            ratelimit,
        }
    }
}

/// Auto-generate a default routing chain from available credentials.
///
/// When no explicit `routing` is configured, generates entries based on:
/// - API keys first (if any exist)
/// - OAuth second (if provider is in `oauth_providers`)
///
/// This preserves the previous default behavior.
pub fn auto_generate_routing<S: BuildHasher>(
    provider: &ProviderId,
    config: &ProviderConfig,
    oauth_providers: &HashSet<ProviderId, S>,
) -> Vec<RoutingEntry> {
    let mut entries = Vec::new();

    if !config.all_api_keys().is_empty() {
        entries.push(RoutingEntry {
            provider: provider.clone(),
            source: CredentialSourceKind::ApiKeys,
            strategy: BalancingStrategy::Failover,
        });
    }

    if oauth_providers.contains(provider) {
        entries.push(RoutingEntry {
            provider: provider.clone(),
            source: CredentialSourceKind::OAuth,
            strategy: BalancingStrategy::Failover,
        });
    }

    entries
}

/// Build routing slots from resolved routing entries.
///
/// For each entry, looks up the provider's config (primary config for the
/// primary provider, `config_fn` for cross-provider entries) and constructs
/// the appropriate `CredentialSource`. Entries with no available credentials
/// are silently skipped.
#[allow(clippy::too_many_arguments)]
pub fn build_routing_slots<S: BuildHasher>(
    entries: &[RoutingEntry],
    primary_provider: &ProviderId,
    primary_config: &ProviderConfig,
    config_fn: impl Fn(&ProviderId) -> Option<ProviderConfig>,
    oauth_providers: &HashSet<ProviderId, S>,
    auth: &Arc<AuthManager>,
    http: &Client,
    cooldown: Duration,
) -> Vec<RoutingSlot> {
    use crate::credentials::{ApiKeySource, OAuthSource};

    entries
        .iter()
        .filter_map(|entry| {
            let entry_config = if entry.provider == *primary_provider {
                primary_config.clone()
            } else {
                config_fn(&entry.provider)?
            };

            let source: Box<dyn CredentialSource> = match entry.source {
                CredentialSourceKind::ApiKeys => {
                    let keys = entry_config.all_api_keys();
                    if keys.is_empty() {
                        return None;
                    }
                    let keys: Vec<String> = keys.into_iter().map(String::from).collect();
                    Box::new(ApiKeySource::new(keys, cooldown))
                }
                CredentialSourceKind::OAuth => {
                    if !oauth_providers.contains(&entry.provider) {
                        return None;
                    }
                    let strategy =
                        crate::build_oauth_strategy(&entry.provider, &entry.strategy, http);
                    Box::new(OAuthSource::new(
                        Arc::clone(auth),
                        entry.provider.clone(),
                        cooldown,
                        strategy,
                    ))
                }
            };

            Some(RoutingSlot {
                provider: entry.provider.clone(),
                source,
                models: crate::models_for_provider(&entry.provider)
                    .into_iter()
                    .collect(),
            })
        })
        .collect()
}

#[async_trait]
impl ProviderExecutor for RoutingExecutor {
    async fn chat_completion(&self, request: ChatRequest) -> ProviderResult<ProviderResponse> {
        let mut last_err = None;

        for slot in &self.slots {
            // Skip slots that can't serve this model.
            if !slot.models.is_empty() && !slot.models.contains(&request.model) {
                continue;
            }

            // Translate model name to the provider's wire format.
            let wire_model =
                crate::registry::api_name_for_provider(&request.model, &slot.provider);
            let mut slot_request = request.clone();
            if wire_model != request.model {
                slot_request.model = wire_model;
            }

            let max_attempts = slot.source.count().await;

            for _ in 0..max_attempts {
                let Some(credential) = slot.source.next().await else {
                    break;
                };

                let executor = match &credential {
                    Credential::ApiKey { .. } => crate::make_executor(
                        &slot.provider,
                        credential.api_key().map(String::from),
                        None,
                        Arc::clone(&self.auth),
                        self.http.clone(),
                        self.ratelimit.clone(),
                    ),
                    Credential::OAuth { .. } => crate::make_executor(
                        &slot.provider,
                        None,
                        credential.original_account_id(),
                        Arc::clone(&self.auth),
                        self.http.clone(),
                        self.ratelimit.clone(),
                    ),
                };

                let Some(executor) = executor else {
                    break;
                };

                match executor.chat_completion(slot_request.clone()).await {
                    Ok(resp) => return Ok(resp),
                    Err(e) => {
                        let should_retry = if credential.is_api_key() {
                            e.is_retryable()
                        } else {
                            e.is_rate_limited()
                        };

                        if should_retry {
                            tracing::warn!(
                                provider = %slot.provider,
                                credential = %credential.id(),
                                error = %e,
                                "retryable error, rotating credential"
                            );
                            slot.source.mark_rate_limited(credential.id());
                            last_err = Some(e);
                            continue;
                        }

                        if e.is_credential_error() {
                            tracing::warn!(
                                provider = %slot.provider,
                                credential = %credential.id(),
                                error = %e,
                                "credential error, cooling down and rotating"
                            );
                            slot.source.mark_rate_limited(credential.id());
                            last_err = Some(e);
                            continue;
                        }

                        tracing::warn!(
                            provider = %slot.provider,
                            error = %e,
                            "non-retryable error, cascading to next slot"
                        );
                        last_err = Some(e);
                        break;
                    }
                }
            }
        }

        Err(last_err.unwrap_or_else(|| {
            byokey_types::ByokError::Auth("all routing slots exhausted".into())
        }))
    }

    fn supported_models(&self) -> Vec<String> {
        self.models.clone()
    }

    async fn forward_request(
        &self,
        format: ApiFormat,
        body: Value,
        stream: bool,
    ) -> ProviderResult<ProviderResponse> {
        let mut last_err = None;
        let model = body.get("model").and_then(Value::as_str).unwrap_or("");

        for slot in &self.slots {
            // Skip slots that don't support this format.
            if !provider_supports_format(&slot.provider, format) {
                continue;
            }
            // Skip slots that can't serve this model.
            if !slot.models.is_empty() && !slot.models.contains(model) {
                continue;
            }

            // Translate model name to the provider's wire format.
            let wire_model = crate::registry::api_name_for_provider(model, &slot.provider);
            let slot_body = if wire_model == model {
                body.clone()
            } else {
                let mut b = body.clone();
                b["model"] = Value::String(wire_model);
                b
            };

            let max_attempts = slot.source.count().await;

            for _ in 0..max_attempts {
                let Some(credential) = slot.source.next().await else {
                    break;
                };

                let executor = match &credential {
                    Credential::ApiKey { .. } => crate::make_executor(
                        &slot.provider,
                        credential.api_key().map(String::from),
                        None,
                        Arc::clone(&self.auth),
                        self.http.clone(),
                        self.ratelimit.clone(),
                    ),
                    Credential::OAuth { .. } => crate::make_executor(
                        &slot.provider,
                        None,
                        credential.original_account_id(),
                        Arc::clone(&self.auth),
                        self.http.clone(),
                        self.ratelimit.clone(),
                    ),
                };

                let Some(executor) = executor else {
                    break;
                };

                match executor.forward_request(format, slot_body.clone(), stream).await {
                    Ok(resp) => return Ok(resp),
                    Err(e) => {
                        let should_retry = if credential.is_api_key() {
                            e.is_retryable()
                        } else {
                            e.is_rate_limited()
                        };

                        if should_retry {
                            tracing::warn!(
                                provider = %slot.provider,
                                credential = %credential.id(),
                                error = %e,
                                "retryable error in forward_request, rotating credential"
                            );
                            slot.source.mark_rate_limited(credential.id());
                            last_err = Some(e);
                            continue;
                        }

                        if e.is_credential_error() {
                            tracing::warn!(
                                provider = %slot.provider,
                                credential = %credential.id(),
                                error = %e,
                                "credential error in forward_request, cooling down"
                            );
                            slot.source.mark_rate_limited(credential.id());
                            last_err = Some(e);
                            continue;
                        }

                        tracing::warn!(
                            provider = %slot.provider,
                            error = %e,
                            "non-retryable error in forward_request, cascading"
                        );
                        last_err = Some(e);
                        break;
                    }
                }
            }
        }

        Err(last_err.unwrap_or_else(|| {
            byokey_types::ByokError::Auth("all routing slots exhausted for forward_request".into())
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::credentials::ApiKeySource;
    use byokey_store::InMemoryTokenStore;
    use std::time::Duration;

    fn make_auth() -> Arc<AuthManager> {
        Arc::new(AuthManager::new(
            Arc::new(InMemoryTokenStore::new()),
            rquest::Client::new(),
        ))
    }

    #[test]
    fn test_routing_executor_supported_models() {
        let slot = RoutingSlot {
            provider: ProviderId::OpenAI,
            source: Box::new(ApiKeySource::new(vec!["k1".into()], Duration::from_secs(30))),
            models: HashSet::new(),
        };
        let executor = RoutingExecutor::new(
            vec![slot],
            vec!["gpt-4o".to_string(), "o3-pro".to_string()],
            make_auth(),
            rquest::Client::new(),
            None,
        );
        assert_eq!(
            executor.supported_models(),
            vec!["gpt-4o".to_string(), "o3-pro".to_string()]
        );
    }

    // --- auto_generate_routing tests ---

    #[test]
    fn test_auto_generate_routing_api_keys_only() {
        let config = ProviderConfig {
            api_key: Some("sk-test".into()),
            ..Default::default()
        };
        let entries = auto_generate_routing(&ProviderId::OpenAI, &config, &HashSet::new());
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].source, CredentialSourceKind::ApiKeys);
        assert_eq!(entries[0].strategy, BalancingStrategy::Failover);
    }

    #[test]
    fn test_auto_generate_routing_oauth_only() {
        let mut oauth = HashSet::new();
        oauth.insert(ProviderId::OpenAI);
        let config = ProviderConfig::default();
        let entries = auto_generate_routing(&ProviderId::OpenAI, &config, &oauth);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].source, CredentialSourceKind::OAuth);
    }

    #[test]
    fn test_auto_generate_routing_both() {
        let mut oauth = HashSet::new();
        oauth.insert(ProviderId::OpenAI);
        let config = ProviderConfig {
            api_key: Some("sk-test".into()),
            ..Default::default()
        };
        let entries = auto_generate_routing(&ProviderId::OpenAI, &config, &oauth);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].source, CredentialSourceKind::ApiKeys);
        assert_eq!(entries[1].source, CredentialSourceKind::OAuth);
    }

    #[test]
    fn test_auto_generate_routing_no_credentials() {
        let config = ProviderConfig::default();
        let entries = auto_generate_routing(&ProviderId::OpenAI, &config, &HashSet::new());
        assert!(entries.is_empty());
    }

    // --- build_routing_slots tests ---

    #[test]
    fn test_build_routing_slots_single_api_key_entry() {
        let entries = vec![RoutingEntry {
            provider: ProviderId::OpenAI,
            source: CredentialSourceKind::ApiKeys,
            strategy: BalancingStrategy::Failover,
        }];
        let config = ProviderConfig {
            api_key: Some("sk-test".into()),
            ..Default::default()
        };
        let auth = make_auth();
        let http = rquest::Client::new();
        let slots = build_routing_slots(
            &entries,
            &ProviderId::OpenAI,
            &config,
            |_| None,
            &HashSet::new(),
            &auth,
            &http,
            Duration::from_secs(30),
        );
        assert_eq!(slots.len(), 1);
        assert_eq!(slots[0].provider, ProviderId::OpenAI);
    }

    #[test]
    fn test_build_routing_slots_cross_provider() {
        let entries = vec![
            RoutingEntry {
                provider: ProviderId::OpenAI,
                source: CredentialSourceKind::ApiKeys,
                strategy: BalancingStrategy::Failover,
            },
            RoutingEntry {
                provider: ProviderId::Copilot,
                source: CredentialSourceKind::ApiKeys,
                strategy: BalancingStrategy::Failover,
            },
        ];
        let config = ProviderConfig {
            api_key: Some("sk-codex".into()),
            ..Default::default()
        };
        let config_fn = |p: &ProviderId| match p {
            ProviderId::Copilot => Some(ProviderConfig {
                api_key: Some("sk-copilot".into()),
                ..Default::default()
            }),
            _ => None,
        };
        let auth = make_auth();
        let http = rquest::Client::new();
        let slots = build_routing_slots(
            &entries,
            &ProviderId::OpenAI,
            &config,
            config_fn,
            &HashSet::new(),
            &auth,
            &http,
            Duration::from_secs(30),
        );
        assert_eq!(slots.len(), 2);
        assert_eq!(slots[0].provider, ProviderId::OpenAI);
        assert_eq!(slots[1].provider, ProviderId::Copilot);
    }

    #[test]
    fn test_build_routing_slots_skips_empty_source() {
        let entries = vec![RoutingEntry {
            provider: ProviderId::Copilot,
            source: CredentialSourceKind::ApiKeys,
            strategy: BalancingStrategy::Failover,
        }];
        let config = ProviderConfig::default();
        let auth = make_auth();
        let http = rquest::Client::new();
        let slots = build_routing_slots(
            &entries,
            &ProviderId::OpenAI,
            &config,
            |_| None,
            &HashSet::new(),
            &auth,
            &http,
            Duration::from_secs(30),
        );
        assert!(slots.is_empty());
    }

    // --- RoutingExecutor construction tests ---

    #[test]
    fn test_routing_executor_single_slot() {
        let slot = RoutingSlot {
            provider: ProviderId::OpenAI,
            source: Box::new(ApiKeySource::new(
                vec!["k1".into(), "k2".into()],
                Duration::from_secs(30),
            )),
            models: HashSet::new(),
        };
        let executor = RoutingExecutor::new(
            vec![slot],
            vec!["gpt-4o".into()],
            make_auth(),
            rquest::Client::new(),
            None,
        );
        assert_eq!(executor.supported_models(), vec!["gpt-4o".to_string()]);
        assert_eq!(executor.slots.len(), 1);
    }

    #[test]
    fn test_routing_executor_multi_slot() {
        let slot1 = RoutingSlot {
            provider: ProviderId::OpenAI,
            source: Box::new(ApiKeySource::new(vec!["k1".into()], Duration::from_secs(30))),
            models: HashSet::new(),
        };
        let slot2 = RoutingSlot {
            provider: ProviderId::Copilot,
            source: Box::new(ApiKeySource::new(vec!["k2".into()], Duration::from_secs(30))),
            models: HashSet::new(),
        };
        let executor = RoutingExecutor::new(
            vec![slot1, slot2],
            vec!["gpt-4o".into()],
            make_auth(),
            rquest::Client::new(),
            None,
        );
        assert_eq!(executor.slots.len(), 2);
        assert_eq!(executor.slots[0].provider, ProviderId::OpenAI);
        assert_eq!(executor.slots[1].provider, ProviderId::Copilot);
    }

    #[tokio::test]
    async fn test_routing_executor_cooldown_propagation() {
        let source = ApiKeySource::new(vec!["k1".into(), "k2".into()], Duration::from_secs(300));
        let slot = RoutingSlot {
            provider: ProviderId::OpenAI,
            source: Box::new(source),
            models: HashSet::new(),
        };
        slot.source.mark_rate_limited("0");
        let cred = slot.source.next().await.unwrap();
        assert_eq!(cred.id(), "1");
    }

    #[test]
    fn test_routing_slot_with_models() {
        let slot = RoutingSlot {
            provider: ProviderId::Copilot,
            source: Box::new(ApiKeySource::new(vec!["k1".into()], Duration::from_secs(30))),
            models: HashSet::from(["gpt-4o".to_string(), "gpt-4o-mini".to_string()]),
        };
        assert!(slot.models.contains("gpt-4o"));
        assert!(!slot.models.contains("claude-sonnet-4-20250514"));
    }

    #[test]
    fn test_routing_slot_empty_models_matches_all() {
        let slot = RoutingSlot {
            provider: ProviderId::Copilot,
            source: Box::new(ApiKeySource::new(vec!["k1".into()], Duration::from_secs(30))),
            models: HashSet::new(),
        };
        assert!(slot.models.is_empty());
    }
}
