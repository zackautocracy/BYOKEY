//! Credential abstraction — unifies API key and OAuth token sources for retry logic.

pub mod api_key;
pub mod oauth;
pub mod quota;

pub use api_key::ApiKeySource;
pub use oauth::{OAuthSource, QuotaFetcher, QuotaSnapshot, SelectionStrategy};
pub use quota::{CachedQuota, QuotaTracker};

use async_trait::async_trait;

/// A credential to use for a provider request.
#[derive(Clone)]
pub enum Credential {
    /// API key — passed to executor as `api_key: Some(key)`.
    ApiKey {
        /// Identifier for cooldown tracking (e.g. key index).
        id: String,
        /// The raw API key string.
        key: String,
    },
    /// OAuth account — executor is created with `api_key: None` and
    /// uses `auth.get_token_for(provider, account_id)`.
    OAuth {
        /// Account ID from the store, used for cooldown tracking and token lookup.
        account_id: String,
    },
}

impl Credential {
    /// Returns the identifier used for cooldown tracking.
    #[must_use]
    pub fn id(&self) -> &str {
        match self {
            Self::ApiKey { id, .. } => id,
            Self::OAuth { account_id, .. } => account_id,
        }
    }

    /// Returns `true` if this is an API key credential.
    #[must_use]
    pub fn is_api_key(&self) -> bool {
        matches!(self, Self::ApiKey { .. })
    }

    /// Returns the original account ID (without chain prefix) for OAuth credentials.
    #[must_use]
    pub fn original_account_id(&self) -> Option<&str> {
        match self {
            Self::OAuth { account_id } => {
                // Strip "idx:" prefix if present.
                Some(
                    account_id
                        .split_once(':')
                        .map_or(account_id.as_str(), |(_, id)| id),
                )
            }
            Self::ApiKey { .. } => None,
        }
    }

    /// Returns the raw API key string for API key credentials.
    #[must_use]
    pub fn api_key(&self) -> Option<&str> {
        match self {
            Self::ApiKey { key, .. } => Some(key),
            Self::OAuth { .. } => None,
        }
    }
}

/// Abstracts credential retrieval for both API keys and OAuth tokens.
///
/// Implementations handle round-robin selection and per-credential cooldowns.
#[async_trait]
pub trait CredentialSource: Send + Sync {
    /// Returns the next available credential, skipping cooled-down ones.
    ///
    /// Returns `None` if all credentials are exhausted or in cooldown.
    async fn next(&self) -> Option<Credential>;

    /// Marks a credential as rate-limited, placing it in cooldown.
    fn mark_rate_limited(&self, id: &str);

    /// Number of available credentials (may query the store).
    async fn count(&self) -> usize;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_credential_id_api_key() {
        let c = Credential::ApiKey {
            id: "0".into(),
            key: "sk-test".into(),
        };
        assert_eq!(c.id(), "0");
        assert!(c.is_api_key());
        assert!(c.original_account_id().is_none());
        assert_eq!(c.api_key().unwrap(), "sk-test");
    }

    #[test]
    fn test_credential_id_oauth() {
        let c = Credential::OAuth {
            account_id: "work".into(),
        };
        assert_eq!(c.id(), "work");
        assert!(!c.is_api_key());
        assert_eq!(c.original_account_id().unwrap(), "work");
        assert!(c.api_key().is_none());
    }

    #[test]
    fn test_original_account_id_strips_chain_prefix() {
        // CredentialChain prefixes IDs as "source_idx:original_id"
        let c = Credential::OAuth {
            account_id: "0:my-account".into(),
        };
        assert_eq!(c.original_account_id().unwrap(), "my-account");
        assert_eq!(c.id(), "0:my-account");
    }

    #[test]
    fn test_api_key_with_chain_prefix() {
        let c = Credential::ApiKey {
            id: "1:2".into(),
            key: "sk-key".into(),
        };
        // api_key() always returns the key regardless of prefix
        assert_eq!(c.api_key().unwrap(), "sk-key");
        // original_account_id is None for API keys
        assert!(c.original_account_id().is_none());
    }
}
