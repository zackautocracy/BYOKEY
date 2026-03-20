//! Rate limit snapshot storage — captures upstream provider rate limit headers.

use crate::ProviderId;
use serde::Serialize;
use std::collections::HashMap;
use std::sync::Mutex;
use utoipa::ToSchema;

/// A snapshot of rate limit headers captured from a single upstream response.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct RateLimitSnapshot {
    /// Raw rate-limit-related header key/value pairs.
    pub headers: HashMap<String, String>,
    /// Unix timestamp (seconds) when this snapshot was captured.
    pub captured_at: u64,
}

/// Thread-safe in-memory store for per-provider, per-account rate limit snapshots.
pub struct RateLimitStore {
    inner: Mutex<HashMap<(ProviderId, String), RateLimitSnapshot>>,
}

impl RateLimitStore {
    /// Creates a new empty store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
        }
    }

    /// Inserts or replaces the snapshot for the given provider + account.
    ///
    /// # Panics
    ///
    /// Panics if the internal mutex is poisoned.
    pub fn update(&self, provider: ProviderId, account_id: String, snapshot: RateLimitSnapshot) {
        let mut map = self.inner.lock().unwrap();
        map.insert((provider, account_id), snapshot);
    }

    /// Returns the snapshot for a specific provider + account, if any.
    ///
    /// # Panics
    ///
    /// Panics if the internal mutex is poisoned.
    #[must_use]
    pub fn get(&self, provider: &ProviderId, account_id: &str) -> Option<RateLimitSnapshot> {
        let map = self.inner.lock().unwrap();
        map.get(&(provider.clone(), account_id.to_string()))
            .cloned()
    }

    /// Returns all stored snapshots.
    ///
    /// # Panics
    ///
    /// Panics if the internal mutex is poisoned.
    #[must_use]
    pub fn all(&self) -> HashMap<(ProviderId, String), RateLimitSnapshot> {
        self.inner.lock().unwrap().clone()
    }
}

impl Default for RateLimitStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_update_and_get() {
        let store = RateLimitStore::new();
        let snap = RateLimitSnapshot {
            headers: HashMap::from([("x-ratelimit-remaining".into(), "50".into())]),
            captured_at: 1_700_000_000,
        };
        store.update(ProviderId::Anthropic, "active".into(), snap);

        let got = store.get(&ProviderId::Anthropic, "active").unwrap();
        assert_eq!(got.headers["x-ratelimit-remaining"], "50");
        assert_eq!(got.captured_at, 1_700_000_000);
    }

    #[test]
    fn test_get_missing() {
        let store = RateLimitStore::new();
        assert!(store.get(&ProviderId::Anthropic, "active").is_none());
    }

    #[test]
    fn test_all() {
        let store = RateLimitStore::new();
        store.update(
            ProviderId::Anthropic,
            "a".into(),
            RateLimitSnapshot {
                headers: HashMap::new(),
                captured_at: 1,
            },
        );
        store.update(
            ProviderId::Gemini,
            "b".into(),
            RateLimitSnapshot {
                headers: HashMap::new(),
                captured_at: 2,
            },
        );
        assert_eq!(store.all().len(), 2);
    }
}
