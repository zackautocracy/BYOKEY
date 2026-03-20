//! Credential routing — round-robin API key selection with error cooldown.

use super::{Credential, CredentialSource};
use async_trait::async_trait;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// A round-robin API key credential source with per-key cooldown.
pub struct ApiKeySource {
    /// API keys available for rotation.
    keys: Vec<String>,
    /// Atomic counter for round-robin selection.
    index: AtomicUsize,
    /// Per-key cooldown state: `Some(until)` means the key is cooled down.
    cooldowns: Mutex<Vec<Option<Instant>>>,
    /// How long a key stays in cooldown after an error.
    cooldown_duration: Duration,
}

impl ApiKeySource {
    /// Creates a new source with the given keys and cooldown duration.
    ///
    /// # Panics
    ///
    /// Panics if `keys` is empty.
    #[must_use]
    pub fn new(keys: Vec<String>, cooldown_duration: Duration) -> Self {
        assert!(
            !keys.is_empty(),
            "ApiKeySource requires at least one key"
        );
        let len = keys.len();
        Self {
            keys,
            index: AtomicUsize::new(0),
            cooldowns: Mutex::new(vec![None; len]),
            cooldown_duration,
        }
    }

    /// Returns the number of configured keys.
    #[must_use]
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Returns `true` if there are no keys.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
}

#[async_trait]
impl CredentialSource for ApiKeySource {
    async fn next(&self) -> Option<Credential> {
        let len = self.keys.len();
        let start = self.index.fetch_add(1, Ordering::Relaxed) % len;
        let now = Instant::now();
        let cooldowns = self.cooldowns.lock().expect("cooldown lock");

        for i in 0..len {
            let idx = (start + i) % len;
            if cooldowns[idx].is_some_and(|until| now < until) {
                continue;
            }
            return Some(Credential::ApiKey {
                id: idx.to_string(),
                key: self.keys[idx].clone(),
            });
        }
        None
    }

    fn mark_rate_limited(&self, id: &str) {
        if let Ok(idx) = id.parse::<usize>()
            && idx < self.keys.len()
        {
            let mut cooldowns = self.cooldowns.lock().expect("cooldown lock");
            cooldowns[idx] = Some(Instant::now() + self.cooldown_duration);
        }
    }

    async fn count(&self) -> usize {
        self.keys.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_round_robin() {
        let source = ApiKeySource::new(
            vec!["key-a".into(), "key-b".into(), "key-c".into()],
            Duration::from_secs(60),
        );
        let c1 = source.next().await.unwrap();
        let c2 = source.next().await.unwrap();
        let c3 = source.next().await.unwrap();
        let c4 = source.next().await.unwrap();
        // Round-robin cycles by index
        assert_eq!(c1.id(), "0");
        assert_eq!(c2.id(), "1");
        assert_eq!(c3.id(), "2");
        assert_eq!(c4.id(), "0");
        // Keys match
        assert_eq!(c1.api_key().unwrap(), "key-a");
        assert_eq!(c2.api_key().unwrap(), "key-b");
        assert_eq!(c3.api_key().unwrap(), "key-c");
        assert_eq!(c4.api_key().unwrap(), "key-a");
    }

    #[tokio::test]
    async fn test_cooldown_skips_key() {
        let source = ApiKeySource::new(
            vec!["key-a".into(), "key-b".into()],
            Duration::from_secs(60),
        );
        // Cool down key-a (index 0)
        source.mark_rate_limited("0");
        // Should skip key-a and return key-b
        let c = source.next().await.unwrap();
        assert_eq!(c.api_key().unwrap(), "key-b");
    }

    #[tokio::test]
    async fn test_all_cooled_returns_none() {
        let source = ApiKeySource::new(
            vec!["key-a".into(), "key-b".into()],
            Duration::from_secs(60),
        );
        source.mark_rate_limited("0");
        source.mark_rate_limited("1");
        assert!(source.next().await.is_none());
    }

    #[tokio::test]
    async fn test_single_key() {
        let source = ApiKeySource::new(vec!["only-key".into()], Duration::from_secs(60));
        assert_eq!(source.next().await.unwrap().api_key().unwrap(), "only-key");
        assert_eq!(source.next().await.unwrap().api_key().unwrap(), "only-key");
    }

    #[test]
    fn test_len() {
        let source = ApiKeySource::new(vec!["a".into(), "b".into()], Duration::from_secs(1));
        assert_eq!(source.len(), 2);
        assert!(!source.is_empty());
    }

    #[test]
    #[should_panic(expected = "at least one key")]
    fn test_empty_keys_panics() {
        let _ = ApiKeySource::new(vec![], Duration::from_secs(1));
    }

    #[tokio::test]
    async fn test_next_returns_api_key_credential() {
        let source = ApiKeySource::new(
            vec!["key-a".into(), "key-b".into()],
            Duration::from_secs(60),
        );
        let cred = source.next().await.unwrap();
        assert!(cred.is_api_key());
        assert_eq!(cred.id(), "0");
    }

    #[tokio::test]
    async fn test_count() {
        let source = ApiKeySource::new(
            vec!["key-a".into(), "key-b".into()],
            Duration::from_secs(60),
        );
        assert_eq!(source.count().await, 2);
    }

    #[tokio::test]
    async fn test_mark_rate_limited_and_skip() {
        let source = ApiKeySource::new(
            vec!["key-a".into(), "key-b".into()],
            Duration::from_secs(60),
        );
        let cred = source.next().await.unwrap(); // key-a, id="0"
        source.mark_rate_limited(cred.id());
        let cred2 = source.next().await.unwrap(); // should skip key-a
        assert_eq!(cred2.id(), "1"); // key-b
    }

    #[tokio::test]
    async fn test_all_rate_limited_returns_none() {
        let source = ApiKeySource::new(vec!["key-a".into()], Duration::from_secs(60));
        source.mark_rate_limited("0");
        assert!(source.next().await.is_none());
    }
}
