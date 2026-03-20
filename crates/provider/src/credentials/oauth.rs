//! OAuth credential source — rotates across enrolled OAuth accounts.

use super::{Credential, CredentialSource};
use crate::credentials::quota::{CachedQuota, QuotaTracker, quota_score};
use async_trait::async_trait;
use byokey_auth::AuthManager;
use byokey_types::ProviderId;
use std::cmp::Ordering as CmpOrdering;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Quota information for a single account.
pub struct QuotaSnapshot {
    /// Percentage of quota remaining (0.0–100.0).
    pub percent_remaining: f64,
    /// Whether this account has unlimited quota.
    pub unlimited: bool,
}

/// Fetches quota/capacity for a provider account.
#[async_trait]
pub trait QuotaFetcher: Send + Sync {
    /// Returns remaining capacity snapshot, or `None` on failure.
    async fn fetch_quota(
        &self,
        auth: &AuthManager,
        provider: &ProviderId,
        account_id: &str,
    ) -> Option<QuotaSnapshot>;
}

/// How `OAuthSource` selects the next account.
pub enum SelectionStrategy {
    /// Use first available (non-cooled) account. Scans from start each time.
    Failover,
    /// True per-request rotation via atomic index, like `ApiKeySource`.
    RoundRobin,
    /// Pick the account with the highest remaining quota. Sticky selection
    /// with periodic rebalancing.
    QuotaAware {
        /// Fetcher that retrieves quota snapshots for accounts.
        quota_fetcher: Arc<dyn QuotaFetcher>,
        /// How often to re-compare accounts.
        rebalance_interval: Duration,
    },
}

/// Rotates OAuth credentials across enrolled accounts for a provider.
pub struct OAuthSource {
    auth: Arc<AuthManager>,
    provider: ProviderId,
    cooldown_duration: Duration,
    /// Per-account cooldown state.
    cooldowns: Mutex<HashMap<String, Instant>>,
    /// Account selection strategy.
    strategy: SelectionStrategy,
    /// Quota tracking state (used only by `QuotaAware`).
    tracker: Mutex<QuotaTracker>,
    /// Atomic index for `RoundRobin` strategy.
    rr_index: AtomicUsize,
}

impl OAuthSource {
    /// Creates a new OAuth credential source.
    pub fn new(
        auth: Arc<AuthManager>,
        provider: ProviderId,
        cooldown_duration: Duration,
        strategy: SelectionStrategy,
    ) -> Self {
        Self {
            auth,
            provider,
            cooldown_duration,
            cooldowns: Mutex::new(HashMap::new()),
            strategy,
            tracker: Mutex::new(QuotaTracker::new()),
            rr_index: AtomicUsize::new(0),
        }
    }
}

#[async_trait]
impl CredentialSource for OAuthSource {
    async fn next(&self) -> Option<Credential> {
        match &self.strategy {
            SelectionStrategy::Failover => self.next_failover().await,
            SelectionStrategy::RoundRobin => self.next_round_robin().await,
            SelectionStrategy::QuotaAware {
                quota_fetcher,
                rebalance_interval,
            } => {
                self.next_quota_aware(quota_fetcher, *rebalance_interval)
                    .await
            }
        }
    }

    fn mark_rate_limited(&self, id: &str) {
        let mut cooldowns = self.cooldowns.lock().expect("cooldown lock");
        cooldowns.insert(id.to_string(), Instant::now() + self.cooldown_duration);
    }

    async fn count(&self) -> usize {
        self.auth
            .get_all_tokens(&self.provider)
            .await
            .map(|v| v.len())
            .unwrap_or(0)
    }
}

impl OAuthSource {
    /// `Failover`: scan from start, return first non-cooled account.
    async fn next_failover(&self) -> Option<Credential> {
        let accounts = self.auth.get_all_tokens(&self.provider).await.ok()?;
        let now = Instant::now();

        for (account_id, _token) in &accounts {
            {
                let cooldowns = self.cooldowns.lock().expect("cooldown lock");
                if cooldowns
                    .get(account_id)
                    .is_some_and(|&until| now < until)
                {
                    continue;
                }
            }

            match self.auth.get_token_for(&self.provider, account_id).await {
                Ok(_) => {
                    return Some(Credential::OAuth {
                        account_id: account_id.clone(),
                    });
                }
                Err(e) => {
                    tracing::warn!(
                        provider = %self.provider,
                        account = %account_id,
                        error = %e,
                        "skipping account with invalid token"
                    );
                }
            }
        }
        None
    }

    /// `RoundRobin`: advance atomic index, skip cooled-down, wrap around.
    async fn next_round_robin(&self) -> Option<Credential> {
        let accounts = self.auth.get_all_tokens(&self.provider).await.ok()?;
        if accounts.is_empty() {
            return None;
        }
        let len = accounts.len();
        let start = self.rr_index.fetch_add(1, Ordering::Relaxed) % len;
        let now = Instant::now();

        for i in 0..len {
            let idx = (start + i) % len;
            let (account_id, _) = &accounts[idx];

            {
                let cooldowns = self.cooldowns.lock().expect("cooldown lock");
                if cooldowns
                    .get(account_id)
                    .is_some_and(|&until| now < until)
                {
                    continue;
                }
            }

            match self.auth.get_token_for(&self.provider, account_id).await {
                Ok(_) => {
                    return Some(Credential::OAuth {
                        account_id: account_id.clone(),
                    });
                }
                Err(e) => {
                    tracing::warn!(
                        provider = %self.provider,
                        account = %account_id,
                        error = %e,
                        "skipping account with invalid token"
                    );
                }
            }
        }
        None
    }

    /// `QuotaAware`: sticky selection with periodic rebalancing, pick highest quota.
    async fn next_quota_aware(
        &self,
        quota_fetcher: &Arc<dyn QuotaFetcher>,
        rebalance_interval: Duration,
    ) -> Option<Credential> {
        let accounts = self.auth.get_all_tokens(&self.provider).await.ok()?;
        if accounts.is_empty() {
            return None;
        }

        let now = Instant::now();

        // Check sticky: current account still valid, not cooled, rebalance interval not elapsed.
        {
            let tracker = self.tracker.lock().expect("tracker lock");
            if let Some(ref current) = tracker.current {
                let cooldowns = self.cooldowns.lock().expect("cooldown lock");
                let still_enrolled = accounts.iter().any(|(id, _)| id == current);
                let not_cooled = cooldowns.get(current).is_none_or(|&until| now >= until);
                let within_interval = tracker
                    .last_rebalance
                    .is_some_and(|t| t.elapsed() < rebalance_interval);
                if still_enrolled && not_cooled && within_interval {
                    return Some(Credential::OAuth {
                        account_id: current.clone(),
                    });
                }
            }
        }

        // Fetch quotas for non-cooled accounts (skip if cached and fresh).
        let quota_cache_ttl = Duration::from_secs(300);
        for (account_id, _) in &accounts {
            {
                let cooldowns = self.cooldowns.lock().expect("cooldown lock");
                if cooldowns
                    .get(account_id)
                    .is_some_and(|&until| now < until)
                {
                    continue;
                }
            }

            {
                let tracker = self.tracker.lock().expect("tracker lock");
                if tracker
                    .quotas
                    .get(account_id)
                    .is_some_and(|q| q.fetched_at.elapsed() < quota_cache_ttl)
                {
                    continue;
                }
            }

            if let Some(snapshot) = quota_fetcher
                .fetch_quota(&self.auth, &self.provider, account_id)
                .await
            {
                let mut tracker = self.tracker.lock().expect("tracker lock");
                tracker.quotas.insert(
                    account_id.clone(),
                    CachedQuota {
                        percent_remaining: snapshot.percent_remaining,
                        unlimited: snapshot.unlimited,
                        fetched_at: Instant::now(),
                    },
                );
            }
        }

        // Pick account with highest quota score (among non-cooled).
        let best = {
            let tracker = self.tracker.lock().expect("tracker lock");
            let cooldowns = self.cooldowns.lock().expect("cooldown lock");
            accounts
                .iter()
                .filter(|(id, _)| cooldowns.get(id).is_none_or(|&until| now >= until))
                .max_by(|(a, _), (b, _)| {
                    let qa = tracker.quotas.get(a);
                    let qb = tracker.quotas.get(b);
                    quota_score(qa)
                        .partial_cmp(&quota_score(qb))
                        .unwrap_or(CmpOrdering::Equal)
                })
                .map(|(id, _)| id.clone())
        };

        let best = best?;

        // Validate the token is usable.
        match self.auth.get_token_for(&self.provider, &best).await {
            Ok(_) => {
                let mut tracker = self.tracker.lock().expect("tracker lock");
                tracker.current = Some(best.clone());
                tracker.last_rebalance = Some(Instant::now());
                Some(Credential::OAuth { account_id: best })
            }
            Err(e) => {
                tracing::warn!(
                    provider = %self.provider,
                    account = %best,
                    error = %e,
                    "best quota account has invalid token, cooling down"
                );
                self.mark_rate_limited(&best);
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use byokey_store::InMemoryTokenStore;
    use byokey_types::OAuthToken;

    fn make_auth() -> Arc<AuthManager> {
        Arc::new(AuthManager::new(
            Arc::new(InMemoryTokenStore::new()),
            rquest::Client::new(),
        ))
    }

    #[tokio::test]
    async fn test_oauth_source_no_accounts() {
        let auth = make_auth();
        let source = OAuthSource::new(auth, ProviderId::OpenAI, Duration::from_secs(300), SelectionStrategy::Failover);
        assert!(source.next().await.is_none());
        assert_eq!(source.count().await, 0);
    }

    #[tokio::test]
    async fn test_oauth_source_single_account() {
        let auth = make_auth();
        auth.save_token_for(
            &ProviderId::OpenAI,
            "work",
            None,
            OAuthToken::new("tok-work").with_expiry(3600),
        )
        .await
        .unwrap();

        let source =
            OAuthSource::new(Arc::clone(&auth), ProviderId::OpenAI, Duration::from_secs(300), SelectionStrategy::Failover);
        let cred = source.next().await.unwrap();
        assert!(!cred.is_api_key());
        assert_eq!(cred.id(), "work");
        assert_eq!(source.count().await, 1);
    }

    #[tokio::test]
    async fn test_oauth_source_rotation_on_cooldown() {
        let auth = make_auth();
        auth.save_token_for(
            &ProviderId::OpenAI,
            "a",
            None,
            OAuthToken::new("tok-a").with_expiry(3600),
        )
        .await
        .unwrap();
        auth.save_token_for(
            &ProviderId::OpenAI,
            "b",
            None,
            OAuthToken::new("tok-b").with_expiry(3600),
        )
        .await
        .unwrap();

        let source =
            OAuthSource::new(Arc::clone(&auth), ProviderId::OpenAI, Duration::from_secs(300), SelectionStrategy::Failover);
        let cred1 = source.next().await.unwrap();
        source.mark_rate_limited(cred1.id());
        let cred2 = source.next().await.unwrap();
        // Should get a different account
        assert_ne!(cred1.id(), cred2.id());
    }

    #[tokio::test]
    async fn test_oauth_source_all_cooled_returns_none() {
        let auth = make_auth();
        auth.save_token_for(
            &ProviderId::OpenAI,
            "only",
            None,
            OAuthToken::new("tok").with_expiry(3600),
        )
        .await
        .unwrap();

        let source =
            OAuthSource::new(Arc::clone(&auth), ProviderId::OpenAI, Duration::from_secs(300), SelectionStrategy::Failover);
        source.mark_rate_limited("only");
        assert!(source.next().await.is_none());
    }

    #[tokio::test]
    async fn test_oauth_source_skips_invalid_tokens() {
        use std::time::{SystemTime, UNIX_EPOCH};
        let auth = make_auth();

        // Account "bad" has an expired token with no refresh token (invalid state)
        let expired_no_refresh = OAuthToken {
            access_token: "old".into(),
            refresh_token: None,
            expires_at: Some(
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs()
                    .saturating_sub(100),
            ),
            token_type: None,
        };
        auth.save_token_for(&ProviderId::OpenAI, "bad", None, expired_no_refresh)
            .await
            .unwrap();

        // Account "good" has a valid token
        auth.save_token_for(
            &ProviderId::OpenAI,
            "good",
            None,
            OAuthToken::new("valid").with_expiry(3600),
        )
        .await
        .unwrap();

        let source =
            OAuthSource::new(Arc::clone(&auth), ProviderId::OpenAI, Duration::from_secs(300), SelectionStrategy::Failover);
        let cred = source.next().await.unwrap();
        assert_eq!(cred.id(), "good");
    }

    /// Mock quota fetcher that returns fixed values per account.
    struct MockQuotaFetcher {
        quotas: std::sync::Mutex<HashMap<String, QuotaSnapshot>>,
    }

    impl MockQuotaFetcher {
        fn new(entries: Vec<(&str, f64, bool)>) -> Self {
            let mut map = HashMap::new();
            for (id, pct, unlimited) in entries {
                map.insert(
                    id.to_string(),
                    QuotaSnapshot {
                        percent_remaining: pct,
                        unlimited,
                    },
                );
            }
            Self {
                quotas: std::sync::Mutex::new(map),
            }
        }
    }

    #[async_trait]
    impl QuotaFetcher for MockQuotaFetcher {
        async fn fetch_quota(
            &self,
            _auth: &AuthManager,
            _provider: &ProviderId,
            account_id: &str,
        ) -> Option<QuotaSnapshot> {
            let map = self.quotas.lock().expect("lock");
            map.get(account_id).map(|q| QuotaSnapshot {
                percent_remaining: q.percent_remaining,
                unlimited: q.unlimited,
            })
        }
    }

    #[tokio::test]
    async fn test_quota_aware_picks_highest() {
        let auth = make_auth();
        auth.save_token_for(
            &ProviderId::OpenAI,
            "low",
            None,
            OAuthToken::new("tok-low").with_expiry(3600),
        )
        .await
        .unwrap();
        auth.save_token_for(
            &ProviderId::OpenAI,
            "high",
            None,
            OAuthToken::new("tok-high").with_expiry(3600),
        )
        .await
        .unwrap();

        let fetcher = Arc::new(MockQuotaFetcher::new(vec![
            ("low", 20.0, false),
            ("high", 80.0, false),
        ]));
        let source = OAuthSource::new(
            Arc::clone(&auth),
            ProviderId::OpenAI,
            Duration::from_secs(300),
            SelectionStrategy::QuotaAware {
                quota_fetcher: fetcher,
                rebalance_interval: Duration::from_secs(300),
            },
        );

        let cred = source.next().await.unwrap();
        assert_eq!(cred.id(), "high");
    }

    #[tokio::test]
    async fn test_quota_aware_sticky_within_interval() {
        let auth = make_auth();
        auth.save_token_for(
            &ProviderId::OpenAI,
            "a",
            None,
            OAuthToken::new("tok-a").with_expiry(3600),
        )
        .await
        .unwrap();
        auth.save_token_for(
            &ProviderId::OpenAI,
            "b",
            None,
            OAuthToken::new("tok-b").with_expiry(3600),
        )
        .await
        .unwrap();

        let fetcher = Arc::new(MockQuotaFetcher::new(vec![
            ("a", 80.0, false),
            ("b", 20.0, false),
        ]));
        let source = OAuthSource::new(
            Arc::clone(&auth),
            ProviderId::OpenAI,
            Duration::from_secs(300),
            SelectionStrategy::QuotaAware {
                quota_fetcher: fetcher,
                rebalance_interval: Duration::from_secs(300),
            },
        );

        let cred1 = source.next().await.unwrap();
        let cred2 = source.next().await.unwrap();
        // Should be sticky — same account both times
        assert_eq!(cred1.id(), cred2.id());
    }

    #[tokio::test]
    async fn test_quota_aware_skips_cooled_sticky() {
        let auth = make_auth();
        auth.save_token_for(
            &ProviderId::OpenAI,
            "a",
            None,
            OAuthToken::new("tok-a").with_expiry(3600),
        )
        .await
        .unwrap();
        auth.save_token_for(
            &ProviderId::OpenAI,
            "b",
            None,
            OAuthToken::new("tok-b").with_expiry(3600),
        )
        .await
        .unwrap();

        let fetcher = Arc::new(MockQuotaFetcher::new(vec![
            ("a", 80.0, false),
            ("b", 60.0, false),
        ]));
        let source = OAuthSource::new(
            Arc::clone(&auth),
            ProviderId::OpenAI,
            Duration::from_secs(300),
            SelectionStrategy::QuotaAware {
                quota_fetcher: fetcher,
                rebalance_interval: Duration::from_secs(300),
            },
        );

        let cred1 = source.next().await.unwrap();
        assert_eq!(cred1.id(), "a"); // highest quota
        source.mark_rate_limited("a");

        // Should skip "a" (cooled) and return "b"
        let cred2 = source.next().await.unwrap();
        assert_eq!(cred2.id(), "b");
    }

    #[tokio::test]
    async fn test_oauth_source_round_robin_rotates() {
        let auth = make_auth();
        for id in ["acct-a", "acct-b", "acct-c"] {
            auth.save_token_for(
                &ProviderId::Copilot,
                id,
                None,
                OAuthToken::new("tok").with_expiry(3600),
            )
            .await
            .unwrap();
        }
        let source = OAuthSource::new(
            auth,
            ProviderId::Copilot,
            Duration::from_secs(300),
            SelectionStrategy::RoundRobin,
        );

        let c1 = source.next().await.unwrap();
        let c2 = source.next().await.unwrap();
        let c3 = source.next().await.unwrap();
        let c4 = source.next().await.unwrap();

        // Should cycle: a, b, c, a (or some deterministic rotation)
        assert_ne!(c1.id(), c2.id());
        assert_ne!(c2.id(), c3.id());
        assert_eq!(c1.id(), c4.id()); // wraps around
    }
}
