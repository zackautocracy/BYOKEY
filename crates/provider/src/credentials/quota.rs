//! Quota tracking state for the `QuotaAware` selection strategy.

use std::collections::HashMap;
use std::time::Instant;

/// Cached quota snapshot for a single account.
pub struct CachedQuota {
    /// Percentage of quota remaining (0.0–100.0).
    pub percent_remaining: f64,
    /// Whether this account has unlimited quota.
    pub unlimited: bool,
    /// When this snapshot was fetched.
    pub fetched_at: Instant,
}

/// Tracks the currently selected account and per-account quota snapshots.
pub struct QuotaTracker {
    /// Currently sticky account id.
    pub current: Option<String>,
    /// When the last rebalance comparison happened.
    pub last_rebalance: Option<Instant>,
    /// Per-account cached quota data.
    pub quotas: HashMap<String, CachedQuota>,
}

impl QuotaTracker {
    /// Creates an empty tracker with no selected account.
    #[must_use]
    pub fn new() -> Self {
        Self {
            current: None,
            last_rebalance: None,
            quotas: HashMap::new(),
        }
    }
}

impl Default for QuotaTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Score a cached quota for account comparison.
///
/// `unlimited` → 100, known quota → `percent_remaining`, unknown → 50 (neutral).
#[must_use]
pub fn quota_score(q: Option<&CachedQuota>) -> f64 {
    match q {
        Some(q) if q.unlimited => 100.0,
        Some(q) => q.percent_remaining,
        None => 50.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quota_score_unlimited() {
        let q = CachedQuota {
            percent_remaining: 10.0,
            unlimited: true,
            fetched_at: Instant::now(),
        };
        assert!((quota_score(Some(&q)) - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_quota_score_known() {
        let q = CachedQuota {
            percent_remaining: 42.5,
            unlimited: false,
            fetched_at: Instant::now(),
        };
        assert!((quota_score(Some(&q)) - 42.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_quota_score_unknown() {
        assert!((quota_score(None) - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_tracker_default() {
        let t = QuotaTracker::new();
        assert!(t.current.is_none());
        assert!(t.last_rebalance.is_none());
        assert!(t.quotas.is_empty());
    }
}
