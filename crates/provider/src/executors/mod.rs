//! Provider executor implementations.
//!
//! Each provider module implements [`ProviderExecutor`] for a specific AI backend.

pub mod antigravity;
pub mod anthropic;
pub mod openai;
pub mod copilot;
pub mod gemini;
pub mod iflow;
pub mod kimi;
pub mod kiro;
pub mod qwen;

pub use antigravity::AntigravityExecutor;
pub use anthropic::AnthropicExecutor;
pub use openai::OpenAiExecutor;
pub use copilot::CopilotExecutor;
pub use gemini::GeminiExecutor;
pub use iflow::IFlowExecutor;
pub use kimi::KimiExecutor;
pub use kiro::KiroExecutor;
pub use qwen::QwenExecutor;

use byokey_auth::AuthManager;
use byokey_types::{ProviderId, RateLimitStore, traits::ProviderExecutor};
use rquest::Client;
use std::sync::Arc;

/// Create a boxed executor for the given provider.
///
/// Returns `None` if the provider is not supported.
pub fn make_executor(
    provider: &ProviderId,
    api_key: Option<String>,
    account_id: Option<&str>,
    auth: Arc<AuthManager>,
    http: Client,
    ratelimit: Option<Arc<RateLimitStore>>,
) -> Option<Box<dyn ProviderExecutor>> {
    let aid = account_id.map(String::from);
    match provider {
        ProviderId::Anthropic => Some(Box::new(AnthropicExecutor::new(
            http, api_key, aid, auth, ratelimit,
        ))),
        ProviderId::OpenAI => Some(Box::new(OpenAiExecutor::new(
            http, api_key, aid, auth, ratelimit,
        ))),
        ProviderId::Gemini => Some(Box::new(GeminiExecutor::new(
            http, api_key, aid, auth, ratelimit,
        ))),
        ProviderId::Kiro => Some(Box::new(KiroExecutor::new(
            http, api_key, aid, auth, ratelimit,
        ))),
        ProviderId::Copilot => Some(Box::new(CopilotExecutor::new(
            http, api_key, aid, auth, ratelimit,
        ))),
        ProviderId::Antigravity => Some(Box::new(AntigravityExecutor::new(
            http, api_key, aid, auth, ratelimit,
        ))),
        ProviderId::Qwen => Some(Box::new(QwenExecutor::new(
            http, api_key, aid, auth, ratelimit,
        ))),
        ProviderId::IFlow => Some(Box::new(IFlowExecutor::new(
            http, api_key, aid, auth, ratelimit,
        ))),
        ProviderId::Kimi => Some(Box::new(KimiExecutor::new(
            http, api_key, aid, auth, ratelimit,
        ))),
    }
}
