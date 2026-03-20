//! Chat completions handler — proxies OpenAI-compatible requests to providers.

use axum::{
    Json,
    body::Body,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use byokey_provider::{make_executor_for_model, parse_qualified_model, resolve_model_id};
use byokey_translate::{apply_thinking, parse_model_suffix};
use byokey_types::{ChatRequest, ProviderId, traits::ProviderResponse};
use futures_util::TryStreamExt as _;
use std::collections::HashSet;
use std::sync::Arc;

use crate::{AppState, error::ApiError};

/// Extract input/output token counts from an OpenAI-compatible usage response.
fn extract_usage_tokens(json: &serde_json::Value) -> (u64, u64) {
    let usage = json.get("usage");
    let input = usage
        .and_then(|u| u.get("prompt_tokens"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let output = usage
        .and_then(|u| u.get("completion_tokens"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    (input, output)
}

/// Handles `POST /copilot/v1/chat/completions` — always routes through Copilot.
pub async fn copilot_chat_completions(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ChatRequest>,
) -> Result<Response, ApiError> {
    chat_completions_inner(state, request, true).await
}

/// Handles `POST /v1/chat/completions` requests.
///
/// Resolves the model to a provider, forwards the request, and returns
/// either a complete JSON response or an SSE stream.
///
/// # Errors
///
/// Returns [`ApiError`] if the model is unsupported or the upstream call fails.
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ChatRequest>,
) -> Result<Response, ApiError> {
    chat_completions_inner(state, request, false).await
}

async fn chat_completions_inner(
    state: Arc<AppState>,
    mut request: ChatRequest,
    force_copilot: bool,
) -> Result<Response, ApiError> {
    let config = state.config.load();

    // Pre-compute which providers have OAuth tokens (async → sync bridge).
    let mut oauth_providers = HashSet::new();
    for p in ProviderId::all() {
        if state.auth.is_authenticated(p).await {
            oauth_providers.insert(p.clone());
        }
    }

    // Resolve model alias before anything else.
    let resolved_model = config.resolve_alias(&request.model);

    // Strip provider qualifier (e.g. "codex/gpt-5.4" → "gpt-5.4").
    let (provider_hint, bare_model) = parse_qualified_model(&resolved_model);

    // Parse thinking suffix from (possibly alias-resolved) model name.
    let suffix = parse_model_suffix(bare_model);

    // Normalize to canonical registry ID (handles dot variants, ag- prefix).
    let canonical_model = resolve_model_id(&suffix.model);

    let config_fn = |p: &ProviderId| {
        let mut pc = config.providers.get(p).cloned().unwrap_or_default();
        if force_copilot && *p != ProviderId::Copilot {
            // Route through Copilot by overriding the routing entries.
            let copilot_config = config
                .providers
                .get(&ProviderId::Copilot)
                .cloned()
                .unwrap_or_default();
            pc.routing = byokey_provider::routing::auto_generate_routing(
                &ProviderId::Copilot,
                &copilot_config,
                &oauth_providers,
            );
        }
        Some(pc)
    };

    let executor = make_executor_for_model(
        &canonical_model,
        config_fn,
        &oauth_providers,
        provider_hint.as_ref(),
        state.auth.clone(),
        state.http.clone(),
        Some(state.ratelimits.clone()),
    )
    .map_err(ApiError::from)?;

    let provider = byokey_provider::resolve_provider(&canonical_model)
        .map_or_else(|| "unknown".to_string(), |p| p.to_string());
    tracing::info!(
        model = %canonical_model,
        provider = %provider,
        stream = request.stream,
        "chat completion request"
    );

    // Replace model name with canonical version
    request.model = canonical_model.clone();

    // Apply thinking config if suffix was parsed
    if let Some(ref thinking) = suffix.thinking {
        let provider =
            byokey_provider::resolve_provider(&canonical_model).unwrap_or(ProviderId::Anthropic);
        let mut body = request.into_body();
        body = apply_thinking(body, &provider, thinking);
        // Re-parse the modified body back into ChatRequest
        request = serde_json::from_value(body)
            .map_err(|e| ApiError::from(byokey_types::ByokError::Translation(e.to_string())))?;
    }

    // Apply payload rules (default/override/filter) based on model name.
    if !config.payload.default.is_empty()
        || !config.payload.r#override.is_empty()
        || !config.payload.filter.is_empty()
    {
        let mut body = request.into_body();
        body = config.apply_payload_rules(body, &canonical_model);
        request = serde_json::from_value(body)
            .map_err(|e| ApiError::from(byokey_types::ByokError::Translation(e.to_string())))?;
    }

    let model_name = canonical_model;
    match executor.chat_completion(request).await {
        Ok(ProviderResponse::Complete(json)) => {
            // Extract token usage from the response if available.
            let (input_tok, output_tok) = extract_usage_tokens(&json);
            state
                .usage
                .record_success(&model_name, input_tok, output_tok);
            tracing::debug!(model = %model_name, "chat completion complete");
            Ok(Json(json).into_response())
        }
        Ok(ProviderResponse::Stream(byte_stream)) => {
            state.usage.record_success(&model_name, 0, 0);
            tracing::debug!(model = %model_name, "streaming chat completion");
            // Convert ByokError → std::io::Error for Body::from_stream
            let mapped = byte_stream.map_err(|e| std::io::Error::other(e.to_string()));
            let body = Body::from_stream(mapped);
            Ok(Response::builder()
                .status(StatusCode::OK)
                .header("content-type", "text/event-stream")
                .header("cache-control", "no-cache")
                .header("x-accel-buffering", "no")
                .body(body)
                .expect("valid response"))
        }
        Err(e) => {
            state.usage.record_failure(&model_name);
            Err(ApiError::from(e))
        }
    }
}
