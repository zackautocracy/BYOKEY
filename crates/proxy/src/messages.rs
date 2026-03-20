//! Anthropic Messages API passthrough handler.
//!
//! Accepts requests in native Anthropic format and forwards them via
//! `ProviderExecutor::forward_request(Anthropic, body, stream)`.
//!
//! Routes:
//! - `POST /v1/messages` — routes through Claude (or Copilot via routing config).
//! - `POST /copilot/v1/messages` — always routes through Copilot.

use axum::{
    body::Body,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use byokey_provider::{make_executor_for_model, resolve_model_id};
use byokey_types::{ByokError, ProviderId};
use byokey_types::traits::{ApiFormat, ProviderResponse};
use futures_util::TryStreamExt as _;
use serde_json::Value;
use std::collections::HashSet;
use std::sync::Arc;

use crate::{AppState, error::ApiError};

/// Strip the `thinking` field when it should not be forwarded to the Anthropic API:
/// 1. `tool_choice.type == "any"` or `"tool"` — API rejects thinking + forced `tool_choice`.
/// 2. `thinking.type == "auto"` — not a valid Anthropic API value; API returns 400.
fn sanitize_thinking(body: &mut Value) {
    let should_remove = {
        let forced_tool = body
            .get("tool_choice")
            .and_then(|tc| tc.get("type"))
            .and_then(Value::as_str)
            .is_some_and(|t| t == "any" || t == "tool");

        let auto_thinking = body
            .get("thinking")
            .and_then(|th| th.get("type"))
            .and_then(Value::as_str)
            .is_some_and(|t| t == "auto");

        forced_tool || auto_thinking
    };

    if should_remove && let Some(obj) = body.as_object_mut() {
        obj.remove("thinking");
    }
}

/// Convert a `ProviderResponse` into an axum HTTP `Response`.
fn forward_provider_response(resp: ProviderResponse) -> Response {
    match resp {
        ProviderResponse::Complete(json) => axum::Json(json).into_response(),
        ProviderResponse::Stream(byte_stream) => {
            let out_body = Body::from_stream(
                byte_stream.map_err(|e| std::io::Error::other(e.to_string())),
            );
            Response::builder()
                .status(StatusCode::OK)
                .header("content-type", "text/event-stream")
                .header("cache-control", "no-cache")
                .header("x-accel-buffering", "no")
                .body(out_body)
                .expect("valid response")
        }
    }
}

/// Build the set of providers that have valid OAuth tokens.
async fn oauth_providers(state: &AppState) -> HashSet<ProviderId> {
    let mut set = HashSet::new();
    for p in ProviderId::all() {
        if state.auth.is_authenticated(p).await {
            set.insert(p.clone());
        }
    }
    set
}

/// Handles `POST /v1/messages` — Anthropic native format passthrough.
pub async fn anthropic_messages(
    State(state): State<Arc<AppState>>,
    body: axum::extract::Json<Value>,
) -> Result<Response, ApiError> {
    let mut body = body.0;
    sanitize_thinking(&mut body);
    let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);

    let model = body
        .get("model")
        .and_then(Value::as_str)
        .ok_or_else(|| ApiError(ByokError::Translation("missing model field".into())))?;
    let canonical = resolve_model_id(model);
    body["model"] = Value::String(canonical.clone());

    let config = state.config.load();
    let oauth_set = oauth_providers(&state).await;

    let executor = make_executor_for_model(
        &canonical,
        |p| config.providers.get(p).cloned(),
        &oauth_set,
        Some(&ProviderId::Anthropic),
        state.auth.clone(),
        state.http.clone(),
        Some(state.ratelimits.clone()),
    )
    .map_err(ApiError::from)?;

    let resp = executor
        .forward_request(ApiFormat::Anthropic, body, stream)
        .await
        .map_err(ApiError::from)?;

    Ok(forward_provider_response(resp))
}

/// Handles `POST /copilot/v1/messages` — always routes through Copilot.
pub async fn copilot_anthropic_messages(
    State(state): State<Arc<AppState>>,
    body: axum::extract::Json<Value>,
) -> Result<Response, ApiError> {
    let mut body = body.0;
    sanitize_thinking(&mut body);
    let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);

    let model = body
        .get("model")
        .and_then(Value::as_str)
        .ok_or_else(|| ApiError(ByokError::Translation("missing model field".into())))?;
    let canonical = resolve_model_id(model);
    body["model"] = Value::String(canonical.clone());

    let config = state.config.load();
    let oauth_set = oauth_providers(&state).await;

    let executor = make_executor_for_model(
        &canonical,
        |p| config.providers.get(p).cloned(),
        &oauth_set,
        Some(&ProviderId::Copilot),
        state.auth.clone(),
        state.http.clone(),
        Some(state.ratelimits.clone()),
    )
    .map_err(ApiError::from)?;

    let resp = executor
        .forward_request(ApiFormat::Anthropic, body, stream)
        .await
        .map_err(ApiError::from)?;

    Ok(forward_provider_response(resp))
}
