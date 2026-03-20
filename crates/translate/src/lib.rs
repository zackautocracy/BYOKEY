//! Request and response translators between LLM API formats.
//!
//! This crate provides bidirectional translation between OpenAI, Anthropic, Gemini,
//! and OpenAI Native (Responses API) message formats. All translators are pure functions
//! with no I/O.

pub mod cache_control;
pub mod anthropic_to_openai;
pub mod openai_native_to_openai;
pub mod gemini_native_to_openai;
pub mod gemini_to_openai;
pub mod merge_messages;
pub mod openai_to_anthropic;
pub mod openai_to_openai_native;
pub mod openai_to_gemini;
pub mod openai_to_gemini_native;
pub mod thinking;

pub use cache_control::inject_cache_control;
pub use anthropic_to_openai::AnthropicToOpenAI;
pub use openai_native_to_openai::OpenAINativeToOpenAI;
pub use gemini_native_to_openai::GeminiNativeRequest;
pub use gemini_to_openai::GeminiToOpenAI;
pub use merge_messages::merge_adjacent_messages;
pub use openai_to_anthropic::OpenAIToAnthropic;
pub use openai_to_openai_native::OpenAIToOpenAINative;
pub use openai_to_gemini::OpenAIToGemini;
pub use openai_to_gemini_native::{OpenAIResponseToGemini, OpenAISseChunk};
pub use thinking::ThinkingExtractor;
pub use thinking::{
    ModelSuffix, ThinkingConfig, ThinkingLevel, apply_thinking, parse_model_suffix,
};
