//! Provider identifiers and protocol format definitions.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Identifies a supported upstream AI provider.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum ProviderId {
    Anthropic,
    #[serde(rename = "openai")]
    OpenAI,
    Gemini,
    Kiro,
    Copilot,
    Antigravity,
    Qwen,
    Kimi,
    #[serde(rename = "iflow")]
    IFlow,
}

impl fmt::Display for ProviderId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Anthropic => write!(f, "anthropic"),
            Self::OpenAI => write!(f, "openai"),
            Self::Gemini => write!(f, "gemini"),
            Self::Kiro => write!(f, "kiro"),
            Self::Copilot => write!(f, "copilot"),
            Self::Antigravity => write!(f, "antigravity"),
            Self::Qwen => write!(f, "qwen"),
            Self::Kimi => write!(f, "kimi"),
            Self::IFlow => write!(f, "iflow"),
        }
    }
}

impl std::str::FromStr for ProviderId {
    type Err = crate::ByokError;

    /// Parse a provider name or well-known alias into a [`ProviderId`].
    ///
    /// # Errors
    ///
    /// Returns [`ByokError::UnsupportedModel`] if the string does not match
    /// any known provider name or alias.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "anthropic" => Ok(Self::Anthropic),
            "openai" => Ok(Self::OpenAI),
            "gemini" | "google" => Ok(Self::Gemini),
            "kiro" => Ok(Self::Kiro),
            "copilot" | "github" => Ok(Self::Copilot),
            "antigravity" => Ok(Self::Antigravity),
            "qwen" | "alibaba" => Ok(Self::Qwen),
            "kimi" | "moonshot" => Ok(Self::Kimi),
            "iflow" | "zai" | "glm" => Ok(Self::IFlow),
            other => Err(crate::ByokError::UnsupportedModel(other.to_string())),
        }
    }
}

impl ProviderId {
    /// Returns a human-readable display name for the provider.
    #[must_use]
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Anthropic => "Claude (Anthropic)",
            Self::OpenAI => "GPT (OpenAI)",
            Self::Gemini => "Gemini (Google)",
            Self::Kiro => "Kiro (AWS)",
            Self::Copilot => "GitHub Copilot",
            Self::Antigravity => "Antigravity",
            Self::Qwen => "Qwen (Alibaba)",
            Self::Kimi => "Kimi (Moonshot)",
            Self::IFlow => "iFlow (Z.ai)",
        }
    }

    /// Returns all known provider variants.
    #[must_use]
    pub fn all() -> &'static [Self] {
        &[
            Self::Anthropic,
            Self::OpenAI,
            Self::Gemini,
            Self::Kiro,
            Self::Copilot,
            Self::Antigravity,
            Self::Qwen,
            Self::Kimi,
            Self::IFlow,
        ]
    }
}

/// The wire protocol format used by a provider's native API.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProtocolFormat {
    OpenAI,
    Anthropic,
    Gemini,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn test_display() {
        assert_eq!(ProviderId::Anthropic.to_string(), "anthropic");
        assert_eq!(ProviderId::OpenAI.to_string(), "openai");
        assert_eq!(ProviderId::Gemini.to_string(), "gemini");
        assert_eq!(ProviderId::Kiro.to_string(), "kiro");
        assert_eq!(ProviderId::Copilot.to_string(), "copilot");
        assert_eq!(ProviderId::Antigravity.to_string(), "antigravity");
        assert_eq!(ProviderId::Qwen.to_string(), "qwen");
        assert_eq!(ProviderId::Kimi.to_string(), "kimi");
        assert_eq!(ProviderId::IFlow.to_string(), "iflow");
    }

    #[test]
    fn test_from_str_canonical() {
        assert_eq!(
            ProviderId::from_str("anthropic").unwrap(),
            ProviderId::Anthropic
        );
        assert_eq!(ProviderId::from_str("openai").unwrap(), ProviderId::OpenAI);
        assert_eq!(ProviderId::from_str("gemini").unwrap(), ProviderId::Gemini);
        assert_eq!(ProviderId::from_str("kiro").unwrap(), ProviderId::Kiro);
        assert_eq!(
            ProviderId::from_str("copilot").unwrap(),
            ProviderId::Copilot
        );
        assert_eq!(
            ProviderId::from_str("antigravity").unwrap(),
            ProviderId::Antigravity
        );
        assert_eq!(ProviderId::from_str("qwen").unwrap(), ProviderId::Qwen);
        assert_eq!(ProviderId::from_str("kimi").unwrap(), ProviderId::Kimi);
        assert_eq!(ProviderId::from_str("iflow").unwrap(), ProviderId::IFlow);
    }

    #[test]
    fn test_from_str_aliases() {
        assert_eq!(ProviderId::from_str("google").unwrap(), ProviderId::Gemini);
        assert_eq!(ProviderId::from_str("github").unwrap(), ProviderId::Copilot);
        assert_eq!(ProviderId::from_str("alibaba").unwrap(), ProviderId::Qwen);
        assert_eq!(ProviderId::from_str("moonshot").unwrap(), ProviderId::Kimi);
        assert_eq!(ProviderId::from_str("zai").unwrap(), ProviderId::IFlow);
        assert_eq!(ProviderId::from_str("glm").unwrap(), ProviderId::IFlow);
    }

    #[test]
    fn test_from_str_unknown() {
        let err = ProviderId::from_str("xyz").unwrap_err();
        assert!(err.to_string().contains("xyz"));
    }

    #[test]
    fn test_serde_roundtrip() {
        for p in [
            ProviderId::Anthropic,
            ProviderId::OpenAI,
            ProviderId::Gemini,
            ProviderId::Kiro,
            ProviderId::Copilot,
            ProviderId::Antigravity,
            ProviderId::Qwen,
            ProviderId::Kimi,
            ProviderId::IFlow,
        ] {
            let json = serde_json::to_string(&p).unwrap();
            let back: ProviderId = serde_json::from_str(&json).unwrap();
            assert_eq!(back, p);
        }
    }

    #[test]
    fn test_hash_in_map() {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(ProviderId::Anthropic, "val");
        assert_eq!(map[&ProviderId::Anthropic], "val");
    }

    #[test]
    fn test_protocol_format_serde() {
        let j = serde_json::to_string(&ProtocolFormat::OpenAI).unwrap();
        let back: ProtocolFormat = serde_json::from_str(&j).unwrap();
        assert_eq!(back, ProtocolFormat::OpenAI);
    }
}
