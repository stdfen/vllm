// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Unified parser registration and selection boundary for `vllm-chat`.

use std::sync::LazyLock;

pub use vllm_parser::unified::{
    Gemma4UnifiedParser, HfTemplateError, HfUnifiedParser, HyV3UnifiedParser, KimiK3UnifiedParser,
    ResponseTemplate, UnifiedParser,
};
use vllm_tokenizer::DynTokenizer;

use crate::parser::ParserFactory;
use crate::request::ChatTool;

/// Canonical public names for registered unified parsers.
pub mod names {
    pub const GEMMA4: &str = "gemma4";
    pub const HY_V3: &str = "hy_v3";
    pub const KIMI_K3: &str = "kimi_k3";

    /// Executes the checkpoint's Hugging Face `response_template`. Registered for
    /// name resolution only: it is built from the model's template, never by its
    /// registry constructor.
    pub const HF: &str = "hf";
}

/// Constructor signature for one registered unified parser implementation.
type UnifiedParserCreator =
    fn(&[ChatTool], DynTokenizer) -> vllm_parser::unified::Result<Box<dyn UnifiedParser>>;

/// Registry and model matcher for unified parsers.
pub type UnifiedParserFactory = ParserFactory<UnifiedParserCreator>;

impl UnifiedParserFactory {
    /// Get the global unified parser factory with built-in registrations and
    /// model mappings.
    pub fn global() -> &'static Self {
        static INSTANCE: LazyLock<UnifiedParserFactory> = LazyLock::new(UnifiedParserFactory::new);
        &INSTANCE
    }

    /// Create the default registry with built-in parser names and model
    /// mappings.
    pub fn new() -> Self {
        let mut factory = Self::default();

        factory.register_parser::<Gemma4UnifiedParser>(names::GEMMA4);
        factory.register_parser::<HyV3UnifiedParser>(names::HY_V3);
        factory.register_parser::<KimiK3UnifiedParser>(names::KIMI_K3);
        // Opt-in only: no model patterns map to it.
        factory.register_parser::<HfUnifiedParser>(names::HF);

        factory
            .register_pattern("gemma-4", names::GEMMA4)
            .register_pattern("gemma4", names::GEMMA4)
            .register_pattern("hy3", names::HY_V3)
            .register_pattern("kimi-k3", names::KIMI_K3)
            .register_pattern("kimi_k3", names::KIMI_K3);

        factory
    }

    /// Register one parser type that exposes a static `create()` constructor.
    pub fn register_parser<T>(&mut self, name: &str) -> &mut Self
    where
        T: UnifiedParser + 'static,
    {
        self.register_creator(name, T::create)
    }

    /// Construct a parser from an exact name.
    pub fn create(
        &self,
        name: &str,
        tools: &[ChatTool],
        tokenizer: DynTokenizer,
    ) -> crate::Result<Box<dyn UnifiedParser>> {
        let creator = self.creator(name).ok_or_else(|| crate::Error::ParserUnavailableByName {
            kind: "unified",
            name: name.to_string(),
            available_names: self.list(),
        })?;

        creator(tools, tokenizer).map_err(|error| crate::Error::ParserInitialization {
            kind: "unified",
            name: name.to_string(),
            error: error.into(),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vllm_tokenizer::test_utils::TestTokenizer;

    use super::{UnifiedParserFactory, names};

    fn tokenizer() -> TestTokenizer {
        TestTokenizer::new()
            .with_regular_token("<|channel>", 256)
            .with_regular_token("<channel|>", 257)
    }

    #[test]
    fn factory_registers_gemma4() {
        let factory = UnifiedParserFactory::new();

        assert!(factory.contains(names::GEMMA4));
        assert_eq!(
            factory.resolve_name_for_model("google/gemma-4-27b-it"),
            Some(names::GEMMA4)
        );
        factory.create(names::GEMMA4, &[], Arc::new(tokenizer())).unwrap();
    }

    #[test]
    fn factory_rejects_removed_inkling() {
        let factory = UnifiedParserFactory::new();
        assert!(!factory.contains("inkling"));
        assert_eq!(
            factory.resolve_name_for_model("thinkingmachines/Inkling"),
            None
        );
        assert!(factory.create("inkling", &[], Arc::new(TestTokenizer::new())).is_err());
    }

    #[test]
    fn factory_registers_kimi_k3() {
        let factory = UnifiedParserFactory::new();
        let tokenizer = TestTokenizer::new()
            .with_regular_token("<|open|>", 1001)
            .with_regular_token("<|close|>", 1002)
            .with_regular_token("<|sep|>", 1003)
            .with_regular_token("<|end_of_msg|>", 1004);

        assert!(factory.contains(names::KIMI_K3));
        assert_eq!(
            factory.resolve_name_for_model("moonshotai/Kimi-K3"),
            Some(names::KIMI_K3)
        );
        factory.create(names::KIMI_K3, &[], Arc::new(tokenizer)).unwrap();
    }

    #[test]
    fn factory_rejects_removed_hy_v4() {
        let factory = UnifiedParserFactory::new();
        assert!(!factory.contains("hy_v4"));
        assert_eq!(factory.resolve_name_for_model("tencent/Hy4-preview"), None);
        assert!(factory.create("hy_v4", &[], Arc::new(TestTokenizer::new())).is_err());
    }
}
