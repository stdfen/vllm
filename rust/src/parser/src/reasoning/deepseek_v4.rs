// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use vllm_tokenizer::{DecodedText, DynTokenizer};
use xgrammar_structural_tag::format::Format;

use crate::output_grammar::{self, OutputGrammarContext};

use super::{
    DelimitedReasoningParser, DelimitedReasoningParserBuilder, ReasoningDelta, ReasoningParser,
    Result,
};

/// Reasoning parser for DeepSeek V4 and V4.1 outputs.
///
/// The delimiters are bare `<think>` / `</think>`, as for DeepSeek V3, but
/// they only count when the model emits their dedicated tokens: an
/// ordinary-token spelling of `</think>` inside reasoning stays reasoning. The
/// DeepSeek reference parser matches these delimiters by token identity too.
///
/// Requires attributed input: fed unattributed text, the parser never leaves
/// its initial state.
pub struct DeepSeekV4ReasoningParser {
    inner: DelimitedReasoningParser,
}

impl DeepSeekV4ReasoningParser {
    /// Create a DeepSeek V4 parser backed by the shared delimited state machine.
    pub fn new(tokenizer: DynTokenizer) -> Result<Self> {
        Ok(Self {
            inner: DelimitedReasoningParserBuilder::new(tokenizer, "<think>", "</think>")
                .with_guarded_delimiters()
                .build()?,
        })
    }
}

impl ReasoningParser for DeepSeekV4ReasoningParser {
    fn create(tokenizer: DynTokenizer) -> Result<Box<dyn ReasoningParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tokenizer)?))
    }

    fn initialize(&mut self, prompt_token_ids: &[u32]) -> Result<()> {
        self.inner.initialize(prompt_token_ids)
    }

    fn wrap_visible_format(
        &self,
        _ctx: &OutputGrammarContext<'_>,
        visible: &Format,
    ) -> output_grammar::Result<Option<Format>> {
        Ok(Some(self.inner.wrap_visible_format(visible)))
    }

    fn push(&mut self, delta: DecodedText) -> Result<ReasoningDelta> {
        Ok(self.inner.push(delta))
    }

    fn finish(&mut self) -> Result<ReasoningDelta> {
        Ok(self.inner.finish())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vllm_tokenizer::test_utils::TestTokenizer;
    use vllm_tokenizer::{DecodedText, Tokenizer as _};

    use super::DeepSeekV4ReasoningParser;
    use crate::reasoning::{ReasoningDelta, ReasoningParser};

    const THINK_START_ID: u32 = 256;
    const THINK_END_ID: u32 = 257;

    /// DeepSeek V4 has `<think>` / `</think>` as added, non-special tokens.
    fn tokenizer() -> TestTokenizer {
        TestTokenizer::new()
            .with_regular_token("<think>", THINK_START_ID)
            .with_regular_token("</think>", THINK_END_ID)
    }

    fn parser(prompt_token_ids: &[u32]) -> DeepSeekV4ReasoningParser {
        let mut parser = DeepSeekV4ReasoningParser::new(Arc::new(tokenizer())).unwrap();
        parser.initialize(prompt_token_ids).unwrap();
        parser
    }

    /// Run token IDs through the real incremental decoder to obtain attributed
    /// output, as the frontend does.
    fn decode_ids(ids: &[u32]) -> DecodedText {
        let tokenizer = tokenizer();
        let mut decoder = tokenizer.create_decode_stream(&[], false, 0);
        for &id in ids {
            decoder.push_token(id).unwrap();
        }
        decoder.flush(None).unwrap().1
    }

    /// Encode with the added tokens, so delimiters are dedicated tokens.
    fn dedicated(text: &str) -> Vec<u32> {
        tokenizer().encode(text, false).unwrap()
    }

    /// Encode as ordinary bytes only, so delimiter spellings are body text.
    fn ordinary(text: &str) -> Vec<u32> {
        tokenizer().encode_ordinary(text).unwrap()
    }

    /// Stream `output` split at the byte offsets `splits`, then finish, and
    /// return the concatenated reasoning and content.
    fn stream(
        parser: &mut DeepSeekV4ReasoningParser,
        output: DecodedText,
        splits: &[usize],
    ) -> (String, String) {
        let mut rest = output;
        let mut consumed = 0;
        let mut deltas = Vec::new();
        for &split in splits {
            deltas.push(parser.push(rest.drain_prefix(split - consumed)).unwrap());
            consumed = split;
        }
        deltas.push(parser.push(rest).unwrap());
        deltas.push(parser.finish().unwrap());
        let text = |part: fn(&ReasoningDelta) -> &Option<DecodedText>| {
            deltas
                .iter()
                .filter_map(|delta| part(delta).as_ref())
                .map(|piece| piece.text.as_str())
                .collect()
        };
        (text(|delta| &delta.reasoning), text(|delta| &delta.content))
    }

    #[test]
    fn deepseek_v4_dedicated_think_end_closes_reasoning() {
        let mut parser = parser(&[THINK_START_ID]);
        let output = decode_ids(&dedicated("reason</think>answer"));

        let (reasoning, content) = stream(&mut parser, output, &[]);
        assert_eq!(reasoning, "reason");
        assert_eq!(content, "answer");
    }

    #[test]
    fn deepseek_v4_ordinary_think_end_stays_reasoning() {
        let mut parser = parser(&[THINK_START_ID]);
        let ids = [
            dedicated("the tag "),
            ordinary("</think>"),
            dedicated(" ends it.</think>answer"),
        ]
        .concat();

        let (reasoning, content) = stream(&mut parser, decode_ids(&ids), &[]);
        assert_eq!(reasoning, "the tag </think> ends it.");
        assert_eq!(content, "answer");
    }

    #[test]
    fn deepseek_v4_dedicated_think_end_split_across_chunks_still_closes() {
        let text = "reason</think>answer";
        let start = text.find("</think>").unwrap();
        for split in start + 1..start + "</think>".len() {
            let mut parser = parser(&[THINK_START_ID]);
            let output = decode_ids(&dedicated(text));

            let (reasoning, content) = stream(&mut parser, output, &[split]);
            assert_eq!(reasoning, "reason", "split at {split}");
            assert_eq!(content, "answer", "split at {split}");
        }
    }

    #[test]
    fn deepseek_v4_ordinary_partial_think_end_streams_immediately() {
        let mut parser = parser(&[THINK_START_ID]);
        let delta = parser.push(decode_ids(&ordinary("reason</thi"))).unwrap();

        // Not held back: the bytes cannot grow into the dedicated token.
        assert_eq!(delta.reasoning.unwrap().text, "reason</thi");
    }

    #[test]
    fn deepseek_v4_ordinary_think_start_stays_content() {
        let mut parser = parser(&[THINK_END_ID]);
        let ids = [ordinary("<think>"), dedicated("x</think>y")].concat();

        let (reasoning, content) = stream(&mut parser, decode_ids(&ids), &[]);
        assert_eq!(reasoning, "");
        assert_eq!(content, "<think>x</think>y");
    }

    #[test]
    fn deepseek_v4_dedicated_think_start_opens_reasoning() {
        let mut parser = parser(&[THINK_END_ID]);
        let output = decode_ids(&dedicated("<think>reason</think>answer"));

        let (reasoning, content) = stream(&mut parser, output, &[]);
        assert_eq!(reasoning, "reason");
        assert_eq!(content, "answer");
    }
}
