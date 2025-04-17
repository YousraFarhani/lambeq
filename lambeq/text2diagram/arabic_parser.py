from __future__ import annotations

__all__ = ['ArabicParser', 'ArabicParseError']

import stanza
import re
from typing import Any, List, Dict
from lambeq.text2diagram.ccg_parser import CCGParser
from lambeq.text2diagram.ccg_rule import CCGRule
from lambeq.text2diagram.ccg_tree import CCGTree
from lambeq.text2diagram.ccg_type import CCGType
from lambeq.core.globals import VerbosityLevel
from lambeq.core.utils import (SentenceBatchType, tokenised_batch_type_check,
                               untokenised_batch_type_check)


class ArabicParseError(Exception):
    def __init__(self, sentence: Any) -> None:
        self.sentence = sentence

    def __str__(self) -> str:
        return f'ArabicParser failed to parse {self.sentence!r}.'


class ArabicParser(CCGParser):
    """CCG parser for Arabic using Stanza and/or ATB dataset files."""

    def __init__(self, verbose: str = VerbosityLevel.PROGRESS.value, **kwargs: Any) -> None:
        if not VerbosityLevel.has_value(verbose):
            raise ValueError(f'`{verbose}` is not valid for ArabicParser.')
        self.verbose = verbose
        stanza.download('ar', processors='tokenize,pos,lemma,depparse')
        self.nlp = stanza.Pipeline(lang='ar', processors='tokenize,pos,lemma,depparse')

    def load_atb_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load ATB file and extract token-level morphology based on the t: tag entries.
        Features list is derived directly from ATB tags (e.g., DET, CASE_DEF_GEN).
        """
        s_lines, t_lines = [], []
        with open(file_path, encoding='utf-8') as f:
            for line in f:
                if line.startswith('s:'):
                    s_lines.append(line.strip())
                elif line.startswith('t:'):
                    t_lines.append(line.strip())
        entries: List[Dict[str, Any]] = []
        for s_line, t_line in zip(s_lines, t_lines):
            s_fields = s_line.split('·')
            word = s_fields[1]
            t_fields = t_line.split('·')
            tag_string = t_fields[1]
            lemma = t_fields[5] if len(t_fields) > 5 else None
            # Split POS+features directly from ATB tag
            pos_parts = tag_string.split('+')
            base_pos = pos_parts[0]
            features = pos_parts[1:]
            entries.append({'word': word, 'lemma': lemma, 'pos': base_pos, 'features': features})
        return entries

    def sentences2trees(
        self,
        sentences: SentenceBatchType,
        tokenised: bool = False,
        suppress_exceptions: bool = False,
        verbose: str | None = None
    ) -> list[CCGTree | None]:
        if verbose is None:
            verbose = self.verbose
        if not VerbosityLevel.has_value(verbose):
            raise ValueError(f'Invalid verbose value: {verbose}')
        if tokenised:
            if not tokenised_batch_type_check(sentences):
                raise ValueError('Batch must be List[List[str]] when tokenised=True.')
        else:
            if not untokenised_batch_type_check(sentences):
                raise ValueError('Batch must be List[str] when tokenised=False.')
            sentences = [self.preprocess(str(s)) for s in sentences]

        trees: list[CCGTree | None] = []
        for sentence in sentences:
            try:
                if isinstance(sentence, list) and sentence and isinstance(sentence[0], dict):
                    atb_entries = sentence
                else:
                    atb_entries = self.parse_stanza(sentence)
                ccg_tree = self.convert_to_ccg(atb_entries)
                trees.append(ccg_tree)
            except Exception as e:
                if suppress_exceptions:
                    trees.append(None)
                else:
                    raise ArabicParseError(sentence) from e
        return trees

    def preprocess(self, sentence: str) -> list[str]:
        """Simple tokenization preserving full words (no splitting)."""
        return re.findall(r'\b\w+\b', sentence)

    def parse_stanza(self, tokens: list[str]) -> list[Dict[str, Any]]:
        """
        Parse a list of Arabic tokens with Stanza.
        Feature list is empty unless augmented by external morphological tools.
        """
        sentence = ' '.join(tokens)
        doc = self.nlp(sentence)
        parsed: List[Dict[str, Any]] = []
        for sent in doc.sentences:
            for w in sent.words:
                parsed.append({
                    'word': w.text,
                    'lemma': w.lemma,
                    'pos': w.xpos,
                    'features': [],  # no ATB morphology here
                    'head': w.head,
                    'dep': w.deprel
                })
        return parsed

    def convert_to_ccg(self, atb_entries: list[Dict[str, Any]]) -> CCGTree:
        nodes: list[CCGTree] = []
        for entry in atb_entries:
            word = entry['word']
            pos = entry['pos']
            features = entry.get('features', [])
            ccg_cat = self.map_pos_to_ccg(pos, features)
            node = CCGTree(text=word, rule=CCGRule.LEXICAL, biclosed_type=ccg_cat)
            node.morphology = {'pos': pos, 'features': features}
            node.semantic = self.map_semantic(pos, features)
            nodes.append(node)
        # build binary tree
        while len(nodes) > 1:
            left = nodes.pop(0)
            right = nodes.pop(0)
            parent = CCGTree(
                text=None,
                rule=CCGRule.FORWARD_APPLICATION,
                children=[left, right],
                biclosed_type=CCGType.SENTENCE
            )
            nodes.insert(0, parent)
        return nodes[0]

    def map_pos_to_ccg(self, pos: str, features: list[str]) -> CCGType:
        base = pos.split('+')[0]
        mapping = {
            'NN': CCGType.NOUN,
            'NOUN': CCGType.NOUN,
            'NOUN_PROP': CCGType.NOUN,
            'NOUN_NUM': CCGType.NOUN,
            'VB': CCGType.SENTENCE.slash('\\', CCGType.NOUN_PHRASE),
            'PREP': CCGType.NOUN_PHRASE.slash('\\', CCGType.NOUN_PHRASE),
            'DET': CCGType.NOUN.slash('/', CCGType.NOUN),
            'JJ': CCGType.NOUN.slash('/', CCGType.NOUN),
            'PRP': CCGType.NOUN_PHRASE,
            'CC': CCGType.CONJUNCTION,
            'RB': CCGType.SENTENCE.slash('/', CCGType.SENTENCE),
            'CD': CCGType.NOUN.slash('/', CCGType.NOUN),
            'UH': CCGType.CONJUNCTION
        }
        # treat determiners as noun modifiers
        if base == 'DET' or 'DET' in features:
            return CCGType.NOUN_PHRASE
        return mapping.get(base, CCGType.NOUN)

    def map_semantic(self, pos: str, features: list[str]) -> str:
        base = pos.split('+')[0]
        sem_map = {
            'NOUN_PROP': 'proper_noun',
            'NOUN_NUM': 'numeric',
            'PREP': 'locative',
            'DET': 'determiner',
            'JJ': 'modifier',
            'VB': 'action',
            'PRP': 'pronoun',
            'CC': 'conjunction',
            'RB': 'adverb',
            'CD': 'quantifier'
        }
        # semantic "definite" if definiteness marked in ATB features
        if any(f.startswith('CASE_DEF') for f in features):
            return 'definite'
        return sem_map.get(base, 'unknown')