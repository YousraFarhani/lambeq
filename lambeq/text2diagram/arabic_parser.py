from __future__ import annotations

__all__ = ['ArabicParser', 'ArabicParseError']

import stanza
from camel_tools.utils.normalize import normalize_unicode
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer
from collections.abc import Iterable
from typing import Any
from lambeq.text2diagram.ccg_parser import CCGParser
from lambeq.text2diagram.ccg_rule import CCGRule
from lambeq.text2diagram.ccg_tree import CCGTree
from lambeq.text2diagram.ccg_type import CCGType
from lambeq.core.globals import VerbosityLevel
from lambeq.core.utils import (SentenceBatchType, tokenised_batch_type_check,
                               untokenised_batch_type_check)


class ArabicParseError(Exception):
    def __init__(self, sentence: str) -> None:
        self.sentence = sentence

    def __str__(self) -> str:
        return f'ArabicParser failed to parse {self.sentence!r}.'


class ArabicParser(CCGParser):
    """CCG parser for Arabic using Stanza and CamelTools."""

    def __init__(self, verbose: str = VerbosityLevel.PROGRESS.value, **kwargs: Any) -> None:
        """Initialize the ArabicParser with required NLP tools."""
        self.verbose = verbose

        if not VerbosityLevel.has_value(verbose):
            raise ValueError(f'`{verbose}` is not a valid verbose value for ArabicParser.')
        
        # Initialize the morphology analyzer
        self.analyzer = Analyzer(MorphologyDB.builtin_db())
        
        # Initialize Stanza for dependency parsing
        stanza.download('ar', processors='tokenize,pos,lemma,depparse')
        self.nlp = stanza.Pipeline(lang='ar', processors='tokenize,pos,lemma,depparse')
    
    def sentences2trees(self,
                        sentences: SentenceBatchType,
                        tokenised: bool = False,
                        suppress_exceptions: bool = False,
                        verbose: str | None = None) -> list[CCGTree | None]:
        """Convert multiple Arabic sentences to CCG trees."""
        
        if verbose is None:
            verbose = self.verbose
        if not VerbosityLevel.has_value(verbose):
            raise ValueError(f'`{verbose}` is not a valid verbose value for ArabicParser.')
        
        if tokenised:
            if not tokenised_batch_type_check(sentences):
                raise ValueError('`tokenised` set to `True`, but variable `sentences` does not have type `List[List[str]]`.')
        else:
            if not untokenised_batch_type_check(sentences):
                raise ValueError('`tokenised` set to `False`, but variable `sentences` does not have type `List[str]`.')
            sent_list: list[str] = [str(s) for s in sentences]
            sentences = [self.preprocess(sentence) for sentence in sent_list]
        
        trees: list[CCGTree] = []
        for sentence in sentences:
            try:
                atb_tree = self.parse_atb(sentence)
                ccg_tree = self.convert_to_ccg(atb_tree)
                trees.append(ccg_tree)
            except Exception as e:
                if suppress_exceptions:
                    trees.append(None)
                else:
                    raise ArabicParseError(' '.join(sentence)) from e
        
        return trees
    
    def preprocess(self, sentence: str) -> list[str]:
        """Normalize and tokenize Arabic text."""
        normalized_text = normalize_unicode(sentence)
        tokens = simple_word_tokenize(normalized_text)
        return tokens

    def parse_atb(self, words: list[str]) -> list[dict]:
        """Parse Arabic sentence using ATB syntactic structures."""
        sentence = " ".join(words)
        doc = self.nlp(sentence)
        parsed_data = []
        
        for sent in doc.sentences:
            for word in sent.words:
                parsed_data.append({
                    "word": word.text,
                    "lemma": word.lemma,
                    "pos": word.xpos,  # ATB POS tags
                    "head": word.head,
                    "dep": word.deprel
                })
        return parsed_data
    
    def convert_to_ccg(self, atb_tree: list[dict]) -> CCGTree:
        """Convert ATB's parse tree into a CCG derivation."""
        children = []
        for entry in atb_tree:
            word = entry["word"]
            pos = entry["pos"]
            dependency = entry["dep"]
            ccg_category = self.map_pos_to_ccg(pos, dependency)
            children.append(CCGTree(text=word, rule=CCGRule.LEXICAL, biclosed_type=ccg_category))
        return CCGTree(text=None, rule=CCGRule.FORWARD_APPLICATION, children=children, biclosed_type=CCGType.SENTENCE)
    
    def map_pos_to_ccg(self, atb_pos: str, dependency: str) -> CCGType:
        """Map Arabic Treebank POS tags & dependencies to CCG-compatible categories."""
        atb_to_ccg_map = {
            "NN": CCGType.NOUN,
            "VB": CCGType.SENTENCE.slash("\\", CCGType.NOUN_PHRASE),
            "IN": CCGType.NOUN_PHRASE.slash("\\", CCGType.NOUN_PHRASE),
            "DT": CCGType.NOUN.slash("/", CCGType.NOUN),
            "JJ": CCGType.NOUN.slash("/", CCGType.NOUN),
            "PRP": CCGType.NOUN_PHRASE,
            "CC": CCGType.CONJUNCTION,
            "RB": CCGType.SENTENCE.slash("/", CCGType.SENTENCE),
            "CD": CCGType.NOUN.slash("/", CCGType.NOUN),
            "UH": CCGType.CONJUNCTION
        }
        
        if dependency in ["amod", "acl"]:
            return CCGType.NOUN.slash("/", CCGType.NOUN)
        elif dependency in ["nsubj", "csubj"]:
            return CCGType.NOUN_PHRASE
        elif dependency in ["obj", "iobj"]:
            return CCGType.NOUN_PHRASE
        
        return atb_to_ccg_map.get(atb_pos, CCGType.NOUN)  # Default to noun if unknown

