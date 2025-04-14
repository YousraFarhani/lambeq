from __future__ import annotations

__all__ = ['ArabicParser', 'ArabicParseError']

import stanza
import re
from collections.abc import Iterable
from typing import Any
from lambeq.text2diagram.ccg_parser import CCGParser
from lambeq.text2diagram.ccg_rule import CCGRule
from lambeq.text2diagram.ccg_tree import CCGTree #Represents the hierarchical CCG structure
from lambeq.text2diagram.ccg_type import CCGType #Represents CCG categories, such as NOUN, VERB
from lambeq.core.globals import VerbosityLevel #Controls the verbosity level of the parser
from lambeq.core.utils import (SentenceBatchType, tokenised_batch_type_check,
                               untokenised_batch_type_check) #Defines type hints for handling batch processing of sentences and validate the input format


class ArabicParseError(Exception):
    def __init__(self, sentence: str) -> None:
        self.sentence = sentence

    def __str__(self) -> str:
        return f'ArabicParser failed to parse {self.sentence!r}.'


class ArabicParser(CCGParser):
    """CCG parser for Arabic using Stanza and ATB dataset."""

    def __init__(self, verbose: str = VerbosityLevel.PROGRESS.value, **kwargs: Any) -> None:
        """Initialize the ArabicParser with required NLP tools."""
        self.verbose = verbose
        
        #setting the verbosity level and Validates the verbosity setting

        if not VerbosityLevel.has_value(verbose):
            raise ValueError(f'`{verbose}` is not a valid verbose value for ArabicParser.')
        
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
        """Normalize and tokenize Arabic text, handling ATB cases."""
        words = re.findall(r'\b\w+\b', sentence)  # Tokenize words correctly
        processed_words = []
        
        for word in words:
            if word.startswith("ال") and len(word) > 2:
                processed_words.append("ال")
                processed_words.append(word[2:])
            else:
                processed_words.append(word)
        
        return processed_words[::-1]  # Reverse word order to maintain correct Arabic rendering

    def parse_atb(self, words: list[str]) -> list[dict]:
        """Parse Arabic sentence using ATB syntactic structures."""
        sentence = " ".join(words)
        doc = self.nlp(sentence)
        parsed_data = []
        
        #Extracts word features
        for sent in doc.sentences:
            for word in sent.words:
                parsed_data.append({
                    "word": word.text,
                    "lemma": word.lemma, #Root form of the word.
                    "pos": word.xpos,  # ATB POS tags
                    "head": word.head, #Syntactic head.
                    "dep": word.deprel #Dependency relation.
                })
        return parsed_data
    
    def convert_to_ccg(self, atb_tree: list[dict]) -> CCGTree:
        """Convert ATB's parse tree into a CCG derivation."""
        nodes = []
        
        for entry in atb_tree:
            word = entry["word"][::-1]  # Reverse Arabic characters for correct display
            pos = entry["pos"]
            dependency = entry["dep"]
            ccg_category = self.map_pos_to_ccg(pos, dependency)
            nodes.append(CCGTree(text=word, rule=CCGRule.LEXICAL, biclosed_type=ccg_category))

        # Ensure binary tree formation
        while len(nodes) > 1:
            left = nodes.pop(0)
            right = nodes.pop(0)
            parent = CCGTree(text=None, rule=CCGRule.FORWARD_APPLICATION, children=[left, right], biclosed_type=CCGType.SENTENCE)
            nodes.insert(0, parent)

        return nodes[0]  # Return root CCGTree
    
    def map_pos_to_ccg(self, atb_pos: str, dependency: str) -> CCGType:
        """Map Arabic Treebank POS tags & dependencies to CCG-compatible categories."""
        #This dictionary maps Arabic Treebank POS tags to their CCG equivalents
        #Slash notation (/ or \) represents combinatory rules
        atb_to_ccg_map = {
            "NN": CCGType.NOUN,                                             #Noun -> N
          #Verb (S\NP) means a verb expects a noun phrase (subject) on its left.
            "VB": CCGType.SENTENCE.slash("\\", CCGType.NOUN_PHRASE),        # Verb -> S\NP
            "IN": CCGType.NOUN_PHRASE.slash("\\", CCGType.NOUN_PHRASE),     # Preposition -> NP\NP
          #Determiner (N/N) means a determiner modifies a noun.
            "DT": CCGType.NOUN.slash("/", CCGType.NOUN),                    # Determiner -> N/N
            "JJ": CCGType.NOUN.slash("/", CCGType.NOUN),                    # Adjective -> N/N
            "PRP": CCGType.NOUN_PHRASE,                                     # Pronoun -> NP
            "CC": CCGType.CONJUNCTION,                                      # Conjunction -> Conj
            "RB": CCGType.SENTENCE.slash("/", CCGType.SENTENCE),            # Adverb -> S/S
            "CD": CCGType.NOUN.slash("/", CCGType.NOUN),                    # Number -> N/N
            "UH": CCGType.CONJUNCTION                                       # Interjection -> Conj
        }
        
        if dependency in ["amod", "acl"]:
            return CCGType.NOUN.slash("/", CCGType.NOUN)                    # Adjective modifier (modifier, clause) (N/N)
        elif dependency in ["nsubj", "csubj"]:
            return CCGType.NOUN_PHRASE                                      # Subject (nominal/clausale subejct)-> NP 
        elif dependency in ["obj", "iobj"]:
            return CCGType.NOUN_PHRASE                                      # Object (direct/indirect) -> NP
        
        return atb_to_ccg_map.get(atb_pos, CCGType.NOUN)  # Default to noun if unknown

        
        if dependency in ["amod", "acl"]:
            return CCGType.NOUN.slash("/", CCGType.NOUN)
        elif dependency in ["nsubj", "csubj"]:
            return CCGType.NOUN_PHRASE
        elif dependency in ["obj", "iobj"]:
            return CCGType.NOUN_PHRASE
        
        return atb_to_ccg_map.get(atb_pos, CCGType.NOUN)  # Default to noun if unknown

#Example::
# الولد--atb: NN , dependency: nsubj, CCGOutput: NP
# يقرا==atb: VB, depenency: root, CCGOutput: S\NP
#الكتاب --atb:NN ,dependency: obj, CCGOutput: NP
# NP S\NP NP