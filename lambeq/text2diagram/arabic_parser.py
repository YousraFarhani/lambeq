from __future__ import annotations

__all__ = ['ArabicParser', 'ArabicParseError']

import stanza
import re
from collections.abc import Iterable
from typing import Any, List, Dict, Optional

from lambeq.text2diagram.ccg_parser import CCGParser
from lambeq.text2diagram.ccg_rule import CCGRule
from lambeq.text2diagram.ccg_tree import CCGTree
from lambeq.text2diagram.ccg_type import CCGType
from lambeq.core.globals import VerbosityLevel
from lambeq.core.utils import (SentenceBatchType, tokenised_batch_type_check,
                               untokenised_batch_type_check)

# Helper classes and functions for PATB tree processing

class ATBNode:
    def __init__(self, tag: str, word: Optional[str] = None, children: Optional[List[ATBNode]] = None):
        self.tag = tag
        self.word = word
        self.children = children if children is not None else []
        self.normalized_tag: Optional[str] = None
        self.constituent_type: Optional[str] = None

    def is_terminal(self) -> bool:
        return len(self.children) == 0

    def __repr__(self) -> str:
        if self.is_terminal():
            return f"({self.tag} {self.word})"
        else:
            return f"({self.tag} {' '.join([repr(child) for child in self.children])})"

def parse_atb_tree(tree_str: str) -> ATBNode:
    tokens = tree_str.replace("(", " ( ").replace(")", " ) ").split()
    def helper(tokens: List[str]) -> ATBNode:
        if not tokens:
            raise ValueError("Unexpected end of tokens.")
        token = tokens.pop(0)
        if token != '(':
            raise ValueError("Expected '('")
        tag = tokens.pop(0)
        children: List[ATBNode] = []
        word: Optional[str] = None
        while tokens and tokens[0] != ')':
            if tokens[0] == '(':
                child = helper(tokens)
                children.append(child)
            else:
                word = tokens.pop(0)
        if tokens and tokens[0] == ')':
            tokens.pop(0)
        return ATBNode(tag, word, children)
    return helper(tokens)

def normalize_tag(node: ATBNode) -> None:
    tag_mapping = {
        "NN": "NN",
        "NOUN": "NN",
        "NOUN_PROP": "NNP",
        "NNS": "NNS",
        "NNP": "NNP",
        "NNPS": "NNPS",
        "VB": "VB",
        "VERB_PERFECT+PVSUFF_SUBJ:3FS": "VB",
        "VERB_IMPERFECT": "VB",
        "VBD": "VBD",
        "VBP": "VBP",
        "ADJ": "JJ",
        "JJ": "JJ",
        "JJR": "JJR",
        "JJS": "JJS",
        "ADV": "RB",
        "RB": "RB",
        "IN": "IN",
        "DT": "DT",
        "DEM": "DT",
        "PRP": "PRP",
        "CC": "CC",
        "PUNC": "PUNC",
        "NON_ALPHABETIC": "PUNC",
        "NON_ARABIC": "FW",
        "NOTAG": "PUNC",
        "NEG_PART+PVSUFF_SUBJ:3MS": "VBP",
        "VBP+NEG": "VBP",
    }
    node.normalized_tag = tag_mapping.get(node.tag, node.tag)
    for child in node.children:
        normalize_tag(child)

def determine_constituent_types(node: ATBNode) -> None:
    if node.is_terminal():
        lower_tag = node.tag.lower()
        if "sbj" in lower_tag:
            node.constituent_type = "subject"
        elif "obj" in lower_tag:
            node.constituent_type = "object"
        elif "adv" in lower_tag or "loc" in lower_tag:
            node.constituent_type = "adjunct"
        else:
            node.constituent_type = "head"
        return
    for child in node.children:
        determine_constituent_types(child)
    explicit = []
    for child in node.children:
        ltag = child.tag.lower()
        if "sbj" in ltag or "obj" in ltag or child.normalized_tag in {"VB", "VBD", "VBP"}:
            explicit.append(child)
    if explicit:
        for i, child in enumerate(node.children):
            if child in explicit and i == 0:
                child.constituent_type = "head"
            else:
                child.constituent_type = "complement"
    else:
        mid = len(node.children) // 2
        for i, child in enumerate(node.children):
            if i == mid:
                child.constituent_type = "head"
            else:
                child.constituent_type = "complement"

def binarize_tree(node: ATBNode) -> ATBNode:
    if node.is_terminal():
        return node
    node.children = [binarize_tree(child) for child in node.children]
    while len(node.children) > 2:
        left = node.children.pop(0)
        right = node.children.pop(0)
        new_node = ATBNode(tag=node.tag + "_BIN", children=[left, right])
        new_node.constituent_type = "head"
        node.children.insert(0, new_node)
    return node

def convert_atb_node_to_ccg(node: ATBNode, parser: ArabicParser) -> CCGTree:
    if node.is_terminal():
        ccg_type = parser.map_pos_to_ccg(node.normalized_tag, "")
        word_text = node.word[::-1] if node.word else ""
        return CCGTree(text=word_text, rule=CCGRule.LEXICAL, biclosed_type=ccg_type)
    children_ccg = [convert_atb_node_to_ccg(child, parser) for child in node.children]
    head_index = None
    for i, child in enumerate(node.children):
        if getattr(child, 'constituent_type', None) == "head":
            head_index = i
            break
    if head_index is None:
        head_index = len(node.children) // 2
    if head_index > 0:
        left_tree = children_ccg[0]
        for i in range(1, head_index + 1):
            left_tree = CCGTree(
                text=None,
                rule=CCGRule.BACKWARD_APPLICATION,
                children=[left_tree, children_ccg[i]],
                biclosed_type=CCGType.SENTENCE
            )
    else:
        left_tree = children_ccg[head_index]
    if head_index < len(children_ccg) - 1:
        right_tree = children_ccg[head_index + 1]
        for i in range(head_index + 2, len(children_ccg)):
            right_tree = CCGTree(
                text=None,
                rule=CCGRule.FORWARD_APPLICATION,
                children=[right_tree, children_ccg[i]],
                biclosed_type=CCGType.SENTENCE
            )
        combined = CCGTree(
            text=None,
            rule=CCGRule.FORWARD_APPLICATION,
            children=[left_tree, right_tree],
            biclosed_type=CCGType.SENTENCE
        )
    else:
        combined = left_tree
    return combined

# Full ArabicParser Implementation

class ArabicParseError(Exception):
    def __init__(self, sentence: str) -> None:
        self.sentence = sentence

    def __str__(self) -> str:
        return f'ArabicParser failed to parse {self.sentence!r}.'

class ArabicParser(CCGParser):
    def __init__(self, verbose: str = VerbosityLevel.PROGRESS.value, **kwargs: Any) -> None:
        self.verbose = verbose
        if not VerbosityLevel.has_value(verbose):
            raise ValueError(f'`{verbose}` is not a valid verbose value for ArabicParser.')
        stanza.download('ar', processors='tokenize,pos,lemma,depparse', verbose=False)
        self.nlp = stanza.Pipeline(lang='ar', processors='tokenize,pos,lemma,depparse', verbose=False)

    # The signature matches the abstract method (with verbose included)
    def sentences2trees(self,
                        sentences: SentenceBatchType,
                        tokenised: bool = False,
                        suppress_exceptions: bool = False,
                        verbose: str | None = None) -> list[CCGTree | None]:
        trees: list[CCGTree | None] = []
        for sentence in sentences:
            try:
                tree = self.sentence2diagram(sentence)
                trees.append(tree)
            except Exception as e:
                if suppress_exceptions:
                    trees.append(None)
                else:
                    raise e
        return trees

    def sentence2diagram(self, sentence: str) -> CCGTree:
        try:
            atb_root = parse_atb_tree(sentence)
            normalize_tag(atb_root)
            determine_constituent_types(atb_root)
            binarized_tree = binarize_tree(atb_root)
            ccg_tree = convert_atb_node_to_ccg(binarized_tree, self)
            return ccg_tree
        except Exception as e:
            raise ArabicParseError(sentence) from e

    def map_pos_to_ccg(self, atb_pos: str, dependency: str) -> CCGType:
        atb_to_ccg_map = {
            "NN": CCGType.NOUN,
            "VB": CCGType.SENTENCE.slash("\\", CCGType.NOUN_PHRASE),
            "IN": CCGType.NOUN_PHRASE.slash("\\", CCGType.NOUN_PHRASE),
            "DT": CCGType.NOUN.slash("/", CCGType.NOUN),
            "JJ": CCGType.NOUN.slash("\\", CCGType.NOUN),
            "PRP": CCGType.NOUN_PHRASE,
            "RB": CCGType.SENTENCE.slash("/", CCGType.SENTENCE),
            "CD": CCGType.NOUN.slash("/", CCGType.NOUN),
            "CC": CCGType.CONJUNCTION,
            "PUNC": CCGType.PUNCTUATION,
            "FW": CCGType.FOREIGN,
        }
        if dependency in ["amod", "acl"]:
            return CCGType.NOUN.slash("\\", CCGType.NOUN)
        elif dependency in ["nsubj", "csubj"]:
            return CCGType.NOUN_PHRASE
        elif dependency in ["obj", "iobj"]:
            return CCGType.NOUN_PHRASE
        return atb_to_ccg_map.get(atb_pos, CCGType.NOUN)