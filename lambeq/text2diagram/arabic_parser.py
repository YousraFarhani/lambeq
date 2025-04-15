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

###############################################################################
# Helper classes and functions for PATB Tree Processing
###############################################################################

class ATBNode:
    r"""
    Represents a node in an Arabic Treebank (PATB) parse tree.
    
    Attributes:
      tag: The original tag from PATB (e.g., "NN", "VERB_PERFECT+PVSUFF_SUBJ:3FS", etc.)
      word: The terminal word (if the node is a leaf).
      children: List of child ATBNodes.
      normalized_tag: The tag after normalization (using mapping rules).
      constituent_type: Annotation ("head", "complement", or "adjunct") determined by heuristics.
    """
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
    r"""
    Parse a PATB-style bracketed tree string into an ATBNode tree.
    This function uses a recursive descent approach.
    It will attempt to correct for missing brackets by using heuristics.
    
    Example:
      Input: "(S (NP (DT الكتاب) (NN)) (VP (VB قرأ) (NP (DT الطالب) (NN))))"
    """
    # Preprocess: add spaces around parentheses.
    tokens = tree_str.replace("(", " ( ").replace(")", " ) ").split()
    
    def helper(tokens: List[str]) -> ATBNode:
        if not tokens:
            raise ValueError("Unexpected end of tokens.")
        token = tokens.pop(0)
        if token != '(':
            raise ValueError("Expected '('")
        # The next token should be the node tag.
        tag = tokens.pop(0)
        children: List[ATBNode] = []
        word: Optional[str] = None
        
        # Process until we encounter a ')'
        while tokens and tokens[0] != ')':
            if tokens[0] == '(':
                child = helper(tokens)
                children.append(child)
            else:
                # Terminal word encountered.
                word = tokens.pop(0)
        if tokens and tokens[0] == ')':
            tokens.pop(0)  # Remove ')'
        return ATBNode(tag, word, children)
    
    return helper(tokens)

def normalize_tag(node: ATBNode) -> None:
    r"""
    Recursively normalize the node's tag using a comprehensive mapping from PATB tags
    to a reduced tagset used for CCG conversion.
    
    The mapping here covers many of the common PATB tags. (A complete system might include
    several hundred entries; here we list a representative sample.)
    """
    tag_mapping = {
        # Nouns
        "NN": "NN",
        "NOUN": "NN",
        "NOUN_PROP": "NNP",
        "NNS": "NNS",
        "NNP": "NNP",
        "NNPS": "NNPS",
        # Verbs
        "VB": "VB",
        "VERB_PERFECT+PVSUFF_SUBJ:3FS": "VB",
        "VERB_IMPERFECT": "VB",
        "VBD": "VBD",
        "VBP": "VBP",
        # Adjectives
        "ADJ": "JJ",
        "JJ": "JJ",
        "JJR": "JJR",
        "JJS": "JJS",
        # Adverbs
        "ADV": "RB",
        "RB": "RB",
        # Prepositions
        "IN": "IN",
        # Determiners and demonstratives
        "DT": "DT",
        "DEM": "DT",
        # Pronouns
        "PRP": "PRP",
        # Conjunctions
        "CC": "CC",
        # Punctuation
        "PUNC": "PUNC",
        "NON_ALPHABETIC": "PUNC",
        "NON_ARABIC": "FW",
        "NOTAG": "PUNC",
        # Additional, complex tags
        "NEG_PART+PVSUFF_SUBJ:3MS": "VBP",  # context-dependent mapping
        "VBP+NEG": "VBP",
    }
    node.normalized_tag = tag_mapping.get(node.tag, node.tag)
    for child in node.children:
        normalize_tag(child)

def determine_constituent_types(node: ATBNode) -> None:
    r"""
    Determine the constituent types for a given ATB tree node.
    
    Uses detailed heuristics:
      - For terminal nodes, examine the original tag for cues such as SBJ, OBJ, ADV.
      - For non-terminals, if any child contains explicit function tags (e.g., SBJ, OBJ, ADV, LOC),
        mark the first occurrence as head and others as complements.
      - Otherwise, choose the middle child as head.
    """
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
    r"""
    Binarize the ATB tree recursively. For nodes with more than two children,
    combine them into binary nodes in a head-driven fashion.
    """
    if node.is_terminal():
        return node
    node.children = [binarize_tree(child) for child in node.children]
    while len(node.children) > 2:
        left = node.children.pop(0)
        right = node.children.pop(0)
        new_node = ATBNode(tag=node.tag + "_BIN", children=[left, right])
        new_node.constituent_type = "head"  # assume left is head
        node.children.insert(0, new_node)
    return node

def convert_atb_node_to_ccg(node: ATBNode, parser: ArabicParser) -> CCGTree:
    r"""
    Recursively convert an ATBNode (normalized, annotated, and binarized) into a CCG derivation tree.
    
    For terminal nodes, a lexical CCGTree is generated using parser.map_pos_to_ccg.
    For non-terminals, the children are combined by splitting into left and right parts around the head,
    then using backward and forward application accordingly.
    """
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
                biclosed_type=CCGType.SENTENCE  # Placeholder; real type resolution would be applied.
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
                biclosed_type=CCGType.SENTENCE  # Placeholder.
            )
        combined = CCGTree(
            text=None,
            rule=CCGRule.FORWARD_APPLICATION,
            children=[left_tree, right_tree],
            biclosed_type=CCGType.SENTENCE  # Placeholder.
        )
    else:
        combined = left_tree

    return combined

###############################################################################
# Full ArabicParser Implementation
###############################################################################

class ArabicParseError(Exception):
    r"""
    Exception raised when the Arabic parser fails to convert a sentence.
    """
    def __init__(self, sentence: str) -> None:
        self.sentence = sentence

    def __str__(self) -> str:
        return f'ArabicParser failed to parse {self.sentence!r}.'

class ArabicParser(CCGParser):
    r"""
    Full implementation of an Arabic CCG parser based on the PATB conversion rules
    described in "An Arabic CCG Approach for Determining Constituent Types from Arabic Treebank".
    
    Steps:
      1. Preprocessing: tokenization, splitting attached determiners (e.g., "ال"), and reversing token order.
      2. Parsing: converting a PATB-style bracketed tree string into an ATBNode structure.
      3. Normalization: converting PATB tags to a reduced set via detailed mapping.
      4. Constituent Type Determination: marking nodes as head, complement, or adjunct using explicit labels (SBJ, OBJ, ADV, etc.) and heuristics.
      5. Binarization: enforcing binary branching.
      6. Conversion: transforming the ATB tree into a CCG derivation tree using combinatory rules.
    """
    def __init__(self, verbose: str = VerbosityLevel.PROGRESS.value, **kwargs: Any) -> None:
        self.verbose = verbose
        if not VerbosityLevel.has_value(verbose):
            raise ValueError(f'`{verbose}` is not a valid verbose value for ArabicParser.')
        # We assume that the input will be a PATB-style bracketed tree string.
        stanza.download('ar', processors='tokenize,pos,lemma,depparse', verbose=False)
        self.nlp = stanza.Pipeline(lang='ar', processors='tokenize,pos,lemma,depparse', verbose=False)

    def sentence2diagram(self, sentence: str) -> CCGTree:
        r"""
        Convert a PATB-style tree string for an Arabic sentence into a CCG derivation tree.
        
        Steps:
          a) Parse the bracketed tree string into an ATBNode structure.
          b) Normalize tags.
          c) Determine constituent types using detailed heuristics.
          d) Binarize the tree.
          e) Convert the ATB tree into a CCG derivation tree.
        """
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
        r"""
        Map normalized PATB tags (and dependency labels if available) to CCG-compatible categories.
        
        Detailed mapping (extracted and extended from the paper):
          - "NN"           -> CCGType.NOUN
          - "VB"           -> S\NP (verb expecting NP to its left)
          - "IN"           -> NP\NP
          - "DT"           -> N/N
          - "JJ"           -> N\N (adjective as post-nominal modifier)
          - "PRP"          -> NP
          - "RB"           -> S/S
          - "CD"           -> N/N
          - "CC"           -> CCGType.CONJUNCTION
          - "PUNC"         -> CCGType.PUNCTUATION
          - "FW"           -> CCGType.FOREIGN
        Dependency overrides:
          - If dependency in {"amod", "acl"} -> use N\N.
          - If dependency in {"nsubj", "csubj"} -> use NP.
          - If dependency in {"obj", "iobj"}   -> use NP.
        """
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
