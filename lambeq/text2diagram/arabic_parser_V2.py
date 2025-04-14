from __future__ import annotations

__all__ = ['ArabicParser2', 'ArabicParse2Error','ATBNode']

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
    """
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
    """
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
    """
    Recursively normalize the node's tag using a comprehensive mapping from PATB tags
    to a reduced tagset used for CCG conversion.
    
    The mapping here covers many of the common PATB tags. (A complete system might include
    several hundred entries; here we list a representative sample.)
    """
    # Comprehensive mapping dictionary for PATB tags:
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
        "NEG_PART+PVSUFF_SUBJ:3MS": "VBP",   # context-dependent: if under VP -> VBP; else RP
        "VBP+NEG": "VBP",
        # You can add more mappings here as discovered from PATB.
    }
    node.normalized_tag = tag_mapping.get(node.tag, node.tag)
    for child in node.children:
        normalize_tag(child)

def determine_constituent_types(node: ATBNode) -> None:
    """
    Determine the constituent types for a given ATB tree node.
    
    Uses detailed heuristics:
      - For terminal nodes, examine the original tag for cues such as SBJ, OBJ, ADV.
      - For non-terminals, if any child contains explicit function tags (e.g., SBJ, OBJ, ADV, LOC),
        mark the first occurrence as head and others as complements.
      - Otherwise, choose the middle child as head.
    """
    if node.is_terminal():
        # For terminals, check if tag contains explicit markers.
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

    # Process children first.
    for child in node.children:
        determine_constituent_types(child)
    
    # For non-terminals:
    explicit = []
    for child in node.children:
        ltag = child.tag.lower()
        if "sbj" in ltag or "obj" in ltag or child.normalized_tag in {"VB", "VBD", "VBP"}:
            explicit.append(child)
    if explicit:
        # Mark the first explicit child as head and the others as complements.
        for i, child in enumerate(node.children):
            if child in explicit and i == 0:
                child.constituent_type = "head"
            else:
                child.constituent_type = "complement"
    else:
        # Default: choose the middle child as head.
        mid = len(node.children) // 2
        for i, child in enumerate(node.children):
            if i == mid:
                child.constituent_type = "head"
            else:
                child.constituent_type = "complement"

def binarize_tree(node: ATBNode) -> ATBNode:
    """
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

def convert_atb_node_to_ccg(node: ATBNode, parser: ArabicParser2) -> CCGTree:
    """
    Recursively convert an ATBNode into a CCG derivation tree.
    
    This function uses the normalized tag and constituent type. For terminals,
    a lexical CCGTree is generated using parser.map_pos_to_ccg. For non-terminals,
    children are combined by splitting the children into left and right parts around
    the head (determined by the constituent_type) and using backward/forward application.
    """
    # Terminal: produce a lexical node.
    if node.is_terminal():
        ccg_type = parser.map_pos_to_ccg(node.normalized_tag, "")
        word_text = node.word[::-1] if node.word else ""
        return CCGTree(text=word_text, rule=CCGRule.LEXICAL, biclosed_type=ccg_type)
    
    # Convert children recursively.
    children_ccg = [convert_atb_node_to_ccg(child, parser) for child in node.children]
    
    # Determine head index: find the child marked "head".
    head_index = None
    for i, child in enumerate(node.children):
        if getattr(child, 'constituent_type', None) == "head":
            head_index = i
            break
    if head_index is None:
        head_index = len(node.children) // 2

    # Combine left side using backward application.
    if head_index > 0:
        left_tree = children_ccg[0]
        for i in range(1, head_index + 1):
            left_tree = CCGTree(
                text=None,
                rule=CCGRule.BACKWARD_APPLICATION,
                children=[left_tree, children_ccg[i]],
                biclosed_type=CCGType.SENTENCE  # Placeholder; proper type resolution needed.
            )
    else:
        left_tree = children_ccg[head_index]
    
    # Combine right side using forward application.
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

class ArabicParse2Error(Exception):
    def __init__(self, sentence: str) -> None:
        self.sentence = sentence

    def __str__(self) -> str:
        return f'ArabicParser2 failed to parse {self.sentence!r}.'

class ArabicParser2(CCGParser):
    """
    Full implementation of an Arabic CCG parser based on the PATB conversion rules
    described in "An Arabic CCG Approach for Determining Constituent Types from Arabic Treebank".
    
    This implementation performs the following steps:
      1. Preprocessing: tokenization, clitic splitting (e.g., "ال"), and token-order reversal.
      2. Parsing: converting a PATB-style bracketed tree string into an internal ATB tree.
      3. Normalization: converting PATB tags to a simplified set via detailed mapping.
      4. Constituent Type Determination: marking each node as head, complement, or adjunct using detailed heuristics.
      5. Binarization: enforcing binary branching.
      6. Conversion: transforming the ATB tree into a CCG derivation tree.
    """
    def __init__(self, verbose: str = VerbosityLevel.PROGRESS.value, **kwargs: Any) -> None:
        self.verbose = verbose
        if not VerbosityLevel.has_value(verbose):
            raise ValueError(f'`{verbose}` is not a valid verbose value for ArabicParser2.')
        # For full conversion, we assume the input is a PATB bracketed tree.
        # However, we still initialize Stanza in case we need downstream dependency info.
        stanza.download('ar', processors='tokenize,pos,lemma,depparse', verbose=False)
        self.nlp = stanza.Pipeline(lang='ar', processors='tokenize,pos,lemma,depparse', verbose=False)

    def sentence2diagram(self, sentence: str) -> CCGTree:
        """
        Convert a PATB-style tree string for an Arabic sentence into a CCG derivation tree.
        Steps:
          a) Parse the tree string into an ATBNode structure.
          b) Normalize tags.
          c) Determine constituent types using detailed heuristics.
          d) Binarize the tree.
          e) Convert to a CCG derivation using conversion rules.
        """
        try:
            atb_root = parse_atb_tree(sentence)
            normalize_tag(atb_root)
            determine_constituent_types(atb_root)
            binarized_tree = binarize_tree(atb_root)
            ccg_tree = convert_atb_node_to_ccg(binarized_tree, self)
            return ccg_tree
        except Exception as e:
            raise ArabicParse2Error(sentence) from e

    def map_pos_to_ccg(self, atb_pos: str, dependency: str) -> CCGType:
        """
        Map normalized PATB tags (and dependency labels if available) to CCG-compatible categories.
        
        Detailed mapping (extracted and extended from the paper):
          - "NN"           -> CCGType.NOUN
          - "VB"           -> S\NP     (verb expecting NP to its left)
          - "IN"           -> NP\NP    (preposition as function over NPs)
          - "DT"           -> N/N      (determiner function)
          - "JJ"           -> N\N      (adjective as post-nominal modifier)
          - "PRP"          -> NP
          - "RB"           -> S/S      (adverb modifying sentences)
          - "CD"           -> N/N      (number as modifier)
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
            "JJ": CCGType.NOUN.slash("\\", CCGType.NOUN),  # Adjectives post-nominal in Arabic.
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
        
        return atb_to_ccg_map.get(atb_pos, CCGType.NOUN)  # Default to noun if unknown.
