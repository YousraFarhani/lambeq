from __future__ import annotations

__all__ = ['ArabicParser', 'ArabicParseError']

import stanza
import re
from collections.abc import Iterable
from typing import Any
import types
from lambeq.text2diagram.ccg_parser import CCGParser
from lambeq.text2diagram.ccg_rule import CCGRule
from lambeq.text2diagram.ccg_tree import CCGTree  # Original CCGTree (unchanged)
from lambeq.text2diagram.ccg_type import CCGType  # Represents CCG categories
from lambeq.core.globals import VerbosityLevel  # Controls verbosity level
from lambeq.core.utils import (
    SentenceBatchType, tokenised_batch_type_check,
    untokenised_batch_type_check
)

##############################################################################
# Subclass CCGTree to include a draw() method.
##############################################################################

class MyCCGTree(CCGTree):
    def draw(self, indent: int = 0) -> None:
        """
        A simple recursive draw function for the CCGTree.
        This method prints the tree structure with indentation.
        """
        label = self.text if self.text is not None else str(self.rule)
        print("  " * indent + label)
        if self.children:
            for child in self.children:
                # If the child has its own draw, call it; otherwise, just print its text.
                if hasattr(child, "draw"):
                    child.draw(indent + 1)
                else:
                    child_label = child.text if child.text is not None else str(child.rule)
                    print("  " * (indent + 1) + child_label)

##############################################################################
# Exception class.
##############################################################################

class ArabicParseError(Exception):
    def __init__(self, sentence: str) -> None:
        self.sentence = sentence

    def __str__(self) -> str:
        return f"ArabicParser failed to parse {self.sentence!r}."

##############################################################################
# ArabicParser class.
##############################################################################

class ArabicParser(CCGParser):
    """
    Enhanced Arabic CCG parser based on Stanza and the methodology
    described in "An Arabic CCG Approach for Determining Constituent Types from Arabic Treebank".

    Features:
      - Preprocessing: normalization (including vowel removal) and segmentation.
      - Extended tag normalization and mapping to CCG types.
      - Simple constituent type determination.
      - Basic binary tree formation.
      - Provides sentence2diagram so a single sentence is converted to a diagram.

    All trees produced are instances of MyCCGTree (or will have draw() attached)
    so that you can call diagram.draw().
    """
    def __init__(self, verbose: str = VerbosityLevel.PROGRESS.value, **kwargs: Any) -> None:
        self.verbose = verbose
        if not VerbosityLevel.has_value(verbose):
            raise ValueError(f"`{verbose}` is not a valid verbose value for ArabicParser.")
        # Download and initialize the Stanza pipeline.
        stanza.download('ar', processors='tokenize,pos,lemma,depparse')
        self.nlp = stanza.Pipeline(lang='ar', processors='tokenize,pos,lemma,depparse')

    def sentences2trees(self,
                        sentences: SentenceBatchType,
                        tokenised: bool = False,
                        suppress_exceptions: bool = False,
                        verbose: str | None = None) -> list[MyCCGTree | None]:
        """
        Convert one or more Arabic sentences to CCG trees.
        Steps:
          1. Preprocess sentences (normalize, remove vowels, segment determiners).
          2. Parse tokens with Stanza for an ATB-like structure.
          3. Convert the ATB parse into a binary CCG derivation.
          4. Determine constituent types (assign head/complement annotations).
        """
        if verbose is None:
            verbose = self.verbose
        if not VerbosityLevel.has_value(verbose):
            raise ValueError(f"`{verbose}` is not a valid verbose value for ArabicParser.")
        if tokenised:
            if not tokenised_batch_type_check(sentences):
                raise ValueError("`tokenised` is True but sentences do not have type List[List[str]].")
        else:
            if not untokenised_batch_type_check(sentences):
                raise ValueError("`tokenised` is False but sentences do not have type List[str].")
            sentences = [self.preprocess(sentence) for sentence in sentences]
        
        trees: list[MyCCGTree] = []
        for sentence in sentences:
            try:
                atb_tree = self.parse_atb(sentence)
                ccg_tree = self.convert_to_ccg(atb_tree)
                ccg_tree = self.determine_constituent_types(ccg_tree)
                trees.append(ccg_tree)
            except Exception as e:
                if suppress_exceptions:
                    trees.append(None)
                else:
                    raise ArabicParseError(" ".join(sentence)) from e
        return trees

    def sentence2diagram(self, sentence: str, tokenised: bool = False,
                         suppress_exceptions: bool = False, verbose: str | None = None) -> MyCCGTree:
        """
        Convert a single Arabic sentence to a CCG diagram (tree).
        Wraps sentences2trees and returns the first tree.
        Also ensures the returned tree has a draw() method.
        """
        trees = self.sentences2trees([sentence],
                                       tokenised=tokenised,
                                       suppress_exceptions=suppress_exceptions,
                                       verbose=verbose)
        if not trees or trees[0] is None:
            raise ArabicParseError(sentence)
        tree = trees[0]
        # If the tree does not have a draw method, attach it dynamically.
        if not hasattr(tree, "draw"):
            tree.draw = types.MethodType(MyCCGTree.draw, tree)
        return tree

    def preprocess(self, sentence: str) -> list[str]:
        """
        Normalize and tokenize Arabic text:
          - Remove vowels/diacritics.
          - Basic word tokenization.
          - Segment attached determiners (e.g. "الكتاب" into "ال" and "كتاب").
          - Reverse token order if required.
        """
        sentence = self.remove_vowels(sentence)
        tokens = re.findall(r'\b\w+\b', sentence)
        processed_tokens = []
        for token in tokens:
            if token.startswith("ال") and len(token) > 2:
                processed_tokens.append("ال")
                processed_tokens.append(token[2:])
            else:
                processed_tokens.append(token)
        return processed_tokens[::-1]

    def remove_vowels(self, sentence: str) -> str:
        """
        Remove Arabic diacritics (vowel marks) from a sentence.
        """
        diacritics = re.compile(r'[\u064B-\u0652]')
        return diacritics.sub('', sentence)

    def parse_atb(self, tokens: list[str]) -> list[dict]:
        """
        Parse the list of tokens using Stanza to produce an ATB-like structure.
        Each token becomes a dictionary with keys: word, lemma, normalized POS,
        head, and dependency relation.
        """
        sentence = " ".join(tokens)
        doc = self.nlp(sentence)
        parsed_data = []
        for sent in doc.sentences:
            for word in sent.words:
                normalized_pos = self.normalize_tag(word.xpos, word.text)
                parsed_data.append({
                    "word": word.text,
                    "lemma": word.lemma,
                    "pos": normalized_pos,
                    "head": word.head,
                    "dep": word.deprel
                })
        return parsed_data

    def normalize_tag(self, atb_pos: str, word: str) -> str:
        """
        Normalize ATB tags to a compact set.
        Handles punctuation, various verb forms, determiners, adjectives,
        adverbs, and other special cases.
        """
        if re.fullmatch(r'[.,?!:;«»()\u060C]', word):
            return word
        if "VERB_PERFECT" in atb_pos:
            return "VBD"
        if "VERB_IMPERFECT" in atb_pos:
            return "VBP"
        if "PVSUFF_SUBJ" in atb_pos:
            if "NEG_PART" in atb_pos:
                return "VBP"
            return "VB"
        if atb_pos == "NO_FUNC":
            if word == "و":
                return "CC"
            return "NNP" if not re.fullmatch(r'[.,?!:;]', word) else word
        if atb_pos in ["NON_ALPHABETIC", "NON_ARABIC"]:
            if word.isnumeric():
                return "CD"
            return "FW"
        tag_map = {
            "NN": "NN",
            "NNS": "NN",
            "NNP": "NNP",
            "NNPS": "NNP",
            "VB": "VB",
            "VBD": "VBD",
            "VBP": "VBP",
            "VBZ": "VBZ",
            "VBN": "VBN",
            "VERB_IMPERFECT": "VBP",
            "IN": "IN",
            "DET": "DT",
            "DEM": "DT",
            "DT": "DT",
            "JJ": "JJ",
            "JJR": "JJ",
            "JJS": "JJ",
            "AD": "JJ",
            "ADV": "RB",
            "ADVP": "RB",
            "RB": "RB",
            "RBR": "RB",
            "RBS": "RB",
            "PRP": "PRP",
            "PRP$": "PRP$",
            "CC": "CC",
            "RP": "RP",
            "CD": "CD",
            "UH": "UH",
            "SYM": "SYM",
        }
        return tag_map.get(atb_pos, atb_pos)

    def convert_to_ccg(self, atb_tree: list[dict]) -> MyCCGTree:
        """
        Convert the ATB parse (list of token dictionaries) into a binary CCG derivation tree.
        Each token is mapped to a preliminary CCG type based on its normalized POS and dependency.
        The tree is constructed using MyCCGTree.
        """
        nodes = []
        for entry in atb_tree:
            word = entry["word"][::-1]  # Reverse for proper Arabic display.
            pos = entry["pos"]
            dependency = entry["dep"]
            ccg_category = self.map_pos_to_ccg(pos, dependency)
            nodes.append(MyCCGTree(text=word, rule=CCGRule.LEXICAL, biclosed_type=ccg_category))
        while len(nodes) > 1:
            left = nodes.pop(0)
            right = nodes.pop(0)
            parent = MyCCGTree(
                text=None,
                rule=CCGRule.FORWARD_APPLICATION,
                children=[left, right],
                biclosed_type=CCGType.SENTENCE
            )
            nodes.insert(0, parent)
        return nodes[0]

    def map_pos_to_ccg(self, norm_pos: str, dependency: str) -> CCGType:
        """
        Map normalized ATB POS tags and dependency relations to preliminary CCG types.
        This includes refinements for determiners, adjectives, adverbs, and verb forms.
        """
        atb_to_ccg = {
            "NN": CCGType.NOUN,
            "NNS": CCGType.NOUN,
            "NNP": CCGType.NOUN,
            "VB": CCGType.SENTENCE.slash("\\", CCGType.NOUN_PHRASE),
            "VBD": CCGType.SENTENCE.slash("\\", CCGType.NOUN_PHRASE),
            "VBP": CCGType.SENTENCE.slash("\\", CCGType.NOUN_PHRASE),
            "VBZ": CCGType.SENTENCE.slash("\\", CCGType.NOUN_PHRASE),
            "VBN": CCGType.SENTENCE.slash("\\", CCGType.NOUN_PHRASE),
            "VERB_IMPERFECT": CCGType.SENTENCE.slash("\\", CCGType.NOUN_PHRASE),
            "VERB_PERFECT": CCGType.SENTENCE.slash("\\", CCGType.NOUN_PHRASE),
            "IN": CCGType.NOUN_PHRASE.slash("\\", CCGType.NOUN_PHRASE),
            "DET": CCGType.NOUN.slash("/", CCGType.NOUN),
            "DEM": CCGType.NOUN.slash("/", CCGType.NOUN),
            "JJ": CCGType.NOUN.slash("/", CCGType.NOUN),
            "JJR": CCGType.NOUN.slash("/", CCGType.NOUN),
            "JJS": CCGType.NOUN.slash("/", CCGType.NOUN),
            "AD": CCGType.NOUN.slash("/", CCGType.NOUN),
            "RB": CCGType.SENTENCE.slash("/", CCGType.SENTENCE),
            "ADV": CCGType.SENTENCE.slash("/", CCGType.SENTENCE),
            "ADVP": CCGType.SENTENCE.slash("/", CCGType.SENTENCE),
            "PRP": CCGType.NOUN_PHRASE,
            "PRP$": CCGType.NOUN_PHRASE,
            "CC": CCGType.CONJUNCTION,
            "RP": CCGType.NOUN.slash("/", CCGType.NOUN),
            "CD": CCGType.NOUN.slash("/", CCGType.NOUN),
            "UH": CCGType.CONJUNCTION,
            "SYM": CCGType.NOUN,
            "FW": CCGType.NOUN,
        }
        if dependency in ["amod", "acl"]:
            return CCGType.NOUN.slash("/", CCGType.NOUN)
        if dependency in ["nsubj", "csubj", "root"]:
            return CCGType.NOUN_PHRASE
        if dependency in ["obj", "iobj"]:
            return CCGType.NOUN_PHRASE
        return atb_to_ccg.get(norm_pos, CCGType.NOUN)

    def determine_constituent_types(self, tree: MyCCGTree) -> MyCCGTree:
        """
        Simple heuristics: mark the first child of a node as the head, and the others as complements.
        This function recursively annotates the tree.
        """
        def assign_types(node: MyCCGTree) -> None:
            if node.children:
                for i, child in enumerate(node.children):
                    child.annotation = "head" if i == 0 else "complement"
                    assign_types(child)
        assign_types(tree)
        return tree

