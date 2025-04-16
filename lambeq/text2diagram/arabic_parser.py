from __future__ import annotations

__all__ = ['ArabicParser', 'ArabicParseError']

import stanza
import re
from collections.abc import Iterable
from typing import Any
from lambeq.text2diagram.ccg_parser import CCGParser
from lambeq.text2diagram.ccg_rule import CCGRule
from lambeq.text2diagram.ccg_tree import CCGTree  # Represents the hierarchical CCG structure
from lambeq.text2diagram.ccg_type import CCGType  # Represents CCG categories, such as NOUN, VERB
from lambeq.core.globals import VerbosityLevel  # Controls the verbosity level of the parser
from lambeq.core.utils import (SentenceBatchType, tokenised_batch_type_check,
                               untokenised_batch_type_check)  # Type hints for sentence batches


class ArabicParseError(Exception):
    def __init__(self, sentence: str) -> None:
        self.sentence = sentence

    def __str__(self) -> str:
        return f'ArabicParser failed to parse {self.sentence!r}.'


# Patch CCGTree with a dummy draw() method to avoid AttributeError.
if not hasattr(CCGTree, 'draw'):
    def dummy_draw(self) -> None:
        """A simple textual display of the tree, implemented recursively."""
        def recurse(node: CCGTree, indent: str = "") -> str:
            node_text = node.text if node.text is not None else f"[{node.rule.name}]"
            s = indent + node_text + "\n"
            for child in getattr(node, 'children', []):
                s += recurse(child, indent + "  ")
            return s
        print(recurse(self))
    CCGTree.draw = dummy_draw  # Monkey-patch the draw method


class ArabicParser(CCGParser):
    """CCG parser for Arabic using Stanza and the Arabic Treebank conversion rules proposed by El‑Taher et al. (2014)."""

    def __init__(self, verbose: str = VerbosityLevel.PROGRESS.value, **kwargs: Any) -> None:
        """Initialize the ArabicParser with required NLP tools."""
        self.verbose = verbose
        if not VerbosityLevel.has_value(verbose):
            raise ValueError(f'`{verbose}` is not a valid verbose value for ArabicParser.')
        
        # Initialize Stanza for dependency parsing.
        stanza.download('ar', processors='tokenize,pos,lemma,depparse')
        self.nlp = stanza.Pipeline(lang='ar', processors='tokenize,pos,lemma,depparse')

    def sentences2trees(
            self,
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
            # Preprocess each sentence: tokenize and, if needed, split clitics.
            sentences = [self.preprocess(sentence) for sentence in sent_list]
        
        trees: list[CCGTree | None] = []
        for sentence in sentences:
            try:
                atb_tree = self.parse_atb(sentence)
                ccg_tree = self.convert_to_ccg(atb_tree)
                trees.append(ccg_tree)
            except Exception as e:
                if suppress_exceptions:
                    trees.append(None)
                else:
                    # Join the sentence tokens back for the error message.
                    raise ArabicParseError(" ".join(sentence)) from e
        
        return trees

    def preprocess(self, sentence: str) -> list[str]:
        """Normalize and tokenize Arabic text.
        
        Splits attached definite articles and returns the token list.
        (Note: tokens are later reversed for dependency order if needed.)
        """
        # Find word tokens (adjust regex if needed for Arabic-specific punctuation)
        words = re.findall(r'\b\w+\b', sentence)
        processed_words = []
        for word in words:
            # If the word starts with the definite article "ال", split it.
            if word.startswith("ال") and len(word) > 2:
                processed_words.append("ال")
                processed_words.append(word[2:])
            else:
                processed_words.append(word)
        # Optionally, you might not need to reverse the word order now that the dependency
        # tree will determine hierarchical structure. (The original reversal was used for
        # rendering; adjust if necessary.)
        return processed_words

    def parse_atb(self, words: list[str]) -> list[dict]:
        """Parse the Arabic sentence using ATB (Arabic Treebank) syntactic structures.
        
        Returns a list of token dictionaries containing word, lemma, POS, head and dependency relation.
        """
        sentence = " ".join(words)
        doc = self.nlp(sentence)
        parsed_data = []
        for sent in doc.sentences:
            for word in sent.words:
                parsed_data.append({
                    "word": word.text,
                    "lemma": word.lemma,
                    "pos": word.xpos,      # ATB POS tags; may require normalization
                    "head": int(word.head),  # head index (0 indicates root)
                    "dep": word.deprel     # dependency relation
                })
        return parsed_data

    def convert_to_ccg(self, atb_tree: list[dict]) -> CCGTree:
        """Convert the ATB parse (list of dicts) into a CCG derivation using a dependency tree.
        
        This function builds a dependency tree from the parsed tokens and recursively constructs
        a binary CCG derivation. The head is determined based on the head indices from the ATB data.
        """
        # Create a dictionary of CCGTree nodes from the tokens.
        nodes: dict[int, CCGTree] = {}
        # Assume tokens are 1-indexed (as output by Stanza); adjust if necessary.
        for i, entry in enumerate(atb_tree):
            # Reverse word characters if needed for display.
            token_text = entry["word"][::-1]
            ccg_category = self.map_pos_to_ccg(entry["pos"], entry["dep"])
            nodes[i + 1] = CCGTree(text=token_text, rule=CCGRule.LEXICAL, biclosed_type=ccg_category)

        # Build a mapping from head index to list of its children indices.
        children_map = {i + 1: [] for i in range(len(atb_tree))}
        root_index = None
        for i, entry in enumerate(atb_tree):
            head = entry["head"]
            current_index = i + 1
            if head == 0:
                root_index = current_index
            else:
                children_map[head].append(current_index)
        
        if root_index is None:
            raise ValueError("No root found in dependency tree.")
        
        # Recursively build the CCG derivation.
        def build_ccg_tree(idx: int) -> CCGTree:
            node = nodes[idx]
            child_indices = children_map.get(idx, [])
            if child_indices:
                # Sort children by their original order (or by dependency relation if more appropriate).
                child_indices.sort()
                # Recursively build each child's subtree.
                child_trees = [build_ccg_tree(child) for child in child_indices]
                # Binarize children: combine them iteratively using FORWARD_APPLICATION.
                # (You might want to choose a different rule based on dependency labels.)
                combined_children = child_trees[0]
                for child in child_trees[1:]:
                    combined_children = CCGTree(
                        text=None,
                        rule=CCGRule.FORWARD_APPLICATION,
                        children=[combined_children, child],
                        biclosed_type=CCGType.SENTENCE  # This is a placeholder; adapt as needed.
                    )
                # Combine the current head with its (binarized) children.
                node = CCGTree(
                    text=None,
                    rule=CCGRule.FORWARD_APPLICATION,
                    children=[node, combined_children],
                    biclosed_type=CCGType.SENTENCE  # Again, adjust according to the constituent type.
                )
            return node
        
        return build_ccg_tree(root_index)

    def map_pos_to_ccg(self, atb_pos: str, dependency: str) -> CCGType:
        """Map ATB POS tags and dependency relations to CCG-compatible categories.
        
        This enhanced mapping includes additional POS tags and handles dependency roles
        (e.g. subject, object, modifier) based on the heuristics in El‑Taher et al. (2014).
        """
        # Extended mapping dictionary: add as many mappings as required.
        atb_to_ccg_map = {
            "NN": CCGType.NOUN,
            "NNS": CCGType.NOUN,
            "NNP": CCGType.NOUN_PHRASE,
            "NNPS": CCGType.NOUN_PHRASE,
            "VB": CCGType.SENTENCE.slash("\\", CCGType.NOUN_PHRASE),
            "VBD": CCGType.SENTENCE.slash("\\", CCGType.NOUN_PHRASE),
            "VBP": CCGType.SENTENCE.slash("\\", CCGType.NOUN_PHRASE),
            "IN": CCGType.NOUN_PHRASE.slash("\\", CCGType.NOUN_PHRASE),
            "DT": CCGType.NOUN.slash("/", CCGType.NOUN),  # For determiners as modifiers.
            "JJ": CCGType.NOUN.slash("/", CCGType.NOUN),
            "PRP": CCGType.NOUN_PHRASE,
            "CC": CCGType.CONJUNCTION,
            "RB": CCGType.SENTENCE.slash("/", CCGType.SENTENCE),
            "CD": CCGType.NOUN.slash("/", CCGType.NOUN),
            "UH": CCGType.CONJUNCTION,
            "PUNC": CCGType.NOUN,  # Alternatively, you might define a specific punctuation type.
            # Additional mappings could be added here (e.g. for NEG_PART, DET attachments, etc.)
        }
        
        # Heuristic adjustments based on dependency relation:
        if dependency in ["amod", "acl"]:
            # Adjective modifiers; represent as functions modifying a noun.
            return CCGType.NOUN.slash("/", CCGType.NOUN)
        elif dependency in ["nsubj", "csubj"]:
            # Subjects: treat as noun phrases.
            return CCGType.NOUN_PHRASE
        elif dependency in ["obj", "iobj"]:
            # Objects: treat as noun phrases.
            return CCGType.NOUN_PHRASE
        # You can add further dependency-based heuristics (e.g., for adjuncts vs. complements) here.
        
        # Default to using the mapping from the POS tag.
        return atb_to_ccg_map.get(atb_pos, CCGType.NOUN)

# Example usage:
# For an Arabic sentence like "الولد يقرا الكتاب":
#   preprocess splits "الولد" into ["ال", "ولد"], and the parser uses the dependency structure
#   to produce a CCG derivation. Any call to draw() on the resulting CCGTree now works because of
#   the dummy_draw patch.
