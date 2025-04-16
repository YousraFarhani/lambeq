from __future__ import annotations
__all__ = ['ArabicParser', 'ArabicParseError']

import sys
import re
from collections.abc import Iterable
from typing import Any
from lambeq.text2diagram.ccg_parser import CCGParser
from lambeq.text2diagram.ccg_rule import CCGRule
from lambeq.text2diagram.ccg_tree import CCGTree  # Represents the hierarchical CCG structure
from lambeq.text2diagram.ccg_type import CCGType  # Represents CCG categories (e.g., NOUN, VERB)
from lambeq.core.globals import VerbosityLevel  # Controls the verbosity level of the parser
from lambeq.core.utils import (SentenceBatchType, tokenised_batch_type_check,
                               untokenised_batch_type_check)
from tqdm import tqdm
import stanza

# -------------------------------------------------------------------------------
# Patch CCGTree so that it has a draw() method.
# In the original parser this was present, so we ensure that our output CCGTree
# retains that method.
if not hasattr(CCGTree, 'draw'):
    def dummy_draw(self) -> None:
        """A simple recursive printer for a CCGTree."""
        def traverse(node: CCGTree, indent: str = "") -> str:
            # Display the node text if available; otherwise show the applied rule.
            text = node.text if node.text is not None else f"[{node.rule.name}]"
            s = indent + text
            if hasattr(node, 'children') and node.children:
                for child in node.children:
                    s += "\n" + traverse(child, indent + "  ")
            return s
        print(traverse(self))
    CCGTree.draw = dummy_draw
# -------------------------------------------------------------------------------

class ArabicParseError(Exception):
    def __init__(self, sentence: str) -> None:
        self.sentence = sentence

    def __str__(self) -> str:
        return f'ArabicParser failed to parse {self.sentence!r}.'

class ArabicParser(CCGParser):
    """
    CCG parser for Arabic using Stanza and conversion rules based on 
    "An Arabic CCG Approach for Determining Constituent Types from Arabic Treebank" (El‑Taher et al., 2014).
    
    This version has been enhanced to use a dependency-based conversion,
    with extended POS & dependency mappings, yet it returns the original
    CCGTree structure (which has a draw() method) so that output type remains
    the same as before.
    """
    
    def __init__(self, verbose: str = VerbosityLevel.PROGRESS.value, **kwargs: Any) -> None:
        self.verbose = verbose
        if not VerbosityLevel.has_value(verbose):
            raise ValueError(f'`{verbose}` is not a valid verbose value for ArabicParser.')
        
        # Download and initialize Stanza for Arabic with the required processors.
        stanza.download('ar', processors='tokenize,pos,lemma,depparse')
        self.nlp = stanza.Pipeline(lang='ar', processors='tokenize,pos,lemma,depparse')

    def sentences2trees(self,
                        sentences: SentenceBatchType,
                        tokenised: bool = False,
                        suppress_exceptions: bool = False,
                        verbose: str | None = None) -> list[CCGTree | None]:
        """Convert multiple sentences into a list of CCGTree objects."""
        if verbose is None:
            verbose = self.verbose
        if not VerbosityLevel.has_value(verbose):
            raise ValueError(f'`{verbose}` is not a valid verbose value for ArabicParser.')
        
        if tokenised:
            if not tokenised_batch_type_check(sentences):
                raise ValueError('`tokenised` set to True, but variable sentences does not have type List[List[str]].')
        else:
            if not untokenised_batch_type_check(sentences):
                raise ValueError('`tokenised` set to False, but variable sentences does not have type List[str].')
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
                    raise ArabicParseError(" ".join(sentence)) from e
        return trees

    def sentences2diagrams(self,
                           sentences: SentenceBatchType,
                           tokenised: bool = False,
                           planar: bool = False,
                           collapse_noun_phrases: bool = True,
                           suppress_exceptions: bool = False,
                           verbose: str | None = None) -> list[CCGTree | None]:
        """
        Convert multiple sentences into a list of diagram objects.
        Here, we simply use the underlying CCGTree (which includes draw())
        instead of converting to another Diagram type.
        """
        trees = self.sentences2trees(sentences,
                                     suppress_exceptions=suppress_exceptions,
                                     tokenised=tokenised,
                                     verbose=verbose)
        diagrams: list[CCGTree | None] = []
        if verbose is None:
            verbose = self.verbose
        if verbose == VerbosityLevel.TEXT.value:
            print('Turning parse trees to diagrams.', file=sys.stderr)
        for tree in tqdm(
                trees,
                desc='Parse trees to diagrams',
                leave=False,
                disable=verbose != VerbosityLevel.PROGRESS.value):
            if tree is not None:
                try:
                    # Instead of converting, we preserve the CCGTree output to maintain draw().
                    diagram = tree  
                except Exception as e:
                    if suppress_exceptions:
                        diagrams.append(None)
                    else:
                        raise e
                else:
                    diagrams.append(diagram)
            else:
                diagrams.append(None)
        return diagrams

    def sentence2diagram(self,
                         sentence: str | list[str],
                         tokenised: bool = False,
                         planar: bool = False,
                         collapse_noun_phrases: bool = True,
                         suppress_exceptions: bool = False) -> CCGTree | None:
        """
        Convert a single sentence into a diagram.
        The returned object is the CCGTree with a working draw() method.
        """
        if tokenised:
            if not isinstance(sentence, list):
                raise ValueError('`tokenised` set to True, but variable sentence does not have type list[str].')
            sent: list[str] = [str(token) for token in sentence]
            return self.sentences2diagrams(
                            [sent],
                            planar=planar,
                            collapse_noun_phrases=collapse_noun_phrases,
                            suppress_exceptions=suppress_exceptions,
                            tokenised=tokenised,
                            verbose=VerbosityLevel.SUPPRESS.value)[0]
        else:
            if not isinstance(sentence, str):
                raise ValueError('`tokenised` set to False, but variable sentence does not have type str.')
            return self.sentences2diagrams(
                            [sentence],
                            planar=planar,
                            collapse_noun_phrases=collapse_noun_phrases,
                            suppress_exceptions=suppress_exceptions,
                            tokenised=tokenised,
                            verbose=VerbosityLevel.SUPPRESS.value)[0]

    def preprocess(self, sentence: str) -> list[str]:
        """
        Normalize and tokenize Arabic text, splitting clitics like the definite article.
        """
        words = re.findall(r'\b\w+\b', sentence)
        processed_words = []
        for word in words:
            if word.startswith("ال") and len(word) > 2:
                processed_words.append("ال")
                processed_words.append(word[2:])
            else:
                processed_words.append(word)
        return processed_words

    def parse_atb(self, words: list[str]) -> list[dict]:
        """
        Parse the Arabic sentence using Stanza, extracting ATB token data.
        Each token dictionary contains the word, lemma, POS tag, head index and dependency relation.
        """
        sentence = " ".join(words)
        doc = self.nlp(sentence)
        parsed_data = []
        for sent in doc.sentences:
            for word in sent.words:
                parsed_data.append({
                    "word": word.text,
                    "lemma": word.lemma,
                    "pos": word.xpos,
                    "head": int(word.head),
                    "dep": word.deprel
                })
        return parsed_data

    def convert_to_ccg(self, atb_tree: list[dict]) -> CCGTree:
        """
        Convert the ATB parse (list of token dicts) into a CCGTree via a dependency-based approach.
        This builds a dependency tree from the ATB output and recursively constructs a binary CCG derivation.
        """
        # Create nodes: assume tokens are 1-indexed (adjust if necessary).
        nodes: dict[int, CCGTree] = {}
        for i, entry in enumerate(atb_tree):
            token_text = entry["word"][::-1]  # Reverse for correct Arabic rendering, if desired.
            ccg_category = self.map_pos_to_ccg(entry["pos"], entry["dep"])
            nodes[i + 1] = CCGTree(text=token_text, rule=CCGRule.LEXICAL, biclosed_type=ccg_category)
        
        # Build a mapping of head indices to child indices.
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
        
        def build_ccg_tree(idx: int) -> CCGTree:
            node = nodes[idx]
            child_indices = children_map.get(idx, [])
            if child_indices:
                child_indices.sort()
                child_trees = [build_ccg_tree(child) for child in child_indices]
                # Binarize children using iterative application of the FORWARD_APPLICATION rule.
                combined_children = child_trees[0]
                for child in child_trees[1:]:
                    combined_children = CCGTree(
                        text=None,
                        rule=CCGRule.FORWARD_APPLICATION,
                        children=[combined_children, child],
                        biclosed_type=CCGType.SENTENCE  # Adjust as appropriate.
                    )
                # Combine the current node with its (binarized) children.
                node = CCGTree(
                    text=None,
                    rule=CCGRule.FORWARD_APPLICATION,
                    children=[node, combined_children],
                    biclosed_type=CCGType.SENTENCE  # Adjust according to the constituent type.
                )
            return node
        
        return build_ccg_tree(root_index)

    def map_pos_to_ccg(self, atb_pos: str, dependency: str) -> CCGType:
        """
        Map ATB POS tags and dependency relations to CCG types.
        The mapping is extended based on heuristics from El‑Taher et al. (2014).
        """
        atb_to_ccg_map = {
            "NN": CCGType.NOUN,
            "NNS": CCGType.NOUN,
            "NNP": CCGType.NOUN_PHRASE,
            "NNPS": CCGType.NOUN_PHRASE,
            "VB": CCGType.SENTENCE.slash("\\", CCGType.NOUN_PHRASE),
            "VBD": CCGType.SENTENCE.slash("\\", CCGType.NOUN_PHRASE),
            "VBP": CCGType.SENTENCE.slash("\\", CCGType.NOUN_PHRASE),
            "IN": CCGType.NOUN_PHRASE.slash("\\", CCGType.NOUN_PHRASE),
            "DT": CCGType.NOUN.slash("/", CCGType.NOUN),
            "JJ": CCGType.NOUN.slash("/", CCGType.NOUN),
            "PRP": CCGType.NOUN_PHRASE,
            "CC": CCGType.CONJUNCTION,
            "RB": CCGType.SENTENCE.slash("/", CCGType.SENTENCE),
            "CD": CCGType.NOUN.slash("/", CCGType.NOUN),
            "UH": CCGType.CONJUNCTION,
            "PUNC": CCGType.NOUN,  # Default treatment for punctuation.
        }
        # Adjust based on dependency relations:
        if dependency in ["amod", "acl"]:
            return CCGType.NOUN.slash("/", CCGType.NOUN)
        elif dependency in ["nsubj", "csubj"]:
            return CCGType.NOUN_PHRASE
        elif dependency in ["obj", "iobj"]:
            return CCGType.NOUN_PHRASE
        return atb_to_ccg_map.get(atb_pos, CCGType.NOUN)

# -------------------------------------------------------------------------------
# Example usage:
if __name__ == "__main__":
    parser = ArabicParser()
    # Example Arabic sentence: "الولد قرأ الكتاب" (The boy read the book).
    diagram = parser.sentence2diagram("الولد قرأ الكتاب", tokenised=False)
    if diagram is not None:
        # Because we preserved the CCGTree as the output, draw() is available.
        diagram.draw()