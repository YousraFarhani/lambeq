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
from lambeq.core.utils import (
    SentenceBatchType, tokenised_batch_type_check,
    untokenised_batch_type_check
)  # Batch processing type hints and validators


class ArabicParseError(Exception):
    def __init__(self, sentence: str) -> None:
        self.sentence = sentence

    def __str__(self) -> str:
        return f"ArabicParser failed to parse {self.sentence!r}."


class ArabicParser(CCGParser):
    """
    Enhanced Arabic CCG parser based on Stanza and the methodology described in:
    "An Arabic CCG Approach for Determining Constituent Types from Arabic Treebank".

    This parser includes:
      - Preprocessing: normalization, vowel removal, and attached-determiner segmentation.
      - Tag normalization covering many ATB tags and special cases such as verb morphology,
        negation, determiners, adjectives, adverbs, particles, symbols, etc.
      - Mapping of normalized ATB tags and dependency relations to preliminary CCG types.
      - A rudimentary constituent-type determination (head versus complement).
      - A simple binary tree formation (to be extended with full binarization methods).
      
    Additionally, a `sentence2diagram` method is provided so that the user can
    directly convert a sentence to a diagram and draw it.
    """

    def __init__(self, verbose: str = VerbosityLevel.PROGRESS.value, **kwargs: Any) -> None:
        self.verbose = verbose
        if not VerbosityLevel.has_value(verbose):
            raise ValueError(f"`{verbose}` is not a valid verbose value for ArabicParser.")
        # Download and initialize the Arabic Stanza pipeline with necessary processors.
        stanza.download('ar', processors='tokenize,pos,lemma,depparse')
        self.nlp = stanza.Pipeline(lang='ar', processors='tokenize,pos,lemma,depparse')

    def sentences2trees(self,
                        sentences: SentenceBatchType,
                        tokenised: bool = False,
                        suppress_exceptions: bool = False,
                        verbose: str | None = None) -> list[CCGTree | None]:
        """
        Convert multiple Arabic sentences to CCG trees.
        Processing steps:
          1. Preprocess sentences (normalize, remove vowels, determiner segmentation).
          2. Parse the preprocessed tokens with Stanza for an ATB-like structure.
          3. Convert the ATB parse into a preliminary CCG derivation.
          4. Determine constituent types (head, complement, adjunct) via heuristics.
        """
        if verbose is None:
            verbose = self.verbose
        if not VerbosityLevel.has_value(verbose):
            raise ValueError(f"`{verbose}` is not a valid verbose value for ArabicParser.")
        if tokenised:
            if not tokenised_batch_type_check(sentences):
                raise ValueError('`tokenised` is True but sentences do not have type List[List[str]].')
        else:
            if not untokenised_batch_type_check(sentences):
                raise ValueError('`tokenised` is False but sentences do not have type List[str].')
            sentences = [self.preprocess(sentence) for sentence in sentences]

        trees: list[CCGTree] = []
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
                         suppress_exceptions: bool = False, verbose: str | None = None) -> CCGTree:
        """
        Convert a single sentence to a CCG diagram (tree).
        This helper method wraps sentences2trees so that a single sentence is processed
        and its corresponding diagram (CCGTree) is returned.
        """
        trees = self.sentences2trees([sentence], tokenised=tokenised,
                                      suppress_exceptions=suppress_exceptions, verbose=verbose)
        if not trees or trees[0] is None:
            raise ArabicParseError(sentence)
        return trees[0]

    def preprocess(self, sentence: str) -> list[str]:
        """
        Normalize and tokenize Arabic text:
         - Remove extraneous vowels/diacritics.
         - Apply basic tokenization.
         - Segment attached determiners (e.g., splitting "الكتاب" into "ال" and "كتاب").
         - Optionally reverse token order for display conventions.
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
        Remove Arabic diacritics (vowel marks) from the sentence.
        """
        diacritics = re.compile(r'[\u064B-\u0652]')
        return diacritics.sub('', sentence)

    def parse_atb(self, tokens: list[str]) -> list[dict]:
        """
        Parse tokens into an ATB-like structure using Stanza.
        Each token is represented as a dictionary with word, lemma, normalized POS, head index, and dependency relation.
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
        Normalize ATB tags to a more compact set.
        
        Covers:
          - Punctuation: Return the punctuation symbol if the word is punctuation.
          - Verb forms: Map VERB_PERFECT, VERB_IMPERFECT, PVSUFF_SUBJ, etc.
          - Determiners: DET, DEM.
          - Adjectives: AD, JJ, JJR, JJS.
          - Adverbs: ADV, ADVP, RB.
          - Particles: RP.
          - Symbols and foreign words.
          - Additional special cases.
        """
        # Check for punctuation symbols
        if re.fullmatch(r'[.,?!:;«»()\u060C]', word):
            return word

        # Special handling for verb forms
        if "VERB_PERFECT" in atb_pos:
            return "VBD"
        if "VERB_IMPERFECT" in atb_pos:
            return "VBP"
        if "PVSUFF_SUBJ" in atb_pos:
            if "NEG_PART" in atb_pos:
                return "VBP"
            return "VB"

        # NO_FUNC handling: if the token is "و", map to conjunction; else try punctuation or proper noun.
        if atb_pos == "NO_FUNC":
            if word == "و":
                return "CC"
            return "NNP" if not re.fullmatch(r'[.,?!:;]', word) else word

        # Handle non-alphabetic or non-Arabic tokens.
        if atb_pos in ["NON_ALPHABETIC", "NON_ARABIC"]:
            if word.isnumeric():
                return "CD"
            return "FW"

        # Extended mapping for common tags and additional cases.
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
            # Additional mappings can be added here as needed.
        }
        return tag_map.get(atb_pos, atb_pos)

    def convert_to_ccg(self, atb_tree: list[dict]) -> CCGTree:
        """
        Convert the ATB parse (list of token dictionaries) into a binary CCG derivation tree.
        Each token is assigned a preliminary CCG type based on its normalized POS and dependency.
        """
        nodes = []
        for entry in atb_tree:
            # Reverse the word text if needed for display.
            word = entry["word"][::-1]
            pos = entry["pos"]
            dependency = entry["dep"]
            ccg_category = self.map_pos_to_ccg(pos, dependency)
            nodes.append(CCGTree(text=word, rule=CCGRule.LEXICAL, biclosed_type=ccg_category))
        # Combine nodes pairwise using a simple forward application.
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

    def map_pos_to_ccg(self, norm_pos: str, dependency: str) -> CCGType:
        """
        Map normalized ATB POS tags and dependency relations to preliminary CCG types.
        
        Extended to cover additional cases, including:
          - Determiners (DET, DEM) as functions from N to N.
          - Adjectives (JJ, AD) as modifiers.
          - Adverbs (ADV, ADVP, RB) producing S/S.
          - Various verb forms mapping to S\NP.
          - Particles, symbols, and foreign words.
        """
        atb_to_ccg = {
            "NN": CCGType.NOUN,
            "NNS": CCGType.NOUN,
            "NNP": CCGType.NOUN,  # Optionally, a specialized proper noun type may be used.
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
            "FW": CCGType.NOUN  # Fallback for foreign words.
            # More mappings can be added here as needed.
        }
        # Additional dependency-based refinements
        if dependency in ["amod", "acl"]:
            return CCGType.NOUN.slash("/", CCGType.NOUN)
        if dependency in ["nsubj", "csubj", "root"]:
            return CCGType.NOUN_PHRASE
        if dependency in ["obj", "iobj"]:
            return CCGType.NOUN_PHRASE
        return atb_to_ccg.get(norm_pos, CCGType.NOUN)

    def determine_constituent_types(self, tree: CCGTree) -> CCGTree:
        """
        Determine constituent types (head, complement, adjunct) in the binary CCG tree.
        A simple heuristic marks the first child as the head and all others as complements.
        Extend this function with further heuristics for adjuncts and complex constructions.
        """
        def assign_types(node: CCGTree) -> None:
            if node.children:
                for i, child in enumerate(node.children):
                    child.annotation = "head" if i == 0 else "complement"
                    assign_types(child)
        assign_types(tree)
        return tree


# Example usage:
if __name__ == '__main__':
    # Sample Arabic sentence: "عذاب اليم" (for example)
    sentences = [
        "عذاب اليم"
    ]
    parser = ArabicParser()
    
    # Convert a single sentence to a diagram using the new sentence2diagram helper
    diagram = parser.sentence2diagram("عذاب اليم")
    diagram.draw()