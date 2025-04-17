from __future__ import annotations
import os
import re
import stanza
import nltk
from nltk import Tree
from lambeq.text2diagram.ccg_parser import CCGParser
from lambeq.text2diagram.ccg_tree import CCGTree
from lambeq.text2diagram.ccg_rule import CCGRule
from lambeq.text2diagram.ccg_type import CCGType
from lambeq.core.globals import VerbosityLevel

class ArabicParseError(Exception):
    def __init__(self, sentence: str) -> None:
        self.sentence = sentence
    def __str__(self) -> str:
        return f"ArabicParser failed to parse {self.sentence!r}."

class ArabicParser(CCGParser):
    """CCG parser for Arabic using bracketed Treebank TXT files and lambeq diagrams."""

    def __init__(self,
                 treebank_txt_root: str,
                 verbose: str = VerbosityLevel.PROGRESS.value,
                 **kwargs) -> None:
        super().__init__(verbose=verbose, **kwargs)
        if not VerbosityLevel.has_value(verbose):
            raise ValueError(f'`{verbose}` is not valid.')
        self.verbose = verbose

        # Download and initialize Stanza for morphological features
        stanza.download('ar', processors='tokenize,mwt,pos,lemma,depparse')
        self.nlp = stanza.Pipeline(lang='ar', processors='tokenize,pos,lemma,depparse')

        # Load all bracketed constituency trees from TXT files
        if not os.path.isdir(treebank_txt_root):
            raise ValueError(f"Treebank directory '{treebank_txt_root}' not found.")
        self.atb_trees: list[nltk.Tree] = []
        for fname in os.listdir(treebank_txt_root):
            if not fname.endswith('.txt'):
                continue
            path = os.path.join(treebank_txt_root, fname)
            with open(path, encoding='utf8') as f:
                for line in f:
                    line = line.strip()
                    # skip empty or non-tree lines
                    if not line or not line.startswith('('):
                        continue
                    try:
                        tree = Tree.fromstring(line)
                        self.atb_trees.append(tree)
                    except Exception:
                        # malformed line
                        continue
        if not self.atb_trees:
            raise ValueError("No valid bracketed trees found in the provided .txt files.")

    def sentences2trees(self,
                        sentences: list[str],
                        tokenised: bool = False,
                        suppress_exceptions: bool = False,
                        verbose: str | None = None) -> list[CCGTree | None]:
        """Convert Arabic sentences to CCG trees using bracketed Treebank data."""
        if verbose is None:
            verbose = self.verbose
        if not VerbosityLevel.has_value(verbose):
            raise ValueError(f'`{verbose}` is not valid.')

        trees: list[CCGTree] = []
        for sent in sentences:
            try:
                tokens = self._normalize(sent)
                atb_tree = self._find_matching_tree(tokens)
                ccg_tree = self._convert_constituency_to_ccg(atb_tree)
                trees.append(ccg_tree)
            except Exception as e:
                if suppress_exceptions:
                    trees.append(None)
                else:
                    raise ArabicParseError(sent) from e
        return trees

    def sentences2diagrams(self,
                            sentences: list[str],
                            tokenised: bool = False,
                            planar: bool = False,
                            collapse_noun_phrases: bool = True,
                            suppress_exceptions: bool = False,
                            verbose: str | None = None):
        """Parse to lambeq diagrams (with .draw()) in batch."""
        trees = self.sentences2trees(
            sentences,
            tokenised=tokenised,
            suppress_exceptions=suppress_exceptions,
            verbose=verbose
        )
        return [None if t is None else self.to_diagram(t, planar=planar, collapse_noun_phrases=collapse_noun_phrases) for t in trees]

    def sentence2diagram(self,
                          sentence: str,
                          tokenised: bool = False,
                          planar: bool = False,
                          collapse_noun_phrases: bool = True,
                          suppress_exceptions: bool = False,
                          verbose: str | None = None):
        """Parse single sentence to lambeq Diagram with .draw()."""
        return self.sentences2diagrams(
            [sentence], tokenised=tokenised,
            planar=planar,
            collapse_noun_phrases=collapse_noun_phrases,
            suppress_exceptions=suppress_exceptions,
            verbose=verbose
        )[0]

    def _normalize(self, sentence: str) -> list[str]:
        """Simple whitespace and punctuation-based tokenization."""
        return re.findall(r'\b\w+\b', sentence)

    def _find_matching_tree(self, tokens: list[str]) -> nltk.Tree:
        """Locate an ATB tree whose leaves match the token list."""
        for tree in self.atb_trees:
            if tree.leaves() == tokens:
                return tree
        raise ArabicParseError("No matching bracketed tree for tokens: " + " ".join(tokens))

    def _convert_constituency_to_ccg(self, tree: nltk.Tree) -> CCGTree:
        """Recursively convert an NLTK Tree to a lambeq CCGTree."""
        if isinstance(tree, str):
            raise ValueError("Unexpected string node encountered.")
        # Leaf preterminals
        if len(tree) == 1 and isinstance(tree[0], str):
            word = tree[0]
            pos = tree.label()
            cat = self._map_pos_to_ccg(pos)
            return CCGTree(text=word, rule=CCGRule.LEXICAL, biclosed_type=cat)
        # Internal: binarize children leftward
        children = [self._convert_constituency_to_ccg(c) for c in tree]
        left = children[0]
        for right in children[1:]:
            left = CCGTree(
                text=None,
                rule=CCGRule.FORWARD_APPLICATION,
                children=[left, right],
                biclosed_type=CCGType.SENTENCE
            )
        return left

    def _map_pos_to_ccg(self, pos: str) -> CCGType:
        """Map PTB-style tags to CCG categories (extend as needed)."""
        m = {
            'NN': CCGType.NOUN,
            'NNS': CCGType.NOUN,
            'VB': CCGType.SENTENCE.slash('\\', CCGType.NOUN_PHRASE),
            'VBD': CCGType.SENTENCE.slash('\\', CCGType.NOUN_PHRASE),
            'DT': CCGType.NOUN.slash('/', CCGType.NOUN),
            'JJ': CCGType.NOUN.slash('/', CCGType.NOUN),
            'IN': CCGType.NOUN_PHRASE.slash('\\', CCGType.NOUN_PHRASE),
        }
        return m.get(pos, CCGType.NOUN)