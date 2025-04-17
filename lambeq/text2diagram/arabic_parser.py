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
    """Arabic CCG Parser with AG-format ATB file support."""

    def __init__(self,
                 ag_txt_root: str,
                 verbose: str = VerbosityLevel.PROGRESS.value,
                 **kwargs) -> None:
        super().__init__(verbose=verbose, **kwargs)
        if not VerbosityLevel.has_value(verbose):
            raise ValueError(f'`{verbose}` is not valid.')
        self.verbose = verbose

        stanza.download('ar', processors='tokenize,pos,lemma,depparse')
        self.nlp = stanza.Pipeline(lang='ar', processors='tokenize,pos,lemma,depparse')

        self.atb_trees: list[nltk.Tree] = []
        for fname in os.listdir(ag_txt_root):
            if not fname.endswith('.txt'):
                continue
            self._parse_ag_file(os.path.join(ag_txt_root, fname))

    def _parse_ag_file(self, filepath: str):
        with open(filepath, encoding='utf-8') as f:
            token_map = {}
            tree_lines = []
            for line in f:
                line = line.strip()
                if line.startswith('s:'):
                    parts = line.split('·')
                    if len(parts) > 2:
                        index = parts[0].split(':')[-1].strip()
                        token = parts[1].strip()
                        token_map[f'W{index}'] = token
                elif line.startswith('TREE:'):
                    tree_lines = []  # reset
                elif line.startswith('(TOP'):
                    tree_lines.append(line)
                    tree_str = ' '.join(tree_lines)
                    for key, value in token_map.items():
                        tree_str = tree_str.replace(f' {key} ', f' {value} ')
                    try:
                        tree = Tree.fromstring(tree_str)
                        self.atb_trees.append(tree)
                    except Exception:
                        continue
                elif tree_lines:
                    tree_lines.append(line)

    def sentences2trees(self,
                        sentences: list[str],
                        tokenised: bool = False,
                        suppress_exceptions: bool = False,
                        verbose: str | None = None) -> list[CCGTree | None]:
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
        return self.sentences2diagrams(
            [sentence], tokenised=tokenised,
            planar=planar,
            collapse_noun_phrases=collapse_noun_phrases,
            suppress_exceptions=suppress_exceptions,
            verbose=verbose
        )[0]

    def _normalize(self, sentence: str) -> list[str]:
        return re.findall(r'\b\w+\b', sentence)

    def _find_matching_tree(self, tokens: list[str]) -> nltk.Tree:
        for tree in self.atb_trees:
            if tree.leaves() == tokens:
                return tree
        raise ArabicParseError("No matching bracketed tree for tokens: " + " ".join(tokens))

    def _convert_constituency_to_ccg(self, tree: nltk.Tree) -> CCGTree:
        if isinstance(tree, str):
            raise ValueError("Unexpected string node encountered.")
        if len(tree) == 1 and isinstance(tree[0], str):
            word = tree[0]
            pos = tree.label()
            cat = self._map_pos_to_ccg(pos)
            return CCGTree(text=word, rule=CCGRule.LEXICAL, biclosed_type=cat)
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