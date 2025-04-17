from __future__ import annotations

__all__ = [
    'ArabicParser',
    'ArabicParseError',
]

import os, re, stanza, nltk
from nltk import Tree
from lambeq.text2diagram.ccg_parser import CCGParser
from lambeq.text2diagram.ccg_tree import CCGTree
from lambeq.text2diagram.ccg_rule import CCGRule
from lambeq.text2diagram.ccg_type import CCGType
from lambeq.core.globals import VerbosityLevel

###############################################################################
# ArabicParser – robust ATB‑to‑CCG converter (AG format)                     #
###############################################################################

class ArabicParseError(Exception):
    """Raised when sentence cannot be matched to any ATB tree."""
    def __init__(self, sentence: str):
        super().__init__(sentence)
        self.sentence = sentence
    def __str__(self):  # pragma: no cover
        return f"ArabicParser failed to parse {self.sentence!r}."

class ArabicParser(CCGParser):
    """Parse Arabic sentences to lambeq CCG diagrams using ATB constituency.

    Parameters
    ----------
    ag_txt_root : str
        Path to directory containing AG‑format *.txt files from the Arabic
        Treebank (ATB v3).
    verbose : str, default: 'progress'
        Lambeq verbosity level.
    """

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def __init__(self, ag_txt_root: str, *, verbose: str = VerbosityLevel.PROGRESS.value, **kwargs):
        super().__init__(verbose=verbose, **kwargs)
        if not VerbosityLevel.has_value(verbose):
            raise ValueError(f"Invalid verbosity: {verbose}")
        self.verbose = verbose

        # Optional Stanza pipeline (not used for constituency but handy for
        # future extensions)
        stanza.download('ar', processors='tokenize,pos,lemma,depparse')
        self.nlp = stanza.Pipeline(lang='ar', processors='tokenize,pos,lemma,depparse')

        # Load ATB constituency trees
        self.atb_trees: list[Tree] = []
        for fn in os.listdir(ag_txt_root):
            if fn.lower().endswith('.txt'):
                self._parse_ag_file(os.path.join(ag_txt_root, fn))
        if not self.atb_trees:
            raise ValueError('No ATB trees found in directory.')

    # ------------------------------------------------------------------
    # AG‑file reader
    # ------------------------------------------------------------------
    def _parse_ag_file(self, path: str):
        token_map, tag_map, buf = {}, {}, []
        with open(path, encoding='utf8', errors='ignore') as fh:
            for raw in fh:
                line = raw.strip()
                # Token lines (s:index · token · ...)
                if line.startswith('s:'):
                    seg = line.split('·')
                    idx = line.split(':', 1)[1].split('·', 1)[0]
                    token_map[f'W{idx}'] = seg[1].strip()
                    continue
                # Tag lines (t:index · TAG · ...)
                if line.startswith('t:'):
                    seg = line.split('·')
                    idx = line.split(':', 1)[1].split('·', 1)[0]
                    tag_map[f'W{idx}'] = seg[1].strip()
                    continue
                # Tree section
                if line.startswith('TREE:'):
                    buf.clear(); continue
                if line.startswith('(TOP') or buf:
                    buf.append(line)
                    if line.endswith('))'):
                        tree_str = ' '.join(buf)
                        # Substitute placeholders W#
                        for wid, tok in token_map.items():
                            tree_str = re.sub(rf'\b{wid}\b', self._expand_leaf(tok, tag_map.get(wid, 'UNK')), tree_str)
                        try:
                            self.atb_trees.append(Tree.fromstring(tree_str))
                        except ValueError:
                            pass
                        buf.clear()

    # ------------------------------------------------------------------
    # Leaf expansion helper (splits proclitic ال)
    # ------------------------------------------------------------------
    def _expand_leaf(self, token: str, tag: str) -> str:
        parts = tag.split('+')
        if parts[0] == 'DET' and token.startswith('ال') and len(token) > 2:
            det_leaf = '(DET ال)'
            rest_tag = '+'.join(parts[1:]) or 'NOUN'
            rest_leaf = f'({self._sanitize(rest_tag)} {token[2:]})'
            return f'{det_leaf} {rest_leaf}'
        return f'({self._sanitize(tag)} {token})'

    @staticmethod
    def _sanitize(tag: str) -> str:
        return re.sub(r'[()\s]', '_', tag)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def sentence2diagram(self, sentence: str, **kw):
        """Parse one sentence to a lambeq diagram."""
        return self.sentences2diagrams([sentence], **kw)[0]

    def sentences2diagrams(self, sentences, **kw):
        trees = self.sentences2trees(sentences, **kw)
        return [None if t is None else self.to_diagram(t) for t in trees]

    def sentences2trees(self, sentences, tokenised: bool = False, suppress_exceptions: bool = False, **kw):
        result = []
        for sent in sentences:
            try:
                tokens = sent if tokenised else self._tokenize(sent)
                atb_tree = self._find_tree(tokens)
                result.append(self._to_ccg(atb_tree))
            except Exception as e:
                if suppress_exceptions:
                    result.append(None)
                else:
                    raise ArabicParseError(sent) from e
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _tokenize(text: str):
        """Simple whitespace+punct tokeniser used for matching with ATB leaves."""
        return re.findall(r'\b\w+\b', text)

    def _find_tree(self, tokens):
        for t in self.atb_trees:
            if t.leaves() == tokens:
                return t
        raise ArabicParseError(' '.join(tokens))

    def _to_ccg(self, t: Tree) -> CCGTree:
        if len(t) == 1 and isinstance(t[0], str):
            return CCGTree(text=t[0], rule=CCGRule.LEXICAL, biclosed_type=self._map_pos(t.label()))
        children = [self._to_ccg(c) for c in t]
        node = children[0]
        for right in children[1:]:
            node = CCGTree(text=None, rule=CCGRule.FORWARD_APPLICATION, children=[node, right], biclosed_type=CCGType.SENTENCE)
        return node

    # ------------------------------------------------------------------ #
    # Public helper: retrieve raw constituency tree                       #
    # ------------------------------------------------------------------ #
    def get_constituency_tree(self, sentence_or_tokens, *, tokenised: bool = False):
        """Expose the original constituency *nltk.Tree* for a sentence.

        Parameters
        ----------
        sentence_or_tokens : str | list[str]
            Untokenised string (default) **or** a list of tokens if
            `tokenised=True`.
        tokenised : bool, default False
            Set to True when *sentence_or_tokens* is already a list of tokens.
        """
        toks = sentence_or_tokens if tokenised else self._tokenize(sentence_or_tokens)
        return self._find_tree(toks)

    # ------------------------------------------------------------------
    # POS→CCG mapping (covers all ATB families)
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------ #
    # Public helper: quick tokenisation                                   #
    # ------------------------------------------------------------------ #
    def get_tokens(self, sentence_or_tokens, *, tokenised: bool = False):
        """Return the token list the parser uses for matching.

        Parameters
        ----------
        sentence_or_tokens : str | list[str]
            Raw sentence (string) **or** list of tokens if `tokenised=True`.
        tokenised : bool, default False
            True when *sentence_or_tokens* is already tokenised.
        """
        return sentence_or_tokens if tokenised else self._tokenize(sentence_or_tokens)


    def _map_pos(self, tag: str) -> CCGType:
        c = tag.split('+')[0]
        if re.match(r'^(NOUN|NOUN_PROP|NOUN_NUM)', c):
            return CCGType.NOUN
        if 'PRON' in c:
            return CCGType.NOUN_PHRASE
        if 'DET' in c or c.startswith('ADJ'):
            return CCGType.NOUN.slash('/', CCGType.NOUN)
        if 'PREP' in c:
            return CCGType.NOUN_PHRASE.slash('\\', CCGType.NOUN_PHRASE)
        if 'REL_PRON' in c or 'SUB_CONJ' in c:
            return CCGType.NOUN_PHRASE.slash('\\', CCGType.SENTENCE)
        if 'FUT_PART' in c or c == 'PRT':
            return CCGType.SENTENCE.slash('/', CCGType.SENTENCE)
        if re.match(r'^(VB|IV|PV)', c):
            return CCGType.SENTENCE.slash('\\', CCGType.NOUN_PHRASE)
        if 'CONJ' in c or 'PUNC' in c:
            return CCGType.CONJUNCTION
        return CCGType.NOUN