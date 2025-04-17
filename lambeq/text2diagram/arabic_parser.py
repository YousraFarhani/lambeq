from __future__ import annotations
import os, re, stanza, nltk
from nltk import Tree
from lambeq.text2diagram.ccg_parser import CCGParser
from lambeq.text2diagram.ccg_tree import CCGTree
from lambeq.text2diagram.ccg_rule import CCGRule
from lambeq.text2diagram.ccg_type import CCGType
from lambeq.core.globals import VerbosityLevel

###############################################################################
# ArabicParser  –  AG‑format ATB  ➔  CCG diagrams with full tag coverage     #
###############################################################################
# Highlights
# ----------
# • Reads ATB "AG" files (token/pos/tree triple‑format) and reconstructs the
#   constituency trees with the *real* Arabic tokens and their full
#   morphological tags.
# • Proclitic splitter: separates determiner (الـ) from its host noun so that
#   diagrams show a DET followed by the noun phrase.
# • Comprehensive POS→CCG mapping: covers **all** coarse tag families found in
#   ATB v3 (NOUN, NOUN_PROP, NOUN_NUM, ADJ, DET, PRON, PREP, SUB_CONJ,
#   REL_PRON, FUT_PART, PRT, IV*, PV*, VB*, CONJ, PUNC, etc.).  Unknown tags
#   default to N (noun).
###############################################################################

class ArabicParseError(Exception):
    def __init__(self, sentence: str):
        self.sentence = sentence
    def __str__(self):  # pragma: no cover
        return f"ArabicParser failed to parse {self.sentence!r}."

class ArabicParser(CCGParser):
    """Arabic CCG parser for ATB (AG‑format).

    Parameters
    ----------
    ag_txt_root : str
        Directory containing ATB *.txt* files in AG format.
    verbose : str, optional
        Lambeq verbosity level.
    """

    # ---------------------------------------------------------------------
    # Initialisation
    # ---------------------------------------------------------------------
    def __init__(self, ag_txt_root: str, *, verbose: str = VerbosityLevel.PROGRESS.value, **kw):
        super().__init__(verbose=verbose, **kw)
        if not VerbosityLevel.has_value(verbose):
            raise ValueError(f"Invalid verbosity: {verbose}")
        self.verbose = verbose

        stanza.download('ar', processors='tokenize,pos,lemma,depparse')
        self.nlp = stanza.Pipeline(lang='ar', processors='tokenize,pos,lemma,depparse')

        self.atb_trees: list[Tree] = []
        for fn in os.listdir(ag_txt_root):
            if fn.endswith('.txt'):
                self._parse_ag_file(os.path.join(ag_txt_root, fn))
        if not self.atb_trees:
            raise ValueError('No ATB trees loaded – check path.')

    # ------------------------------------------------------------------
    # AG‑file reader (reconstruct constituency trees)
    # ------------------------------------------------------------------
    def _parse_ag_file(self, path: str):
        token_map, tag_map, buf = {}, {}, []
        with open(path, encoding='utf8', errors='ignore') as fh:
            for raw in fh:
                line = raw.strip()
                if line.startswith('s:'):
                    idx = line.split(':', 1)[1].split('·', 1)[0]
                    token = line.split('·')[1].strip()
                    token_map[f'W{idx}'] = token
                elif line.startswith('t:'):
                    idx = line.split(':', 1)[1].split('·', 1)[0]
                    tag = line.split('·')[1].strip()
                    tag_map[f'W{idx}'] = tag
                if line.startswith('TREE:'):
                    buf.clear(); continue
                if line.startswith('(TOP') or buf:
                    buf.append(line)
                    if line.endswith('))'):
                        tree_str = ' '.join(buf)
                        # Replace placeholders W# with (TAG token)
                        for wid, tok in token_map.items():
                            tree_str = re.sub(rf'\b{wid}\b', self._expand_leaf(tok, tag_map.get(wid, 'UNK')), tree_str)
                        try:
                            self.atb_trees.append(Tree.fromstring(tree_str))
                        except ValueError:
                            pass
                        buf.clear()

    # ------------------------------------------------------------------
    # Leaf expansion (handles proclitics like الـ)
    # ------------------------------------------------------------------
    def _expand_leaf(self, token: str, tag: str) -> str:
        parts = tag.split('+')
        # Split definite article proclitic if DET prefix and token starts with ال
        if parts[0] == 'DET' and token.startswith('ال') and len(token) > 2:
            det_leaf = '(DET ال)'
            rest = token[2:]
            rest_tag = '+'.join(parts[1:]) or 'NOUN'
            rest_leaf = f'({self._sanitize(rest_tag)} {rest})'
            return f'{det_leaf} {rest_leaf}'
        return f'({self._sanitize(tag)} {token})'

    @staticmethod
    def _sanitize(tag: str) -> str:
        """Remove parentheses/spaces so tags are valid tree labels."""
        return re.sub(r'[()\s]', '_', tag)

    # ------------------------------------------------------------------
    # Public API wrappers
    # ------------------------------------------------------------------
    def sentence2diagram(self, s: str, **kw):
        return self.sentences2diagrams([s], **kw)[0]

    def sentences2diagrams(self, sents, **kw):
        return [None if t is None else self.to_diagram(t) for t in self.sentences2trees(sents, **kw)]

    def sentences2trees(self, sents, suppress_exceptions=False, **kw):
        out = []
        for s in sents:
            try:
                out.append(self._to_ccg(self._find_tree(self._tokenize(s))))
            except Exception as e:
                if suppress_exceptions:
                    out.append(None)
                else:
                    raise ArabicParseError(s) from e
        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _tokenize(text: str):
        return re.findall(r'\b\w+\b', text)

    def _find_tree(self, tokens):
        for t in self.atb_trees:
            if t.leaves() == tokens:
                return t
        raise ArabicParseError(' '.join(tokens))

    def _to_ccg(self, t: Tree) -> CCGTree:
        if len(t) == 1 and isinstance(t[0], str):
            return CCGTree(text=t[0], rule=CCGRule.LEXICAL, biclosed_type=self._map_pos(t.label()))
        kids = [self._to_ccg(c) for c in t]
        node = kids[0]
        for r in kids[1:]:
            node = CCGTree(text=None, rule=CCGRule.FORWARD_APPLICATION, children=[node, r], biclosed_type=CCGType.SENTENCE)
        return node

    # ------------------------------------------------------------------
    # Coarse POS → CCG mapping (covers all ATB tag families)
    # ------------------------------------------------------------------
    def _map_pos(self, tag: str) -> CCGType:
        coarse = tag.split('+')[0]
        # Nouns & proper nouns & numerals
        if re.match(r'^(NOUN|NOUN_PROP|NOUN_NUM)', coarse):
            return CCGType.NOUN
        # Pronouns (independent or suffix)
        if 'PRON' in coarse:
            return CCGType.NOUN_PHRASE
        # Determiner (when not split) and adjectives → modifier N/N
        if 'DET' in coarse or coarse.startswith('ADJ'):
            return CCGType.NOUN.slash('/', CCGType.NOUN)
        # Prepositions
        if 'PREP' in coarse:
            return CCGType.NOUN_PHRASE.slash('\\', CCGType.NOUN_PHRASE)
        # Relative / subordinating conjunctions act like NP\S or S/S; treat as NP backslash S for now
        if 'REL_PRON' in coarse or 'SUB_CONJ' in coarse:
            return CCGType.NOUN_PHRASE.slash('\\', CCGType.SENTENCE)
        # Particles (future, negative, etc.) act as S/S
        if 'FUT_PART' in coarse or coarse == 'PRT':
            return CCGType.SENTENCE.slash('/', CCGType.SENTENCE)
        # Verbs (perfect/imperfect/passive) – S\NP
        if re.match(r'^(VB|IV|PV)', coarse):
            return CCGType.SENTENCE.slash('\\', CCGType.NOUN_PHRASE)
        # Conjunction & punctuation
        if 'CONJ' in coarse or 'PUNC' in coarse:
            return CCGType.CONJUNCTION
        # Fallback
        return CCGType.NOUN