from __future__ import annotations
"""Arabic → CCG parser for lambeq diagrams (AG‑format ATB).

Key improvements in this version
--------------------------------
1. **Arabic reshaping + BiDi** — tokens are passed through `arabic‑reshaper` and
   `python‑bidi`, so diagram labels render right‑to‑left correctly.
2. **Proclitic splitting** — determiners (الـ) are split into a separate `DET`
   leaf so your diagram shows the expected lexical item.
3. **Automatic POS→CCG mapping** — uses the coarse POS from the ATB tag; no
   fixed dictionary required.

Install extras once:
```
pip install arabic-reshaper python-bidi
```
"""

import os, re, stanza, nltk, arabic_reshaper
from bidi.algorithm import get_display
from nltk import Tree
from lambeq.text2diagram.ccg_parser import CCGParser
from lambeq.text2diagram.ccg_tree import CCGTree
from lambeq.text2diagram.ccg_rule import CCGRule
from lambeq.text2diagram.ccg_type import CCGType
from lambeq.core.globals import VerbosityLevel

# ---------------------------------------------------------------------------
# Utilities for Arabic shaping
# ---------------------------------------------------------------------------

def _shape(token: str) -> str:
    """Return a visually correct RTL string for diagram labels."""
    return get_display(arabic_reshaper.reshape(token))

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class ArabicParseError(Exception):
    def __init__(self, sentence: str) -> None:
        self.sentence = sentence
    def __str__(self) -> str:  # pragma: no cover
        return f"ArabicParser failed to parse {self.sentence!r}."

# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------
class ArabicParser(CCGParser):
    """Parse Arabic sentences into CCG derivations using ATB constituency
    trees stored in AG format, then convert to lambeq diagrams.
    """

    def __init__(self, ag_txt_root: str, verbose: str = VerbosityLevel.PROGRESS.value, **kw):
        super().__init__(verbose=verbose, **kw)
        if not VerbosityLevel.has_value(verbose):
            raise ValueError(f"Invalid verbosity: {verbose}")
        self.verbose = verbose

        # (Optional) dependency pipeline if you need it elsewhere
        stanza.download('ar', processors='tokenize,pos,lemma,depparse')
        self.nlp = stanza.Pipeline(lang='ar', processors='tokenize,pos,lemma,depparse')

        # Load all AG‑format files
        self.atb_trees: list[Tree] = []
        for fn in os.listdir(ag_txt_root):
            if fn.endswith('.txt'):
                self._parse_ag_file(os.path.join(ag_txt_root, fn))
        if not self.atb_trees:
            raise ValueError('No ATB trees found in the provided directory.')

    # ------------------------------------------------------------------
    # AG‑file reader with DET splitting & token shaping
    # ------------------------------------------------------------------
    def _parse_ag_file(self, path: str):
        token_map, tag_map, buf = {}, {}, []
        with open(path, encoding='utf-8') as fh:
            for raw in fh:
                line = raw.strip()
                # token lines
                if line.startswith('s:'):
                    idx = line.split(':', 1)[1].split('·', 1)[0]
                    token = line.split('·')[1].strip()
                    token_map[f'W{idx}'] = token
                # tag lines
                elif line.startswith('t:'):
                    idx = line.split(':', 1)[1].split('·', 1)[0]
                    tag = line.split('·')[1].strip()
                    tag_map[f'W{idx}'] = tag
                # tree buffering
                if line.startswith('TREE:'):
                    buf.clear(); continue
                if line.startswith('(TOP') or buf:
                    buf.append(line)
                    if line.endswith('))'):
                        tree_str = ' '.join(buf)
                        # replace placeholders
                        for wid, tok in token_map.items():
                            replacement = self._morph_replacement(tok, tag_map.get(wid, 'UNK'))
                            tree_str = re.sub(rf'\b{re.escape(wid)}\b', replacement, tree_str)
                        try:
                            self.atb_trees.append(Tree.fromstring(tree_str))
                        except ValueError:
                            pass
                        buf.clear()

    # ------------------------------------------------------------------
    # Morphological expansion helper (splits DET + shapes tokens)
    # ------------------------------------------------------------------
    def _morph_replacement(self, token: str, tag: str) -> str:
        parts = tag.split('+')
        shaped_tok = _shape(token)
        if parts[0] == 'DET' and token.startswith('ال'):
            det_leaf = f'(DET {_shape("ال")})'
            rest_tok = token[2:] or token
            rest_leaf = f'({"+".join(parts[1:]) or "NOUN"} {_shape(rest_tok)})'
            return f'{det_leaf} {rest_leaf}'
        safe_tag = re.sub(r'[()\s]', '_', tag)
        return f'({safe_tag} {shaped_tok})'

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def sentence2diagram(self, s: str, **kw):
        return self.sentences2diagrams([s], **kw)[0]

    def sentences2diagrams(self, sentences, **kw):
        return [None if t is None else self.to_diagram(t) for t in self.sentences2trees(sentences, **kw)]

    def sentences2trees(self, sentences, suppress_exceptions=False, **kw):
        res = []
        for s in sentences:
            try:
                tree = self._find_matching_tree(self._norm(s))
                res.append(self._to_ccg(tree))
            except Exception as e:
                res.append(None if suppress_exceptions else (_ for _ in ()).throw(e))
        return res

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _norm(text: str):
        return re.findall(r'\b\w+\b', text)

    def _find_matching_tree(self, tokens):
        for t in self.atb_trees:
            if t.leaves() == tokens:
                return t
        raise ArabicParseError(' '.join(tokens))

    def _to_ccg(self, t: Tree) -> CCGTree:
        # pre‑terminal
        if len(t) == 1 and isinstance(t[0], str):
            return CCGTree(text=t[0], rule=CCGRule.LEXICAL, biclosed_type=self._map_pos(t.label()))
        children = [self._to_ccg(c) for c in t]
        node = children[0]
        for right in children[1:]:
            node = CCGTree(text=None, rule=CCGRule.FORWARD_APPLICATION, children=[node, right], biclosed_type=CCGType.SENTENCE)
        return node

    # ------------------------------------------------------------------
    # Heuristic POS → CCG mapping
    # ------------------------------------------------------------------
    def _map_pos(self, tag: str) -> CCGType:
        c = tag.split('+')[0]
        if 'DET' in c:  return CCGType.NOUN.slash('/', CCGType.NOUN)
        if 'NOUN' in c: return CCGType.NOUN
        if 'PRON' in c: return CCGType.NOUN_PHRASE
        if 'ADJ'  in c: return CCGType.NOUN.slash('/', CCGType.NOUN)
        if 'PREP' in c: return CCGType.NOUN_PHRASE.slash('\\', CCGType.NOUN_PHRASE)
        if any(v in c for v in ('VB', 'IV', 'PV')):
            return CCGType.SENTENCE.slash('\\', CCGType.NOUN_PHRASE)
        if 'CONJ' in c or 'PUNC' in c: return CCGType.CONJUNCTION
        return CCGType.NOUN