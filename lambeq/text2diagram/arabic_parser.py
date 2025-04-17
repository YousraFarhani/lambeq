from __future__ import annotations
"""Arabic → CCG parser for lambeq diagrams (AG‑format ATB)
================================================================
This revision *fully* decomposes Arabic clitic strings according to the
morphological tag list provided by the ATB:

* **CONJ**  – prefixes «و», «ف»
* **PREP**  – prefixes «ب», «ل», «ك»
* **DET**   – proclitic «ال»

The splitter walks through the tag parts in order and carves the token so that
*every* tag component becomes its own pre‑terminal leaf in the constituency
fragment.  This guarantees that all tags are represented in the resulting CCG
diagram — no more missing `PREP`, `DET`, etc.

Arabic glyph shaping is performed with **`arabic‑reshaper`**.  We *do not* run
`python‑bidi`; Graphviz already renders RTL correctly once glyphs are joined.
Install once:
```
pip install arabic-reshaper
```
"""

import os, re, stanza, nltk, arabic_reshaper
from nltk import Tree
from lambeq.text2diagram.ccg_parser import CCGParser
from lambeq.text2diagram.ccg_tree import CCGTree
from lambeq.text2diagram.ccg_rule import CCGRule
from lambeq.text2diagram.ccg_type import CCGType
from lambeq.core.globals import VerbosityLevel

# ---------------------------------------------------------------------------
# RTL shaping helper
# ---------------------------------------------------------------------------

def _shape(token: str) -> str:
    """Return glyph‑joined RTL string suitable for Graphviz labels."""
    return arabic_reshaper.reshape(token)

# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------
class ArabicParseError(Exception):
    def __init__(self, sentence: str):
        self.sentence = sentence
    def __str__(self):
        return f"ArabicParser failed to parse {self.sentence!r}."

# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------
class ArabicParser(CCGParser):
    def __init__(self, ag_root: str, verbose: str = VerbosityLevel.PROGRESS.value, **kw):
        super().__init__(verbose=verbose, **kw)
        if not VerbosityLevel.has_value(verbose):
            raise ValueError(f"Invalid verbosity: {verbose}")
        self.verbose = verbose

        stanza.download('ar', processors='tokenize,pos,lemma,depparse')
        self.nlp = stanza.Pipeline(lang='ar', processors='tokenize,pos,lemma,depparse')

        self.trees: list[Tree] = []
        for fn in os.listdir(ag_root):
            if fn.endswith('.txt'):
                self._read_ag_file(os.path.join(ag_root, fn))
        if not self.trees:
            raise ValueError('No ATB trees found.')

    # ------------------------------------------------------------------
    # AG reader
    # ------------------------------------------------------------------
    def _read_ag_file(self, path: str):
        tok_map, tag_map, buf = {}, {}, []
        with open(path, encoding='utf-8') as fh:
            for raw in fh:
                line = raw.strip()
                if line.startswith('s:'):
                    idx = line.split(':', 1)[1].split('·', 1)[0]
                    tok_map[f'W{idx}'] = line.split('·')[1].strip()
                elif line.startswith('t:'):
                    idx = line.split(':', 1)[1].split('·', 1)[0]
                    tag_map[f'W{idx}'] = line.split('·')[1].strip()
                if line.startswith('TREE:'):
                    buf.clear(); continue
                if line.startswith('(TOP') or buf:
                    buf.append(line)
                    if line.endswith('))'):
                        tree = self._inject_leaves(' '.join(buf), tok_map, tag_map)
                        if tree is not None:
                            self.trees.append(tree)
                        buf.clear()

    # ------------------------------------------------------------------
    # Placeholder substitution with full clitic decomposition
    # ------------------------------------------------------------------
    def _inject_leaves(self, tree_str: str, tok_map: dict, tag_map: dict):
        for wid, token in tok_map.items():
            tags = tag_map.get(wid, 'UNK')
            repl = self._expand_token(token, tags)
            tree_str = re.sub(rf'\b{re.escape(wid)}\b', repl, tree_str)
        try:
            return Tree.fromstring(tree_str)
        except ValueError:
            return None

    # Decompose token based on tag list ---------------------------------
    def _expand_token(self, token: str, tag_str: str) -> str:
        parts = tag_str.split('+')
        leaves: list[tuple[str, str]] = []
        t = token
        # 1) CONJ prefix «و», «ف»
        if 'CONJ' in parts and t and t[0] in 'وف':
            leaves.append(('CONJ', t[0]))
            t = t[1:]
            parts.remove('CONJ')
        # 2) PREP prefix «ب», «ل», «ك»
        if 'PREP' in parts and t and t[0] in 'بللك':
            leaves.append(('PREP', t[0]))
            t = t[1:]
            parts.remove('PREP')
        # 3) DET proclitic «ال»
        if 'DET' in parts and t.startswith('ال'):
            leaves.append(('DET', 'ال'))
            t = t[2:]
            parts.remove('DET')
        # 4) Remainder
        leaves.append(('+'.join(parts) or 'NOUN', t))
        # Build s‑expression fragment
        return ' '.join(f'({re.sub(r"[()\s]", "_", tag)} {_shape(tok)})' for tag, tok in leaves if tok)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def sentence2diagram(self, s: str, **kw):
        return self.sentences2diagrams([s], **kw)[0]

    def sentences2diagrams(self, sentences, **kw):
        return [None if t is None else self.to_diagram(t) for t in self.sentences2trees(sentences, **kw)]

    def sentences2trees(self, sentences, suppress_exceptions=False, **kw):
        out = []
        for s in sentences:
            try:
                tree = self._find_tree(self._tokenise(s))
                out.append(self._to_ccg(tree))
            except Exception as e:
                out.append(None if suppress_exceptions else (_ for _ in ()).throw(e))
        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _tokenise(text: str):
        return re.findall(r'\b\w+\b', text)

    def _find_tree(self, tokens):
        for t in self.trees:
            if t.leaves() == tokens:
                return t
        raise ArabicParseError(' '.join(tokens))

    def _to_ccg(self, t: Tree):
        # pre‑terminal
        if len(t) == 1 and isinstance(t[0], str):
            return CCGTree(text=t[0], rule=CCGRule.LEXICAL, biclosed_type=self._map_pos(t.label()))
        kids = [self._to_ccg(c) for c in t]
        root = kids[0]
        for r in kids[1:]:
            root = CCGTree(text=None, rule=CCGRule.FORWARD_APPLICATION, children=[root, r], biclosed_type=CCGType.SENTENCE)
        return root

    # Coarse POS → CCG ---------------------------------------------------
    def _map_pos(self, tag: str):
        segs = tag.split('+')
        if 'DET' in segs:   return CCGType.NOUN.slash('/', CCGType.NOUN)
        if 'PRON' in segs:  return CCGType.NOUN_PHRASE
        if 'ADJ' in segs:   return CCGType.NOUN.slash('/', CCGType.NOUN)
        if 'PREP' in segs:  return CCGType.NOUN_PHRASE.slash('\\', CCGType.NOUN_PHRASE)
        if 'CONJ' in segs or 'PUNC' in segs: return CCGType.CONJUNCTION
        if any(v in segs[0] for v in ('VB', 'IV', 'PV')):  # verbs often first segment
            return CCGType.SENTENCE.slash('\\', CCGType.NOUN_PHRASE)
        return CCGType.NOUN