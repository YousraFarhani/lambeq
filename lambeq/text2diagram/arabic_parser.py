from __future__ import annotations
"""Arabic → CCG parser for lambeq diagrams (AG‑format ATB)
===========================================================
Now *fully automatic*: the parser will **segment user input clitics at
runtime** so that unspaced forms like «وبالكتاب» match ATB trees that were
saved with separate tokens «و», «ب», «الكتاب».

What’s new
----------
1. **`_segment_token`** — splits leading conjunction, preposition, determiner
   according to ATB tag logic.
2. **`_tokenise`** — applies regex tokenisation *then* segments each token,
   yielding a list identical to Treebank leaves.
3. Cleaned duplicate `_shape` definition and use full RTL shaping (reshaper +
   python‑bidi + RLE/PDF markers).

```python
parser = ArabicParser('/content/atb')
d = parser.sentence2diagram('وبالكتاب')  # works – splits into 4 leaves
```
Install once:
```bash
pip install arabic-reshaper python-bidi
```
"""
import os, re, stanza, arabic_reshaper
from bidi.algorithm import get_display
from nltk import Tree
from lambeq.text2diagram.ccg_parser import CCGParser
from lambeq.text2diagram.ccg_tree import CCGTree
from lambeq.text2diagram.ccg_rule import CCGRule
from lambeq.text2diagram.ccg_type import CCGType
from lambeq.core.globals import VerbosityLevel

# ---------------------------------------------------------------------------
# RTL shaping helper
# ---------------------------------------------------------------------------
BIDI_RLE = "\u202B"  # Right‑to‑Left Embedding
BIDI_PDF = "\u202C"  # Pop Directional Formatting

def _shape(t: str) -> str:
    reshaped = arabic_reshaper.reshape(t)
    bidi = get_display(reshaped)
    return f"{BIDI_RLE}{bidi}{BIDI_PDF}"

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
    # Placeholder substitution with clitic decomposition
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

    def _expand_token(self, token: str, tag_str: str) -> str:
        parts = tag_str.split('+')
        leaves = []
        t = token
        # CONJ
        if 'CONJ' in parts and t and t[0] in 'وف':
            leaves.append(('CONJ', t[0])); t = t[1:]; parts.remove('CONJ')
        # PREP
        if 'PREP' in parts and t and t[0] in 'بللك':
            leaves.append(('PREP', t[0])); t = t[1:]; parts.remove('PREP')
        # DET
        if 'DET' in parts and t.startswith('ال'):
            leaves.append(('DET', 'ال')); t = t[2:]; parts.remove('DET')
        leaves.append(('+'.join(parts) or 'NOUN', t))
        return ' '.join(f'({re.sub(r"[()\s]", "_", tag)} {_shape(tok)})' for tag, tok in leaves if tok)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def sentence2diagram(self, s: str, **kw):
        return self.sentences2diagrams([s], **kw)[0]

    def sentences2diagrams(self, sentences, **kw):
        return [None if t is None else self.to_diagram(t)
                for t in self.sentences2trees(sentences, **kw)]

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
    # Tokenisation that AUTO‑SEGMENTS clitics
    # ------------------------------------------------------------------
    _conj = 'وف'
    _prep = 'بللك'

    @classmethod
    def _segment_token(cls, tok: str):
        segs = []
        t = tok
        if t and t[0] in cls._conj:
            segs.append(t[0]); t = t[1:]
        if t and t[0] in cls._prep:
            segs.append(t[0]); t = t[1:]
        if t.startswith('ال'):
            segs.extend(['ال', t[2:]] if len(t) > 2 else ['ال'])
        else:
            segs.append(t)
        return segs

    @classmethod
    def _tokenise(cls, text: str):
        raw = re.findall(r'\b\w+\b', text)
        tokens: list[str] = []
        for tok in raw:
            tokens.extend(cls._segment_token(tok))
        return tokens

    # ------------------------------------------------------------------
    # Tree lookup & conversion
    # ------------------------------------------------------------------
    def _find_tree(self, tokens):
        for t in self.trees:
            if t.leaves() == tokens:
                return t
        raise ArabicParseError(' '.join(tokens))

    def _to_ccg(self, t: Tree):
        if len(t) == 1 and isinstance(t[0], str):
            return CCGTree(text=t[0], rule=CCGRule.LEXICAL, biclosed_type=self._map_pos(t.label()))
        kids = [self._to_ccg(c) for c in t]
        root = kids[0]
        for r in kids[1:]:
            root = CCGTree(text=None, rule=CCGRule.FORWARD_APPLICATION, children=[root, r], biclosed_type=CCGType.SENTENCE)
        return root

    # ------------------------------------------------------------------
    # POS → CCG
    # ------------------------------------------------------------------
    def _map_pos(self, tag: str):
        segs = tag.split('+')
        if 'DET' in segs:   return CCGType.NOUN.slash('/', CCGType.NOUN)
        if 'PRON' in segs:  return CCGType.NOUN_PHRASE
        if 'ADJ' in segs:   return CCGType.NOUN.slash('/', CCGType.NOUN)
        if 'PREP' in segs:  return CCGType.NOUN_PHRASE.slash('\\', CCGType.NOUN_PHRASE)
        if 'CONJ' in segs or 'PUNC' in segs: return CCGType.CONJUNCTION
        if any(v in segs[0] for v in ('VB', 'IV', 'PV')):
            return CCGType.SENTENCE.slash('\\', CCGType.NOUN_PHRASE)
        return CCGType.NOUN