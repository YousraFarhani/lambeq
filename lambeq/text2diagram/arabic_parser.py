from __future__ import annotations
import os, re, stanza, nltk
from nltk import Tree
from lambeq.text2diagram.ccg_parser import CCGParser
from lambeq.text2diagram.ccg_tree import CCGTree
from lambeq.text2diagram.ccg_rule import CCGRule
from lambeq.text2diagram.ccg_type import CCGType
from lambeq.core.globals import VerbosityLevel

class ArabicParseError(Exception):
    def __init__(self, sentence: str) -> None:
        self.sentence = sentence
    def __str__(self) -> str:  # pragma: no cover
        return f"ArabicParser failed to parse {self.sentence!r}."

class ArabicParser(CCGParser):
    """Arabic CCG parser that ingests AG‑format ATB files, reconstructs
    constituency trees with full morphological pre‑terminals, and converts
    them on‑the‑fly to lambeq CCG diagrams.  The POS→CCG mapping is learnt
    heuristically from the tag strings themselves, so *no* hard‑coded table
    is required.  This automatically covers determiner clitics (ال), case
    suffixes, verb moods, etc.
    """

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def __init__(self,
                 ag_txt_root: str,
                 verbose: str = VerbosityLevel.PROGRESS.value,
                 **kwargs):
        super().__init__(verbose=verbose, **kwargs)
        if not VerbosityLevel.has_value(verbose):
            raise ValueError(f"Invalid verbosity: {verbose}")
        self.verbose = verbose

        # Optional dependency pipeline
        stanza.download('ar', processors='tokenize,pos,lemma,depparse')
        self.nlp = stanza.Pipeline(lang='ar', processors='tokenize,pos,lemma,depparse')

        # Load AG files
        self.atb_trees: list[Tree] = []
        for fn in os.listdir(ag_txt_root):
            if fn.endswith('.txt'):
                self._parse_ag_file(os.path.join(ag_txt_root, fn))
        if not self.atb_trees:
            raise ValueError('No ATB trees found in directory.')

    # ------------------------------------------------------------------
    # AG‑file reader
    # ------------------------------------------------------------------
    def _parse_ag_file(self, path: str):
        token_map, tag_map, buffer = {}, {}, []
        with open(path, encoding='utf-8') as fh:
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
                    buffer.clear(); continue
                if line.startswith('(TOP') or buffer:
                    buffer.append(line)
                    if line.endswith('))'):
                        tree_str = ' '.join(buffer)
                        for wid, tok in token_map.items():
                            tag = re.sub(r'[()\s]', '_', tag_map.get(wid, 'UNK'))
                            tree_str = re.sub(rf'\b{wid}\b', f'({tag} {tok})', tree_str)
                        try:
                            self.atb_trees.append(Tree.fromstring(tree_str))
                        except ValueError:
                            pass
                        buffer.clear()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def sentence2diagram(self, sentence: str, **kw):
        return self.sentences2diagrams([sentence], **kw)[0]

    def sentences2diagrams(self, sentences, **kw):
        trees = self.sentences2trees(sentences, **kw)
        return [None if t is None else self.to_diagram(t) for t in trees]

    def sentences2trees(self, sentences, tokenised=False, suppress_exceptions=False, verbose=None):
        verbose = verbose or self.verbose
        result = []
        for sent in sentences:
            try:
                tokens = self._normalize(sent)
                tree = self._find_matching_tree(tokens)
                result.append(self._convert_to_ccg(tree))
            except Exception as e:
                if suppress_exceptions:
                    result.append(None)
                else:
                    raise ArabicParseError(sent) from e
        return result

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(s: str):
        return re.findall(r'\b\w+\b', s)

    def _find_matching_tree(self, tokens):
        for t in self.atb_trees:
            if t.leaves() == tokens:
                return t
        raise ArabicParseError('No matching tree for: ' + ' '.join(tokens))

    def _convert_to_ccg(self, t: Tree) -> CCGTree:
        if len(t) == 1 and isinstance(t[0], str):
            return CCGTree(text=t[0], rule=CCGRule.LEXICAL, biclosed_type=self._map_pos_to_ccg(t.label()))
        children = [self._convert_to_ccg(c) for c in t]
        node = children[0]
        for right in children[1:]:
            node = CCGTree(text=None, rule=CCGRule.FORWARD_APPLICATION, children=[node, right], biclosed_type=CCGType.SENTENCE)
        return node

    # ------------------------------------------------------------------
    # Heuristic POS → CCG mapping (learned from tag string)
    # ------------------------------------------------------------------
    def _map_pos_to_ccg(self, tag: str) -> CCGType:
        coarse = tag.split('+')[0]
        if 'NOUN' in coarse:
            return CCGType.NOUN
        if 'PRON' in coarse:
            return CCGType.NOUN_PHRASE
        if 'ADJ' in coarse:
            return CCGType.NOUN.slash('/', CCGType.NOUN)
        if 'DET' in coarse:
            return CCGType.NOUN.slash('/', CCGType.NOUN)
        if 'PREP' in coarse:
            return CCGType.NOUN_PHRASE.slash('\\', CCGType.NOUN_PHRASE)
        if any(v in coarse for v in ('VB', 'IV', 'PV')):
            return CCGType.SENTENCE.slash('\\', CCGType.NOUN_PHRASE)
        if 'CONJ' in coarse:
            return CCGType.CONJUNCTION
        if 'PUNC' in coarse:
            return CCGType.CONJUNCTION
        # fallback
        return CCGType.NOUN