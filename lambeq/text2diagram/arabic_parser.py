# arabic_ccg_parser.py
from __future__ import annotations

__all__ = ["ArabicParser", "ArabicParseError"]

import os, re, stanza, nltk
from nltk import Tree
from lambeq.text2diagram.ccg_parser import CCGParser
from lambeq.text2diagram.ccg_tree import CCGTree
from lambeq.text2diagram.ccg_rule import CCGRule
from lambeq.text2diagram.ccg_type import CCGType
from lambeq.core.globals import VerbosityLevel


class ArabicParseError(Exception):
    """Raised when a sentence cannot be matched to any ATB tree."""
    def __init__(self, sentence: str):
        super().__init__(sentence)
        self.sentence = sentence

    def __str__(self):
        return f"ArabicParser failed to parse {self.sentence!r}."


class ArabicParser(CCGParser):
    """Parse Arabic sentences to lambeq CCG diagrams using ATB constituency."""

    # ------------------------------------------------------------------ #
    # Initialisation                                                     #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        ag_txt_root: str,
        *,
        verbose: str = VerbosityLevel.PROGRESS.value,
        split_det: bool = True,            # ⇠ split الـ proclitic?
        **kwargs,
    ):
        super().__init__(verbose=verbose, **kwargs)

        if not VerbosityLevel.has_value(verbose):
            raise ValueError(f"Invalid verbosity: {verbose}")
        self.verbose   = verbose
        self.split_det = split_det

        # (Optional) Stanza pipeline – not used by default, but handy later
        stanza.download("ar", processors="tokenize,pos,lemma,depparse")
        self.nlp = stanza.Pipeline(lang="ar", processors="tokenize,pos,lemma,depparse")

        # Load ATB constituency trees
        self.atb_trees: list[Tree] = []
        for fn in os.listdir(ag_txt_root):
            if fn.lower().endswith(".txt"):
                self._parse_ag_file(os.path.join(ag_txt_root, fn))

        if not self.atb_trees:
            raise ValueError("No ATB trees found — check directory path.")

    # ------------------------------------------------------------------ #
    # Reading AG‑format files                                            #
    # ------------------------------------------------------------------ #
    def _parse_ag_file(self, path: str):
        token_map, tag_map, buf = {}, {}, []
        with open(path, encoding="utf8", errors="ignore") as fh:
            for raw in fh:
                line = raw.strip()

                # token lines  s:<idx> · TOKEN · …
                if line.startswith("s:"):
                    idx  = line.split(":", 1)[1].split("·", 1)[0]
                    tok  = line.split("·")[1].strip()
                    token_map[f"W{idx}"] = tok
                    continue

                # tag lines    t:<idx> · TAG   · …
                if line.startswith("t:"):
                    idx  = line.split(":", 1)[1].split("·", 1)[0]
                    tag  = line.split("·")[1].strip()
                    tag_map[f"W{idx}"] = tag
                    continue

                # Tree section
                if line.startswith("TREE:"):
                    buf.clear()
                    continue

                if line.startswith("(TOP") or buf:
                    buf.append(line)
                    if line.endswith("))"):                # end of tree
                        tree_str = " ".join(buf)
                        # Replace placeholders W# by (TAG token)
                        for wid, tok in token_map.items():
                            tag = tag_map.get(wid, "UNK")
                            tree_str = re.sub(
                                rf"\b{wid}\b",
                                self._expand_leaf(tok, tag),
                                tree_str,
                            )
                        try:
                            self.atb_trees.append(Tree.fromstring(tree_str))
                        except ValueError:                 # malformed → skip
                            pass
                        buf.clear()

    # ------------------------------------------------------------------ #
    # Leaf expansion helper                                              #
    # ------------------------------------------------------------------ #
    def _expand_leaf(self, token: str, tag: str) -> str:
        """Return an S‑expression fragment for a single word."""
        if (
            self.split_det
            and tag.split("+")[0] == "DET"
            and token.startswith("ال")
            and len(token) > 2
        ):
            det   = "(DET ال)"
            rest  = token[2:]
            rtag  = "+".join(tag.split("+")[1:]) or "NOUN"
            rest_leaf = f"({self._sanitize(rtag)} {rest})"
            return f"{det} {rest_leaf}"

        return f"({self._sanitize(tag)} {token})"

    @staticmethod
    def _sanitize(tag: str) -> str:
        """Make tag safe as an S‑expression label."""
        return re.sub(r"[()\\s]", "_", tag)

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def sentence2diagram(self, sentence: str, **kw):
        return self.sentences2diagrams([sentence], **kw)[0]

    def sentences2diagrams(self, sentences, **kw):
        trees = self.sentences2trees(sentences, **kw)
        return [None if t is None else self.to_diagram(t) for t in trees]

    def sentences2trees(
        self,
        sentences,
        *,
        tokenised: bool = False,
        suppress_exceptions: bool = False,
        **kw,
    ):
        out = []
        for sent in sentences:
            try:
                tokens = sent if tokenised else self._tokenize(sent)
                atb    = self._find_tree(tokens)
                out.append(self._to_ccg(atb))
            except Exception as e:
                if suppress_exceptions:
                    out.append(None)
                else:
                    raise ArabicParseError(sent) from e
        return out

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _tokenize(text: str):
        """Very simple whitespace / punctuation tokenizer."""
        return re.findall(r"\\b\\w+\\b", text)

    def _find_tree(self, tokens):
        for t in self.atb_trees:
            if t.leaves() == tokens:
                return t
        raise ArabicParseError(" ".join(tokens))

    def _to_ccg(self, t: Tree) -> CCGTree:
        if len(t) == 1 and isinstance(t[0], str):          # pre‑terminal
            return CCGTree(
                text=t[0],
                rule=CCGRule.LEXICAL,
                biclosed_type=self._map_pos(t.label()),
            )
        kids = [self._to_ccg(k) for k in t]
        node = kids[0]
        for r in kids[1:]:
            node = CCGTree(
                text=None,
                rule=CCGRule.FORWARD_APPLICATION,
                children=[node, r],
                biclosed_type=CCGType.SENTENCE,
            )
        return node

    # ------------------------------------------------------------------ #
    # POS  →  CCG mapping (covers all ATB families)                      #
    # ------------------------------------------------------------------ #
    def _map_pos(self, tag: str) -> CCGType:
        c = tag.split("+")[0]

        if re.match(r"^(NOUN|NOUN_PROP|NOUN_NUM)", c):
            return CCGType.NOUN
        if "PRON" in c:
            return CCGType.NOUN_PHRASE
        if "DET" in c or c.startswith("ADJ"):
            return CCGType.NOUN.slash("/", CCGType.NOUN)
        if "PREP" in c:
            return CCGType.NOUN_PHRASE.slash("\\\\", CCGType.NOUN_PHRASE)
        if "REL_PRON" in c or "SUB_CONJ" in c:
            return CCGType.NOUN_PHRASE.slash("\\\\", CCGType.SENTENCE)
        if "FUT_PART" in c or c == "PRT":
            return CCGType.SENTENCE.slash("/", CCGType.SENTENCE)
        if re.match(r"^(VB|IV|PV)", c):
            return CCGType.SENTENCE.slash("\\\\", CCGType.NOUN_PHRASE)
        if "CONJ" in c or "PUNC" in c:
            return CCGType.CONJUNCTION

        return CCGType.NOUN