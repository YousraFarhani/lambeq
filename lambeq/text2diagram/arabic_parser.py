from __future__ import annotations

__all__ = ['ArabicParser', 'ArabicParseError']

import stanza
import re
from collections.abc import Iterable
from typing import Any, Dict, List, Optional, Tuple
from lambeq.text2diagram.ccg_parser import CCGParser
from lambeq.text2diagram.ccg_rule import CCGRule
from lambeq.text2diagram.ccg_tree import CCGTree
from lambeq.text2diagram.ccg_type import CCGType
from lambeq.core.globals import VerbosityLevel
from lambeq.core.utils import (SentenceBatchType, tokenised_batch_type_check,
                             untokenised_batch_type_check)


class ArabicParseError(Exception):
    def __init__(self, sentence: str) -> None:
        self.sentence = sentence

    def __str__(self) -> str:
        return f'ArabicParser failed to parse {self.sentence!r}.'


class ArabicParser(CCGParser):
    """CCG parser for Arabic using Stanza and ATB dataset.
    
    This parser implements the approach described in the research paper:
    "An Arabic CCG Approach for Determining Constituent Types from Arabic Treebank"
    """

    # ATB POS tag to CCG category mapping based on the paper
    ATB_TO_CCG_MAP = {
        "NN": CCGType.NOUN,                                            # Noun -> N
        "VB": CCGType.SENTENCE.slash("\\", CCGType.NOUN_PHRASE),       # Verb -> S\NP
        "IN": CCGType.NOUN_PHRASE.slash("\\", CCGType.NOUN_PHRASE),    # Preposition -> NP\NP
        "DT": CCGType.NOUN.slash("/", CCGType.NOUN),                   # Determiner -> N/N
        "JJ": CCGType.NOUN.slash("/", CCGType.NOUN),                   # Adjective -> N/N
        "PRP": CCGType.NOUN_PHRASE,                                    # Pronoun -> NP
        "CC": CCGType.CONJUNCTION,                                     # Conjunction -> Conj
        "RB": CCGType.SENTENCE.slash("/", CCGType.SENTENCE),           # Adverb -> S/S
        "CD": CCGType.NOUN.slash("/", CCGType.NOUN),                   # Number -> N/N
        "UH": CCGType.CONJUNCTION,                                     # Interjection -> Conj
        "RP": CCGType.VERB_PHRASE.slash("/", CCGType.VERB_PHRASE),     # Particle -> VP/VP
        "WP": CCGType.NOUN_PHRASE,                                     # Wh-pronoun -> NP
        "PUNC": CCGType.PUNCTUATION,                                   # Punctuation
        "NNP": CCGType.NOUN_PHRASE,                                    # Proper noun -> NP
        "NNPS": CCGType.NOUN_PHRASE,                                   # Plural proper noun -> NP 
        "NNS": CCGType.NOUN,                                           # Plural noun -> N
        "VBD": CCGType.SENTENCE.slash("\\", CCGType.NOUN_PHRASE),      # Past tense verb -> S\NP
        "VBN": CCGType.VERB_PHRASE.slash("\\", CCGType.NOUN_PHRASE),   # Past participle -> VP\NP
        "VBP": CCGType.SENTENCE.slash("\\", CCGType.NOUN_PHRASE),      # Present tense verb -> S\NP
        "MD": CCGType.SENTENCE.slash("/", CCGType.VERB_PHRASE)         # Modal -> S/VP
    }
    
    # ATB dependency to CCG category mapping based on the paper
    DEP_TO_CCG_MAP = {
        "amod": CCGType.NOUN.slash("/", CCGType.NOUN),                 # Adjective modifier -> N/N
        "acl": CCGType.NOUN.slash("/", CCGType.NOUN),                  # Adjectival clause -> N/N
        "nsubj": CCGType.NOUN_PHRASE,                                  # Nominal subject -> NP
        "csubj": CCGType.NOUN_PHRASE,                                  # Clausal subject -> NP
        "obj": CCGType.NOUN_PHRASE,                                    # Object -> NP
        "iobj": CCGType.NOUN_PHRASE,                                   # Indirect object -> NP
        "obl": CCGType.NOUN_PHRASE,                                    # Oblique nominal -> NP
        "nmod": CCGType.NOUN.slash("\\", CCGType.NOUN),                # Nominal modifier -> N\N
        "advmod": CCGType.VERB_PHRASE.slash("/", CCGType.VERB_PHRASE), # Adverbial modifier -> VP/VP
        "mark": CCGType.CONJUNCTION,                                   # Marker -> Conj
        "det": CCGType.NOUN.slash("/", CCGType.NOUN),                  # Determiner -> N/N
        "case": CCGType.NOUN_PHRASE.slash("\\", CCGType.NOUN_PHRASE),  # Case marking -> NP\NP
        "root": CCGType.SENTENCE                                       # Root -> S
    }

    # Punctuation mapping from the paper
    PUNCTUATION_MAP = {
        ".": ".",
        "?": ".",
        "!": ".",
        ",": ",",
        ";": ":",
        ":": ":",
        "(": "-LRB-",
        ")": "-RRB-",
        "[": "-LRB-",
        "]": "-RRB-",
        "\"": "\"",
        "'": "''",
        """: "``",
        """: "''",
        "_": ":",
        "...": ":",
        "&": "SYM",
        "@": "SYM",
        "=": "SYM",
        "+": "SYM",
        "*": "SYM",
        "#": "#",
        "$": "$",
        "%": "NN",
    }

    def __init__(self, verbose: str = VerbosityLevel.PROGRESS.value, **kwargs: Any) -> None:
        """Initialize the ArabicParser with required NLP tools."""
        self.verbose = verbose
        
        # Validate the verbosity setting
        if not VerbosityLevel.has_value(verbose):
            raise ValueError(f'`{verbose}` is not a valid verbose value for ArabicParser.')
        
        # Initialize Stanza for dependency parsing with more processors
        # Using full pipeline for detailed morphological analysis
        try:
            stanza.download('ar', processors='tokenize,mwt,pos,lemma,depparse,ner')
            self.nlp = stanza.Pipeline(lang='ar', 
                                      processors='tokenize,mwt,pos,lemma,depparse,ner',
                                      tokenize_pretokenized=True)
        except Exception as e:
            print(f"Stanza initialization error: {e}")
            print("Falling back to basic pipeline...")
            stanza.download('ar', processors='tokenize,pos,lemma,depparse')
            self.nlp = stanza.Pipeline(lang='ar', processors='tokenize,pos,lemma,depparse')
    
    def sentences2trees(self,
                      sentences: SentenceBatchType,
                      tokenised: bool = False,
                      suppress_exceptions: bool = False,
                      verbose: str | None = None) -> list[CCGTree | None]:
        """Convert multiple Arabic sentences to CCG trees.
        
        Args:
            sentences: List of sentences or tokenized sentences
            tokenised: Whether the input is already tokenized
            suppress_exceptions: Whether to return None instead of raising exceptions
            verbose: Verbosity level
            
        Returns:
            List of CCG trees or None values for failed parses
        """
        if verbose is None:
            verbose = self.verbose
        if not VerbosityLevel.has_value(verbose):
            raise ValueError(f'`{verbose}` is not a valid verbose value for ArabicParser.')
        
        if tokenised:
            if not tokenised_batch_type_check(sentences):
                raise ValueError('`tokenised` set to `True`, but variable `sentences` does not have type `List[List[str]]`.')
        else:
            if not untokenised_batch_type_check(sentences):
                raise ValueError('`tokenised` set to `False`, but variable `sentences` does not have type `List[str]`.')
            sent_list: list[str] = [str(s) for s in sentences]
            # Apply preprocessing to each sentence
            sentences = [self.preprocess(sentence) for sentence in sent_list]
        
        trees: list[CCGTree | None] = []
        for i, sentence in enumerate(sentences):
            try:
                # Parse sentence using Arabic Treebank structure
                atb_tree = self.parse_atb(sentence)
                # Convert ATB parse to CCG structure
                ccg_tree = self.convert_to_ccg(atb_tree)
                trees.append(ccg_tree)
                
                if verbose == VerbosityLevel.TEXT.value:
                    print(f"Parsed sentence {i+1}/{len(sentences)}")
                
            except Exception as e:
                if suppress_exceptions:
                    trees.append(None)
                    if verbose == VerbosityLevel.TEXT.value:
                        print(f"Failed to parse sentence {i+1}: {' '.join(sentence) if isinstance(sentence, list) else sentence}")
                else:
                    sentence_text = ' '.join(sentence) if isinstance(sentence, list) else sentence
                    raise ArabicParseError(sentence_text) from e
        
        return trees
    
    def preprocess(self, sentence: str) -> list[str]:
        """Normalize and tokenize Arabic text with morphological awareness.
        
        This implements preprocessing described in the paper section 4.1-4.4.
        """
        # First, handle basic tokenization
        tokens = []
        
        # Simple regex-based tokenization - will be enhanced by Stanza later
        raw_tokens = re.findall(r'[\u0600-\u06FF\u0750-\u077F]+|[^\s\u0600-\u06FF\u0750-\u077F]+', sentence)
        
        for token in raw_tokens:
            # Handle basic Arabic morphological patterns
            if re.match(r'^[\u0600-\u06FF\u0750-\u077F]+$', token):  # If token is Arabic
                # Handle common clitics as described in the paper section 4.3
                # Process determiner "ال" (al)
                if token.startswith("ال") and len(token) > 2:
                    # Following the paper's determiner segmentation approach
                    tokens.append(token)  # Keep original for Stanza's MWT processor
                # Handle prepositions like ب (bi), ل (li), etc.
                elif token.startswith("ب") and len(token) > 1:
                    tokens.append(token)  # Keep for Stanza's MWT
                elif token.startswith("ل") and len(token) > 1:
                    tokens.append(token)  # Keep for Stanza's MWT
                elif token.startswith("و") and len(token) > 1:
                    tokens.append(token)  # Keep for Stanza's MWT
                else:
                    tokens.append(token)
            else:
                # Non-Arabic token
                tokens.append(token)
        
        # Process tokens to match ATB conventions as described in section 4.1 of the paper
        return tokens
    
    def normalize_tag(self, tag: str, word: str) -> str:
        """Normalize ATB tag to standard Penn Treebank tag.
        
        This implements the tag conversion described in section 4.2 of the paper.
        """
        # Handle special patterns
        if tag.startswith("NEG_PART+PVSUFF_SUBJ:3MS"):
            return "VBP"  # imperfect verb
        
        # Handle various NO_FUNC, NOTAG cases as described in the paper
        if tag == "NO_FUNC":
            if word == "و":  # waw conjunction
                return "CC"
            # Check if it's punctuation
            elif word in self.PUNCTUATION_MAP:
                return self.PUNCTUATION_MAP[word]
            else:
                return "NNP"  # Default to proper noun
        
        # Handle non-alphabetic and non-Arabic
        if tag in ["NON_ALPHABETIC", "NON_ARABIC", "NOTAG"]:
            if word in self.PUNCTUATION_MAP:
                return self.PUNCTUATION_MAP[word]
            elif word.isdigit():
                return "CD"  # Cardinal number
            else:
                return "FW"  # Foreign word
                
        # Handle punctuation
        if tag == "PUNC":
            if word in self.PUNCTUATION_MAP:
                return self.PUNCTUATION_MAP[word]
            return "PUNC"
        
        # Basic tag mappings
        tag_mapping = {
            "NOUN": "NN",
            "NOUN_PROP": "NNP",
            "NOUN_NUM": "CD",
            "ADJ": "JJ",
            "ADJ_COMP": "JJR",
            "ADV": "RB",
            "PRON": "PRP",
            "DEM_PRON": "DT",
            "REL_PRON": "WP",
            "INTERROG_PRON": "WP",
            "VERB_PERFECT": "VBD",
            "VERB_IMPERFECT": "VBP",
            "VERB_IMPERATIVE": "VB",
            "VERB_PASSIVE": "VBN",
            "PREP": "IN",
            "CONJ": "CC",
            "SUB_CONJ": "IN",
            "INTERJ": "UH",
            "PART_NEG": "RB",
            "PART_VERB": "RP",
            "PART_VOC": "UH",
            "PART_FOCUS": "RP",
            "PART_FUT": "MD",
            "DET": "DT"
        }
        
        # Return mapped tag or original if no mapping exists
        return tag_mapping.get(tag.split("+")[0], tag)
    
    def parse_atb(self, words: list[str]) -> list[dict]:
        """Parse Arabic sentence using ATB syntactic structures and Stanza.
        
        This implements the parsing approach described in sections 4 and 5 of the paper.
        """
        # Join words for Stanza processing
        text = " ".join(words)
        
        # Process with Stanza
        doc = self.nlp(text)
        parsed_data = []
        
        # Extract word features from Stanza parse results
        for sent in doc.sentences:
            for word in sent.words:
                # Normalize the POS tag
                normalized_tag = self.normalize_tag(word.xpos or word.pos, word.text)
                
                parsed_data.append({
                    "word": word.text,
                    "lemma": word.lemma or word.text,
                    "pos": normalized_tag,
                    "upos": word.pos,  # Universal POS
                    "xpos": word.xpos,  # Language-specific POS
                    "head": word.head,  # Syntactic head
                    "dep": word.deprel,  # Dependency relation
                    "feats": word.feats  # Morphological features
                })
        
        return parsed_data
    
    def determine_head(self, subtree: list[dict]) -> int:
        """Determine the head node of a subtree.
        
        This implements head-finding heuristics from section 5.1 of the paper.
        
        Args:
            subtree: List of nodes in the subtree
            
        Returns:
            Index of the head node
        """
        # Check for head-marking dependency relations
        for i, node in enumerate(subtree):
            if node["dep"] == "root":
                return i
        
        # Fallback head-finding heuristics based on POS hierarchy
        # Priority order for heads: Verb > Noun > Adjective > Others
        pos_priority = {
            "VBD": 1, "VBP": 1, "VB": 1,  # Verbs highest priority
            "NN": 2, "NNP": 2, "NNS": 2,  # Nouns second priority
            "JJ": 3,  # Adjectives third priority
        }
        
        # Find node with highest priority
        best_priority = float('inf')
        head_idx = 0
        
        for i, node in enumerate(subtree):
            priority = pos_priority.get(node["pos"], 10)  # Default low priority
            if priority < best_priority:
                best_priority = priority
                head_idx = i
        
        return head_idx
    
    def identify_complements(self, subtree: list[dict], head_idx: int) -> list[int]:
        """Identify complement nodes in the subtree.
        
        This implements section 5.2 of the paper for identifying complements.
        
        Args:
            subtree: List of nodes in the subtree
            head_idx: Index of the head node
            
        Returns:
            List of indices of complement nodes
        """
        complement_deps = ["nsubj", "csubj", "obj", "iobj", "ccomp", "xcomp", "nmod"]
        complements = []
        
        # Check for dependencies that typically indicate complements
        for i, node in enumerate(subtree):
            if i != head_idx:
                # If dependency indicates a complement relation to the head
                if node["dep"] in complement_deps:
                    complements.append(i)
                # Special case for "BNF" benefactive case (indicating second object)
                elif node["dep"] == "obl" and "Case=Ben" in (node["feats"] or ""):
                    complements.append(i)
        
        return complements
    
    def convert_to_ccg(self, atb_tree: list[dict]) -> CCGTree:
        """Convert ATB's parse tree into a CCG derivation.
        
        This creates a CCG derivation from an ATB-style parse tree
        as described in sections 5-6 of the paper.
        """
        if not atb_tree:
            raise ArabicParseError("Empty parse tree")
        
        # Step 1: Determine the head of the sentence
        head_idx = self.determine_head(atb_tree)
        head_node = atb_tree[head_idx]
        
        # Step 2: Identify complements (arguments) and adjuncts
        complement_indices = self.identify_complements(atb_tree, head_idx)
        
        # All other nodes are adjuncts
        adjunct_indices = [i for i in range(len(atb_tree)) 
                          if i != head_idx and i not in complement_indices]
        
        # Step 3: Create CCG nodes for all words
        nodes = []
        for i, entry in enumerate(atb_tree):
            word = entry["word"]  # Use original word without reversal
            pos = entry["pos"]
            dependency = entry["dep"]
            
            # Determine CCG category based on POS and dependency
            if dependency in self.DEP_TO_CCG_MAP:
                ccg_category = self.DEP_TO_CCG_MAP[dependency]
            elif pos in self.ATB_TO_CCG_MAP:
                ccg_category = self.ATB_TO_CCG_MAP[pos]
            else:
                # Default to noun if unknown
                ccg_category = CCGType.NOUN
            
            # Create lexical node
            nodes.append(CCGTree(text=word, rule=CCGRule.LEXICAL, 
                              biclosed_type=ccg_category))
        
        # Step 4: Build CCG tree from the nodes
        if len(nodes) == 1:
            # Single word sentence
            return nodes[0]
        
        # Create a binary tree following head-complement-adjunct structure
        remaining_nodes = list(range(len(nodes)))
        tree_nodes = [nodes[i] for i in remaining_nodes]
        
        # First combine head with complements
        while complement_indices:
            # Process one complement at a time
            comp_idx = complement_indices.pop(0)
            head_tree = nodes[head_idx]
            comp_tree = nodes[comp_idx]
            
            # Determine rule based on word order
            if comp_idx < head_idx:  # Complement is to the left of head
                parent = CCGTree(
                    text=None,
                    rule=CCGRule.BACKWARD_APPLICATION,
                    children=[comp_tree, head_tree], 
                    biclosed_type=head_tree.biclosed_type.result
                )
            else:  # Complement is to the right of head
                parent = CCGTree(
                    text=None,
                    rule=CCGRule.FORWARD_APPLICATION,
                    children=[head_tree, comp_tree],
                    biclosed_type=head_tree.biclosed_type.result
                )
            
            # Update head node
            nodes[head_idx] = parent
        
        # Then combine with adjuncts
        while adjunct_indices:
            # Process one adjunct at a time
            adj_idx = adjunct_indices.pop(0)
            head_tree = nodes[head_idx]
            adj_tree = nodes[adj_idx]
            
            # Determine rule based on word order
            if adj_idx < head_idx:  # Adjunct is to the left of head
                parent = CCGTree(
                    text=None,
                    rule=CCGRule.BACKWARD_APPLICATION,
                    children=[adj_tree, head_tree],
                    biclosed_type=head_tree.biclosed_type
                )
            else:  # Adjunct is to the right of head
                parent = CCGTree(
                    text=None,
                    rule=CCGRule.FORWARD_APPLICATION,
                    children=[head_tree, adj_tree],
                    biclosed_type=head_tree.biclosed_type
                )
            
            # Update head node
            nodes[head_idx] = parent
        
        # Return the final CCG tree (should be at the head index)
        return nodes[head_idx]
    
    def sentence2diagram(self, sentence: str) -> CCGTree:
        """Convert a single sentence to a CCG diagram.
        
        This ensures compatibility with the original API.
        
        Args:
            sentence: Input Arabic sentence
            
        Returns:
            CCG tree that can be used with the draw() function
        """
        trees = self.sentences2trees([sentence], suppress_exceptions=False)
        if not trees or trees[0] is None:
            raise ArabicParseError(sentence)
        return trees[0]
    
    def map_pos_to_ccg(self, atb_pos: str, dependency: str) -> CCGType:
        """Map Arabic Treebank POS tags & dependencies to CCG-compatible categories.
        
        This is maintained for backward compatibility.
        """
        # Check dependency first
        if dependency in self.DEP_TO_CCG_MAP:
            return self.DEP_TO_CCG_MAP[dependency]
        
        # Fall back to POS tag
        return self.ATB_TO_CCG_MAP.get(atb_pos, CCGType.NOUN)