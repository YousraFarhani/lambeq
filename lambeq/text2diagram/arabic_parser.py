from lambeq.text2diagram.ccg_parser import CCGParser
import stanza  # Stanford NLP for Arabic
from camel_tools.utils.normalize import normalize_unicode
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer

class ATBParser(CCGParser):
    """
    Arabic Treebank (ATB) Parser for Quantum NLP with Lambeq.
    Converts Arabic syntactic structures from ATB into Combinatory Categorial Grammar (CCG)
    to be used with Lambeq's quantum NLP framework.
    """

    def __init__(self):
        super().__init__()

        # Initialize CamelTools Analyzer
        self.analyzer = Analyzer(MorphologyDB.builtin_db())

        # Load Stanford NLP for syntactic parsing
        stanza.download('ar', processors='tokenize,pos,lemma,depparse')
        self.nlp = stanza.Pipeline(lang='ar', processors='tokenize,pos,lemma,depparse')

    def sentence2diagram(self, sentence: str):
        """
        Convert an Arabic sentence from ATB into a CCG-compatible derivation.
        """
        # Step 1: Preprocess Arabic Text
        words = self.preprocess(sentence)

        # Step 2: Parse Sentence with ATB-compatible NLP
        atb_parse_tree = self.parse_atb(words)

        # Step 3: Convert ATB Treebank output into CCG derivations
        ccg_tree = self.convert_to_ccg(atb_parse_tree)

        return ccg_tree

    def preprocess(self, sentence: str):
        """
        Normalize and tokenize Arabic text for ATB parsing.
        """
        normalized_text = normalize_unicode(sentence)  # Normalize Unicode encoding
        tokens = simple_word_tokenize(normalized_text)  # Tokenize Arabic words
        return tokens

    def parse_atb(self, words):
        """
        Parse Arabic sentence using ATB syntactic structures.
        """
        sentence = " ".join(words)
        doc = self.nlp(sentence)

        parsed_data = []
        for sent in doc.sentences:
            for word in sent.words:
                parsed_data.append({
                    "word": word.text,
                    "lemma": word.lemma,
                    "pos": word.xpos,  # ATB POS tags
                    "head": word.head,
                    "dep": word.deprel
                })

        return parsed_data

    def convert_to_ccg(self, atb_tree):
        """
        Convert ATB's constituency parse tree into a CCG derivation.
        """
        ccg_derivation = []
        for entry in atb_tree:
            word = entry["word"]
            pos = entry["pos"]
            dependency = entry["dep"]

            # Get CCG Category
            ccg_category = self.map_pos_to_ccg(pos, dependency)

            ccg_derivation.append((word, ccg_category))

        return ccg_derivation

    def map_pos_to_ccg(self, atb_pos, dependency):
        """
        Map Arabic Treebank POS tags & dependencies to CCG-compatible categories.
        """
        atb_to_ccg_map = {
            "NN": "N",      # Noun
            "VB": "S\\NP",  # Verb
            "IN": "NP\\NP", # Preposition
            "DT": "N/N",    # Determiner
            "JJ": "N/N",    # Adjective
            "PRP": "NP",    # Pronoun
            "CC": "Conj",   # Conjunction
            "RB": "S/S",    # Adverb
            "CD": "N/N",    # Numbers
            "UH": "Interj", # Interjections
        }

        # Handle Dependency Modifiers
        if dependency in ["amod", "acl"]:
            return "N/N"  # Adjective modifiers
        elif dependency in ["nsubj", "csubj"]:
            return "NP"  # Subjects
        elif dependency in ["obj", "iobj"]:
            return "NP"  # Objects

        return atb_to_ccg_map.get(atb_pos, "N")  # Default to Noun if unknown
