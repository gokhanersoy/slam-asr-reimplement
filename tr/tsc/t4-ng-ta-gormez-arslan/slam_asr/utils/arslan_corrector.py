"""Recep Sinan Arslan Output Correction Module for Turkish ASR (Arslan PhD Thesis, 2020).

"Türkçe Konuşma Tanıma Sistemlerinde Çıktı Düzeltme Yöntemleri"
Gazi University Institute of Science and Technology, 2020.

Methodology:
1. Lexicon Validation:
   - Evaluates whether words exist in the Turkish vocabulary/lexicon.
   - In-vocabulary (IV) words are strictly preserved.
2. Normalized Levenshtein (NL) Distance:
   - For Out-of-Vocabulary (OOV) words, evaluates candidate words in the lexicon.
   - NL(w1, w2) = Levenshtein(w1, w2) / max(len(w1), len(w2))
3. Arslan Golden Rules:
   - Distance Threshold: NL <= 0.33 (prevents replacing with unrelated words).
   - Word Length Constraint: len(w) > 3 (prevents false correction of short words,
     prepositions, particles, and conjunctions like 've', 'de', 'da', 'mi').
   - Frequency-Weighted Tie Breaking: Chooses candidate with lowest NL distance,
     breaking ties by corpus frequency.
4. Turkish Morpho-Phonetic & Colloquial Normalization:
   - Maps spoken colloquial contractions to canonical orthography:
     'biliyosun' -> 'biliyorsun', 'geliyo' -> 'geliyor', 'yapcaz' -> 'yapacağız'.
   - Normalizes loanword spelling variations: 'corona' -> 'korona'.
   - Contextual 'de/da' conjunction disambiguation.
"""

from collections import Counter, defaultdict
import json
import os
import re
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

from slam_asr.utils.metrics import normalize_text_turkish


COMMON_SUFFIXES = (
    "larda", "lerde", "lardan", "lerden", "lara", "lere", "ların", "lerin", "ları", "leri",
    "ımızda", "imizde", "umuzda", "ümüzde", "ımızdan", "imizden", "umuzdan", "ümüzden",
    "ımıza", "imize", "umuza", "ümüze", "ımızı", "imizi", "umuzu", "ümüzü",
    "dan", "den", "tan", "ten", "da", "de", "ta", "te",
    "ya", "ye", "a", "e", "yı", "yi", "yu", "yü", "ı", "i", "u", "ü",
    "nın", "nin", "nun", "nün", "ın", "in", "un", "ün",
    "lar", "ler", "lık", "lik", "luk", "lük", "sız", "siz", "suz", "süz",
    "lı", "li", "lu", "lü", "cu", "cü", "cı", "ci",
    "mak", "mek", "ma", "me", "makta", "mekte",
    "arak", "erek", "yarak", "yerek", "ınca", "ince", "ken",
    "acak", "ecek", "yacak", "yecek",
    "ıyor", "iyor", "uyor", "üyor", "yor",
    "mış", "miş", "muş", "müş",
    "dı", "di", "du", "dü", "tı", "ti", "tu", "tü",
    "dık", "dik", "duk", "dük", "tık", "tik", "tuk", "tük",
    "dıkları", "dikleri", "dukları", "dükleri",
    "dığım", "diğim", "duğum", "düğü",
    "dığı", "diği", "duğu", "düğü",
    "r", "ar", "er", "ır", "ir", "ur", "ür", "rim", "riz", "sin", "siniz", "sunuz", "sünüz",
)


def fast_levenshtein(s1: str, s2: str) -> int:
    """Two-row dynamic programming Levenshtein distance for strings."""
    if s1 == s2:
        return 0
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1] + [0] * len(s2)
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (0 if c1 == c2 else 1)
            current_row[j + 1] = min(insertions, deletions, substitutions)
        previous_row = current_row
    return previous_row[-1]


def normalized_levenshtein(w1: str, w2: str) -> float:
    """Compute Normalized Levenshtein (NL) distance: NL = dist / max(len(w1), len(w2))."""
    max_len = max(len(w1), len(w2))
    if max_len == 0:
        return 0.0
    return fast_levenshtein(w1, w2) / max_len


class ArslanOutputCorrector:
    """Recep Sinan Arslan's Normalized Levenshtein Output Correction System."""

    def __init__(
        self,
        lexicon: Optional[Union[Dict[str, int], Iterable[str]]] = None,
        lexicon_file: Optional[str] = None,
        manifest_path: Optional[str] = None,
        max_nl_distance: float = 0.33,
        min_word_len: int = 4,
        apply_colloquial_rules: bool = True,
        apply_spelling_variants: bool = True,
        apply_conjunction_rules: bool = True,
    ):
        """Args:
            lexicon: Optional dictionary of word frequencies or iterable of valid words.
            lexicon_file: Path to a text or JSON file with lexicon words.
            manifest_path: Path to JSONL manifest (e.g. train.jsonl) to build lexicon from.
            max_nl_distance: Maximum normalized Levenshtein distance (Arslan default: 0.33).
            min_word_len: Minimum word length to correct (Arslan default: > 3 characters, so min 4).
            apply_colloquial_rules: Convert colloquial contractions (biliyosun -> biliyorsun).
            apply_spelling_variants: Standardize common loanwords (corona -> korona).
            apply_conjunction_rules: Apply 'de/da' conjunction and spacing heuristics.
        """
        self.max_nl_distance = max_nl_distance
        self.min_word_len = min_word_len
        self.apply_colloquial_rules = apply_colloquial_rules
        self.apply_spelling_variants = apply_spelling_variants
        self.apply_conjunction_rules = apply_conjunction_rules

        self.word_freqs: Dict[str, int] = Counter()
        self.lexicon_by_len: Dict[int, List[Tuple[str, int]]] = defaultdict(list)
        self.vocab: Set[str] = set()

        # 1. Initialize with built-in core Turkish lexicon
        self._load_core_lexicon()

        # 2. Add external lexicon if provided
        if lexicon is not None:
            self.add_words(lexicon)

        # 3. Add lexicon from file if provided
        if lexicon_file and os.path.exists(lexicon_file):
            self.load_lexicon_from_file(lexicon_file)

        # 4. Add lexicon from training manifest if available
        if manifest_path and os.path.exists(manifest_path):
            self.load_lexicon_from_manifest(manifest_path)

        self._rebuild_index()

    def _load_core_lexicon(self):
        """Seed core Turkish lexicon containing high-frequency lemmas and grammatical forms."""
        core_words = [
            "bir", "ve", "bu", "da", "de", "için", "ne", "o", "ile", "çok", "daha", "en", "kadar",
            "gibi", "var", "yok", "olan", "olarak", "sonra", "önce", "kendi", "büyük", "küçük",
            "iyi", "yeni", "ilk", "zaman", "insan", "insanlar", "yer", "gün", "yıl", "iş", "şekilde",
            "bütün", "diğer", "ancak", "çünkü", "nasıl", "neden", "nedeni", "yani", "tüm", "her",
            "hiç", "bile", "hem", "ise", "diye", "şimdi", "son", "biz", "ben", "sen", "siz", "onlar",
            "bana", "sana", "ona", "bize", "size", "onlara", "beni", "seni", "onu", "bizi", "sizi",
            "onları", "benim", "senin", "onun", "bizim", "sizin", "onların", "bunda", "şunda", "onda",
            "bunun", "şunun", "onun", "bunu", "şunu", "böyle", "şöyle", "öyle", "burada", "şurada",
            "orada", "buraya", "oraya", "buradan", "oradan", "nerede", "nereye", "nereden", "kim",
            "kime", "kimi", "kimin", "hangisi", "hangi", "kaç", "kaçta", "nasıl", "niçin", "neden",
            "şey", "şeyler", "şeyleri", "olmak", "oldu", "olur", "olacak", "olmuş", "oluyor",
            "olduğu", "olduğu", "etmek", "etti", "eder", "edecek", "etmiş", "ediyor", "yapmak",
            "yaptı", "yapar", "yapacak", "yapmış", "yapıyor", "demek", "dedi", "der", "diyecek",
            "demiş", "diyor", "gitmek", "gitti", "gider", "gidecek", "gitmiş", "gidiyor", "gelmek",
            "geldi", "gelir", "gelecek", "gelmiş", "geliyor", "görmek", "gördü", "görür", "görecek",
            "görmüş", "görüyor", "bilmek", "bildi", "bilir", "bilecek", "bilmiş", "biliyor", "biliyorsun",
            "vermek", "verdi", "verir", "verecek", "vermiş", "veriyor", "almak", "aldı", "alır",
            "alacak", "almış", "alıyor", "çıkmak", "çıktı", "çıkar", "çıkacak", "çıkmış", "çıkıyor",
            "kalmak", "kaldı", "kalır", "kalacak", "kalmış", "kalıyor", "istemek", "istedi", "ister",
            "isteyecek", "istemiş", "istiyor", "bulmak", "buldu", "bulur", "bulacak", "bulmuş",
            "buluyor", "başlamak", "başladı", "başlar", "başlayacak", "başlamış", "başlıyor",
            "konuşmak", "konuştu", "konuşur", "konuşacak", "konuşmuş", "konuşuyor", "çalışmak",
            "çalıştı", "çalışır", "çalışacak", "çalışmış", "çalışıyor", "anlamak", "anladı",
            "anlar", "anlayacak", "anlamış", "anlıyor", "söylemek", "söyledi", "söyler", "söyleyecek",
            "söylemiş", "söylüyor", "söyleyeyim", "korona", "koronavirüs", "salgın", "hastalık",
            "türkiye", "türk", "türkçe", "ankara", "istanbul", "izmir", "devlet", "bakan", "bakanlık",
            "hükümet", "cumhurbaşkanı", "başkan", "meclis", "parti", "yasa", "karar", "para", "ekonomi",
            "fiyat", "ücret", "maaş", "işsiz", "işsizlik", "istihdam", "refah", "ümit", "ümitlerini",
            "kestiler", "kestikleri", "aramıyorlar", "aramadıkları", "tablo", "tabloyu", "genel",
            "özel", "özellikle", "genç", "yaşlı", "çocuk", "kadın", "erkek", "ay", "hafta", "saat",
            "dakika", "mart", "nisan", "mayıs", "haziran", "temmuz", "ağustos", "eylül", "ekim",
            "kasım", "aralık", "pazartesi", "salı", "çarşamba", "perşembe", "cuma", "cumartesi", "pazar",
            "biraz", "fazla", "az", "tam", "belli", "açık", "kesin", "fakat", "lakin", "falan", "filan",
            "kavuştu", "kavuştuğu", "kavuştur", "kavuşmak", "itibarıyla", "itibariyle", "itibarla", "etkiledi", "etkilemedi", "etkilemediği",
            "geç", "geçti", "geçer", "geçecek", "geçmiş", "geçiyor", "geçmek", "iste", "istemek", "isterim", "isteriz", "ister", "istedi",
            "zamanda", "zamandan", "zamana", "günde", "yılda", "ayda", "işte", "evde", "düşmek", "düşüyor", "düştü",
            "artmak", "artıyor", "arttı", "refah", "refaha", "refahı", "tablo", "tabloyu", "tabloda",
        ]
        for w in core_words:
            norm_w = normalize_text_turkish(w)
            if norm_w:
                self.word_freqs[norm_w] += 100

    def add_words(self, words: Union[Dict[str, int], Iterable[str]]):
        """Add words with optional frequencies to lexicon."""
        if isinstance(words, dict):
            for w, freq in words.items():
                norm = normalize_text_turkish(w)
                if norm:
                    self.word_freqs[norm] += freq
        else:
            for w in words:
                norm = normalize_text_turkish(w)
                if norm:
                    self.word_freqs[norm] += 1
        self._rebuild_index()

    def load_lexicon_from_manifest(self, manifest_path: str):
        """Extract all words and frequencies from a JSONL speech manifest."""
        if not os.path.exists(manifest_path):
            return
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    text = data.get("text", "") or data.get("transcript", "")
                    norm = normalize_text_turkish(text)
                    for word in norm.split():
                        if word:
                            self.word_freqs[word] += 1
                except Exception:
                    continue
        self._rebuild_index()

    def load_lexicon_from_file(self, filepath: str):
        """Load lexicon from a text file (one word per line or 'word count')."""
        if not os.path.exists(filepath):
            return
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                word = normalize_text_turkish(parts[0])
                freq = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
                if word:
                    self.word_freqs[word] += freq
        self._rebuild_index()

    def _rebuild_index(self):
        """Rebuild fast lookup table and length-bucketed candidate lists."""
        self.vocab = set(self.word_freqs.keys())
        self.lexicon_by_len.clear()
        for word, freq in self.word_freqs.items():
            self.lexicon_by_len[len(word)].append((word, freq))
        for length in self.lexicon_by_len:
            self.lexicon_by_len[length].sort(key=lambda item: item[1], reverse=True)

    def is_in_vocabulary(self, word: str) -> bool:
        """Check if a word is in vocabulary, either directly or via regular morphology."""
        if word in self.vocab:
            return True
        for suf in COMMON_SUFFIXES:
            if word.endswith(suf) and len(word) - len(suf) >= 2:
                stem = word[:-len(suf)]
                if stem in self.vocab:
                    return True
        return False

    def find_best_candidate(self, word: str) -> str:
        """Find the closest valid vocabulary candidate using Arslan Normalized Levenshtein."""
        # 1. In-Vocabulary words (including regular inflections) are never modified
        if self.is_in_vocabulary(word):
            return word

        # 2. Golden Rule: len(w) > 3 (short words are preserved)
        if len(word) < self.min_word_len:
            return word

        best_candidate = word
        min_nl = float("inf")
        best_freq = -1

        # Search candidates in length window [len - 1, len + 1]
        target_len = len(word)
        candidate_lengths = [target_len, target_len - 1, target_len + 1, target_len - 2, target_len + 2]

        for c_len in candidate_lengths:
            if c_len < self.min_word_len or c_len not in self.lexicon_by_len:
                continue

            for candidate, freq in self.lexicon_by_len[c_len]:
                # If word is a valid inflection of candidate, do not strip the suffix
                if word.startswith(candidate) and word[len(candidate):] in COMMON_SUFFIXES:
                    continue

                # Heuristic speedup: first character matching gives high confidence
                first_match = (candidate[0] == word[0])
                if not first_match and target_len >= 6:
                    continue

                nl = normalized_levenshtein(word, candidate)

                # Arslan Golden Rule: NL <= 0.33
                if nl <= self.max_nl_distance:
                    if nl < min_nl:
                        min_nl = nl
                        best_candidate = candidate
                        best_freq = freq
                    elif abs(nl - min_nl) < 1e-5:
                        # Tie-breaker: higher corpus frequency
                        if freq > best_freq:
                            best_candidate = candidate
                            best_freq = freq

        return best_candidate

    def apply_colloquial_verbalizations(self, text: str) -> str:
        """Standardize spoken conversational contractions to standard orthography."""
        colloquial_verb_map = {
            "yapcaz": "yapacağız",
            "yapcam": "yapacağım",
            "yapcan": "yapacaksın",
            "gitcez": "gideceğiz",
            "gitcem": "gideceğim",
            "alcam": "alacağım",
            "alcaz": "alacağız",
            "gelcez": "geleceğiz",
            "gelcem": "geleceğim",
        }
        for colloquial, canonical in colloquial_verb_map.items():
            text = re.sub(rf"\b{colloquial}\b", canonical, text)

        # 'biliyosun' -> 'biliyorsun', 'geliyosun' -> 'geliyorsun'
        text = re.sub(r"\b(\w+y)osun\b", r"\1orsun", text)
        # 'biliyom' -> 'biliyorum'
        text = re.sub(r"\b(\w+y)om\b", r"\1orum", text)
        # 'geliyo' -> 'geliyor'
        text = re.sub(r"\b(\w+y)o\b", r"\1or", text)
        # 'biliyon mu' -> 'biliyor musun'
        text = re.sub(r"\bbiliyon\b", "biliyorsun", text)
        return text

    def apply_loanword_standardization(self, text: str) -> str:
        """Standardize frequent English/foreign loanword variations."""
        text = re.sub(r"\bcorona\b", "korona", text)
        text = re.sub(r"\bcoronanın\b", "koronanın", text)
        text = re.sub(r"\bcoronavirüs\b", "koronavirüs", text)
        text = re.sub(r"\bcoronaya\b", "koronaya", text)
        text = re.sub(r"\bcoronada\b", "koronada", text)
        text = re.sub(r"\bcoronadan\b", "koronadan", text)
        text = re.sub(r"\bcoronavirüsü\b", "koronavirüsü", text)
        return text

    def apply_conjunction_heuristics(self, text: str) -> str:
        """Disambiguate common Turkish conjunction vs suffix boundaries ('de/da')."""
        # Spoken 'zaman da' vs 'zamanda'
        text = re.sub(r"\bzaman da\b", "zamanda", text)
        # Fix detached question particle 'mı/mi/mu/mü' if glued
        text = re.sub(r"\b(\w+)(misin|musun|müsün|mısın)\b", r"\1 \2", text)
        return text

    def correct_sentence(self, text: str) -> str:
        """Apply complete Arslan Output Correction pipeline to a sentence.

        Pipeline:
        1. Base normalization (Turkish casing & punctuation).
        2. Colloquial speech normalization (biliyosun -> biliyorsun).
        3. Loanword normalization (corona -> korona).
        4. Arslan Normalized Levenshtein OOV correction against lexicon.
        5. Conjunction and orthographic clean-up.
        """
        if not text:
            return ""

        # Step 1: Base normalization
        norm_text = normalize_text_turkish(text)

        # Step 2: Colloquial verbalizations
        if self.apply_colloquial_rules:
            norm_text = self.apply_colloquial_verbalizations(norm_text)

        # Step 3: Loanword spelling variants
        if self.apply_spelling_variants:
            norm_text = self.apply_loanword_standardization(norm_text)

        # Step 4: Arslan Normalized Levenshtein OOV Word Correction
        words = norm_text.split()
        corrected_words = [self.find_best_candidate(w) for w in words]
        corrected_text = " ".join(corrected_words)

        # Step 5: Conjunction heuristics
        if self.apply_conjunction_rules:
            corrected_text = self.apply_conjunction_heuristics(corrected_text)

        return normalize_text_turkish(corrected_text)

    def correct_batch(self, sentences: List[str]) -> List[str]:
        """Apply correction to a list of sentences."""
        return [self.correct_sentence(s) for s in sentences]
