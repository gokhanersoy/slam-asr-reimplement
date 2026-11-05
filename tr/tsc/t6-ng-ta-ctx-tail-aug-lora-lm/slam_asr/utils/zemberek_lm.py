"""Zemberek Turkish Morphological Language Model & Output Rescorer.

References:
- Akin, A. A., & Akin, M. D. "Zemberek: An open source NLP framework for Turkic languages."
- Görmez, Y. (2024). "Customized deep learning based Turkish automatic speech recognition
  system supported by language model." PeerJ Computer Science, 10, e1981.

This module provides a unified interface for Turkish Language Model (LM) rescoring and
morphological output correction, supporting both:
1. Native Zemberek Java Bridge (via JPype1 and zemberek-full.jar) when available.
2. Built-in Pure-Python Turkish Morphological & N-Gram Language Model Engine (Zero-dependency fallback),
   ensuring full operational functionality across any platform (Colab, Linux, Mac, Windows).
"""

from collections import Counter, defaultdict
import json
import math
import os
import re
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

from slam_asr.utils.metrics import normalize_text_turkish


# ============================================================================
# Core Turkish Phonological and Grammatical Constants
# ============================================================================

TURKISH_FRONT_VOWELS = set("eiöüEİÖÜ")
TURKISH_BACK_VOWELS = set("aıouAIOU")
TURKISH_ALL_VOWELS = TURKISH_FRONT_VOWELS | TURKISH_BACK_VOWELS

TURKISH_VOICELESS_CONSONANTS = set("çfhkpstşÇFHKPSTŞ")  # Fıstıkçı Şahap
TURKISH_VOICED_CONSONANTS = set("bcdgğjlmnrvyzBCDGĞJLMNRVYZ")

# Suffix templates according to 2-way (A/E) and 4-way (I/İ/U/Ü) vowel harmony
A_TYPE_HARMONY = {"front": "e", "back": "a"}
I_TYPE_HARMONY = {"front_unrounded": "i", "front_rounded": "ü", "back_unrounded": "ı", "back_rounded": "u"}

COMMON_TURKISH_SUFFIXES = [
    # Plural
    "lar", "ler",
    # Case endings
    "da", "de", "ta", "te", "dan", "den", "tan", "ten", "a", "e", "ya", "ye", "ı", "i", "u", "ü", "yı", "yi", "yu", "yü",
    "ın", "in", "un", "ün", "nın", "nin", "nun", "nün",
    # Possessive
    "ım", "im", "um", "üm", "mız", "miz", "muz", "müz", "ımız", "imiz", "umuz", "ümüz",
    "ın", "in", "un", "ün", "nız", "niz", "nuz", "nüz", "ınız", "iniz", "unuz", "ünüz",
    "ı", "i", "u", "ü", "sı", "si", "su", "sü", "ları", "leri",
    # Derivational / Copula
    "lı", "li", "lu", "lü", "sız", "siz", "suz", "süz", "lık", "lik", "luk", "lük",
    "dır", "dir", "dur", "dür", "tır", "tir", "tur", "tür",
    # Verbal tenses & aspects
    "dı", "di", "du", "dü", "tı", "ti", "tu", "tü",
    "mış", "miş", "muş", "müş",
    "ıyor", "iyor", "uyor", "üyor",
    "acak", "ecek", "yacak", "yecek",
    "malı", "meli", "ar", "er", "ır", "ir", "ur", "ür",
    "maz", "mez", "ma", "me",
    # Verbal person
    "k", "n", "m", "nız", "niz", "nuz", "nüz",
    # Combined multi-suffix clusters
    "larında", "lerinde", "larından", "lerinden", "larına", "lerine",
    "ımızda", "imizde", "umuzda", "ümüzde", "ımızdan", "imizden", "umuzdan", "ümüzden",
    "ındaki", "indeki", "unduki", "ündeki", "undaki", "ündeki",
    "acaktı", "ecekti", "yacaktı", "yecekti", "mıştı", "mişti", "muştu", "müştü",
    "ıyordu", "iyordu", "uyordu", "üyordu",
]


def check_vowel_harmony(stem: str, suffix: str) -> bool:
    """Check whether a suffix satisfies Turkish major vowel harmony (Kalınlık-İncelik)."""
    stem_vowels = [c for c in stem if c in TURKISH_ALL_VOWELS]
    suffix_vowels = [c for c in suffix if c in TURKISH_ALL_VOWELS]
    if not stem_vowels or not suffix_vowels:
        return True
    last_stem_vowel = stem_vowels[-1]
    is_stem_back = last_stem_vowel in TURKISH_BACK_VOWELS
    for sv in suffix_vowels:
        is_suffix_back = sv in TURKISH_BACK_VOWELS
        if is_stem_back != is_suffix_back:
            # Special exceptions in Turkish (e.g. -ken, -ki, -yor)
            if suffix in ("ken", "ki", "yor", "iyor", "ıyor", "uyor", "üyor"):
                continue
            return False
    return True


def fast_levenshtein(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
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


# ============================================================================
# Built-in Core Turkish Lexicon Stems & Lemmas
# ============================================================================

CORE_TURKISH_ROOTS = {
    # Pronouns & Conjunctions & Particles
    "ben", "sen", "o", "biz", "siz", "onlar", "bu", "şu", "o", "bunlar", "şunlar", "onlar",
    "biri", "birisi", "hepsi", "hiçbiri", "herkes", "kimse", "şey", "şeyler",
    "ve", "ile", "veya", "ya", "yahut", "ama", "fakat", "lakin", "ancak", "çünkü",
    "da", "de", "ki", "mi", "mı", "mu", "mü", "ise", "diye", "gibi", "kadar", "için", "göre",
    "daha", "en", "çok", "az", "fazla", "hiç", "bile", "zaten", "artık", "henüz", "yalnız",
    # Question words
    "ne", "kim", "nasıl", "nerede", "nereden", "nereye", "neden", "niçin", "neye", "hangi", "kaç",
    # Common Nouns
    "adam", "kadın", "çocuk", "insan", "kişi", "halk", "millet", "devlet", "ülke", "dünya", "hayat", "yaşam",
    "ev", "oda", "kapı", "pencere", "sokak", "cadde", "şehir", "kent", "köy", "bölge", "alan", "yer",
    "yol", "araba", "tren", "uçak", "otobüs", "istasyon", "durak", "köprü", "deniz", "göl", "nehir", "su",
    "gün", "ay", "yıl", "hafta", "saat", "dakika", "saniye", "zaman", "vakit", "sabah", "öğle", "akşam", "gece",
    "dün", "bugün", "yarın", "şimdi", "sonra", "önce", "erken", "geç", "daima", "bazen", "ara", "sıra",
    "iş", "güç", "çalışma", "şirket", "fabrika", "okul", "üniversite", "öğrenci", "öğretmen", "hoca", "doktor",
    "kitap", "defter", "kalem", "yazı", "mektup", "gazete", "haber", "bilgi", "fikir", "düşünce", "kelime", "söz",
    "el", "kol", "ayak", "bacak", "baş", "kafa", "göz", "kulak", "burun", "ağız", "yüz", "kalp", "yürek",
    "bölüm", "dinleyici", "yayın", "ses", "konuşma", "metin", "program", "kanal", "radyo", "televizyon",
    # Common Verbs (Stems)
    "al", "ver", "gel", "git", "yap", "et", "ol", "kal", "bak", "gör", "duy", "işit",
    "bil", "anla", "öğren", "anlat", "söyle", "de", "yaz", "oku", "konuş", "sor", "cevapla",
    "dur", "başla", "bit", "bitir", "çalış", "otur", "kalk", "yürü", "koş", "aç", "kapa",
    "sev", "iste", "düşün", "bul", "ara", "kaybet", "bekle", "geç", "ulaş", "sağla", "kur",
}


# ============================================================================
# Zemberek Language Model & Output Rescorer
# ============================================================================

class ZemberekLanguageModel:
    """Zemberek Morphological Language Model & Output Rescorer for Turkish ASR.

    Implements:
    - Morphological validity check and root-suffix decomposition.
    - N-gram statistical language model scoring (Unigram, Bigram, Trigram).
    - Optional Java Bridge via JPype1 to native Zemberek (zemberek-full.jar).
    - High-performance pure-Python fallback morphology & n-gram language model.
    """

    def __init__(
        self,
        zemberek_jar_path: Optional[str] = None,
        train_manifest: Optional[str] = None,
        lexicon_file: Optional[str] = None,
        ngram_order: int = 3,
        alpha_smoothing: float = 0.05,
        lm_weight: float = 0.35,
        max_edit_distance: int = 2,
        use_jpype_if_available: bool = True,
    ):
        """Initialize Zemberek Turkish Language Model.

        Args:
            zemberek_jar_path: Path to zemberek-full.jar (optional).
            train_manifest: Path to JSONL train manifest (e.g. data/issai_tsc/train.jsonl)
                            to learn n-gram transition probabilities matching Görmez (2024).
            lexicon_file: Optional path to text or JSON lexicon dictionary.
            ngram_order: Maximum n-gram order for statistical language model (default: 3).
            alpha_smoothing: Laplace/Jelinek-Mercer smoothing parameter (default: 0.05).
            lm_weight: Interpolation weight between acoustic similarity and LM likelihood.
            max_edit_distance: Maximum character edit distance for candidate morphology generation.
            use_jpype_if_available: Whether to attempt loading Java Zemberek if JPype1 is present.
        """
        self.zemberek_jar_path = zemberek_jar_path
        self.ngram_order = max(1, min(ngram_order, 3))
        self.alpha_smoothing = alpha_smoothing
        self.lm_weight = lm_weight
        self.max_edit_distance = max_edit_distance

        # Statistical N-Gram counts
        self.unigrams: Counter = Counter()
        self.bigrams: Counter = Counter()
        self.trigrams: Counter = Counter()
        self.total_tokens: int = 0
        self.vocab: Set[str] = set()
        self.vocab_by_len: Dict[int, Set[str]] = defaultdict(set)

        # Pre-populate with core Turkish roots and common inflected words
        self._init_core_vocabulary()

        # Load training manifest if provided
        if train_manifest and os.path.exists(train_manifest):
            self.load_from_manifest(train_manifest)

        # Load custom lexicon if provided
        if lexicon_file and os.path.exists(lexicon_file):
            self.load_lexicon_file(lexicon_file)

        # Attempt to initialize Java Zemberek Bridge
        self.java_morphology = None
        self.java_spell_checker = None
        self.java_lm = None
        self.is_java_active = False

        if use_jpype_if_available and zemberek_jar_path and os.path.exists(zemberek_jar_path):
            self._try_init_java_zemberek(zemberek_jar_path)

    def _register_token(self, token: str, count: int = 1):
        """Register a token into vocabulary, length index, and unigram counters."""
        token = token.lower().strip()
        if not token:
            return
        self.vocab.add(token)
        self.vocab_by_len[len(token)].add(token)
        self.unigrams[token] += count
        self.total_tokens += count

    def _init_core_vocabulary(self):
        """Seed vocabulary and n-gram priors with core Turkish morphology."""
        for root in CORE_TURKISH_ROOTS:
            self._register_token(root, count=5)
            # Synthesize valid morphological forms following vowel harmony
            for suffix in COMMON_TURKISH_SUFFIXES:
                if check_vowel_harmony(root, suffix):
                    inflected = root + suffix
                    self._register_token(inflected, count=2)

    def _try_init_java_zemberek(self, jar_path: str):
        """Attempt to boot Java VM and load Zemberek classes via JPype."""
        try:
            import jpype
            import jpype.imports

            if not jpype.isJVMStarted():
                jpype.startJVM(
                    jpype.getDefaultJVMPath(),
                    f"-Djava.class.path={jar_path}",
                    "-Xmx2g",
                    convertStrings=True,
                )

            from zemberek.morphology import TurkishMorphology
            from zemberek.normalization import TurkishSpellChecker

            self.java_morphology = TurkishMorphology.createWithDefaults()
            self.java_spell_checker = TurkishSpellChecker(self.java_morphology)
            self.is_java_active = True
            print(f"[*] Native Java Zemberek initialized successfully from {jar_path}")
        except Exception as e:
            self.is_java_active = False
            self.java_morphology = None
            self.java_spell_checker = None
            print(f"[i] Running built-in Python Turkish Morphological & N-Gram LM Engine (JPype notice: {e})")

    def load_from_manifest(self, manifest_path: str):
        """Train N-gram Language Model statistics on Turkish training transcripts."""
        count = 0
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    text = record.get("transcript") or record.get("text") or record.get("sentence") or ""
                    if text:
                        self.train_on_sentence(text)
                        count += 1
                except Exception:
                    continue
        print(f"[*] Zemberek LM trained on {count:,} sentences from {manifest_path} (Vocab: {len(self.vocab):,})")

    def load_lexicon_file(self, lexicon_path: str):
        """Load external wordlist or frequency dictionary."""
        with open(lexicon_path, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip().lower()
                if word:
                    self._register_token(word, count=1)

    def train_on_sentence(self, text: str):
        """Update n-gram frequency distributions from a sentence."""
        norm_text = normalize_text_turkish(text)
        words = norm_text.split()
        if not words:
            return

        for w in words:
            self._register_token(w, count=1)

        # Bigrams: (w1, w2)
        for i in range(len(words) - 1):
            self.bigrams[(words[i], words[i + 1])] += 1

        # Trigrams: (w1, w2, w3)
        for i in range(len(words) - 2):
            self.trigrams[(words[i], words[i + 1], words[i + 2])] += 1

    def is_valid_word(self, word: str) -> bool:
        """Check if a word is morphologically valid in Turkish."""
        word = word.lower().strip()
        if not word:
            return False

        # 1. Native Java Zemberek check
        if self.is_java_active and self.java_morphology is not None:
            try:
                results = self.java_morphology.analyze(word)
                if results and not results.getAnalysisResults().isEmpty():
                    return True
            except Exception:
                pass

        # 2. Known in vocabulary
        if word in self.vocab:
            return True

        # 3. Rule-based morphological parsing: Root + valid suffix cascade
        for suffix in COMMON_TURKISH_SUFFIXES:
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if stem in self.vocab or stem in CORE_TURKISH_ROOTS:
                    if check_vowel_harmony(stem, suffix):
                        return True
        return False

    def get_word_prob(self, word: str, prev_word: Optional[str] = None, prev_prev_word: Optional[str] = None) -> float:
        """Compute conditional probability P(word | context) with backoff smoothing."""
        vocab_size = max(len(self.vocab), 10000)
        unigram_prob = (self.unigrams[word] + self.alpha_smoothing) / (self.total_tokens + self.alpha_smoothing * vocab_size)

        if self.ngram_order == 1 or prev_word is None:
            return max(unigram_prob, 1e-8)

        # Bigram probability
        bigram_count = self.bigrams.get((prev_word, word), 0)
        prev_count = self.unigrams.get(prev_word, 0)
        if prev_count > 0:
            bigram_prob = (bigram_count + self.alpha_smoothing) / (prev_count + self.alpha_smoothing * vocab_size)
        else:
            bigram_prob = unigram_prob

        if self.ngram_order == 2 or prev_prev_word is None:
            return 0.7 * bigram_prob + 0.3 * unigram_prob

        # Trigram probability
        trigram_count = self.trigrams.get((prev_prev_word, prev_word, word), 0)
        bi_context_count = self.bigrams.get((prev_prev_word, prev_word), 0)
        if bi_context_count > 0:
            trigram_prob = (trigram_count + self.alpha_smoothing) / (bi_context_count + self.alpha_smoothing * vocab_size)
        else:
            trigram_prob = bigram_prob

        return 0.5 * trigram_prob + 0.35 * bigram_prob + 0.15 * unigram_prob

    def score_sentence(self, sentence: str) -> float:
        """Compute total log-likelihood score of a sentence under the Zemberek Language Model."""
        norm_text = normalize_text_turkish(sentence)
        words = norm_text.split()
        if not words:
            return -100.0

        total_log_prob = 0.0
        for i, word in enumerate(words):
            prev = words[i - 1] if i > 0 else None
            prev_prev = words[i - 2] if i > 1 else None
            prob = self.get_word_prob(word, prev, prev_prev)
            total_log_prob += math.log(prob)
        return total_log_prob

    def get_perplexity(self, sentence: str) -> float:
        """Compute model perplexity on a sentence."""
        norm_text = normalize_text_turkish(sentence)
        words = norm_text.split()
        if not words:
            return float("inf")
        log_prob = self.score_sentence(sentence)
        return math.exp(-log_prob / len(words))

    def generate_candidates(self, word: str) -> List[str]:
        """Generate morphologically plausible replacement candidates for a word."""
        candidates = {word}

        # 1. Java Zemberek suggestions
        if self.is_java_active and self.java_spell_checker is not None:
            try:
                suggestions = self.java_spell_checker.suggestForWord(word)
                for s in suggestions:
                    candidates.add(str(s).lower())
            except Exception:
                pass

        # 2. Suffix decomposition candidates
        for suffix in COMMON_TURKISH_SUFFIXES:
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if stem in self.vocab or stem in CORE_TURKISH_ROOTS:
                    candidates.add(stem)
                    # Alternate suffixes matching harmony
                    for alt_suffix in ("lar", "ler", "da", "de", "dan", "den", "ı", "i", "e", "a"):
                        if check_vowel_harmony(stem, alt_suffix):
                            candidates.add(stem + alt_suffix)

        # 3. Known vocabulary words within edit distance <= max_edit_distance (length-bucketed search)
        target_len = len(word)
        min_len = max(1, target_len - self.max_edit_distance)
        max_len = target_len + self.max_edit_distance
        for l in range(min_len, max_len + 1):
            for cand in self.vocab_by_len.get(l, ()):
                if fast_levenshtein(word, cand) <= self.max_edit_distance:
                    candidates.add(cand)

        # Filter by morphological validity
        valid_candidates = [c for c in candidates if self.is_valid_word(c)]
        return valid_candidates if valid_candidates else [word]

    def correct_sentence(self, sentence: str) -> str:
        """Apply Zemberek Morphological Language Model rescoring to an ASR hypothesis."""
        norm_text = normalize_text_turkish(sentence)
        words = norm_text.split()
        if not words:
            return sentence

        rescored_words = []
        for i, word in enumerate(words):
            # If word is already morphologically valid, keep it
            if self.is_valid_word(word):
                rescored_words.append(word)
                continue

            candidates = self.generate_candidates(word)
            if len(candidates) == 1 and candidates[0] == word:
                rescored_words.append(word)
                continue

            prev = rescored_words[-1] if rescored_words else None
            prev_prev = rescored_words[-2] if len(rescored_words) > 1 else None

            best_word = word
            best_score = float("-inf")

            for cand in candidates:
                # Acoustic / orthographic fidelity score
                dist = fast_levenshtein(word, cand)
                fidelity_score = -1.5 * dist

                # N-Gram Language Model score
                prob = self.get_word_prob(cand, prev, prev_prev)
                lm_score = math.log(prob)

                # Morphological validity bonus
                morph_bonus = 1.0 if self.is_valid_word(cand) else -2.0

                total_score = fidelity_score + self.lm_weight * lm_score + morph_bonus
                if total_score > best_score:
                    best_score = total_score
                    best_word = cand

            rescored_words.append(best_word)

        return " ".join(rescored_words)


# Alias for compatibility with corrector pipelines
ZemberekLMCorrector = ZemberekLanguageModel
