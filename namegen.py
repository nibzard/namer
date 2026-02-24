#!/usr/bin/env python3
"""Style-aware, reusable fake-name generator.

Pipeline:
- learn syllable + character transitions from a style corpus
- generate candidate names with constrained beam sampling
- filter by pronounceability/morphology
- rank with combined model-based score
- return top-N names
"""

from __future__ import annotations

import argparse
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

START = "<START>"
END = "<END>"
START_STYLE_CORPUS = "scandi"
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS = str(PROJECT_ROOT / "data" / "default_corpus.txt")
DEFAULT_STYLES_DIR = str(PROJECT_ROOT / "data" / "styles")


@dataclass(frozen=True)
class StyleProfile:
    name: str
    display: str
    vowels: str = "aeiouy"
    allowed_chars: str = "abcdefghijklmnopqrstuvwxyz"
    min_len: int = 4
    max_len: int = 12
    min_syllables: int = 2
    max_syllables: int = 4
    novelty_cutoff: int = 1
    sample_pool: int = 40
    beam: int = 80
    branch: int = 24
    temperature: float = 0.95
    min_vowel_ratio: float = 0.28
    max_vowel_ratio: float = 0.67
    morph_endings: Tuple[str, ...] = ()


STYLE_PROFILES: Dict[str, StyleProfile] = {
    "scandi": StyleProfile(
        name="scandi",
        display="Scandinavian",
        vowels="aeiouyåäö",
        allowed_chars="åäö",
        min_len=4,
        max_len=12,
        min_syllables=2,
        max_syllables=4,
        novelty_cutoff=1,
        sample_pool=40,
        beam=80,
        branch=24,
        temperature=0.92,
        morph_endings=("berg", "son", "holm", "dal", "stad", "vik", "by", "lund", "fors", "strand", "borg", "ström", "havn"),
    ),
    "italian": StyleProfile(
        name="italian",
        display="Italian",
        vowels="aeiouy",
        min_len=4,
        max_len=11,
        min_syllables=2,
        max_syllables=4,
        novelty_cutoff=1,
        sample_pool=45,
        beam=96,
        branch=30,
        temperature=0.95,
        min_vowel_ratio=0.30,
        max_vowel_ratio=0.64,
        morph_endings=("o", "i", "a", "e", "io", "ini", "ini", "ello", "elli", "etti", "etto", "ini"),
    ),
    "greek": StyleProfile(
        name="greek",
        display="Greek",
        vowels="aeiouy",
        min_len=4,
        max_len=12,
        min_syllables=2,
        max_syllables=5,
        novelty_cutoff=1,
        sample_pool=40,
        beam=90,
        branch=28,
        temperature=0.95,
        min_vowel_ratio=0.29,
        max_vowel_ratio=0.65,
        morph_endings=("os", "is", "as", "ou", "aki", "idis", "opoulos", "akis", "ou", "akis", "iadis"),
    ),
    "klingon": StyleProfile(
        name="klingon",
        display="Klingon",
        vowels="aeiouy",
        allowed_chars="qwxz",  # keep style-specific letters for Klingon flavor
        min_len=4,
        max_len=10,
        min_syllables=1,
        max_syllables=4,
        novelty_cutoff=1,
        sample_pool=50,
        beam=110,
        branch=36,
        temperature=1.05,
        min_vowel_ratio=0.18,
        max_vowel_ratio=0.62,
        morph_endings=("gh", "q", "th", "kh", "ng", "v", "r", "k"),
    ),
    "roman": StyleProfile(
        name="roman",
        display="Roman",
        vowels="aeiouy",
        min_len=4,
        max_len=12,
        min_syllables=2,
        max_syllables=5,
        novelty_cutoff=1,
        sample_pool=45,
        beam=90,
        branch=30,
        temperature=0.95,
        min_vowel_ratio=0.28,
        max_vowel_ratio=0.66,
        morph_endings=("us", "a", "um", "is", "ior", "ius", "ius", "iana", "a", "icus", "ius", "ianus"),
    ),
}


DEFAULT_PROFILE = StyleProfile(
    name="generic",
    display="Generic",
    vowels="aeiouy",
    allowed_chars="",
    min_len=4,
    max_len=12,
    min_syllables=2,
    max_syllables=4,
    novelty_cutoff=1,
    sample_pool=40,
    beam=80,
    branch=24,
    temperature=0.95,
)


BUILTIN_STYLE_CORPORA: Dict[str, List[str]] = {
    "scandi": [
        "stockholm",
        "gothenburg",
        "uppsala",
        "berlin",
        "lund",
        "vikar",
        "forsmark",
        "havn",
        "berg",
        "stavik",
        "norrland",
        "gotland",
        "kiruna",
        "malmo",
        "solna",
        "bjorna",
        "halmstad",
        "oslo",
        "bergen",
        "aarhus",
        "kristiansand",
        "strand",
        "fjord",
        "nordby",
        "kristi",
        "alborg",
    ],
    "italian": [
        "roma",
        "milano",
        "venezia",
        "genova",
        "firenze",
        "bologna",
        "torino",
        "napoli",
        "verona",
        "palermo",
        "bari",
        "trieste",
        "padova",
        "parma",
        "ancona",
        "perugia",
        "pisa",
        "pavia",
        "udine",
        "ferrara",
        "modena",
        "catania",
        "trieste",
        "salerno",
        "pescara",
        "trento",
        "bergamo",
        "mantova",
        "alessandria",
        "varese",
        "laquila",
        "roccia",
        "valdagno",
        "montagna",
        "marconi",
        "rossi",
        "bianchi",
        "ferrari",
        "romano",
        "esposito",
        "conti",
        "ricci",
        "bruni",
        "marino",
        "monti",
        "greco",
        "fontana",
        "montoro",
        "viano",
        "novella",
        "moretti",
        "pellegrini",
        "bellini",
        "galliano",
        "lombardi",
        "verdi",
        "rossini",
        "bianchini",
        "fiorini",
        "montanari",
        "valente",
        "marini",
        "santoro",
        "serrano",
        "giordani",
        "caruso",
        "lazzari",
        "bianchiere",
        "ferrante",
        "veneziani",
        "santini",
        "lorenzi",
        "bellandi",
        "colombo",
        "baro",
        "pallini",
        "montorsi",
        "carbona",
        "venturi",
        "mancini",
        "marchetti",
        "rinaldi",
        "fabbri",
        "guerra",
        "lucente",
        "nobili",
    ],
    "greek": [
        "athina",
        "thessaloniki",
        "patra",
        "larisa",
        "irakli",
        "ioannina",
        "kalamata",
        "volos",
        "chania",
        "rhodes",
        "xanthi",
        "mytilene",
        "karpenisi",
        "piraeus",
        "kavala",
        "lamia",
        "naxos",
        "samos",
        "mykonos",
        "korinthi",
        "heraklion",
        "alexandroupoli",
        "stavros",
        "dimitri",
        "petros",
        "nikolaos",
        "georgiou",
        "papadopoulos",
        "michalis",
        "alexandros",
        "antoniadis",
        "kallianiotis",
        "pappas",
        "stavridis",
        "karelis",
        "mandaris",
        "alexiou",
        "petrakis",
        "kalogeris",
        "christos",
        "kostas",
        "panagiotis",
        "sotiris",
        "argyros",
        "kyriakos",
        "tsakalos",
        "dimaratos",
        "theodorakis",
        "samaras",
        "vassilis",
        "kallikrates",
        "lykos",
        "theodoros",
        "alexandra",
        "stefanos",
        "nikos",
        "katerina",
        "eleni",
        "maria",
        "kostantinou",
        "petrou",
        "papadaki",
        "anastasios",
        "marinos",
        "nikitas",
        "kyris",
    ],
    "klingon": [
        "qapla",
        "qeyl",
        "kohr",
        "gortha",
        "khargh",
        "vokra",
        "thok",
        "krel",
        "qet",
        "tuvak",
        "qorath",
        "gargh",
        "mohla",
        "mogh",
        "nakri",
        "tuhla",
        "qen",
        "korgh",
        "vash",
        "thluth",
        "mrel",
        "qir",
        "targ",
        "khagh",
        "qah",
        "zhoq",
        "qem",
        "talon",
        "kargh",
        "drovak",
        "ghor",
        "qel",
        "narq",
        "korg",
        "vokor",
        "thra",
        "qim",
        "varkh",
        "qhul",
        "grog",
        "qos",
        "khelen",
        "thol",
        "qeb",
        "korv",
        "ghora",
        "qot",
        "vuh",
        "qath",
        "mrahl",
        "qetlar",
        "qahri",
        "vokh",
        "thrash",
        "qorv",
        "kharak",
        "vokri",
        "qarn",
    ],
    "roman": [
        "augustus",
        "aurelius",
        "cassius",
        "valerius",
        "flavian",
        "maximus",
        "marcellus",
        "severus",
        "tiberius",
        "marcus",
        "lucius",
        "quintus",
        "julius",
        "julians",
        "constantine",
        "cicero",
        "horatius",
        "virgil",
        "antonius",
        "seneca",
        "varro",
        "livia",
        "liviae",
        "claudia",
        "octavia",
        "cloelia",
        "drusian",
        "fabius",
        "pompeius",
        "augusta",
        "flavius",
        "nerva",
        "trajan",
        "marcellinus",
        "valerian",
        "constantinus",
        "justinian",
        "aurelia",
        "severina",
        "marcellina",
        "lucilla",
        "florian",
        "aegina",
        "venantius",
        "felix",
        "octavian",
        "domitian",
        "hadrian",
        "lucian",
        "brutus",
        "nicol",
        "tiber",
        "florianus",
        "drago",
        "agrippa",
        "maxima",
        "silvanus",
        "nerva",
        "valens",
        "marcus",
        "aemilian",
        "tullius",
        "gaius",
        "marian",
        "carina",
        "verona",
        "maro",
        "livius",
        "orantius",
        "marcellus",
        "cassian",
    ],
}


@dataclass
class Candidate:
    word: str
    syllables: List[str]
    score: float
    details: Dict[str, float]


def normalize(text: str, profile: StyleProfile, ascii_only: bool = False) -> str:
    text = text.strip().lower()
    if ascii_only:
        text = (
            unicodedata.normalize("NFKD", text)
            .encode("ascii", "ignore")
            .decode("ascii", "ignore")
        )

    allowed = "".join(sorted(set(profile.allowed_chars)))
    if allowed:
        text = re.sub(rf"[^a-z{re.escape(allowed)}]", "", text.lower())
    else:
        text = re.sub(r"[^a-z]", "", text.lower())
    return text


def is_vowel_char(ch: str, vowels: Set[str]) -> bool:
    return ch in vowels


def syllabify(word: str, vowels: Set[str]) -> List[str]:
    if not word:
        return []

    vowel_groups: List[Tuple[int, int]] = []
    i = 0
    n = len(word)

    while i < n:
        if not is_vowel_char(word[i], vowels):
            i += 1
            continue
        start = i
        while i < n and is_vowel_char(word[i], vowels):
            i += 1
        vowel_groups.append((start, i))
        while i < n and not is_vowel_char(word[i], vowels):
            i += 1

    if not vowel_groups:
        return [word]

    syllables: List[str] = []
    start = 0
    for idx, (vs, ve) in enumerate(vowel_groups):
        if idx == len(vowel_groups) - 1:
            end = n
        else:
            nxt_vs = vowel_groups[idx + 1][0]
            gap = nxt_vs - ve
            if gap <= 2:
                end = nxt_vs
            else:
                end = max(ve + 1, nxt_vs - 2)
        syllables.append(word[start:end])
        start = end

    return [s for s in syllables if s]


def weighted_choice(items: List[str], log_weights: List[float], temperature: float = 1.0) -> str:
    if not items:
        raise ValueError("No candidates to sample")
    if temperature <= 0:
        return items[0]
    weights = [max(1e-12, math.exp(w / temperature)) for w in log_weights]
    total = sum(weights)
    r = random.random() * total
    c = 0.0
    for it, w in zip(items, weights):
        c += w
        if r <= c:
            return it
    return items[-1]


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


class NameModel:
    def __init__(self, names: Iterable[str], profile: StyleProfile, smoothing: float = 0.7):
        self.names = list(dict.fromkeys(name for name in names if name))
        if not self.names:
            raise ValueError("No names provided to train model")

        self.profile = profile
        self.vowels = set(profile.vowels)
        self.smoothing = smoothing

        self.length_counter = Counter()
        self.trans_counter: Dict[str, Counter] = defaultdict(Counter)
        self.syllable_counter = Counter()
        self.char_counter = Counter()
        self.char_pair_counter: Dict[str, Counter] = defaultdict(Counter)
        self.suffix_counters: Dict[int, Counter] = {2: Counter(), 3: Counter(), 4: Counter()}
        self.run_counter = Counter()
        self.names_by_length: Dict[int, List[str]] = defaultdict(list)
        self.first2_buckets: Dict[str, List[str]] = defaultdict(list)
        self.last2_buckets: Dict[str, List[str]] = defaultdict(list)

        self.allowed_runs2: Set[str] = set()
        self.allowed_runs3: Set[str] = set()
        self.vocab_size = 0

        for word in self.names:
            if len(word) >= 2:
                self.first2_buckets[word[:2]].append(word)
                self.last2_buckets[word[-2:]].append(word)
            self.names_by_length[len(word)].append(word)
            syllables = syllabify(word, self.vowels)
            if not syllables:
                continue
            self.length_counter[len(syllables)] += 1

            prev = START
            for syll in syllables:
                self.syllable_counter[syll] += 1
                self.trans_counter[prev][syll] += 1
                prev = syll
            self.trans_counter[prev][END] += 1

            seq = "^" + word + "$"
            for ch in seq:
                self.char_counter[ch] += 1
            for a, b in zip(seq[:-1], seq[1:]):
                self.char_pair_counter[a][b] += 1

            for n in (2, 3, 4):
                if len(word) >= n:
                    self.suffix_counters[n][word[-n:]] += 1

            for i, ch in enumerate(word):
                if is_vowel_char(ch, self.vowels):
                    continue
                j = i
                while j < len(word) and not is_vowel_char(word[j], self.vowels):
                    j += 1
                run = word[i:j]
                if 1 <= len(run) <= 4:
                    self.run_counter[run] += 1
                i = j

        if not self.length_counter:
            raise ValueError("Training failed: cannot extract syllables from corpus")

        self.allowed_runs2 = {run for run, c in self.run_counter.items() if len(run) == 2 and c >= 1}
        self.allowed_runs3 = {run for run, c in self.run_counter.items() if len(run) == 3 and c >= 1}
        if not self.allowed_runs2:
            self.allowed_runs2 = {"st", "nd", "ll", "nn", "rk", "rt", "ld", "ng", "gr", "fr"}
        if not self.allowed_runs3:
            self.allowed_runs3 = {"str", "skr", "ndr", "ntr"}

        self.top_suffixes = set()
        for n, cnt in self.suffix_counters.items():
            for suff, c in cnt.most_common(100):
                if c >= 1:
                    self.top_suffixes.add(suff)
        if not self.top_suffixes:
            self.top_suffixes = {"en", "on", "in", "a", "o", "e", "us", "um", "stad", "berg", "vik"}

        self.top_suffixes_3_4 = {s for s in self.top_suffixes if len(s) >= 3}
        if not self.top_suffixes_3_4:
            self.top_suffixes_3_4 = set(self.top_suffixes)

        self.name_set = set(self.names)
        self.vocab_size = max(len(self.syllable_counter), 1)

    def transition_logp(self, prev: str, nxt: str) -> float:
        next_map = self.trans_counter.get(prev)
        if not next_map:
            next_map = self.trans_counter.get(START, Counter())
        total = sum(next_map.values()) or 1
        return math.log((next_map.get(nxt, 0) + self.smoothing) / (total + self.smoothing * self.vocab_size))

    def transition_logprobs(
        self, prev: str, include_end: bool = False, top_k: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        options = self.trans_counter.get(prev)
        if not options:
            options = self.trans_counter.get(START, Counter())
        out: List[Tuple[str, float]] = []
        for syl in options:
            if not include_end and syl == END:
                continue
            out.append((syl, self.transition_logp(prev, syl)))
        if not out:
            for syl in self.syllable_counter:
                out.append((syl, math.log(1.0 / max(len(self.syllable_counter), 1))))
        out.sort(key=lambda x: x[1], reverse=True)
        if top_k is not None:
            out = out[:top_k]
        return out

    def sample_length(self, min_len: int, max_len: int) -> int:
        candidates = [(n, c) for n, c in self.length_counter.items() if min_len <= n <= max_len]
        if not candidates:
            min_len = max(min_len, min(self.length_counter))
            max_len = min(max_len, max(self.length_counter))
            candidates = [(n, c) for n, c in self.length_counter.items() if min_len <= n <= max_len]
            if not candidates:
                return random.randint(min_len, max_len)
        lengths, weights = zip(*candidates)
        return random.choices(lengths, weights=weights, k=1)[0]

    def sequence_logprob(self, syllables: List[str]) -> float:
        prev = START
        score = math.log(self.length_counter[len(syllables)] / (sum(self.length_counter.values()) + 1))
        for syll in syllables:
            score += self.transition_logp(prev, syll)
            prev = syll
        score += self.transition_logp(prev, END)
        return score

    def char_logprob(self, word: str) -> float:
        seq = "^" + word + "$"
        if not seq:
            return -100.0

        total_chars = max(len(self.char_counter), 1)
        result = 0.0
        for a, b in zip(seq[:-1], seq[1:]):
            nxts = self.char_pair_counter.get(a)
            if not nxts:
                result += -math.log(total_chars)
                continue
            total = sum(nxts.values())
            result += math.log((nxts.get(b, 0) + self.smoothing) / (total + self.smoothing * total_chars))
        return result

    def suffix_score(self, word: str) -> float:
        # Give zero reward if the word ends with a learned longer suffix,
        # small penalty otherwise.
        for n in (4, 3):
            if len(word) >= n and word[-n:] in self.top_suffixes_3_4:
                return 0.0
        return -1.2

    def beam_generate_one(self, target_syllables: int, beam: int, branch: int, temperature: float) -> List[Tuple[str, float, List[str]]]:
        if target_syllables <= 0:
            return []
        states = [([], START, 0.0)]

        for _ in range(target_syllables):
            next_states: List[Tuple[List[str], str, float]] = []
            for seq, prev, score in states:
                options = self.transition_logprobs(prev, include_end=False, top_k=branch)
                if not options:
                    continue
                syms, lps = zip(*options)
                # keep a little randomness while preserving high-probability exploration
                attempts = min(3, len(syms))
                choices = {weighted_choice(list(syms), list(lps), temperature) for _ in range(attempts)}
                for nxt in choices:
                    next_states.append((seq + [nxt], nxt, score + self.transition_logp(prev, nxt)))

            if not next_states:
                break
            next_states.sort(key=lambda x: x[2], reverse=True)
            states = next_states[:beam]

        out = []
        for seq, prev, score in states:
            if len(seq) != target_syllables:
                continue
            out.append(("".join(seq), score + self.transition_logp(prev, END), seq))
        out.sort(key=lambda x: x[1], reverse=True)
        return out


class Generator:
    def __init__(
        self,
        model: NameModel,
        min_len: int,
        max_len: int,
        min_syllables: int,
        max_syllables: int,
        novelty_cutoff: int,
        profile: StyleProfile,
    ) -> None:
        self.model = model
        self.vowels = set(profile.vowels)
        self.min_len = min_len
        self.max_len = max_len
        self.min_syllables = min_syllables
        self.max_syllables = max_syllables
        self.novelty_cutoff = max(0, novelty_cutoff)
        self.profile = profile
        self.allowed_suffixes = sorted(model.top_suffixes, key=len, reverse=True)
        self.suffix_pool = [s for s in model.top_suffixes if len(s) >= 3]
        self.min_vowel_ratio = profile.min_vowel_ratio
        self.max_vowel_ratio = profile.max_vowel_ratio
        self.morph_endings = tuple(profile.morph_endings) if profile.morph_endings else ()

    def _vowel_ratio(self, word: str) -> float:
        return sum(1 for ch in word if ch in self.vowels) / max(len(word), 1)

    def _has_forbidden_runs(self, word: str) -> bool:
        i = 0
        while i < len(word):
            if word[i] in self.vowels:
                i += 1
                continue
            j = i
            while j < len(word) and word[j] not in self.vowels:
                j += 1
            run = word[i:j]
            if len(run) >= 4:
                return True
            if len(run) == 2 and run not in self.model.allowed_runs2:
                return True
            if len(run) == 3 and run not in self.model.allowed_runs3:
                return True
            i = j
        return False

    def _ends_with_allowed_suffix(self, word: str) -> bool:
        if len(word) < 3:
            return False
        for ending in self.morph_endings:
            if word.endswith(ending):
                return True
        for suff in self.suffix_pool:
            if word.endswith(suff):
                return True
        for suff in self.allowed_suffixes:
            if len(suff) <= len(word) and word.endswith(suff):
                return True
        return False

    def _distance_to_train(self, word: str) -> int:
        candidates: Set[str] = set()
        if len(word) >= 2:
            candidates.update(self.model.first2_buckets.get(word[:2], []))
            candidates.update(self.model.last2_buckets.get(word[-2:], []))
        for n in range(max(1, len(word) - 4), len(word) + 5):
            candidates.update(self.model.names_by_length.get(n, []))

        if not candidates:
            candidates = set(self.model.name_set)
        elif len(candidates) > 250:
            candidates = set(random.sample(list(candidates), 250))

        best = 999
        for train in candidates:
            d = levenshtein(word, train)
            if d < best:
                best = d
                if best <= self.novelty_cutoff:
                    return best
        return best

    def is_valid(self, word: str, syllables: List[str]) -> bool:
        if word in self.model.name_set:
            return False
        if not (self.min_len <= len(word) <= self.max_len):
            return False
        if not (self.min_syllables <= len(syllables) <= self.max_syllables):
            return False
        ratio = self._vowel_ratio(word)
        if not (self.min_vowel_ratio <= ratio <= self.max_vowel_ratio):
            return False
        if re.search(r"(.)\\1\\1", word):
            return False
        if self._has_forbidden_runs(word):
            return False
        if not self._ends_with_allowed_suffix(word):
            return False
        if self._distance_to_train(word) <= self.novelty_cutoff:
            return False
        return True

    def score(self, word: str, syllables: List[str], sequence_score: float) -> float:
        char = self.model.char_logprob(word) / max(len(word), 1)
        suffix = self.model.suffix_score(word)
        length_pref = -abs(len(word) - 8) / 8.0
        return (
            0.50 * sequence_score
            + 0.26 * char
            + 0.15 * suffix
            + 0.09 * length_pref
        )

    def generate_many(self, n: int, sample_pool: int, beam: int, branch: int, temperature: float) -> List[Candidate]:
        raw: List[Tuple[str, List[str], float]] = []
        seen: Set[str] = set()
        attempts = 0
        max_attempts = max(sample_pool * 6, max(400, n * 25))

        while len(raw) < sample_pool and attempts < max_attempts:
            attempts += 1
            target = self.model.sample_length(self.min_syllables, self.max_syllables)
            beam_output = self.model.beam_generate_one(
                target, beam=beam, branch=branch, temperature=temperature
            )
            if not beam_output:
                continue

            weights = [math.exp(score) for _, score, __ in beam_output]
            for idx in random.choices(range(len(beam_output)), weights=weights, k=min(4, len(beam_output))):
                word, __, sylls = beam_output[idx]
                if word in seen:
                    continue
                if self.is_valid(word, sylls):
                    seen.add(word)
                    raw.append((word, sylls, self.model.sequence_logprob(sylls)))

        scored: List[Candidate] = []
        for word, sylls, logp in raw:
            scored.append(
                Candidate(
                    word=word,
                    syllables=sylls,
                    score=self.score(word, sylls, logp),
                    details={
                        "sequence_logprob": logp,
                        "char_logprob": self.model.char_logprob(word),
                        "suffix_score": self.model.suffix_score(word),
                        "distance_to_train": self._distance_to_train(word),
                    },
                )
            )
        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:n]


def dedupe_preserve_order(names: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for n in names:
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def load_styled_corpus(path: Path, ascii_only: bool, style: str) -> Tuple[Dict[str, List[str]], List[str]]:
    by_style: Dict[str, List[str]] = {}
    style_names: List[str] = []
    if not path or not path.exists():
        return by_style, style_names

    raw: Dict[str, List[str]] = defaultdict(list)
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        row = raw_line.strip()
        if not row or row.startswith("#"):
            continue
        if "\t" in row:
            style_name, name = row.split("\t", 1)
            style_name = style_name.strip().lower()
        else:
            style_name = START_STYLE_CORPUS
            name = row

        # Normalize lazily; actual profile-specific filtering happens later.
        norm = re.sub(r"[\s\-']", "", name.lower())
        if not norm:
            continue
        raw[style_name].append(name.strip())

    by_style = {k: dedupe_preserve_order(v) for k, v in raw.items()}
    style_names = by_style.get(style, [])
    return by_style, style_names


def load_style_file(styles_dir: Path, style: str, profile: StyleProfile, ascii_only: bool) -> List[str]:
    path = styles_dir / f"{style}.txt"
    if not path.exists():
        return []
    names: List[str] = []
    for row in path.read_text(encoding="utf-8").splitlines():
        row = row.strip()
        if not row or row.startswith("#"):
            continue
        norm = normalize(row, profile, ascii_only=ascii_only)
        if norm:
            names.append(norm)
    return dedupe_preserve_order(names)


def get_profile_for_style(style: str) -> StyleProfile:
    normalized = style.lower().strip()
    if normalized in STYLE_PROFILES:
        return STYLE_PROFILES[normalized]

    label = normalized.replace("-", " ").replace("_", " ").title()
    return StyleProfile(
        name=normalized,
        display=label,
        vowels=DEFAULT_PROFILE.vowels,
        allowed_chars=DEFAULT_PROFILE.allowed_chars,
        min_len=DEFAULT_PROFILE.min_len,
        max_len=DEFAULT_PROFILE.max_len,
        min_syllables=DEFAULT_PROFILE.min_syllables,
        max_syllables=DEFAULT_PROFILE.max_syllables,
        novelty_cutoff=DEFAULT_PROFILE.novelty_cutoff,
        sample_pool=DEFAULT_PROFILE.sample_pool,
        beam=DEFAULT_PROFILE.beam,
        branch=DEFAULT_PROFILE.branch,
        temperature=DEFAULT_PROFILE.temperature,
        min_vowel_ratio=DEFAULT_PROFILE.min_vowel_ratio,
        max_vowel_ratio=DEFAULT_PROFILE.max_vowel_ratio,
        morph_endings=DEFAULT_PROFILE.morph_endings,
    )


def gather_corpus(style: str, corpus_path: str, styles_dir: str, ascii_only: bool) -> List[str]:
    profile = get_profile_for_style(style)
    path = Path(corpus_path)
    by_style, _ = load_styled_corpus(path, ascii_only, style)

    names = by_style.get(style.lower(), [])
    if not names and path.exists() and style.lower() == START_STYLE_CORPUS:
        names = by_style.get(START_STYLE_CORPUS, [])

    names.extend(load_style_file(Path(styles_dir), style.lower(), profile, ascii_only))
    if not names:
        names = BUILTIN_STYLE_CORPORA.get(style.lower(), [])

    # normalize based on style profile and remove empties
    names = dedupe_preserve_order(normalize(name, profile, ascii_only=ascii_only) for name in names)
    return names


def pick_profile_arg(value: int, fallback: int, sentinel: int = 0) -> int:
    return fallback if value == sentinel else value


def pick_profile_arg_float(value: float, fallback: float, sentinel: float = -1.0) -> float:
    return fallback if value == sentinel else value


def run_generation(args: argparse.Namespace) -> None:
    style = args.style.lower().strip()
    profile = get_profile_for_style(style)

    names = gather_corpus(
        style=style,
        corpus_path=args.corpus,
        styles_dir=args.styles_dir,
        ascii_only=args.ascii,
    )
    if not names:
        raise ValueError(f"No corpus available for style '{style}'")

    model = NameModel(names, profile=profile, smoothing=args.smoothing)

    min_len = pick_profile_arg(args.min_len, profile.min_len, 0)
    max_len = pick_profile_arg(args.max_len, profile.max_len, 0)
    min_syllables = pick_profile_arg(args.min_syllables, profile.min_syllables, 0)
    max_syllables = pick_profile_arg(args.max_syllables, profile.max_syllables, 0)
    novelty_cutoff = pick_profile_arg(args.novelty_cutoff, profile.novelty_cutoff, -1)
    sample_pool = pick_profile_arg(args.sample_pool, profile.sample_pool, 0)
    beam = pick_profile_arg(args.beam, profile.beam, 0)
    branch = pick_profile_arg(args.branch, profile.branch, 0)
    temperature = pick_profile_arg_float(args.temperature, profile.temperature, -1.0)

    gen = Generator(
        model=model,
        min_len=min_len,
        max_len=max_len,
        min_syllables=min_syllables,
        max_syllables=max_syllables,
        novelty_cutoff=novelty_cutoff,
        profile=profile,
    )

    out = gen.generate_many(
        n=args.n,
        sample_pool=sample_pool,
        beam=beam,
        branch=branch,
        temperature=temperature,
    )

    print(
        f"style={profile.display} requested={args.n} output={len(out)} "
        f"corpus={len(names)} unique={len(set(names))}"
    )
    for i, c in enumerate(out, start=1):
        parts = ".".join(c.syllables)
        print(f"{i:02d}: {c.word:12} syll={len(c.syllables):1d} score={c.score: .3f} parts={parts}")


def run_self_test(args: argparse.Namespace) -> None:
    for n in (10, 20):
        print(f"\n== selftest n={n} ==")
        a = argparse.Namespace(**vars(args))
        a.n = n
        run_generation(a)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reusable style-aware name generator")
    p.add_argument("--corpus", default=DEFAULT_CORPUS)
    p.add_argument("--styles-dir", default=DEFAULT_STYLES_DIR)
    p.add_argument(
        "--style",
        default="scandi",
        help="style profile to use (e.g. scandi, italian, greek, klingon, roman)",
    )
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--sample-pool", type=int, default=0, help="0 uses style defaults")
    p.add_argument("--beam", type=int, default=0, help="0 uses style defaults")
    p.add_argument("--branch", type=int, default=0, help="0 uses style defaults")
    p.add_argument("--temperature", type=float, default=-1.0, help="0 uses style defaults")
    p.add_argument("--min-len", type=int, default=0, help="0 uses style defaults")
    p.add_argument("--max-len", type=int, default=0, help="0 uses style defaults")
    p.add_argument("--min-syllables", type=int, default=0, help="0 uses style defaults")
    p.add_argument("--max-syllables", type=int, default=0, help="0 uses style defaults")
    p.add_argument("--novelty-cutoff", type=int, default=-1, help="0 uses style defaults")
    p.add_argument("--smoothing", type=float, default=0.7)
    p.add_argument("--ascii", action="store_true", help="normalize by dropping diacritics")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    if args.seed is not None:
        random.seed(args.seed)

    if args.self_test:
        run_self_test(args)
    else:
        run_generation(args)


if __name__ == "__main__":
    main()
