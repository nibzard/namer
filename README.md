# Name Generator

A reusable fake-name generator with style-aware generation and filtering.

The script learns phonotactic patterns from a corpus of real names (or theme examples),
then generates pronounceable, novel names using:
- syllable transition probabilities
- character-level probabilities
- suffix and vowel/rhyme filters
- optional novelty filtering against training names

## Files

- `namegen.py` — generator code
- `data/default_corpus.txt` — default Swedish/Nordic-style corpus
- `data/styles/` — per-style custom corpora (one plain text file per style)

## Quick Start

Run directly via `uvx` (Astral `uv`):

```bash
uvx --with-editable . namegen --style scandi --n 10
```

From GitHub (once this repo has been pushed):

```bash
uvx --from git+https://github.com/nibzard/namer.git namegen --style roman --n 10
```

Equivalent direct execution (local checkout):

```bash
python3 namegen.py --style scandi --n 10
python3 namegen.py --style italian --n 10
python3 namegen.py --style greek --n 10
python3 namegen.py --style klingon --n 10
python3 namegen.py --style roman --n 10
```

Use a seed for deterministic output:

```bash
python3 namegen.py --style scandi --n 10 --seed 13
```

## Available built-in styles

- `scandi`
- `italian`
- `greek`
- `klingon`
- `roman`

Any other style name is allowed; if a corresponding file exists under `data/styles/<style>.txt`, it will be used automatically.

## Adding a new style/theme

Create a file at:

```bash
data/styles/<style>.txt
```

with one seed word per line, for example:

```text
glistaris
xalthor
sorvane
nyxaris
```

Then run:

```bash
python3 namegen.py --style <style> --styles-dir data/styles --n 8
uvx --with-editable . namegen --style <style> --styles-dir data/styles --n 8
```

If no file/known built-in style exists, generation fails with:

`ValueError: No corpus available for style '<style>'`.

## Included theme corpora

These are already present under `data/styles/`:
- `arcane`
- `bioforge`
- `vaporwave`
- `asteroid`
- `mythic`
- `jungle`
- `steampunk`
- `tundra`
- `noir`
- `haunted`

## Useful flags

```bash
--style <name>          style profile or custom theme
--styles-dir <path>     directory with style files (default: data/styles)
--corpus <path>         main corpus (default: project's bundled data/default_corpus.txt)
--n <int>               number of final names to emit (default: 20)
--sample-pool <int>     internal candidate attempts (0 = style default)
--beam <int>            beam width (0 = style default)
--branch <int>           branch width in beam expansion (0 = style default)
--temperature <float>    sampling temperature (0 = style default)
--min-len/--max-len     character length bounds (0 = style default)
--min-syllables/--max-syllables  syllable count bounds (0 = style default)
--novelty-cutoff <int>  min Levenshtein distance from training names (0 = style default)
--smoothing <float>     smoothing for log-probability estimates
--ascii                 normalize by dropping diacritics
--seed <int>            deterministic output
--self-test              run two quick self tests (n=10 and n=20)
```

## How scoring/filtering works (high level)

- Generates many candidates from style-specific syllable Markov chains.
- Filters invalid candidates by:
  - length and syllable constraints
  - vowel ratio
  - disallowed repeated letters/consonant runs
  - ending plausibility
  - duplicate/training-name similarity checks
- Ranks survivors with combined sequence + character + suffix scores and returns top `n`.

## Example output

```text
style=Scandinavian requested=8 output=8 corpus=57137
01: inen       syll=2 score=-7.753 parts=in.en
02: södrala    syll=3 score=-9.358 parts=södr.al.a
...
```
