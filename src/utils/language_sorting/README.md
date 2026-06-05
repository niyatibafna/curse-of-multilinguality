# Language Sorting Utilities

## Sort Languages by Wikipedia Article Count

```bash
python src/utils/language_sorting/sort_by_wiki_count.py langs.txt > sorted_langs.csv
python src/utils/language_sorting/sort_by_wiki_count.py --dataset floresplus --quiet > sorted_langs.csv
```

Defaults:

- `language_to_wiki.csv`: editable benchmark-code to Wikipedia-code mapping.
- `wiki_counts.csv`: normalized Wikipedia article counts.
- `dataset_languages.csv`: language lists for the current `bouquet`,
  `floresplus`, and `wmt24pp` runs.

The sorter also accepts the raw Wikistats CSV with `prefix,good` columns:

```bash
python src/utils/language_sorting/sort_by_wiki_count.py langs.txt language_to_wiki.csv raw_wikistats.csv
```

Refresh the local count snapshot with one Wikistats download:

```bash
python src/utils/language_sorting/update_wiki_counts.py
```
