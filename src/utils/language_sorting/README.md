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

## Sort Languages by MADLAD Resource Size

```bash
python src/utils/language_sorting/sort_by_madlad_resource.py --dataset floresplus --quiet > sorted_langs.csv
python src/utils/language_sorting/sort_by_madlad_resource.py langs.txt --key clean_docs > sorted_langs.csv
python src/utils/language_sorting/sort_by_madlad_resource.py --all-madlad > madlad_langs_by_clean_bytes.csv
```

Defaults:

- `language_to_madlad.csv`: editable benchmark-code to MADLAD-code mapping.
- `madlad_counts.csv`: normalized counts parsed from the MADLAD dataset-card
  final table.

The default sort key is `clean_bytes`. Other keys include `clean_docs`,
`clean_sents`, `clean_tokens`, `clean_chars`, and the corresponding `noisy_*`
columns.

Refresh the local MADLAD count snapshot from the dataset-card table:

```bash
python src/utils/language_sorting/update_madlad_counts.py
```
