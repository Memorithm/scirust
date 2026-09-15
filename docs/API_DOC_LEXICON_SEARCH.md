# Searching the API lexicon

The source lexicon supports package/domain/text filters and missing-summary work
queues. Typical uses:

```bash
python3 scripts/api-lexicon.py --package scirust-core --query tensor
python3 scripts/api-lexicon.py --package scirust-core --missing-docs
python3 scripts/api-lexicon.py --list-domains
```

Use filtered output for navigation/review queues. Documentation-debt baseline
checks intentionally require the unfiltered lexicon so a filter cannot hide a
regression elsewhere.

The future v2 search should additionally filter by reachability, example status
and contract-section observations while preserving these existing queries.
