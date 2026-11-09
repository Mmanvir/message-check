# Source pack

One `.txt` file per source document. Header lines, then a line with only `---`, then the saved text.
Blank lines separate passages (each passage is one search unit).

```
id: PIB-2026-03-14-02          # unique, cited in verdicts
title: ...
date: 2026-03-14               # date of the source
category: government | health
url: https://...               # where it was saved from (never fetched by the app)
licence: ...
---
Paragraph one.

Paragraph two.
```

The `SAMPLE-*` files are fictional fixtures so the pipeline can be tested. Replace them with real
saved PIB Fact Check, press release, WHO and state health advisory texts, and update `eval/cases` to match.
