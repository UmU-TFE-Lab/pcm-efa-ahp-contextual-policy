# GitHub release checklist

- [ ] Select and add an explicit software license after all authors approve it.
- [ ] Replace repository and archival DOI placeholders in the manuscript after the public release exists.
- [ ] Confirm that all authors approve publishing the expert-judgment constants.
- [ ] Run `python scripts/check_repository.py` and resolve every finding.
- [ ] Run `python -m compileall -q pcm_efa_ahp pcm_efa_ahp_rl pcm_journal_extension scripts`.
- [ ] Run `pytest` without access to the private study table.
- [ ] Reproduce the full analysis locally with the authorized input and compare the reported tables.
- [ ] Inspect vector figures for font substitution and clipping; keep editable PowerPoint sources outside this repository.
- [ ] Run `git status --short --ignored` and verify that private data and generated outputs appear only as ignored files.
- [ ] Review `git diff --cached --stat` before the first commit.

Suggested first publication commands, after creating an empty GitHub repository:

```bash
git add .
python scripts/check_repository.py
git commit -m "Release manuscript analysis code"
git remote add origin https://github.com/OWNER/REPOSITORY.git
git push -u origin main
```

Do not use `git add -f` for any ignored data or output file.
