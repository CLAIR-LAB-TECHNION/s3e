# Pending GitHub Actions workflows

`tests.yml` and `draft-pdf.yml` belong in `.github/workflows/`. They are
committed here only because the credential used to push this branch lacks
GitHub's `workflow` scope, which is required to create files under
`.github/workflows/`. Move them with a credential that has that scope (or
recreate them through the GitHub web UI):

```bash
mkdir -p .github/workflows
git mv .github/pending-workflows/tests.yml .github/pending-workflows/draft-pdf.yml .github/workflows/
git rm .github/pending-workflows/README.md
git commit -m "ci: enable test, docs, and draft-PDF workflows"
```
