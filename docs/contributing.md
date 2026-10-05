# Contributing and support

- **Questions and bug reports:** open an issue on the
  [GitHub issue tracker](https://github.com/CLAIR-LAB-TECHNION/s3e/issues/new/choose).
- **Contributing code:** see
  [`CONTRIBUTING.md`](https://github.com/CLAIR-LAB-TECHNION/s3e/blob/main/CONTRIBUTING.md)
  for the development setup, test commands, and style conventions.
- **Code of conduct:** participation is governed by the
  [Contributor Covenant](https://github.com/CLAIR-LAB-TECHNION/s3e/blob/main/CODE_OF_CONDUCT.md).

To build this documentation locally:

```bash
pip install -e ".[docs]"
sphinx-build -W -b html docs docs/_build/html
```
