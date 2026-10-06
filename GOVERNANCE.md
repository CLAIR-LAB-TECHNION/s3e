# Governance

S3E is developed at the [CLAIR Lab](https://github.com/CLAIR-LAB-TECHNION)
at the Taub Faculty of Computer Science, Technion – Israel Institute of
Technology, which uses it in its own research.

## Maintainers

- Guy Azran ([@guyazran](https://github.com/guyazran)), lead maintainer

## How changes are made

Contributions are made through pull requests, which a maintainer reviews and
merges once continuous integration passes. The lead maintainer decides on the
project's direction, on changes to the public API, and on releases.
[`CONTRIBUTING.md`](CONTRIBUTING.md) describes how to propose a change.

## Releases and compatibility

S3E follows [semantic versioning](https://semver.org/). Before version 1.0,
a minor release (0.x.0) may change the public API; every such change is marked
**Breaking** in the [changelog](CHANGELOG.md). Releases are published on PyPI,
tagged `vX.Y.Z`, and announced as GitHub Releases, following the steps in
[`CONTRIBUTING.md`](CONTRIBUTING.md#releasing).

## Support

Questions, bug reports, and feature requests go to the
[issue tracker](https://github.com/CLAIR-LAB-TECHNION/s3e/issues/new/choose),
where the maintainers triage them on a best-effort basis. Security problems
should be reported privately, as described in [`SECURITY.md`](SECURITY.md).
