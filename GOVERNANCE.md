# Governance

`s3e` is developed at the [CLAIR Lab](https://github.com/CLAIR-LAB-TECHNION)
at the Taub Faculty of Computer Science, Technion – Israel Institute of
Technology, which uses it in its own research.

## Maintainers

- Guy Azran ([@guyazran](https://github.com/guyazran)), lead maintainer

## How changes are made

Every change lands through a pull request that passes continuous integration
and is reviewed by a maintainer. The lead maintainer decides on the project's
direction, on changes to the public API, and on releases.
[`CONTRIBUTING.md`](CONTRIBUTING.md) describes how to propose a change.

## Releases and compatibility

`s3e` follows [semantic versioning](https://semver.org/). Before version 1.0,
a minor release (0.x.0) may change the public API; every such change is marked
**Breaking** in the [changelog](CHANGELOG.md). Each release is tagged `vX.Y.Z`, published on PyPI, and announced as a
GitHub Release; the steps are in [`CONTRIBUTING.md`](CONTRIBUTING.md#releasing).

## Support

Questions, bug reports, and feature requests go to the
[issue tracker](https://github.com/CLAIR-LAB-TECHNION/s3e/issues/new/choose),
where the maintainers triage them on a best-effort basis. Security problems
should be reported privately, as described in [`SECURITY.md`](SECURITY.md).
