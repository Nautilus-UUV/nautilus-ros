# Contributing

## Adding Code/Packages
When contributing new packages or modules to the project:

1. **Package Structure**: Add your package following this structure:
   ```
   src/py_pkg/py_pkg/
   ├── __init__.py
   └── your_package_name/
       ├── __init__.py          # Expose main classes/functions
       ├── README.md            # Package documentation
       ├── examples/            # Usage examples
       │   └── basic_usage.py
       ├── module1.py           # Your implementation files
       └── module2.py
   ```

2. **Main Package Integration**: Update `src/py_pkg/py_pkg/__init__.py` to import your package:
   ```python
   from . import your_package_name
   ```
3. **Add to [README](../README.md)**: One line description and link to your package in the **Current packages** section

## Branching
For collaborative development create a branch named <github_username>/<feature_name>, e.g.
> massarin/polarisutils

## Pull requests
Always PR into `dev`, this way we can test interactions between merged features before pushing to `main`. From here the moderator of the repository will PR into `main`.

## Committing
Using [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/) allows for automated release and changelog generation, see
- [https://github.com/marketplace/actions/conventional-changelog-action](https://github.com/marketplace/actions/conventional-changelog-action), or
- [release-please](https://github.com/marketplace/actions/release-please-action) by google