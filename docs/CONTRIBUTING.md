# Contributing

## Branching
For collaborative development create a branch named <github_username>/<feature_name>, e.g.
> massarin/polarisutils

## Pull requests
Always PR into `dev`, this way we can test interactions between merged features before pushing to `main`. From here the moderator of the repository will PR into `main`.

## Committing
Using [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/) allows for automated release and changelog generation, see
- [https://github.com/marketplace/actions/conventional-changelog-action](https://github.com/marketplace/actions/conventional-changelog-action), or
- [release-please](https://github.com/marketplace/actions/release-please-action) by google

## Linting
Install [ms-python.black-formatter](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter) on vscode, which will format your code automatically on save due to [settings.json](../.vscode/settings.json)

## Continuous Integration

Currently there is a CI pipeline that will build the project inside a docker container running `ROS2 Humble` on `Ubuntu:latest`. If the build fails or the smoke test does not pass, your PR or Push will be flagged.