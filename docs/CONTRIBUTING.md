# Contributing

## Adding Code/Packages
When contributing new packages or modules to the project:

### Python
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
3. **Node integration**: 
Add an entry to `console_scripts` in `src/py_pkg/setup.py`
```Python
entry_points={
    'console_scripts': [
        ...,
        '<YOUR_NODE> = py_pkg.<YOUR_NODE>:main',
        ...
    ],
},
```

4. **Add to [README](../README.md)**: One line description and link to your package in the **Current packages** section

### C++
1. **Package Structure**: Add your package following this structure:
   ```
   src/cpp_pkg/
   ├── src/
   │   └── your_package_name/
   │       ├── your_node.cpp        # Your ROS2 node implementation
   │       ├── your_library.cpp     # Additional implementation files
   │       └── examples/            # Usage examples
   │           └── basic_usage.cpp
   ├── include/cpp_pkg/
   │   └── your_package_name/
   │       ├── your_node.hpp        # Header files
   │       └── your_library.hpp
   └── README.md                    # Package documentation
   ```

2. **CMakeLists.txt Integration**: Register your executable in `src/cpp_pkg/CMakeLists.txt`:
   ```cmake
   # Add executable
   add_executable(<YOUR_NODE> src/your_package_name/<YOUR_NODE>.cpp)

   # Link dependencies
   ament_target_dependencies(<YOUR_NODE> rclcpp std_msgs geometry_msgs sensor_msgs)

   # Install executable
   install(TARGETS <YOUR_NODE> DESTINATION lib/${PROJECT_NAME})

   # Add include directories
   target_include_directories(<YOUR_NODE> PUBLIC
     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
     $<INSTALL_INTERFACE:include>)
   ```

3. **Dependencies**: If you need additional ROS2 packages, add them to both:
   - `find_package()` in CMakeLists.txt
   - `<depend>` tags in package.xml

4. **Add to [README](../README.md)**: One line description and link to your package in the **Current packages** section

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

## Continuous Intergration

Currently there is a CI pipeline that will build the project inside a docker container running `ROS2 Humble` on `Ubuntu:latest`. If the build fails or the smoke test does not pass, your PR or Push will be flagged.