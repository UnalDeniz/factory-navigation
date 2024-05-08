# A Robotic Simulation Environment

## Get started

1. Install necessary system packages on our Ubuntu machine using the command:

   ```
   sudo apt install git python3-pip python3-venv python-is-python3
   ```

   1. `python3-pip` is the package manager for Python 3, which allows you to install and manage Python packages.
   2. `python3-venv` is the module for creating isolated Python environments, which are useful for keeping the dependencies for different projects separate.
   3. `python-is-python3` is a utility package to ensure that we use Python 3 when we type `python`.

   Note that the `sudo` command is used to run the command as a superuser, which is necessary when installing packages globally on the system. The `apt` command stands for Advanced Packaging Tool and is used to manage packages on Debian-based systems like Ubuntu.

2. gcc compiler and gnu make should be installed by default in your system. If you are using a minimal installation and do not have access to those, install them by using your package manager.

3. Clone the repository and go to the project root directory
   ```
   git clone https://github.com/UnalDeniz/factory-navigation
   cd factory-navigation
   ```

4. Run the code by using make command:
   ```
   make run
   ```

5. (Optional) Clean the python environment and c shared library using make command:
   ```
   make clean
   ```
