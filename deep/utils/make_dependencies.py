"""
This script extracts third-party dependencies from Python scripts and Jupyter notebooks in a directory and writes them to a text file.
"""

import os
import re
import json
import importlib.util

def extract_imports(filepath):
    """Extracts import statements based on the file type (script or notebook)."""
    
    def _as_code(code):
        """Parses given code and returns all valid instances of import statements."""
        imports = set()
        for line in code.splitlines():
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            # Strip inline comments
            line = line.split('#')[0].strip()
            match = re.match(r'^(?:from|import)\s+([a-zA-Z0-9_\.]+)', line)
            if match:
                module = match.group(1).split('.')[0]
                imports.add(module)
        return imports

    def _from_notebook(nb_path):
        """Extracts import statements from Jupyter notebook and returns them."""
        with open(nb_path, 'r', encoding='utf-8') as f:
            try:
                nb = json.load(f)
                code_cells = [
                    cell['source'] for cell in nb.get('cells', [])
                    if cell.get('cell_type') == 'code'
                ]
                combined_code = '\n'.join([''.join(cell) for cell in code_cells])
                return _as_code(combined_code)
            except Exception as e:
                print(f"Error parsing notebook {nb_path}: {e}")
                return set()

    def _from_script(filepath):
        """Extracts import statements from a Python script and returns them."""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return _as_code(content)

    # Determine file type and call the respective private method
    if filepath.endswith('.py'):
        return _from_script(filepath)
    elif filepath.endswith('.ipynb'):
        return _from_notebook(filepath)
    else:
        return set()

def is_third_party(module_name):
    """Checks if a module is a third-party library."""
    try:
        spec = importlib.util.find_spec(module_name)
        if spec and spec.origin:
            return 'site-packages' in spec.origin
        return False
    except Exception:
        return False

def collect_dependencies(directory):
    """Collects third-party dependencies from all Python scripts and notebooks in a directory."""
    dependencies = set()
    
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                imports = extract_imports(file_path)  # Use the extract_imports function to handle both .py and .ipynb files
                third_party = {imp for imp in imports if is_third_party(imp)}
                dependencies.update(third_party)
            except Exception as e:
                print(f"Could not process {file_path}: {e}")

    return sorted(dependencies)

def write_dependencies_to_file(dependencies, output_path):
    """Writes collected dependencies to a text file."""
    with open(output_path, 'w') as f:
        for dep in dependencies:
            f.write(dep + '\n')

if __name__ == "__main__":
    try:
        # Build package root path
        current_file = os.path.abspath(__file__)
        package_root = os.path.abspath(os.path.join(current_file, '..', '..', '..'))

        # Collect dependencies
        dependencies = collect_dependencies(package_root)

        # Build file path
        output_file = os.path.join(package_root, 'dependencies.txt')
        
        # Write file to path
        write_dependencies_to_file(dependencies, output_file)

        # Assert completion
        print(f"Dependencies written to: {output_file}")
    except RuntimeError as rte:
        print(f"Fatal error:  {rte}; \n\t While attempting to build requirements file, aborting now.")
