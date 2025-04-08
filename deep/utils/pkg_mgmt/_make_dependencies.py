"""
Parses all imports in a directory to build a minimal requirements.txt.
"""

import os
import re
import json
import subprocess
import tempfile

def extract_imports(filepath):
    """Extracts import statements based on the file type (script or notebook)."""
    
    def _as_code(code):
        """Parses given code and returns all valid instances of import statements."""
        imports = set()
        for line in code.splitlines():
            line = line.strip()
            if not line:
                continue
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

def load_installed_modules():
    """Loads the installed system modules from the sys_modules.txt file."""
    sys_modules_path = os.path.join(package_root, 'sys_modules.txt')
    installed_modules = set()

    if os.path.exists(sys_modules_path):
        with open(sys_modules_path, 'r', encoding='utf-8') as f:
            installed_modules = {line.strip() for line in f.readlines()}
    
    return installed_modules

def collect_dependencies(directory):
    """Collects third-party dependencies from all Python scripts and notebooks in a directory."""
    dependencies = set()
    installed_modules = load_installed_modules()

    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                imports = extract_imports(file_path)
                third_party = {imp for imp in imports if imp not in installed_modules}
                
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
        current_file = os.path.abspath(__file__)
        package_root = os.path.abspath(os.path.join(current_file, '..', '..', '..', '..'))

        # Collect dependencies
        dependencies = collect_dependencies(package_root)

        if not dependencies:
            raise RuntimeError("No third-party dependencies found.")

        # Create a temporary .in file with the dependencies
        with tempfile.NamedTemporaryFile('w+', suffix='.in', delete=False) as temp_in:
            temp_in.write('\n'.join(dependencies) + '\n')
            temp_in_path = temp_in.name
        
        # Output path for final resolved requirements
        output_path = os.path.join(package_root, 'requirements.txt')

        # Call pip-compile to resolve versions and write to requirements.txt
        result = subprocess.run(
            ['pip-compile', temp_in_path, '--output-file', output_path],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"pip-compile failed:\n{result.stderr}")
        else:
            print(f"requirements.txt generated successfully at: {output_path}")
        
        # Clean up temporary .in file
        os.remove(temp_in_path)

    except RuntimeError as rte:
        print(f"Fatal error: {rte}; \n\tWhile attempting to build requirements file, aborting now.")
