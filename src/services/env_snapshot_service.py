"""
Service de snapshot d'environnement Python.

Extrait les modules importés d'un script train.py et résout leurs versions
installées pour générer un requirements.txt reproductible.
"""

import ast
import importlib.metadata
import sys

_MODULE_TO_PACKAGE: dict[str, str] = {
    "sklearn": "scikit-learn",
    "dotenv": "python-dotenv",
    "PIL": "Pillow",
    "cv2": "opencv-python",
    "bs4": "beautifulsoup4",
}


def extract_imports(source: str) -> set[str]:
    """Parse l'AST et retourne les noms de modules top-level importés."""
    tree = ast.parse(source)
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.add(node.module.split(".")[0])
    return modules


def dependencies_to_requirements_txt(deps: dict[str, str]) -> str:
    """Convertit un dict {package: version} en contenu requirements.txt trié."""
    lines = [f"{pkg}=={ver}" for pkg, ver in sorted(deps.items()) if ver]
    return "\n".join(lines) + "\n" if lines else "# No external dependencies detected\n"


def generate_requirements_txt(source: str) -> str:
    """
    Génère le contenu d'un requirements.txt à partir du source d'un script train.py.

    Filtre les modules stdlib, résout les versions installées via importlib.metadata.
    Retourne un fichier vide commenté si aucune dépendance externe n'est détectée.
    """
    stdlib = getattr(sys, "stdlib_module_names", set())
    modules = extract_imports(source)
    lines: list[str] = []
    for mod in sorted(modules):
        if mod in stdlib:
            continue
        pkg = _MODULE_TO_PACKAGE.get(mod, mod)
        try:
            ver = importlib.metadata.version(pkg)
            lines.append(f"{pkg}=={ver}")
        except importlib.metadata.PackageNotFoundError:
            pass
    return "\n".join(lines) + "\n" if lines else "# No external dependencies detected\n"
