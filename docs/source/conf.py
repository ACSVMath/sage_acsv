import os
import re
import importlib.metadata
from sphinx import highlighting
from IPython.lib.lexers import IPythonConsoleLexer, IPyLexer
from sage_docbuild.conf import (
    html_theme,
    html_theme_options,
    pygments_style,
    pygments_dark_style,
    html_css_files,
    skip_TESTS_block,
    mathjax3_config,
    default_role,
)

extensions = [
    # We need to use SageMath's autodoc to render nested classes in categories
    # correctly. Otherwise they just render as "alias for" in the
    # documentation.
    "sage_docbuild.ext.sage_autodoc",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    # "myst_nb",
    # "jupyter_sphinx",
]

# Extensions when rendering .ipynb/.md notebooks
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
]

# The suffix of source filenames.
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "sage-acsv"
copyright = "2025, the sage-acsv authors"
release = importlib.metadata.version("sage-acsv")

source_repository = 'https://github.com/ACSVMath/sage_acsv/'
source_branch = 'main'

# Allow linking to external projects, e.g., SageMath
intersphinx_mapping = {"sage": ("https://doc.sagemath.org/html/en/reference", None)}

# -- Options for HTML output ----------------------------------------------
print(html_css_files)
if html_css_files != ["custom-furo.css", "custom-jupyter-sphinx.css", "custom-codemirror-monokai.css", "custom-tabs.css"]:
    raise NotImplementedError(
        "CSS customization has changed in SageMath. The configuration of sage_acsv "
        "documentation build needs to be updated."
    )

html_css_files = [
    "https://doc.sagemath.org/html/en/reference/_static/custom-furo.css",
    "https://doc.sagemath.org/html/en/reference/_static/custom-jupyter-sphinx.css",
    "https://doc.sagemath.org/html/en/reference/_static/custom-codemirror-monokai.css",
    "jupyter_execute.css"
]

html_theme_options["light_logo"] = html_theme_options["dark_logo"] = "acsv-logo.svg"
html_static_path = ["_static"]

html_theme_options.update({
    "source_repository": source_repository,
    "source_branch": source_branch,
    'source_view_link': os.path.join(source_repository, 'blob/main/docs/source/{filename}'),
    'source_edit_link': os.path.join(source_repository, 'edit/main/docs/source/{filename}'),
})

# Output file base name for HTML help builder.
htmlhelp_basename = "sage_acsvdoc"

# Options for jupyter-sphinx
jupyter_execute_default_kernel = "sagemath"

# Only rerender example notebooks when the cache is stale.
nb_execution_mode = "cache"

highlighting.lexers['ipycon'] = IPythonConsoleLexer(in1_regex=r'(sage:|>>>)', in2_regex=r'([.][.][.][.]:|[.][.][.])')
highlighting.lexers['ipython'] = IPyLexer()
highlight_language = 'ipycon'

def setup(app):
    app.connect('autodoc-process-docstring', skip_TESTS_block)


github_ref = os.environ.get('GITHUB_REF', '')
if github_ref:
    match = re.search(r'refs/pull/(\d+)/merge', github_ref)
    if match:
        pr_number = match.group(1)
is_for_github_pr = github_ref and match and pr_number

if is_for_github_pr:  # condition for announcement banner
    pr_url = f'https://github.com/ACSVMath/sage_acsv/pull/{pr_number}'
    banner = (
        f'This is a documentation preview for sage-acsv version {release} '
        f'built with GitHub PR <a href="{pr_url}">#{pr_number}</a>.'
    )
    html_theme_options.update({ "announcement": banner })
