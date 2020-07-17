# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import inspect
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'Splice MLManager'
copyright = '2020, Splice Machine'
author = 'Ben Epstein'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx.ext.autosummary'
]


# To allow sphinx to document the mlflow_support functions since they are private functions
#autodoc_default_flags = ['members', 'undoc-members', 'private-members', 'special-members', 'inherited-members', 'show-inheritance']
autodoc_default_options = {
    'members':True,
    'private-members':True,
    'inherited-members':True,
    'undoc-members': False, 
    'exclude-members': '_check_for_splice_ctx,_dropTableIfExists, _generateDBSchema,_getCreateTableSchema,_jstructtype,_spliceSparkPackagesName,_splicemachineContext,apply_patches, main'
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'en'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['**mlflow_support.utilities.*','_build', 'Thumbs.db', '.DS_Store','*test/*', '*.test.*','*test*']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Removes the full module path for the function name
add_module_names = False
# Allows us to remove the _ from the mlflow_support functions
autodoc_docstring_signature=True
# So we don't have to have pyspark
#autodoc_mock_imports = ["pyspark"]
# Set master doc
master_doc = 'index'

# To skip the mlflow_support.utilities/constants and spark.constants modules
def check_skip_member(app, what, name, obj, skip, options):
    try:
        mro = obj.__module__
    except:
        mro = ''
    #print('WHAT:', what, 'NAME:', name, 'OBJ:', mro, end='==>')
    if "constants" in mro or 'utilities' in mro:
        return True
    else:
        return None # Default behavior

def setup(app):
    app.connect("autodoc-skip-member", check_skip_member)

# -- Extension configuration -------------------------------------------------

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True
