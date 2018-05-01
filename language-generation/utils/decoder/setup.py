#!/usr/bin/env python

"""
setup.py file for SWIG example
"""

from distutils.core import setup, Extension


trie_module = Extension('_trie',
                        sources=['trie_wrap.cxx', 'trie.cpp'],
                        )

setup (name = 'trie',
       version = '0.1',
       author      = "SWIG Docs",
       description = """Simple swig example from docs""",
       ext_modules = [trie_module],
       py_modules = ["trie"],
       )
