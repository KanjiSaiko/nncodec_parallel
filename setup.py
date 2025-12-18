import sys
import platform
import os
from setuptools import setup, Extension, find_packages
from glob import glob
from setuptools.command.build_ext import build_ext
import setuptools # Importa setuptools para usar errors
import tempfile # Importa tempfile aqui

MIN_PYTHON = (3, 6)

if sys.version_info < MIN_PYTHON:
    sys.exit( "Python {}.{} or later is required.".format( *MIN_PYTHON ) )

__version__ = '0.3.1'

# --- CAMINHOS PTHREADS CORRIGIDOS ---
# Assume que pthreads está em 'extensions/pthreads/'
pthreads_dir = os.path.join('extensions', 'pthreads')
pthreads_include_dir = os.path.join(pthreads_dir, 'include')
pthreads_lib_dir = os.path.join(pthreads_dir, 'lib', 'x64') # Assume 64 bits
# ------------------------------------


# --- DEFINIÇÃO DE FLAGS E LIBS ---
extra_compile_args = []
extra_link_args = []
libraries = []
library_dirs = []

if platform.system() == "Windows":
    # Flags para MSVC + Pthreads
    extra_compile_args = [
        '/EHsc', '/O2',
        '/DVERSION_INFO=\\"{}\\"'.format(__version__),
        '/DHAVE_STRUCT_TIMESPEC=1'
    ]
    libraries = ['pthreadVC2']
    library_dirs = [pthreads_lib_dir]
    extra_link_args = [] # MSVC geralmente encontra libs nos library_dirs
else:
    # Flags para GCC/Clang com Pthreads (geralmente padrão)
    extra_compile_args = [
        '-pthread', '-O3', '-std=c++11',
        '-DVERSION_INFO=\\"{}\\"'.format(__version__)
    ]
    extra_link_args = ['-pthread']
# --- FIM DEFINIÇÃO DE FLAGS E LIBS ---



class get_pybind_include(object):
    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)


sources = glob( "extensions/deepCABAC/source/*.cpp*" ) + \
          glob( "extensions/deepCABAC/source/Lib/CommonLib/*.cpp*" ) + \
          glob( "extensions/deepCABAC/source/Lib/EncLib/*.cpp*" ) + \
          glob( "extensions/deepCABAC/source/Lib/DecLib/*.cpp*" )

ext_modules = [
    Extension(
        'deepCABAC',
        sources=sources,
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            pthreads_include_dir,
            get_pybind_include(user=True),
            "extensions/deepCABAC/source",
            "extensions/deepCABAC/source/Lib",
            "extensions/deepCABAC/source/Lib/CommonLib"
            "extensions/deepCABAC/source/Lib/EncLib"
            "extensions/deepCABAC/source/Lib/DecLib"
        ],
        language='c++',
                # Passa os argumentos definidos acima
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        libraries=libraries,
        library_dirs=library_dirs
    ),
]

# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    flags = ['-std=c++17', '-std=c++14', '-std=c++11']

    for flag in flags:
        if has_flag(compiler, flag): return flag

    raise RuntimeError('Unsupported compiler -- at least C++11 support '
                       'is needed!')


# --- CLASSE BuildExt MODIFICADA (Mantida como antes) ---
class BuildExt(build_ext):
    c_opts = { 'msvc': ['/EHsc'], 'unix': [] }
    l_opts = { 'msvc': [], 'unix': [] }

    # Adiciona opções específicas do Darwin (macOS) se necessário
    if sys.platform == 'darwin':
        darwin_opts = ['-stdlib=libc++', '-mmacosx-version-min=10.14']
        c_opts['unix'] += darwin_opts
        l_opts['unix'] += darwin_opts

    def build_extensions(self):
        ct = self.compiler.compiler_type
        # Pega as opções padrão
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])

        # Adiciona flags específicos do sistema/compilador
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            # Adiciona flag C++11 se necessário E ainda não presente
            needed_cpp_flag = cpp_flag(self.compiler)
            if needed_cpp_flag not in opts:
                opts.append(needed_cpp_flag)
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
             # A versão já está em extra_compile_args
             pass

        # Adiciona os flags padrão aos flags específicos que já definimos
        for ext in self.extensions:
            # Adiciona os flags de opts que ainda não estão lá
            ext.extra_compile_args.extend([opt for opt in opts if opt not in ext.extra_compile_args])
            # Adiciona os flags de link_opts que ainda não estão lá
            ext.extra_link_args.extend([opt for opt in link_opts if opt not in ext.extra_link_args])

        # Chama a função original para realmente construir
        build_ext.build_extensions(self)

setup(
    name='NNC',
    version=__version__,
    packages=find_packages(),
    author='Paul Haase, Daniel Becking',
    author_email='paul.haase@hhi.fraunhofer.de, daniel.becking@hhi.fraunhofer.de',
    url='https://hhi.fraunhofer.de',
    description='Neural Network Codec. deepCABAC C++ binding using pybind11.',
    long_description='',
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.3'],
    setup_requires=['pybind11>=2.3'],
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
    package_data={"": ["*.txt"]},
    include_package_data=True,
)
