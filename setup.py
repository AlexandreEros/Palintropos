"""Build of the native CPU substrate library (the rest of the configuration
is in pyproject.toml).

The library is an ordinary setuptools extension with a plain C ABI, compiled
with the flags of tropoi/temporal/native/build_identity.py, which also
supplies the build digest compiled into it (checked at load time). It is
optional: without a C++ compiler the package installs and GPU execution works;
CPU execution then fails with instructions to build it. Linux and macOS
(GCC or Clang) only for now: MSVC does not accept the flags.
"""
import importlib.util
import pathlib
import subprocess

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "_pt_build_identity", HERE / "src" / "tropoi" / "temporal" / "native" / "build_identity.py")
identity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(identity)


class BuildNative(build_ext):
    """Record the compiler identity next to the flags in the library."""

    def build_extensions(self):
        cxx = self.compiler.compiler_so[0] if getattr(self.compiler, "compiler_so", None) else "?"
        try:
            version = subprocess.run([cxx, "--version"], capture_output=True, text=True,
                                     timeout=60).stdout.splitlines()[0]
        except Exception:                                       # noqa: BLE001
            version = "unknown"
        for ext in self.extensions:
            ext.define_macros.append(
                ("PT_BUILD_FLAGS", '"%s"' % (f"{version}; " + " ".join(identity.HOST_FLAGS))
                 .replace('"', "'")))
        super().build_extensions()


setup(
    ext_modules=[Extension(
        "tropoi.temporal._substrate_native",
        sources=["src/tropoi/temporal/native/drivers/host_driver.cpp"],
        include_dirs=[str(d) for d in identity.INCLUDE_DIRS] + [str(identity.DRIVERS)],
        depends=[str(p) for p in identity.source_files()],
        define_macros=[("PT_BUILD_DIGEST", '"%s"' % identity.digest())],
        extra_compile_args=list(identity.HOST_FLAGS),
        language="c++",
        optional=True)],
    cmdclass={"build_ext": BuildNative},
)
