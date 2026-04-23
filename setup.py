import os
import sys
import glob
import shutil
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class F2pyBuildExt(build_ext):
    def build_extension(self, ext):
        if ext.name != "custom_colours.cc_api":
            super().build_extension(ext)
            return

        src_dir = os.path.abspath(os.path.dirname(__file__))
        pkg_dir = os.path.join(src_dir, "custom_colours")

        # Ensure meson/ninja installed in the build env are on PATH.
        # pip installs them alongside the Python executable.
        env = os.environ.copy()
        python_bin = os.path.dirname(sys.executable)
        env["PATH"] = python_bin + os.pathsep + env.get("PATH", "")

        subprocess.check_call(
            [
                sys.executable, "-m", "numpy.f2py", "-c",
                os.path.join(src_dir, "fortran", "cc_kernels.f90"),
                os.path.join(src_dir, "fortran", "cc_api.f90"),
                "-m", "cc_api",
            ],
            cwd=src_dir,
            env=env,
        )

        # f2py drops the .so in cwd; move it into the package.
        built = (
            glob.glob(os.path.join(src_dir, "cc_api*.so")) +
            glob.glob(os.path.join(src_dir, "cc_api*.pyd"))
        )
        for f in built:
            dest = os.path.join(pkg_dir, os.path.basename(f))
            shutil.move(f, dest)

        # For non-editable installs, also copy to the build tree.
        ext_path = self.get_ext_fullpath(ext.name)
        os.makedirs(os.path.dirname(ext_path), exist_ok=True)
        installed = (
            glob.glob(os.path.join(pkg_dir, "cc_api*.so")) +
            glob.glob(os.path.join(pkg_dir, "cc_api*.pyd"))
        )
        for f in installed:
            shutil.copy2(f, ext_path)


setup(
    ext_modules=[Extension("custom_colours.cc_api", sources=[])],
    cmdclass={"build_ext": F2pyBuildExt},
)
