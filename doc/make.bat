@ECHO OFF

set SOURCEDIR=.
set BUILDDIR=_build

if "%SPHINXBUILD%" == "" (
    set SPHINXBUILD=sphinx-build
)

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR%
