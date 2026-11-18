"""Interface to Julia via juliacall (PythonCall.jl).

Julia is an optional dependency of ``sage_acsv``, like Macaulay2: nothing in
the package needs it unless the Julia backend is used. Using the backend
requires three things, which users install themselves. Each one is checked
when the backend is first used, and a :class:`JuliaBackendError` saying how to
fix the problem is raised if it is missing:

* The ``juliacall`` Python package, installed into Sage's Python::

      sage -pip install juliacall

* Julia itself, for example installed with
  `juliaup <https://github.com/JuliaLang/juliaup>`_. juliacall only supports
  some Julia versions: on a Sage whose Python uses an OpenSSL older than 3.5,
  it needs Julia 1.11 or older (``juliaup add 1.11``).

* Any Julia packages a computation uses, such as
  `RationalUnivariateRepresentation.jl
  <https://gitlab.inria.fr/NewRUR/RationalUnivariateRepresentation.jl>`_.
  These are installed into juliacall's Julia environment from Sage::

      sage: # import juliapkg
      sage: # juliapkg.add("RationalUnivariateRepresentation", "17246082-4fa7-4287-87d4-6a5617fca855")
      sage: # juliapkg.resolve()

Choosing the Julia executable:

* To use a specific Julia, call :meth:`.ACSVSettings.set_julia_path` before
  the first Julia computation::

      sage: # from sage_acsv import ACSVSettings
      sage: # ACSVSettings.set_julia_path("/path/to/julia")

* Otherwise the backend looks for an installed Julia that juliacall supports:
  the ``PYTHON_JULIAPKG_EXE`` environment variable, Julia versions installed
  with juliaup, then ``julia`` on the ``PATH``. Jupyter kernels often start
  with a shorter ``PATH`` than a terminal, so ``~/.juliaup/bin`` is added to
  the ``PATH`` of the current process when it exists.

In both cases the chosen executable is passed to ``juliapkg`` (juliacall's
package manager) through ``PYTHON_JULIAPKG_EXE``, so it never downloads or
installs a Julia by itself.

Julia runs inside the Sage process and can only be started once per session.
The first start sets up juliacall's Julia environment, which needs network
access and can take a few minutes. This happens again whenever the Julia
executable or version changes. Later starts are fast.

.. NOTE::

    On macOS, Julia can leave the main thread marked by the kernel as
    running on its alternate signal stack, although it is not. This happens,
    for example, when Julia leaves a top-level ``try`` block, which includes
    juliacall's own startup code. The Sage Jupyter kernel then re-installs
    the cysignals handlers before every cell. When that ``sigaltstack()`` call
    fails, cysignals calls ``exit(1)`` and the kernel dies on the cell after
    the first Julia call. This module clears the stale flag after each Julia
    call it makes, and after every IPython cell once Julia is loaded; see
    :func:`_clear_stale_sigaltstack_flag`.
"""

import contextlib
import importlib.util
import io
import os
import shutil
import sys

from sage_acsv.data.exceptions import ACSVException

# Name and UUID of the RationalUnivariateRepresentation and Nemo Julia package.
RUR_PACKAGE = ("RationalUnivariateRepresentation", "17246082-4fa7-4287-87d4-6a5617fca855")
NEMO_PACKAGE = ("Nemo", "2edaba10-b0f1-5616-af89-8c11ac63239a")

# Cached Julia runtime handle (``juliacall.Main``), created on first use.
_julia_main = None

# Cached libc ``(sigaltstack, __sigreturn, stack_t)`` on macOS, or ``False``
# if unavailable.
_darwin_sigaltstack_api = None

# ``ss_flags`` bit meaning "currently executing on the alternate stack".
_SS_ONSTACK = 1
# ``__sigreturn`` style that clears the kernel's per-thread on-alt-stack flag
# (``UC_RESET_ALT_STACK = 0x80000000`` in libplatform), as a signed C int.
_UC_RESET_ALT_STACK = -0x80000000


class JuliaBackendError(ACSVException):
    """Raised when the Julia backend cannot be used because juliacall, Julia,
    or a required Julia package is missing or cannot be started.
    """


def _get_darwin_sigaltstack_api():
    """Return cached ctypes bindings used to inspect and reset the alt stack.

    Returns ``False`` on non-macOS platforms or if the symbols are missing.
    """
    global _darwin_sigaltstack_api
    if _darwin_sigaltstack_api is None:
        _darwin_sigaltstack_api = False
        if sys.platform == "darwin":
            try:
                import ctypes

                class stack_t(ctypes.Structure):
                    _fields_ = [
                        ("ss_sp", ctypes.c_void_p),
                        ("ss_size", ctypes.c_size_t),
                        ("ss_flags", ctypes.c_int),
                    ]

                libc = ctypes.CDLL(None)
                sigaltstack = libc.sigaltstack
                sigaltstack.argtypes = [ctypes.c_void_p, ctypes.POINTER(stack_t)]
                sigaltstack.restype = ctypes.c_int
                sigreturn = libc.__sigreturn
                sigreturn.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
                sigreturn.restype = ctypes.c_int
                _darwin_sigaltstack_api = (sigaltstack, sigreturn, stack_t)
            except (OSError, AttributeError):
                pass
    return _darwin_sigaltstack_api


def _clear_stale_sigaltstack_flag():
    """Clear a stale "on alternate signal stack" flag on the current thread.

    macOS only; does nothing on other platforms. Returns ``True`` if a stale
    flag was found and cleared.

    On macOS, ``siglongjmp`` restores the alternate-signal-stack state that
    ``sigsetjmp`` saved, by calling ``__sigreturn(NULL, UC_SET_ALT_STACK)`` or
    ``__sigreturn(NULL, UC_RESET_ALT_STACK)``. With Julia embedded in Sage via
    juliacall, Julia's interpreter calls ``siglongjmp`` when it leaves a
    top-level ``try`` block, and the call sets the flag even though the thread
    is on its normal stack. While the flag is set, the kernel rejects any
    change to the alternate stack with ``EPERM``. cysignals'
    ``init_cysignals()``, which the Sage Jupyter kernel runs before every cell,
    then aborts the process with ``cysignals sigaltstack: Operation not
    permitted``.

    This is only called from Python code, which never runs on the alternate
    signal stack, so the flag is known to be stale whenever it is set here.
    Clearing it is the same operation that libplatform's ``longjmp`` performs.
    """
    api = _get_darwin_sigaltstack_api()
    if not api:
        return False
    sigaltstack, sigreturn, stack_t = api
    current = stack_t()
    if sigaltstack(None, current) != 0 or not current.ss_flags & _SS_ONSTACK:
        return False
    return sigreturn(None, _UC_RESET_ALT_STACK, 0) == 0


def _register_ipython_cleanup():
    """Clear the stale flag after every IPython cell, if running in IPython.

    Covers Julia code that does not go through this module, such as direct
    ``juliacall`` calls or the ``%julia`` magic. The ``post_execute`` event
    runs at the end of each cell, before the Sage kernel calls
    ``init_cysignals()`` for the next one.
    """
    ipython = sys.modules.get("IPython")
    shell = ipython.get_ipython() if ipython is not None else None
    if shell is None:
        return
    callbacks = shell.events.callbacks.get("post_execute", [])
    if _clear_stale_sigaltstack_flag not in callbacks:
        shell.events.register("post_execute", _clear_stale_sigaltstack_flag)


def _ensure_juliaup_on_path():
    """Add juliaup's directory ``~/.juliaup/bin`` to the ``PATH`` of the
    current process, if it exists and is missing.

    juliaup installs its ``julia`` and ``juliaup`` commands there, but that
    directory is not always on the ``PATH`` that Sage or Jupyter starts with.
    """
    juliaup_bin = os.path.join(os.path.expanduser("~"), ".juliaup", "bin")
    path = os.environ.get("PATH", "")
    if os.path.isdir(juliaup_bin) and juliaup_bin not in path.split(os.pathsep):
        os.environ["PATH"] = juliaup_bin + os.pathsep + path


def running_julia_executable():
    """Return the path of the Julia executable running in this process, or
    ``None`` if Julia has not been started yet.

    This also detects Julia started by importing ``juliacall`` directly.
    """
    juliacall = sys.modules.get("juliacall")
    config = getattr(juliacall, "CONFIG", None) or {}
    if config.get("inited"):
        return config.get("exepath")
    return None


def _pin_julia_executable(path):
    """Make ``juliapkg`` use the Julia executable at ``path``.

    Sets ``PYTHON_JULIAPKG_EXE``, which ``juliapkg`` reads when it is first
    imported. If ``juliapkg`` was already imported, its state is updated too.
    With this set, ``juliapkg`` never installs a Julia by itself.
    """
    os.environ["PYTHON_JULIAPKG_EXE"] = path
    juliapkg_state = sys.modules.get("juliapkg.state")
    if juliapkg_state is not None:
        state = juliapkg_state.STATE
        if state.get("override_executable") != path:
            state["override_executable"] = path
            state["resolved"] = False


def _juliacall_julia_compat():
    """Return the Julia versions juliacall supports in this Sage, as a
    ``juliapkg`` ``Compat`` object, or ``None`` if they cannot be determined.

    This depends on the installed Python packages: for example juliacall
    limits Julia to 1.11 or older when Python's OpenSSL is older than 3.5.
    """
    try:
        from juliapkg.deps import find_requirements

        with contextlib.redirect_stdout(io.StringIO()):
            compat, _ = find_requirements()
        return compat
    except Exception:
        return None


def _julia_version(exe):
    """Return the version of the Julia executable ``exe``, or ``None``."""
    from juliapkg.find_julia import julia_version

    return julia_version(exe)


def _find_any_julia():
    """Return ``(executable, version)`` of some installed Julia, ignoring
    version requirements, or ``None`` if no Julia is installed.
    """
    from juliapkg.find_julia import ju_find_julia_noinstall

    with contextlib.redirect_stdout(io.StringIO()):
        try:
            found = ju_find_julia_noinstall()
        except Exception:
            found = None
    if found:
        return found
    exe = shutil.which("julia")
    if exe is not None:
        version = _julia_version(exe)
        if version is not None:
            return exe, version
    return None


def _locate_julia(configured_path):
    """Return the Julia executable to start, or raise a
    :class:`JuliaBackendError` if no usable Julia is installed.

    ``configured_path`` is the path set with
    :meth:`.ACSVSettings.set_julia_path`, or ``None``.
    """
    from juliapkg.find_julia import find_julia
    from juliapkg.state import STATE

    from sage_acsv.debug import acsv_logger

    compat = _juliacall_julia_compat()
    explicit = configured_path or os.environ.get("PYTHON_JULIAPKG_EXE")
    if explicit:
        version = _julia_version(explicit)
        if version is None:
            raise JuliaBackendError(
                f"{explicit!r} is not a working Julia executable. "
                "Set the correct path with ACSVSettings.set_julia_path(...)."
            )
        if compat is not None and version not in compat:
            acsv_logger.warning(
                f"Julia {version} at {explicit!r} is not supported by juliacall "
                f"in this Sage (needs Julia {compat}); it may fail or crash."
            )
        return explicit

    _ensure_juliaup_on_path()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exe, _ = find_julia(compat=compat, prefix=STATE["install"], install=False)
        return exe
    except Exception:
        pass

    found = _find_any_julia()
    if found is None:
        raise JuliaBackendError(
            "Julia was not found. Install it (https://julialang.org/install/) "
            "or set its location with ACSVSettings.set_julia_path(...)."
        )
    exe, version = found
    raise JuliaBackendError(
        f"Found Julia {version}, but juliacall in this Sage needs Julia {compat}. "
        "Install a version in that range (with juliaup: juliaup add <version>) "
        "or set its location with ACSVSettings.set_julia_path(...)."
    )


def get_julia():
    """Return the Julia runtime (``juliacall.Main``), starting it on first use.

    Julia is started with the executable set by
    :meth:`.ACSVSettings.set_julia_path`, or with an installed Julia found
    automatically (see the module documentation). The runtime is cached for
    later calls.

    Raises a :class:`JuliaBackendError` if juliacall or a usable Julia is not
    installed, or if Julia fails to start.
    """
    global _julia_main
    if _julia_main is not None:
        return _julia_main

    from sage_acsv.settings import ACSVSettings

    path = ACSVSettings.get_julia_path()
    running = running_julia_executable()
    if running is not None:
        # Julia was already started, e.g. by importing juliacall directly.
        if path is not None and not (
            os.path.exists(running) and os.path.samefile(path, running)
        ):
            raise RuntimeError(
                f"Julia is already running from {running!r} in this session, "
                f"not from the configured path {path!r}; restart Sage (or the "
                "Jupyter kernel) to use the configured Julia."
            )
    else:
        if importlib.util.find_spec("juliacall") is None:
            raise JuliaBackendError(
                "The Python package juliacall is not installed. "
                "Install it with: sage -pip install juliacall"
            )
        _pin_julia_executable(_locate_julia(path))

    # Load juliacall's IPython integration explicitly (rather than letting it
    # autodetect) so that it does not print a one-time banner to stdout when
    # first imported inside an interactive Sage/IPython session. A user-set
    # value is left untouched.
    os.environ.setdefault("PYTHON_JULIACALL_AUTOLOAD_IPYTHON_EXTENSION", "yes")
    try:
        from juliacall import Main as jl
    except Exception as e:
        raise JuliaBackendError(f"Julia could not be started: {e}") from e
    finally:
        # juliacall's startup script is a top-level ``try`` block, which
        # leaves the stale alt-stack flag set on macOS.
        _clear_stale_sigaltstack_flag()

    _register_ipython_cleanup()
    _julia_main = jl
    return _julia_main


def get_julia_package(name, uuid):
    """Load the Julia package ``name`` and return it as a Julia module.

    Raises a :class:`JuliaBackendError` saying how to install the package if
    it is not installed, or if Julia itself is not available (see
    :func:`get_julia`).

    INPUT:

    * ``name`` -- the name of a Julia package
    * ``uuid`` -- the UUID of the package, used in the install instructions
    """
    if not name.isidentifier():
        raise ValueError(f"invalid Julia package name: {name!r}")
    jl = get_julia()
    try:
        installed = jl.seval(f'Base.find_package("{name}") !== nothing')
        if not installed:
            raise JuliaBackendError(
                f"The Julia package {name} is not installed. Install it from Sage with:\n"
                f'    import juliapkg; juliapkg.add("{name}", "{uuid}"); juliapkg.resolve()'
            )
        try:
            jl.seval(f"import {name}")
        except Exception as e:
            raise JuliaBackendError(
                f"The Julia package {name} is installed but failed to load: {e}"
            ) from e
        return jl.seval(name)
    finally:
        _clear_stale_sigaltstack_flag()


def get_rur():
    """Return the RationalUnivariateRepresentation Julia package.

    Raises a :class:`JuliaBackendError` saying how to install it if the
    package, Julia, or juliacall is missing.

    """
    return get_julia_package(*RUR_PACKAGE)


def get_nemo():
    """Return the RationalUnivariateRepresentation Julia package.

    Raises a :class:`JuliaBackendError` saying how to install it if the
    package, Julia, or juliacall is missing.

    """
    return get_julia_package(*NEMO_PACKAGE)


def julia_is_available():
    """Return whether a Julia runtime can be initialized via juliacall.

    This never raises: it returns ``False`` if juliacall is not installed or
    a Julia runtime cannot be started.
    """
    try:
        get_julia()
    except Exception:
        return False
    return True


def run_julia(*args):
    """Evaluate a Julia expression and return the result.

    This is a small smoke test for the Sage-to-Julia bridge. Julia scalar
    and string results are converted to native Python values by juliacall.

    INPUT:

    * ``expression`` -- a string containing a Julia expression (default:
      ``"1 + 1"``).
    """
    jl = get_julia()
    try:
        for expression in args:
            res = jl.seval(expression)
        return res
    finally:
        _clear_stale_sigaltstack_flag()

def get_parametrization(vs, system):
    get_rur()
    get_nemo()
    res = run_julia(
        f"using Nemo, RationalUnivariateRepresentation",
        f"R, ({', '.join([str(v) for v in vs])}) = QQ{str([str(v) for v in vs])}",
        f"sys = {system}".replace("/", "//"),
        f"zdim_parameterization(sys, get_separating_element = true, verbose = false)"
    )
    return res