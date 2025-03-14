"""Global settings for the package."""

from __future__ import annotations

from enum import Enum

from sage_acsv.debug import acsv_logger


class OutputFormat(Enum):
    """Output options for displaying the asymptotic behavior determined
    by :func:`.diagonal_asy`.

    See also:

    - :func:`.diagonal_asy`
    - :class:`.ACSVSettings`

    """
    ASYMPTOTIC = "asymptotic"
    SYMBOLIC = "symbolic"
    TUPLE = "tuple"

class KroneckerBackend(Enum):
    """Options for computing Kronecker representations determined by
    :func:`.ContributingCombinatorial`, :func:`.MinimalCriticalCombinatorial`, and
    :func:`.CriticalPoints`.

    See also:

    - :func:`.ContributingCombinatorial`
    - :func:`.MinimalCriticalCombinatorial`
    - :func:`.CriticalPoints`
    - :class:`.ACSVSettings`

    """
    SAGE_ACSV = "sage_acsv"
    MSOLVE = "msolve"

class GroebnerBackend(Enum):
    """Options for computing Groebner Bases and related ideal functions
    :func:`.GroebnerBasis`, :func:`.PrimaryDecomposition`, :func:`.Saturate`, and
    :func:`.CriticalPoints`.

    See also:

    - :func:`.GroebnerBasis`
    - :func:`.PrimaryDecomposition`
    - :func:`.Saturate`
    - :func:`.Radical`
    - :class:`.ACSVSettings`

    """
    SINGULAR = "singular"
    MACAULAY2 = "macaulay2"

DEFAULT_OUTPUT_FORMAT = OutputFormat.ASYMPTOTIC
DEFAULT_KRONECKER_BACKEND = KroneckerBackend.SAGE_ACSV
DEFAULT_GROEBNER_BACKEND = GroebnerBackend.SINGULAR

class ACSVSettings:
    """Global settings for the package.

    This class is used to store global settings for the package. It
    provides a mechanism to set and get the default settings.

    Examples::

        sage: from sage_acsv import ACSVSettings
        sage: list(ACSVSettings.Output)
        [<OutputFormat.ASYMPTOTIC: 'asymptotic'>,
         <OutputFormat.SYMBOLIC: 'symbolic'>,
         <OutputFormat.TUPLE: 'tuple'>]
        sage: ACSVSettings.Output.ASYMPTOTIC
        <OutputFormat.ASYMPTOTIC: 'asymptotic'>
        sage: ACSVSettings.set_default_output_format("tuple")
        sage: ACSVSettings.get_default_output_format()
        <OutputFormat.TUPLE: 'tuple'>
        sage: ACSVSettings.set_default_output_format("symbolic")
        sage: ACSVSettings.get_default_output_format()
        <OutputFormat.SYMBOLIC: 'symbolic'>
        sage: ACSVSettings.set_default_output_format()
        sage: ACSVSettings.get_default_output_format()
        <OutputFormat.ASYMPTOTIC: 'asymptotic'>

    """
    Output = OutputFormat
    Kronecker = KroneckerBackend
    Groebner  = GroebnerBackend
    _default_output_format = DEFAULT_OUTPUT_FORMAT
    _default_kronecker_backend = DEFAULT_KRONECKER_BACKEND
    _default_groebner_backend = DEFAULT_GROEBNER_BACKEND
    _m2_installation_path = None

    MAX_MIN_CRIT_RETRIES = 5  # Maximum number of retries for critical point detection

    @classmethod
    def set_default_output_format(cls, output_format: OutputFormat | str | None = None) -> None:
        """Set the default output format chosen when not explicitly
        specifying one.

        INPUT:

        * ``output_format`` -- a given :class:`.OutputFormat`, a string identifying one,
          or ``None`` to restore the default behavior.
        """
        if output_format is None:
            cls._default_output_format = DEFAULT_OUTPUT_FORMAT
        else:
            cls._default_output_format = OutputFormat(output_format)
            

    @classmethod
    def get_default_output_format(cls) -> OutputFormat:
        """Get the default output format chosen when not explicitly
        specifying one.
        """
        return cls._default_output_format

    @classmethod
    def get_logging_level(cls) -> int:
        """Get the logging level for the package."""
        return acsv_logger.level

    @classmethod
    def set_logging_level(cls, level: int) -> None:
        """Set the logging level for the package.

        INPUT:

        * ``level`` -- a logging level.

        EXAMPLES::

            sage: from sage_acsv import ACSVSettings
            sage: import logging
            sage: ACSVSettings.set_logging_level(logging.DEBUG)
            sage: ACSVSettings.get_logging_level()
            10
            sage: ACSVSettings.set_logging_level(logging.INFO)
            sage: ACSVSettings.get_logging_level()
            20
        """
        acsv_logger.setLevel(level)
    
    @classmethod
    def set_default_kronecker_backend(cls, backend: KroneckerBackend | str | None) -> None:
        """Set the preferred method for computing a Kronecker Representation. Will default
        to the sage_acsv implementation if not specified.
        
        See :func:`._kronecker_representation`.

        INPUT:

        * ``backend`` -- a given :class:`.KroneckerBackend`, a string identifying one,
          or ``None`` to restore the default behavior.
        """
        if backend is None:
            cls._default_kronecker_backend = DEFAULT_KRONECKER_BACKEND
        else:
            cls._default_kronecker_backend = KroneckerBackend(backend)

    @classmethod
    def get_default_kronecker_backend(cls) -> KroneckerBackend:
        """Get the default kronecker representation backend."""
        return cls._default_kronecker_backend

    @classmethod
    def set_default_groebner_backend(cls, backend: GroebnerBackend | str | None) -> None:
        """Set the preferred method for performing Groebner Bases and related computations,
        including :func:`.GroebnerBasis`, :func:`.PrimaryDecomposition`, :func:`.Saturate`,
        and :func:`.Radical`

        Will default to singular.

        INPUT:

        * ``backend`` -- a given :class:`.GroebnerBackend`, a string identifying one,
          or ``None`` to restore the default behavior.
        """
        if backend is None:
            cls._default_groebner_backend = DEFAULT_GROEBNER_BACKEND
        else:
            cls._default_groebner_backend = GroebnerBackend(backend)

    @classmethod
    def get_default_groebner_backend(cls) -> KroneckerBackend:
        """Get the default groebner basis backend."""
        return cls._default_groebner_backend
    
    @classmethod
    def set_macaulay2_path(cls, path: str | None) -> None:
        """Set the path of the Macaulay2 executable.

        INPUT:

        * ``path`` -- a filepath string or `None` to run from anywhere
        """
        cls._m2_installation_path = path

    @classmethod
    def get_macaulay2_path(cls) -> KroneckerBackend:
        """Get the Macaulay2 installation path."""
        return cls._m2_installation_path