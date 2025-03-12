"""Global settings for the package."""

from __future__ import annotations

from enum import Enum


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


DEFAULT_OUTPUT_FORMAT = OutputFormat.ASYMPTOTIC

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
    _default_output_format = DEFAULT_OUTPUT_FORMAT

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