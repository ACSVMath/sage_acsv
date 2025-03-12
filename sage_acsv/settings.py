"""Global settings for the package."""

from __future__ import annotations

from enum import Enum


class OutputFormat(Enum):
    """Output options for displaying the asymptotic behavior determined
    by :func:`.diagonal_asy`.

    See also:

    - :func:`.diagonal_asy`

    Examples::

        sage: from sage_acsv import OutputFormat
        sage: OutputFormat("asymptotic")
        <OutputFormat.ASYMPTOTIC: 'asymptotic'>

    Check that the mechanism for setting default values
    works as intended::

        sage: OutputFormat.get_default()
        <OutputFormat.ASYMPTOTIC: 'asymptotic'>
        sage: OutputFormat.set_default("tuple")
        sage: OutputFormat.get_default()
        <OutputFormat.TUPLE: 'tuple'>
        sage: OutputFormat.set_default()
        sage: OutputFormat.get_default()
        <OutputFormat.ASYMPTOTIC: 'asymptotic'>

    """
    ASYMPTOTIC = "asymptotic"
    SYMBOLIC = "symbolic"
    TUPLE = "tuple"
    
    @classmethod
    def set_default(cls, output_format: OutputFormat | str | None = None) -> None:
        """Set the default output format chosen when not explicitly
        specifying one.

        INPUT:

        * ``output_format`` -- a given `OutputFormat`, a string identifying one,
          or ``None`` (which restores the original behavior)
        """
        if output_format is None:
            del cls._default
        else:
            cls._default = OutputFormat(output_format)

    @classmethod
    def get_default(cls) -> OutputFormat:
        """Get the default output format chosen when not explicitly
        specifying one.
        """
        if hasattr(cls, '_default'):
            return cls._default
        return cls.ASYMPTOTIC