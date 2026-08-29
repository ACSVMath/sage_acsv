"""Interface to msolve."""

import os
import tempfile
import subprocess

from sage.misc.sage_eval import sage_eval
from sage.features.msolve import msolve
from sage_acsv.helpers import ACSVException

def _parse_parametrization_output(output):
    r"""Parse the raw output printed by ``msolve -P 2`` on its standard
    output into a (nested) list.

    msolve may print diagnostic messages (e.g., ``Restarting with
    another random linear form``) to its standard output before the
    computed parametrization, so any content before the start of the
    output list is discarded.

    INPUT:

    * ``output`` - string, the standard output of a ``msolve -P 2`` run

    EXAMPLES:

    Parsing the output of a run of ``msolve -P 2`` on the system
    `x^2 - 1 = y - x = 0`::

        sage: from sage_acsv.backends.msolve import _parse_parametrization_output
        sage: output = (
        ....:     "[0, [0, \n2, \n2, \n['x', 'y'],\n[0, 1],\n[1,\n"
        ....:     "[[2, [-1, 0, 1]],\n[1, [0, 2]],\n[\n[[1, [-2, 0]],\n1]\n]]]]]:\n"
        ....: )
        sage: _parse_parametrization_output(output)
        [0, [0, 2, 2, ['x', 'y'], [0, 1], [1, [[2, [-1, 0, 1]], [1, [0, 2]], [[[1, [-2, 0]], 1]]]]]]

    Diagnostic messages printed before the parametrization are ignored::

        sage: _parse_parametrization_output(
        ....:     "Restarting with another random linear form\n" + output
        ....: ) == _parse_parametrization_output(output)
        True

    Output that does not contain a parametrization is rejected::

        sage: _parse_parametrization_output("some msolve error message\n")
        Traceback (most recent call last):
        ...
        ACSVException: Unable to parse msolve output: 'some msolve error message\n'
    """
    start = output.find("[")
    if start == -1:
        raise ACSVException(f"Unable to parse msolve output: {output!r}")
    # the output list is followed by a trailing ":\n"
    return sage_eval(output[start:-2])

def get_parametrization(vs, system):
    filename = msolve().absolute_filename()
    msolve_in = tempfile.NamedTemporaryFile(mode="w", encoding="ascii", delete=False)
    command = [filename, "-f", msolve_in.name, "-P", "2"]

    system = list(str(e) for e in system)
    try:
        print(",".join([str(v) for v in vs]), file=msolve_in)
        print(0, file=msolve_in)
        print(*(pol.replace(" ", "") for pol in system), sep=",\n", file=msolve_in)
        msolve_in.close()
        msolve_out = subprocess.run(command, capture_output=True, text=True)
    finally:
        os.unlink(msolve_in.name)

    msolve_out.check_returncode()

    result = _parse_parametrization_output(msolve_out.stdout)

    if result[0] != 0:
        raise ACSVException(
            "Issue with msolve parametrization - system does not have finitely many solutions"
        )

    return result
