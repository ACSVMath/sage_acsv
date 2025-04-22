"""Interface to msolve."""

import os
import tempfile
import subprocess

from sage.misc.sage_eval import sage_eval
from sage.features.msolve import msolve
from sage_acsv.helpers import ACSVException


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

    result = msolve_out.stdout
    result = sage_eval(result[:-2])

    if result[0] != 0:
        raise ACSVException(
            "Issue with msolve parametrization - system does not have finitely many solutions"
        )

    return result
