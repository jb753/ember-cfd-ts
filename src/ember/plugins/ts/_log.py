"""Solver log-file housekeeping shared by the TS3 and TS4 drivers.

Both solvers redirect their output to a fixed filename inside the working
directory, so successive runs in one directory would otherwise clobber each
other's logs. This module keeps the archiving convention in one place; it is
the only subprocess-adjacent machinery the two drivers share, since TS4 detects
divergence in-process and so needs none of TS3's live log polling.
"""

from glob import glob
import os
import re

_re_log_index = re.compile(r"^log_(\d+)\.txt$")


def archive_log(workdir, name="log.txt"):
    """Rename an existing log in ``workdir`` to the next free ``log_NNN.txt``.

    The live log must keep its fixed name -- the shell redirect that creates it
    and TS3's live NaN poll both hardcode it -- so instead of writing per-run
    names directly we archive the *previous* run's log at the start of each run.
    Successive runs in one workdir (e.g. the levels of a full-multigrid startup)
    therefore each keep their own ``log_000.txt``, ``log_001.txt``, ... A legacy
    ``log_old.txt`` is ignored by the numbering.

    Parameters
    ----------
    workdir : str or Path
        Working directory holding the log.
    name : str, optional
        Filename of the live log. TS3 writes ``log.txt``; TS4 writes
        ``log_ts4.txt``. The archived names are the same in both cases, so a
        directory used by both solvers keeps one numbering sequence.
    """
    log_path = os.path.join(workdir, name)
    if not os.path.exists(log_path):
        return
    indices = [
        int(m.group(1))
        for f in glob(os.path.join(workdir, "log_*.txt"))
        if (m := _re_log_index.match(os.path.basename(f)))
    ]
    n = max(indices, default=-1) + 1
    os.rename(log_path, os.path.join(workdir, f"log_{n:03d}.txt"))
