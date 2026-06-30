ARTIQ Releases
##############

ARTIQ follows a rolling release model, with beta, stable, and legacy channels. Different releases are saved as different branches on the M-Labs `ARTIQ repository <https://git.m-labs.hk/M-Labs/artiq>`_. The ``master`` branch represents the beta version, where at any time the next stable release of ARTIQ is currently in development. This branch is unstable and does not yet guarantee reliability or consistency, but may also already offer new features and improvements; see the `beta release notes <https://git.m-labs.hk/M-Labs/artiq/src/branch/master/RELEASE_NOTES.rst>`_ for the most up-to-date information. The ``release-[number]`` branches represent stable releases, of which the most recent is considered the current stable version, and the second-most recent the current legacy version.

Releases are numbered. A particular ARTIQ installation can be described by its "major version", which is its release, and its "minor version", which represents the precise revision of the release. Once a release is stable, it receives no more significant updates, only bugfixes, so minor versions within a stable release do not differ notably and are not usually relevant.

To install the current stable version of ARTIQ, consult the *current* `Installing ARTIQ <https://m-labs.hk/artiq/manual/installing.html>`_ page. To install beta or legacy versions, consult the same page in their respective manuals. Instructions given in pre-legacy versions of the manual may or may not install their corresponding ARTIQ systems, and are no longer supported. Regardless, all out-of-date versions remain available as complete source code in the repository.

The beta manual is hosted `here <https://m-labs.hk/artiq/manual-beta/>`_. The current manual is hosted `here <https://m-labs.hk/artiq/manual/>`__. The legacy manual is hosted `here <https://m-labs.hk/artiq/manual-legacy/>`__. Older versions of the manual can be rebuilt from the source files in ``doc/manual``, retrieved from the respective branch.

.. attention::
    This is the **|channel|** manual, currently documenting **ARTIQ-|major_version|**.

.. include:: ../../RELEASE_NOTES.rst
