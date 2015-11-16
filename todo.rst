TODO
----

- If outliers span both sides of transit, remove transit.
- For usps: ensure that data points don't get included in neighboring transits!
- Add ``clobber`` override options to all functions
- Put crowding in ``pysyzygy``
- ``alt`` subtraction not working on Hyak
- When user quits out of interactive inspect, cancel qsub job
- Multi-planet system handling
- Fit parameters simultaneously; beware overlapping transits
- Use Rowe's transit times for TTV systems
- Libralato et al. - K2 PSF photomoetry
- Fabien Bastien: Flicker can give stellar density within ~ 30%
- Resolve tension between ``inject`` and ``trninfo`` in ``DownloadInfo()``

NOTE
----

- I had to edit the matplotlibrc file to set ``Agg`` as the default backend

TARGETS
-------

17.01 - Hot Jupiter
18.01 - Hot Jupiter
142.01 - "King of TTVs"
254.01 - Hot Jupiter, M dwarf
3284.01 - Earth-size, potentially habitable, M dwarf
4087.01 - Super-Earth, potentially habitable, early M dwarf
KIC 7869590 - Very variable red giant --> use for crowding tests w/ Russell
7592.01 - Super Earth beyond HZ of M dwarf; I don't detect it!