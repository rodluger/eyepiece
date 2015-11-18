TODO
----

- Multi-planet system handling; fit parameters simultaneously; beware overlapping transits
- Use Rowe's transit times for TTV systems
- Resolve tension between ``inject`` and ``trninfo`` in ``DownloadInfo()``?
- Fix ``pysyzygy`` memory issues. Separate it from plotting routine. Put crowding in ``pysyzygy``

NOTES
-----

- Libralato et al. - K2 PSF photometry
- Fabien Bastien: Flicker can give stellar density within ~ 30%

INSTRUCTIONS
------------

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