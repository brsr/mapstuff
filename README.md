# mapstuff

A package for experimenting with map projections and other cartography stuff. Not a practical cartography package, most of the stuff included here is not optimized for speed or memory.

Depends on:
* [geopandas](https://github.com/geopandas/geopandas),
* [pyproj](https://github.com/pyproj4/pyproj),
* [shapely](https://github.com/Toblerity/Shapely),
* Other packages that are required by those packages

Map projections implemented here:
* Linear trimetric (new, a variation of Chamberlin that's more numerically tractable)
* Areal (new, an analog of [areal barycentric coordinates](https://en.wikipedia.org/wiki/Barycentric_coordinate_system) in the plane)
* Fuller's projection (Crider, John E. "Exact Equations for Fuller's Map Projection and Inverse." Cartographica, vol. 43, no. 1, 2008, pp. 67-72. [DOI: 10.3138/carto.43.1.67](https://doi.org/10.3138/carto.43.1.67))
* Crider's quadrilateral version of the Fuller projection (Crider, John E. "A geodesic map projection for quadrilaterals." Cartography and Geographic Information Science, vol. 36, no. 2, 2009, p. 131+. [DOI: 10.1559/152304009788188781](https://doi.org/10.1559/152304009788188781))
* Snyder Equal-Area (exists in PROJ, but not in general form. Snyder, John P. "An Equal-Area Map Projection For Polyhedral Globes." Cartographica, vol. 29, no. 1, 1992, pp. 10-21. [DOI: 10.3138/27H7-8K88-4882-1752](https://doi.org/10.3138/27H7-8K88-4882-1752))
* Some variations of the above
* Conformal polygonal (probably some horrible circle-packing thing, to be implemented in the future)
