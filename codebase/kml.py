# -*- coding: utf-8 -*-
"""
Keyhole Markup Language-related
"""
import matplotlib.pyplot as plt

import simplekml

class KmlWriter():
    """KML class"""
    def __init__(self, kml_file):
        """Initialize"""
        self.kml_file = kml_file
        self.kml = simplekml.Kml()

    def write(self):
        """Save file"""
        self.kml.save(self.kml_file)

    def add_point(
        self,
        lat,
        lon,
        rgba=(128, 128, 128, 255),
        name='',
        description='',
        **kwargs
    ):
        """Add point to KML

        Parameters
        ----------
        lat : float
            Latitudes, degrees

        lon : float
            Longitudes, degrees

        rgba : (4) int
            Red green blue alpha from [0, 255]

        name : str
            Name of item

        description : str
            Description of item

        **kwargs : {}
            Additional keyword arguments for simplekml writer
        """
        point = self.kml.newpoint(
            name=name,
            description=description,
            coords=[(lon, lat)],
            **kwargs
        )
        point.color = self._rgba_to_kmlhex(rgba)

    def add_path(
        self,
        lats,
        lons,
        rgba=(128, 128, 128, 255),
        name='',
        description='',
        **kwargs
    ):
        """Add path to KML
        
        Parameters
        ----------
        lats : (N) np.ndarray float
            Latitudes, degrees

        lons : (N) np.ndarray float
            Longitudes, degrees

        rgba : (4) int
            Red green blue alpha from [0, 255]

        name : str
            Name of item

        description : str
            Description of item

        **kwargs : {}
            Additional keyword arguments for simplekml writer
        """
        lon_lats = self._lon_lat_to_kml(lons, lats)
        path = self.kml.newlinestring(
            name=name,
            description=description,
            coords=lon_lats,
            **kwargs
        )

        # Get color  
        path.style.linestyle.width = 5
        path.style.linestyle.color = self._rgba_to_kmlhex(rgba)

    def add_polygon(
        self,
        lats,
        lons,
        rgba=(128, 128, 128, 255),
        name='',
        description='',
        **kwargs
    ):
        """Add an enclosed polygon with option to fill
        
        Parameters
        ----------
        lats : (N) np.ndarray float
            Latitudes, degrees

        lons : (N) np.ndarray float
            Longitudes, degrees

        rgba : (4) int
            Red green blue alpha from [0, 255]

        name : str
            Name of item

        description : str
            Description of item

        **kwargs : {}
            Additional keyword arguments for simplekml writer
        """
        lon_lats = self._lon_lat_to_kml(lons, lats)
        polygon = self.kml.newpolygon(
            name=name,
            description=description,
            outerboundaryis=lon_lats,
            **kwargs
            # innerboundaryis=[]  # not currently using
        )
        polygon.color = self._rgba_to_kmlhex(rgba)

    def add_contours(self, lat_grid, lon_grid, values, levels=5):
        """Add contour lines to KML file
    
        Parameters
        ----------
        lat_grid : (n_lat, n_lon) np.ndarray float
            Latitude grid

        lon_grid : (n_lat, n_lon) np.ndarray float
            Longitude grid
    
        values : (n_lat, n_lon) np.ndarray float
            Value grid
    
        levels : [] or int
            Specific level values for contours or number of levels
        """
        contours =  plt.contour(
            lat_grid,
            lon_grid,
            values,
            levels=levels)
    
        for collection, level in zip(contours.collections, contours.levels):
            paths = collection.get_paths()
            color = (255 * collection.get_edgecolor()).astype(int).flatten()
    
            for path in paths:
                self.add_path(
                    lats=path.vertices[:, 0],
                    lons=path.vertices[:, 1],
                    rgba=color,
                    name='{}'.format(level),
                    description='{}'.format(level)
                )

    @staticmethod
    def _rgba_to_kmlhex(rgba):
        """Convert RGBA to ARGB in Hex for KML"""
        r, g, b, a = rgba
        hex_color = '#{:02X}{:02X}{:02X}{:02X}'.format(a, r, g, b)
        return hex_color

    @staticmethod
    def _lon_lat_to_kml(lons, lats):
        """Convert 1D array of lon and lat to kml format"""
        lon_lats = \
            [list(lon_lat) for lon_lat in zip(lons, lats)]
        return lon_lats