package org.openmaptiles.layers;

import static com.onthegomap.planetiler.TestUtils.rectangle;

import com.onthegomap.planetiler.VectorTile;
import com.onthegomap.planetiler.geo.GeoUtils;
import com.onthegomap.planetiler.geo.GeometryException;
import com.onthegomap.planetiler.reader.SimpleFeature;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.openmaptiles.OpenMapTilesProfile;

class LanduseTest extends AbstractLayerTest {

  @Test
  void testNaturalEarthUrbanAreas() {
    assertFeatures(0, List.of(Map.of(
      "_layer", "landuse",
      "class", "residential",
      "_buffer", 4d,
      "_minzoom", 4
    )), process(SimpleFeature.create(
      GeoUtils.worldToLatLonCoords(rectangle(0, Math.sqrt(1))),
      Map.of("scalerank", 1.9),
      OpenMapTilesProfile.NATURAL_EARTH_SOURCE,
      "ne_50m_urban_areas",
      0
    )));
    assertFeatures(0, List.of(Map.of(
      "_layer", "landuse",
      "class", "residential",
      "_buffer", 4d,
      "_minzoom", 5
    )), process(SimpleFeature.create(
      GeoUtils.worldToLatLonCoords(rectangle(0, Math.sqrt(1))),
      Map.of("scalerank", 2.1),
      OpenMapTilesProfile.NATURAL_EARTH_SOURCE,
      "ne_50m_urban_areas",
      0
    )));
  }

  @Test
  void testOsmLanduse() {
    assertFeatures(13, List.of(
      Map.of("_layer", "poi", "_type", "polygon"),
      Map.of("_layer", "poi", "_type", "point"),
      Map.of(
        "_layer", "landuse",
        "class", "railway",
        "_minpixelsize", 4d,
        "_minzoom", 9,
        "_maxzoom", 14
      )), process(polygonFeature(Map.of(
        "landuse", "railway",
        "amenity", "school"
      ))));
    assertFeatures(13, List.of(
      Map.of("_layer", "poi", "_type", "polygon"),
      Map.of("_layer", "poi", "_type", "point"),
      Map.of(
        "_layer", "landuse",
        "class", "school",
        "_minpixelsize", 4d,
        "_minzoom", 9,
        "_maxzoom", 14
      )), process(polygonFeature(Map.of(
        "amenity", "school"
      ))));
  }

  @Test
  void testGraveYardBecomesCemetery() {
    assertFeatures(14, List.of(
      Map.of("_layer", "poi", "_type", "polygon"),
      Map.of("_layer", "poi", "_type", "point"),
      Map.of(
        "_layer", "landuse",
        "class", "cemetery"
      )), process(polygonFeature(Map.of(
        "amenity", "grave_yard"
      ))));
  }

  @Test
  void testOsmLanduseLowerZoom() {
    assertFeatures(6, List.of(Map.of(
      "_layer", "landuse",
      "class", "suburb",
      "_minzoom", 6,
      "_maxzoom", 14,
      "_minpixelsize", 1d
    )), process(polygonFeature(Map.of(
      "place", "suburb"
    ))));
    assertFeatures(7, List.of(Map.of(
      "_layer", "landuse",
      "class", "residential",
      "_minzoom", 6,
      "_maxzoom", 14,
      "_minpixelsize", 0.1d
    )), process(polygonFeature(Map.of(
      "landuse", "residential"
    ))));
  }

  @Test
  void testOsmLanduseName() {
    // The landuse table has no name column, so the name is read off the source feature - at max
    // zoom, as everywhere else. That names quarries, industrial estates and military areas, and
    // the place=suburb/quarter/neighbourhood polygons, none of which carried one before.
    assertFeatures(14, List.of(Map.of(
      "_layer", "landuse",
      "class", "quarry",
      "name", "Craigpark Quarry",
      "_minzoom", 9,
      "_maxzoom", 14
    )), process(polygonFeature(Map.of(
      "landuse", "quarry",
      "name", "Craigpark Quarry"
    ))));
    assertFeatures(13, List.of(Map.of(
      "_layer", "landuse",
      "class", "quarry",
      "name", "<null>"
    )), process(polygonFeature(Map.of(
      "landuse", "quarry",
      "name", "Craigpark Quarry"
    ))));
    assertFeatures(14, List.of(Map.of(
      "_layer", "landuse",
      "class", "quarry",
      "name", "<null>"
    )), process(polygonFeature(Map.of(
      "landuse", "quarry"
    ))));
  }

  @Test
  void testFarmyard() {
    // landuse=farmyard is in no OpenMapTiles mapping, so before this the farm buildings and
    // yard never reached the tiles at all. Max zoom only: going through the handler above would
    // have given it minzoom 9 and put it on the graphical map.
    assertFeatures(14, List.of(Map.of(
      "_layer", "landuse",
      "_type", "polygon",
      "class", "farmyard",
      "name", "Douchlage",
      "_minzoom", 14,
      "_maxzoom", 14
    )), process(polygonFeature(Map.of(
      "landuse", "farmyard",
      "name", "Douchlage"
    ))));
    // emitted whether named or not, as every other class in this layer is
    assertFeatures(14, List.of(Map.of(
      "_layer", "landuse",
      "class", "farmyard",
      "name", "<null>",
      "_minzoom", 14
    )), process(polygonFeature(Map.of(
      "landuse", "farmyard"
    ))));
    // a node tagged farmyard isn't an area and has nothing to contribute
    assertFeatures(14, List.of(), process(pointFeature(Map.of(
      "landuse", "farmyard"
    ))));
  }

  @Test
  void testNamedResidentialDoesNotMergeAtMaxZoom() throws GeometryException {
    // mergeMultiPolygon() groups on the exact attribute set, so at max zoom - the only zoom
    // where the name exists - two differently named estates stay separate. That is the point of
    // having the name; unnamed residential still collapses into a single multipolygon.
    var westerton = residential(rectangle(10, 20), "Westerton");
    var kilmardinny = residential(rectangle(20, 10, 22, 20), "Kilmardinny");
    var moreWesterton = residential(rectangle(20, 10, 22, 20), "Westerton");
    var unnamed = residential(rectangle(10, 20, 20, 22), null);

    Assertions.assertEquals(3,
      profile.postProcessLayerFeatures(Landuse.LAYER_NAME, 14, List.of(westerton, kilmardinny, unnamed)).size(),
      "differently named estates must not be merged into one another");
    Assertions.assertEquals(1,
      profile.postProcessLayerFeatures(Landuse.LAYER_NAME, 14, List.of(westerton, moreWesterton)).size(),
      "two parts of the same named estate must still merge");
  }

  private VectorTile.Feature residential(org.locationtech.jts.geom.Polygon geom, String name) {
    Map<String, Object> attrs = new HashMap<>();
    attrs.put("class", "residential");
    if (name != null) {
      attrs.put("name", name);
    }
    return new VectorTile.Feature(Landuse.LAYER_NAME, 1, VectorTile.encodeGeometry(geom), attrs, 0);
  }

  @Test
  void testMergePolygonsZ12() throws GeometryException {
    var poly1 = new VectorTile.Feature(
      Landuse.LAYER_NAME,
      1,
      VectorTile.encodeGeometry(rectangle(10, 20)),
      Map.of("class", "residential"),
      0
    );
    var poly2 = new VectorTile.Feature(
      Landuse.LAYER_NAME,
      1,
      VectorTile.encodeGeometry(rectangle(20, 10, 22, 20)),
      Map.of("class", "residential"),
      0
    );
    var poly3 = new VectorTile.Feature(
      Landuse.LAYER_NAME,
      1,
      VectorTile.encodeGeometry(rectangle(10, 20, 20, 22)),
      Map.of("class", "suburb"),
      0
    );

    Assertions.assertEquals(
      List.of(1, 2),
      profile.postProcessLayerFeatures(Landuse.LAYER_NAME, 13, List.of(poly1, poly2, poly3)).stream()
        .map(d -> {
          try {
            return d.geometry().decode().getNumGeometries();
          } catch (GeometryException e) {
            throw new AssertionError(e);
          }
        }).toList()
    );
    Assertions.assertEquals(
      List.of(1, 1),
      profile.postProcessLayerFeatures(Landuse.LAYER_NAME, 12, List.of(poly1, poly2, poly3)).stream()
        .map(d -> {
          try {
            return d.geometry().decode().getNumGeometries();
          } catch (GeometryException e) {
            throw new AssertionError(e);
          }
        }).toList()
    );
  }
}
