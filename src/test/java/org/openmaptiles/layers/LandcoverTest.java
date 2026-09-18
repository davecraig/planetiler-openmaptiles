package org.openmaptiles.layers;

import static com.onthegomap.planetiler.TestUtils.rectangle;

import com.onthegomap.planetiler.FeatureCollector;
import com.onthegomap.planetiler.VectorTile;
import com.onthegomap.planetiler.geo.GeoUtils;
import com.onthegomap.planetiler.geo.GeometryException;
import com.onthegomap.planetiler.reader.SimpleFeature;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.StreamSupport;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.openmaptiles.OpenMapTilesProfile;

class LandcoverTest extends AbstractLayerTest {

  @Test
  void testNaturalEarthGlaciers() {
    var glacier1 = process(SimpleFeature.create(
      GeoUtils.worldToLatLonCoords(rectangle(0, Math.sqrt(1))),
      Map.of(),
      OpenMapTilesProfile.NATURAL_EARTH_SOURCE,
      "ne_110m_glaciated_areas",
      0
    ));
    var glacier2 = process(SimpleFeature.create(
      GeoUtils.worldToLatLonCoords(rectangle(0, Math.sqrt(1))),
      Map.of(),
      OpenMapTilesProfile.NATURAL_EARTH_SOURCE,
      "ne_50m_glaciated_areas",
      0
    ));
    var glacier3 = process(SimpleFeature.create(
      GeoUtils.worldToLatLonCoords(rectangle(0, Math.sqrt(1))),
      Map.of(),
      OpenMapTilesProfile.NATURAL_EARTH_SOURCE,
      "ne_10m_glaciated_areas",
      0
    ));
    assertFeatures(0, List.of(Map.of(
      "_layer", "landcover",
      "subclass", "glacier",
      "class", "ice",
      "_buffer", 4d
    )), glacier1);
    assertFeatures(0, List.of(Map.of(
      "_layer", "landcover",
      "subclass", "glacier",
      "class", "ice",
      "_buffer", 4d
    )), glacier2);
    assertFeatures(0, List.of(Map.of(
      "_layer", "landcover",
      "subclass", "glacier",
      "class", "ice",
      "_buffer", 4d
    )), glacier3);
    assertCoversZoomRange(0, 6, "landcover",
      glacier1,
      glacier2,
      glacier3
    );
  }

  @Test
  void testNaturalEarthAntarcticIceShelves() {
    var ice1 = process(SimpleFeature.create(
      GeoUtils.worldToLatLonCoords(rectangle(0, Math.sqrt(1))),
      Map.of(),
      OpenMapTilesProfile.NATURAL_EARTH_SOURCE,
      "ne_50m_antarctic_ice_shelves_polys",
      0
    ));
    var ice2 = process(SimpleFeature.create(
      GeoUtils.worldToLatLonCoords(rectangle(0, Math.sqrt(1))),
      Map.of(),
      OpenMapTilesProfile.NATURAL_EARTH_SOURCE,
      "ne_10m_antarctic_ice_shelves_polys",
      0
    ));
    assertFeatures(0, List.of(Map.of(
      "_layer", "landcover",
      "subclass", "ice_shelf",
      "class", "ice",
      "_buffer", 4d
    )), ice1);
    assertFeatures(0, List.of(Map.of(
      "_layer", "landcover",
      "subclass", "ice_shelf",
      "class", "ice",
      "_buffer", 4d
    )), ice2);
    assertCoversZoomRange(2, 6, "landcover",
      ice1,
      ice2
    );
  }

  @Test
  void testOsmLandcover() {
    assertFeatures(13, List.of(Map.of(
      "_layer", "landcover",
      "subclass", "wood",
      "class", "wood",
      "_minpixelsize", 8d,
      "_numpointsattr", "_numpoints",
      "_minzoom", 7,
      "_maxzoom", 14
    )), process(polygonFeature(Map.of(
      "natural", "wood"
    ))));
    assertFeatures(12, List.of(Map.of(
      "_layer", "landcover",
      "subclass", "forest",
      "class", "wood",
      "_minpixelsize", 8d,
      "_minzoom", 7,
      "_maxzoom", 14
    )), process(polygonFeature(Map.of(
      "landuse", "forest"
    ))));
    assertFeatures(10, List.of(Map.of(
      "_layer", "landcover",
      "subclass", "dune",
      "class", "sand",
      "_minpixelsize", 4d,
      "_minzoom", 7,
      "_maxzoom", 14
    )), process(polygonFeature(Map.of(
      "natural", "dune"
    ))));
    assertFeatures(10, List.of(), process(polygonFeature(Map.of(
      "landuse", "park"
    ))));
  }

  @Test
  void testOsmLandcoverName() {
    assertFeatures(14, List.of(Map.of(
      "_layer", "landcover",
      "subclass", "heath",
      "class", "grass",
      "name", "Drumclog Moor"
    )), process(polygonFeature(Map.of(
      "natural", "heath",
      "name", "Drumclog Moor"
    ))));
    assertFeatures(14, List.of(Map.of(
      "_layer", "landcover",
      "subclass", "forest",
      "class", "wood",
      "name", "Garadhban"
    )), process(polygonFeature(Map.of(
      "landuse", "forest",
      "name", "Garadhban"
    ))));
    // farmland is named by the same change, which is one of the ways a farm reaches the tiles
    assertFeatures(14, List.of(Map.of(
      "_layer", "landcover",
      "subclass", "farmland",
      "class", "farmland",
      "name", "Farmland name"
    )), process(polygonFeature(Map.of(
      "landuse", "farmland",
      "name", "Farmland name"
    ))));
    // max zoom only - below it there must be no name key at all, which is what leaves the
    // z7-13 merging in postProcess() seeing exactly the attributes it saw before
    assertFeatures(13, List.of(Map.of(
      "_layer", "landcover",
      "subclass", "forest",
      "class", "wood",
      "name", "<null>"
    )), process(polygonFeature(Map.of(
      "landuse", "forest",
      "name", "Garadhban"
    ))));
    // and an unnamed polygon never gains the key
    assertFeatures(14, List.of(Map.of(
      "_layer", "landcover",
      "subclass", "wood",
      "class", "wood",
      "name", "<null>"
    )), process(polygonFeature(Map.of(
      "natural", "wood"
    ))));
  }

  @Test
  void testNamesDoNotSplitTheMergeBelowMaxZoom() {
    // mergeOverlappingPolygons() groups on the exact attribute set, so the whole safety of
    // adding a name rests on differently named woods being indistinguishable below max zoom.
    var garadhban = attrsAtZoom(process(polygonFeature(Map.of(
      "natural", "wood",
      "name", "Garadhban"
    ))), 13);
    var drumclog = attrsAtZoom(process(polygonFeature(Map.of(
      "natural", "wood",
      "name", "Drumclog Wood"
    ))), 13);
    var unnamed = attrsAtZoom(process(polygonFeature(Map.of(
      "natural", "wood"
    ))), 13);

    Assertions.assertEquals(garadhban, drumclog, "two named woods must merge as one at z13");
    Assertions.assertEquals(garadhban, unnamed, "a named wood must merge with an unnamed one at z13");
    Assertions.assertNotEquals(
      attrsAtZoom(process(polygonFeature(Map.of("natural", "wood", "name", "Garadhban"))), 14),
      attrsAtZoom(process(polygonFeature(Map.of("natural", "wood", "name", "Drumclog Wood"))), 14),
      "at max zoom they must be told apart, or the name is doing nothing"
    );
  }

  @Test
  void testMergeForestsBuNumPointsZ9to13() throws GeometryException {
    Map<String, Object> map = Map.of("subclass", "wood");

    assertMerges(List.of(map, map, map, map, map, map), List.of(
      // don't merge any
      feature(rectangle(10, 20), Map.of("_numpoints", 48, "subclass", "wood")),
      feature(rectangle(10, 20), Map.of("_numpoints", 49, "subclass", "wood")),
      feature(rectangle(12, 18), Map.of("_numpoints", 50, "subclass", "wood")),
      feature(rectangle(12, 18), Map.of("_numpoints", 299, "subclass", "wood")),
      feature(rectangle(12, 18), Map.of("_numpoints", 300, "subclass", "wood")),
      feature(rectangle(12, 18), Map.of("_numpoints", 301, "subclass", "wood"))
    ), 14);
    assertMerges(List.of(map, map, map), List.of(
      // < 300 - merge
      feature(rectangle(10, 20), Map.of("_numpoints", 48, "subclass", "wood")),
      feature(rectangle(10, 20), Map.of("_numpoints", 49, "subclass", "wood")),
      feature(rectangle(12, 18), Map.of("_numpoints", 50, "subclass", "wood")),
      feature(rectangle(12, 18), Map.of("_numpoints", 299, "subclass", "wood")),
      // >= 300 - don't merge
      feature(rectangle(12, 18), Map.of("_numpoints", 300, "subclass", "wood")),
      feature(rectangle(12, 18), Map.of("_numpoints", 301, "subclass", "wood"))
    ), 13);
    assertMerges(List.of(map, map), List.of(
      // < 300 - merge
      feature(rectangle(10, 20), Map.of("_numpoints", 48, "subclass", "wood")),
      feature(rectangle(10, 20), Map.of("_numpoints", 49, "subclass", "wood")),
      feature(rectangle(12, 18), Map.of("_numpoints", 50, "subclass", "wood")),
      feature(rectangle(12, 18), Map.of("_numpoints", 299, "subclass", "wood")),
      // >= 300 - merge
      feature(rectangle(12, 18), Map.of("_numpoints", 300, "subclass", "wood")),
      feature(rectangle(12, 18), Map.of("_numpoints", 301, "subclass", "wood"))
    ), 9);
  }

  @Test
  void testMergeNonForestsBelowZ8() throws GeometryException {
    Map<String, Object> map = Map.of("subclass", "dune");

    assertMerges(List.of(map, map), List.of(
      feature(rectangle(10, 20), Map.of("_numpoints", 48, "subclass", "dune")),
      feature(rectangle(12, 18), Map.of("_numpoints", 301, "subclass", "dune"))
    ), 8);
    assertMerges(List.of(map), List.of(
      feature(rectangle(10, 20), Map.of("_numpoints", 48, "subclass", "dune")),
      feature(rectangle(12, 18), Map.of("_numpoints", 301, "subclass", "dune"))
    ), 7);
    assertMerges(List.of(map, map), List.of(
      feature(rectangle(10, 20), Map.of("_numpoints", 48, "subclass", "dune")),
      feature(rectangle(12, 18), Map.of("_numpoints", 301, "subclass", "dune"))
    ), 6);
  }


  /** The attributes the single landcover feature in {@code features} would carry at {@code zoom}. */
  private static Map<String, Object> attrsAtZoom(FeatureCollector features, int zoom) {
    var feature = StreamSupport.stream(features.spliterator(), false)
      .filter(f -> Landcover.LAYER_NAME.equals(f.getLayer()))
      .findFirst()
      .orElseThrow();
    return feature.getAttrsAtZoom(zoom);
  }

  private VectorTile.Feature feature(org.locationtech.jts.geom.Polygon geom, Map<String, Object> m) {
    return new VectorTile.Feature(
      "landcover",
      1,
      VectorTile.encodeGeometry(geom),
      new HashMap<>(m),
      0
    );
  }

  private void assertMerges(List<Map<String, Object>> expected, List<VectorTile.Feature> in, int zoom)
    throws GeometryException {
    Assertions.assertEquals(expected,
      profile.postProcessLayerFeatures("landcover", zoom, in).stream().map(
        VectorTile.Feature::tags)
        .toList());
  }
}
