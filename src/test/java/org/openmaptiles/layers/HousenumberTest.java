package org.openmaptiles.layers;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.onthegomap.planetiler.geo.GeometryException;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

class HousenumberTest extends AbstractLayerTest {

  @Test
  void testHousenumber() {
    assertFeatures(14, List.of(Map.of(
      "_layer", "housenumber",
      "_type", "point",
      "_minzoom", 14,
      "_maxzoom", 14,
      "_buffer", 8d
    )), process(pointFeature(Map.of(
      "addr:housenumber", "10"
    ))));
    assertFeatures(14, List.of(Map.of(
      "_layer", "housenumber",
      "_type", "point",
      "_minzoom", 14,
      "_maxzoom", 14,
      "_buffer", 8d
    )), process(polygonFeature(Map.of(
      "addr:housenumber", "10"
    ))));
  }

  @ParameterizedTest
  @CsvSource({
    "1, 1",
    "1;1a;2;2/b;20;3, 1–3",
    "1;1a;2;2/b;20;3;, 1–3",
    "1;2;20;3, 1–20",
    "1;2;20;3;, 1–20",
    ";, ;",
    ";;, ;;",
    "2712;935803935803, 2712–935803935803",
  })
  void testDisplayHousenumber(String outlier, String expected) {
    assertEquals(expected, Housenumber.displayHousenumber(outlier));
  }

  @Test
  void testAddressAttrs() {
    assertFeatures(14, List.of(Map.of(
      "housenumber", "765/6",
      "street", "street",
      "housename", "house",
      "_has_name", "<null>",
      "_partition", "<null>"
    )), process(polygonFeature(Map.of(
      "addr:housenumber", "765/6",
      "addr:block_number", "X",
      "addr:street", "street",
      "addr:housename", "house",
      "name", "name"
    ))));
  }

  @Test
  void testNonduplicateHousenumber() throws GeometryException {
    var layerName = Housenumber.LAYER_NAME;
    var hn1 = pointFeature(
      layerName,
      Map.of("housenumber", "764/2"),
      1
    );
    var hn2 = pointFeature(
      layerName,
      Map.of("housenumber", "765/6"),
      1
    );

    Assertions.assertEquals(
      2,
      profile.postProcessLayerFeatures(layerName, 14, List.of(hn1, hn2)).size()
    );
  }

  @Test
  void testNonduplicateStreet() throws GeometryException {
    var layerName = Housenumber.LAYER_NAME;
    var housenumber = "765/6";
    var hn1 = pointFeature(
      layerName,
      Map.of(
        "housenumber", housenumber,
        "street", "street 1"
      ),
      1
    );
    var hn2 = pointFeature(
      layerName,
      Map.of(
        "housenumber", housenumber,
        "street", "street 2"
      ),
      1
    );

    Assertions.assertEquals(
      2, // same housenumber on different streets => kept apart
      profile.postProcessLayerFeatures(layerName, 14, List.of(hn1, hn2)).size()
    );
  }

  @Test
  void testDuplicateHousenumber() throws GeometryException {
    var layerName = Housenumber.LAYER_NAME;
    var tags = Map.<String, Object>of(
      "housenumber", "765/6",
      "street", "street"
    );
    var hn1 = pointFeature(layerName, tags, 1);
    var hn2 = pointFeature(layerName, tags, 1);

    var result = profile.postProcessLayerFeatures(layerName, 14, List.of(hn1, hn2));

    Assertions.assertEquals(
      1, // duplicates are no longer dropped, but identical ones are merged into one multipoint
      result.size()
    );
    Assertions.assertEquals(
      5, // two points in multipoint => 5 commands
      result.getFirst().geometry().commands().length);
  }
}
