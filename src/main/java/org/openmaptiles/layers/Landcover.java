/*
Copyright (c) 2023, MapTiler.com & OpenMapTiles contributors.
All rights reserved.

Code license: BSD 3-Clause License

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Design license: CC-BY 4.0

See https://github.com/openmaptiles/openmaptiles/blob/master/LICENSE.md for details on usage
*/
package org.openmaptiles.layers;

import static org.openmaptiles.util.Utils.nullIfEmpty;

import com.onthegomap.planetiler.FeatureCollector;
import com.onthegomap.planetiler.FeatureMerge;
import com.onthegomap.planetiler.ForwardingProfile;
import com.onthegomap.planetiler.VectorTile;
import com.onthegomap.planetiler.config.PlanetilerConfig;
import com.onthegomap.planetiler.expression.MultiExpression;
import com.onthegomap.planetiler.geo.GeometryException;
import com.onthegomap.planetiler.reader.SourceFeature;
import com.onthegomap.planetiler.stats.Stats;
import com.onthegomap.planetiler.util.Translations;
import com.onthegomap.planetiler.util.ZoomFunction;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.openmaptiles.OpenMapTilesProfile;
import org.openmaptiles.generated.OpenMapTilesSchema;
import org.openmaptiles.generated.Tables;

/**
 * Defines the logic for generating map elements for natural land cover polygons like ice, sand, and forest in the
 * {@code landcover} layer from source features.
 * <p>
 * This class is ported to Java from
 * <a href="https://github.com/openmaptiles/openmaptiles/tree/master/layers/landcover">OpenMapTiles landcover sql
 * files</a>.
 */
public class Landcover implements
  OpenMapTilesSchema.Landcover,
  OpenMapTilesProfile.NaturalEarthProcessor,
  Tables.OsmLandcoverPolygon.Handler,
  ForwardingProfile.LayerPostProcessor {

  /*
   * Large ice areas come from natural earth and the rest come from OpenStreetMap at higher zoom
   * levels. At render-time, postProcess() merges polygons into larger connected area based
   * on the number of points in the original area.  Since postProcess() only has visibility into
   * features on a single tile, process() needs to pass the number of points the original feature
   * had through using a temporary "_numpoints" attribute.
   */

  public static final ZoomFunction<Number> MIN_PIXEL_SIZE_THRESHOLDS = ZoomFunction.fromMaxZoomThresholds(Map.of(
    13, 8,
    10, 4,
    9, 2
  ));
  private static final String TEMP_NUM_POINTS_ATTR = "_numpoints";
  private static final Set<String> WOOD_OR_FOREST = Set.of(
    FieldValues.SUBCLASS_WOOD,
    FieldValues.SUBCLASS_FOREST
  );
  private final MultiExpression.Index<String> classMapping;
  private final PlanetilerConfig config;

  public Landcover(Translations translations, PlanetilerConfig config, Stats stats) {
    this.classMapping = FieldMappings.Class.index();
    this.config = config;
  }

  private String getClassFromSubclass(String subclass) {
    return subclass == null ? null : classMapping.getOrElse(Map.of(Fields.SUBCLASS, subclass), null);
  }

  @Override
  public void processNaturalEarth(String table, SourceFeature feature,
    FeatureCollector features) {
    record LandcoverInfo(String subclass, int minzoom, int maxzoom) {}
    LandcoverInfo info = switch (table) {
      case "ne_110m_glaciated_areas" -> new LandcoverInfo(FieldValues.SUBCLASS_GLACIER, 0, 1);
      case "ne_50m_glaciated_areas" -> new LandcoverInfo(FieldValues.SUBCLASS_GLACIER, 2, 4);
      case "ne_10m_glaciated_areas" -> new LandcoverInfo(FieldValues.SUBCLASS_GLACIER, 5, 6);
      case "ne_50m_antarctic_ice_shelves_polys" -> new LandcoverInfo("ice_shelf", 2, 4);
      case "ne_10m_antarctic_ice_shelves_polys" -> new LandcoverInfo("ice_shelf", 5, 6);
      default -> null;
    };
    if (info != null) {
      String clazz = getClassFromSubclass(info.subclass);
      if (clazz != null) {
        features.polygon(LAYER_NAME).setBufferPixels(BUFFER_SIZE)
          .setAttr(Fields.CLASS, clazz)
          .setAttr(Fields.SUBCLASS, info.subclass)
          .setZoomRange(info.minzoom, info.maxzoom);
      }
    }
  }

  @Override
  public void process(Tables.OsmLandcoverPolygon element, FeatureCollector features) {
    String subclass = element.subclass();
    String clazz = getClassFromSubclass(subclass);
    if (clazz != null) {
      var feature = features.polygon(LAYER_NAME).setBufferPixels(BUFFER_SIZE)
        .setMinPixelSizeOverrides(MIN_PIXEL_SIZE_THRESHOLDS)
        // default is 0.1, this helps reduce size of some heavy z7-10 tiles
        .setPixelToleranceBelowZoom(10, 0.25)
        .setAttr(Fields.CLASS, clazz)
        .setAttr(Fields.SUBCLASS, subclass)
        .setNumPointsAttr(TEMP_NUM_POINTS_ATTR)
        .setMinZoom(7);

      /*
       * A named wood, forest or moor is often the only way to describe where you are in open
       * country - "Garadhban", "Drumclog Moor". The landcover table has no name column, so the
       * name is read straight off the source feature, and "name" is a literal because the
       * landcover layer schema has no name field of its own.
       *
       * Max zoom only, as elsewhere - which is also what leaves postProcess() alone. It merges
       * polygons at z7-13 and returns early above that, and below max zoom this attribute
       * resolves to null and is dropped, so the attribute sets it groups on are unchanged. Were
       * the build's maxzoom ever dropped to 13 or below, differently named woods would stop
       * merging into one another and those tiles would grow.
       */
      String name = nullIfEmpty(element.source().getString("name"));
      if (name != null) {
        // Guarded rather than set unconditionally: a null value still wraps the attribute in a
        // ZoomFunction, which would cost every unnamed polygon in the planet's largest polygon
        // layer a separate attribute map per zoom, for a value that is never there.
        feature.setAttrWithMinzoom("name", name, config.maxzoom());
      }
    }
  }

  @Override
  public List<VectorTile.Feature> postProcess(int zoom, List<VectorTile.Feature> items) throws GeometryException {
    if (zoom < 7 || zoom > 13) {
      for (var item : items) {
        item.tags().remove(TEMP_NUM_POINTS_ATTR);
      }
      return items;
    } else { // z7-13
      // merging only merges polygons with the same attributes, so use this temporary key
      // to separate features into layers that will be merged separately
      String tempGroupKey = "_group";
      List<VectorTile.Feature> result = new ArrayList<>();
      List<VectorTile.Feature> toMerge = new ArrayList<>();
      for (var item : items) {
        Map<String, Object> attrs = item.attrs();
        Object numPointsObj = attrs.remove(TEMP_NUM_POINTS_ATTR);
        Object subclassObj = attrs.get(Fields.SUBCLASS);
        if (numPointsObj instanceof Number num && subclassObj instanceof String subclass) {
          long numPoints = num.longValue();
          if (zoom >= 10) {
            if (WOOD_OR_FOREST.contains(subclass) && numPoints < 300) {
              attrs.put(tempGroupKey, "<300");
              toMerge.add(item);
            } else { // don't merge
              result.add(item);
            }
          } else if (zoom >= 8 && zoom <= 9) {
            if (WOOD_OR_FOREST.contains(subclass)) {
              attrs.put(tempGroupKey, numPoints < 300 ? "<300" : ">300");
              toMerge.add(item);
            } else { // don't merge
              result.add(item);
            }
          } else { // zoom 7
            toMerge.add(item);
          }
        } else {
          result.add(item);
        }
      }
      var merged = FeatureMerge.mergeOverlappingPolygons(toMerge, 4);
      for (var item : merged) {
        item.tags().remove(tempGroupKey);
      }
      result.addAll(merged);
      return result;
    }
  }
}
