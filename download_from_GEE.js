
var AOIs = [
  {name: 'LoireValley', geometry: ee.Geometry.Rectangle([0.25, 47.25, 0.5, 47.5])},
  {name: 'Provence', geometry: ee.Geometry.Rectangle([5.25, 43.25, 5.5, 43.5])},
  {name: 'Tuscany', geometry: ee.Geometry.Rectangle([11.0, 43.0, 11.25, 43.25])},
  {name: 'BavarianAlps', geometry: ee.Geometry.Rectangle([10.75, 47.5, 11.0, 47.75])},
  {name: 'Transylvania', geometry: ee.Geometry.Rectangle([25.0, 45.75, 25.25, 46.0])},
  {name: 'SierraNevada', geometry: ee.Geometry.Rectangle([-3.5, 37.0, -3.25, 37.25])},
  {name: 'NorwegianFjords', geometry: ee.Geometry.Rectangle([6.75, 60.25, 7.0, 60.5])},
  {name: 'SwissJura', geometry: ee.Geometry.Rectangle([6.0, 47.0, 6.25, 47.25])},
  {name: 'Eifel', geometry: ee.Geometry.Rectangle([6.75, 50.0, 7.0, 50.25])},
  {name: 'PicosdeEuropa', geometry: ee.Geometry.Rectangle([-4.75, 43.0, -4.5, 43.25])},
  {name: 'DanubeDelta', geometry: ee.Geometry.Rectangle([29.25, 45.0, 29.5, 45.25])},
  {name: 'Alps', geometry: ee.Geometry.Rectangle([10.0, 46.0, 10.25, 46.25])},
  {name: 'Camargue', geometry: ee.Geometry.Rectangle([4.5, 43.5, 4.75, 43.75])},
  {name: 'LaMancha', geometry: ee.Geometry.Rectangle([-3.0, 39.0, -2.75, 39.25])},
  {name: 'BlackForest', geometry: ee.Geometry.Rectangle([8.0, 48.0, 8.25, 48.25])},
  {name: 'ScottishHighlands', geometry: ee.Geometry.Rectangle([-5.0, 57.0, -4.75, 57.25])},
  {name: 'VenetianLagoon', geometry: ee.Geometry.Rectangle([12.25, 45.25, 12.5, 45.5])},
  {name: 'Peloponnese', geometry: ee.Geometry.Rectangle([21.75, 37.25, 22.0, 37.5])},
  {name: 'DalmatianCoast', geometry: ee.Geometry.Rectangle([16.0, 43.5, 16.25, 43.75])},
  {name: 'CarpathianMountains', geometry: ee.Geometry.Rectangle([24.0, 47.5, 24.25, 47.75])}

];

function maskS2clouds(image) {
  var qa = image.select('QA60');

  // Bits 10 and 11 are clouds and cirrus, respectively.
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;

  // Both flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
      .and(qa.bitwiseAnd(cirrusBitMask).eq(0));

  return image.updateMask(mask).divide(10000);
}

AOIs.forEach(function(aoi) {
  var dataset = ee.ImageCollection('COPERNICUS/S2_SR')
                    .filterDate('2023-06-01', '2023-09-01')
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',10))
                    .filterBounds(aoi.geometry)
                    .map(maskS2clouds);
                    
  var reqBands = ['B4', 'B3', 'B2'];
  
  dataset = dataset.median().select(reqBands);
  
  Export.image.toDrive({
    image: dataset,
    description: 'sentinel2_' + aoi.name + '_s23_small',
    region: aoi.geometry,
    scale: 10
  });
});
