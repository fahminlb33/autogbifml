// ----------------------------------------
// STATIC DATA
// ----------------------------------------

mapboxgl.accessToken =
  "pk.eyJ1IjoiZmFobWlubGIzMyIsImEiOiJjbTEyOG5sdnQxMDhiMnJzZmZidWo1Zm85In0.5_Rc4JZq2SK1flcZ6N2hPQ";

const MONTHS = [
  "Januari",
  "Februari",
  "Maret",
  "April",
  "Mei",
  "Juni",
  "Juli",
  "Agustus",
  "September",
  "Oktober",
  "November",
  "Desember",
];

const PREDICTION_CLASS = {
  0: "Tidak ada paus",
  1: "Ada kemunculan paus",
};

function getPredictionSources() {
  const defaultPaint = {
    "fill-opacity": 0.7,
    "fill-color": [
      "interpolate",
      ["linear"],
      ["get", "proba"],
      0,
      "#fde0c5",
      1,
      "#eb4a40",
    ],
  };

  return [
    {
      key: "africa",
      sourceOptions: {
        type: "geojson",
        data: "/assets/maps/africa.json",
      },
      layerOptions: {
        type: "fill",
        paint: structuredClone(defaultPaint),
      },
    },
    {
      key: "australia",
      sourceOptions: {
        type: "geojson",
        data: "/assets/maps/australia.json",
      },
      layerOptions: {
        type: "fill",
        paint: structuredClone(defaultPaint),
      },
    },
  ];
}

function getCMEMSUrl(productId, datasetId, variable, time = 0) {
  // base URI
  let url =
    "https://wmts.marine.copernicus.eu/teroWmts/?service=WMTS&version=1.0.0&request=GetTile&tilematrixset=EPSG:3857&tilematrix={z}&tilecol={x}&tilerow={y}";

  // add time query
  if (time !== 0) {
    url += "&time=2023-" + time.toString().padStart(2, "0") + "-01";
  }

  // add layer
  url += `&layer=${productId}/${datasetId}/${variable}`;

  return url;
}

// https://help.marine.copernicus.eu/en/articles/6478168-how-to-use-wmts-to-visualize-data
// https://help.marine.copernicus.eu/en/articles/9527599-how-to-visualize-wmts-layers-on-google-earth-web-app
// https://wmts.marine.copernicus.eu/teroWmts/GLOBAL_ANALYSISFORECAST_PHY_001_024/cmems_mod_glo_phy-cur_anfc_0.083deg_P1M-m_202406?request=GetCapabilities&service=WMS
function* getCMEMSSources() {
  const datasets = {
    GLOBAL_ANALYSISFORECAST_PHY_001_024: {
      "cmems_mod_glo_phy_anfc_0.083deg_P1M-m_202406": [
        "zos",
        "pbo",
        "tob",
        "sob",
      ],
      "cmems_mod_glo_phy-cur_anfc_0.083deg_P1M-m_202406": ["uo", "vo"],
      "cmems_mod_glo_phy-so_anfc_0.083deg_P1M-m_202406": ["so"],
      "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1M-m_202406": ["thetao"],
      "cmems_mod_glo_phy-wcur_anfc_0.083deg_P1M-m_202406": ["wo"],
    },
    GLOBAL_ANALYSISFORECAST_BGC_001_028: {
      "cmems_mod_glo_bgc-car_anfc_0.25deg_P1D-m_202311": ["nppv", "o2"],
      "cmems_mod_glo_bgc-car_anfc_0.25deg_P1M-m_202311": [
        "talk",
        "dissic",
        "ph",
      ],
      "cmems_mod_glo_bgc-co2_anfc_0.25deg_P1M-m_202311": ["spco2"],
      "cmems_mod_glo_bgc-nut_anfc_0.25deg_P1M-m_202311": [
        "no3",
        "po4",
        "si",
        "fe",
      ],
      "cmems_mod_glo_bgc-pft_anfc_0.25deg_P1M-m_202311": ["chl", "phyc"],
    },
  };

  const variableNames = {
    // CMEMS_GLOBAL_ANALYSISFORECAST_PHY_001_024
    zos: "Fisika: Tinggi permukaan laut (m)",
    pbo: "Fisika: Tekanan air pada dasar laut (m)",
    tob: "Fisika: Suhu air pada dasar laut (°C)",
    sob: "Fisika: Salinitas air pada dasar laut (psu)",
    so: "Fisika: Salinitas air pada permukaan (psu)",
    thetao: "Fisika: Suhu air pada kedalaman tertentu (°C)",
    uo: "Fisika: Kecepatan arus laut ke arah timur (m s<sup>-1</sup>)",
    vo: "Fisika: Kecepatan arus laut ke arah utara (m s<sup>-1</sup>)",
    wo: "Fisika: Kecepatan arus laut ke arah permukaan (m s<sup>-1</sup>)",

    // GLOBAL_ANALYSISFORECAST_BGC_001_028
    nppv: "Biogeokimia: Produksi biomassa primer (mg m<sup>-3</sup> hari<sup>-1</sup>)",
    o2: "Biogeokimia: Konsentrasi oksigen terlarut dalam air (mmol m<sup>-3</sup>)",
    talk: "Biogeokimia: Total basa/alkalinitas (mol m<sup>-3</sup>)",
    dissic:
      "Biogeokimia: Konsentrasi karbon anorganik terlarut dalam air (mol m<sup>-3</sup>)",
    ph: "Biogeokimia: pH",
    spco2: "Biogeokimia: Tekanan karbon dioksida pada permukaan laut (Pa)",
    no3: "Biogeokimia: Nitrat terlarut dalam air (mmol m<sup>-3</sup>)",
    po4: "Biogeokimia: Fosfat terlarut dalam air (mmol m<sup>-3</sup>)",
    si: "Biogeokimia: Silika terlarut dalam air (mmol m<sup>-3</sup>)",
    fe: "Biogeokimia: Besi terlarut dalam air (mmol m<sup>-3</sup>)",
    chl: "Biogeokimia: Total massa klorofil terlarut dalam air (mg m<sup>-3</sup>)",
    phyc: "Biogeokimia: Konsentrasi fitoplankton terlarut dalam air (mmol m<sup>-3</sup>)",
  };

  // process all products
  for (const productId in datasets) {
    for (const datasetId in datasets[productId]) {
      for (const variable of datasets[productId][datasetId]) {
        yield {
          key: `cmems_${variable}`,
          label: variableNames[variable],
          productId,
          datasetId,
          variable,
          sourceOptions: {
            type: "raster",
            tileSize: 256,
            attribution:
              "&copy; Copernicus Marine Environment Monitoring Service (CMEMS)",
            tiles: [getCMEMSUrl(productId, datasetId, variable)],
          },
          layerOptions: {
            type: "raster",
            layout: {
              visibility: "none",
            },
          },
        };
      }
    }
  }
}

var BASEMAPS = [
  // ESRI
  {
    // source: https://leaflet-extras.github.io/leaflet-providers/preview/
    // source: https://www.arcgis.com/home/item.html?id=5ae9e138a17842688b0b79283a4353f6
    // alternative: https://docs.mapbox.com/mapbox-gl-js/example/style-ocean-depth-data/
    key: "esri_ocean",
    label: "ESRI World Ocean",
    sourceOptions: {
      type: "raster",
      tiles: [
        "https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}",
      ],
      tileSize: 256,
      attribution:
        "Tiles &copy; Esri &mdash; Sources: GEBCO, NOAA, CHS, OSU, UNH, CSUMB, National Geographic, DeLorme, NAVTEQ, and Esri",
    },
    layerOptions: {
      type: "raster",
    },
  },

  // CMEMS
  ...getCMEMSSources(),
];

// ----------------------------------------
// ALPINE APP
// ----------------------------------------

var map = null;

document.addEventListener("alpine:init", () => {
  Alpine.data("app", () => ({
    // states
    monthText: "",
    selectedMonth: 5,
    selectedLayer: "esri_ocean",
    props: null,
    basemaps: Object.freeze(structuredClone(BASEMAPS)),

    get selectedProps() {
      if (this.props === null) {
        return {
          show: false,
          zone_id: "-",
          target: "-",
          predicted: "-",
          proba: "-",
          proba_log: "-",

          sob_mean: "-",
          sob_sum: "-",
          fe_mean: "-",
          fe_sum: "-",
          so_sum: "-",
          po4_mean: "-",
          pbo_mean: "-",
          pbo_sum: "-",
          tob_mean: "-",
          tob_sum: "-",
        };
      }

      return {
        show: true,
        zone_id: this.props.zone_id,
        target: PREDICTION_CLASS[this.props.target],
        predicted: PREDICTION_CLASS[this.props.predicted],
        proba: (this.props.proba * 100).toFixed(4) + "%",
        proba_log: this.props.proba_log.toFixed(4),

        sob_mean: this.props.sob_mean.toFixed(4),
        sob_sum: this.props.sob_sum.toFixed(4),
        fe_mean: this.props.fe_mean.toFixed(4),
        fe_sum: this.props.fe_sum.toFixed(4),
        so_sum: this.props.so_sum.toFixed(4),
        po4_mean: this.props.po4_mean.toFixed(4),
        pbo_mean: this.props.pbo_mean.toFixed(4),
        pbo_sum: this.props.pbo_sum.toFixed(4),
        tob_mean: this.props.tob_mean.toFixed(4),
        tob_sum: this.props.tob_sum.toFixed(4),
      };
    },

    // initialization
    init() {
      // create Mapbox map
      map = new mapboxgl.Map({
        container: "map",
        center: { lng: 42.3586993633067, lat: -18.7128841592949 },
        zoom: 3,
        minZoom: 3,
        maxZoom: 8,
        style: "mapbox://styles/mapbox/light-v11",
        // style: 'mapbox://styles/mapbox/streets-v11',
        projection: {
          name: "mercator",
        },
      });

      // add zoom controls
      map.addControl(new mapboxgl.NavigationControl());

      // bind event handlers
      map.on("load", this.onMapLoad.bind(this));
      map.on("mousemove", "africa", this.onMapMouseMove.bind(this));
      map.on("mouseleave", "africa", this.onMapMouseLeave.bind(this));
      map.on("mousemove", "australia", this.onMapMouseMove.bind(this));
      map.on("mouseleave", "australia", this.onMapMouseLeave.bind(this));

      // control events
      this.$watch("selectedMonth", this.onFilterChange.bind(this));
      this.$watch("selectedLayer", this.onBasemapChange.bind(this));
    },

    // methods
    filterBy(month) {
      // set prediction filters
      const filter = ["==", "ts_month", month];
      map.setFilter("africa", filter);
      map.setFilter("australia", filter);

      // set basemap
      if (this.selectedLayer !== "esri_ocean") {
        // get currently active basemap
        const basemap = BASEMAPS.find((x) => x.key === this.selectedLayer);

        // get new tiles URL
        const url = getCMEMSUrl(
          basemap.productId,
          basemap.datasetId,
          basemap.variable,
          month,
        );
        map.getSource(this.selectedLayer).setTiles([url]);
      }

      // Set the label to the month
      this.monthText = MONTHS[month];
    },

    // event handlers
    onFilterChange(value) {
      this.filterBy(value);
    },

    onBasemapChange(value, oldValue) {
      // https://docs.mapbox.com/mapbox-gl-js/example/toggle-layers/
      map.setLayoutProperty(oldValue, "visibility", "none");
      map.setLayoutProperty(value, "visibility", "visible");
    },

    onMapMouseMove(e) {
      // return if no feature is selected
      if (e.features.length < 1) {
        return;
      }

      // Change the cursor style as a UI indicator.
      map.getCanvas().style.cursor = "pointer";

      // Copy coordinates array.
      this.props = structuredClone(e.features[0].properties);
    },

    onMapMouseLeave() {
      map.getCanvas().style.cursor = "";
      this.props = null;
    },

    onMapLoad() {
      // add sources and layers
      for (const source of BASEMAPS) {
        map.addSource(source.key, source.sourceOptions);
        map.addLayer({
          id: source.key,
          source: source.key,
          ...source.layerOptions,
        });
      }

      // add predictions
      for (const source of getPredictionSources()) {
        map.addSource(source.key, source.sourceOptions);
        map.addLayer({
          id: source.key,
          source: source.key,
          ...source.layerOptions,
        });
      }

      // show initial
      this.filterBy(5, map);
    },
  }));
});
