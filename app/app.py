"""
True Density - Urban Morphology Analysis App
==============================================
Calculates morphological density measurements using Open Street Map data.
"""
import streamlit as st
import pandas as pd
import geopandas as gpd
import osmnx as ox
import momepy
import plotly.express as px
from typing import Optional, Tuple
import logging
import time
import geocoder
from libpysal import graph


import streamlit as st

bg_url = "https://raw.githubusercontent.com/teemuja/urban_density/main/assets/background.jpg"

page_bg_css = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("{bg_url}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}}

[data-testid="stHeader"] {{
    background: rgba(0,0,0,0);  /* make header transparent */
}}

[data-testid="stSidebar"] {{
    background: rgba(255,255,255,0.7); /* optional styling */
}}
</style>
"""
st.markdown(page_bg_css, unsafe_allow_html=True)


def getlatlon(add: str) -> Optional[Tuple[float, float]]:
    """Get latitude and longitude from address using Mapbox geocoder."""
    try:
        loc = geocoder.mapbox(add, key=st.secrets['MAPBOX_TOKEN'])
        if loc.ok:
            lat = loc.lat
            lon = loc.lng
            return (lat, lon)
        else:
            logger.warning(f"Geocoder could not find location: {add}")
            return None
    except Exception as e:
        logger.error(f"Geocoding error for '{add}': {type(e).__name__}")
        return None
    
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prevent geocoder from logging URLs with tokens
logging.getLogger('geocoder').setLevel(logging.WARNING)
logging.getLogger('geocoder.base').setLevel(logging.WARNING)

# Configure OSMnx settings for better reliability
ox.settings.max_query_area_size = 50000 * 50000  # Limit query area
ox.settings.useful_tags_way = ox.settings.useful_tags_way + ['building', 'building:levels']

# Page configuration
st.set_page_config(
    page_title="True Density - Urban Morphology Analysis", 
    layout="wide", 
    initial_sidebar_state='expanded'
)

# Custom CSS
st.markdown("""
<style>
button[title="View fullscreen"]{
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# Initialize secrets with error handling
try:
    px.set_mapbox_access_token(st.secrets['MAPBOX_TOKEN'])
    MAPBOX_STYLE = st.secrets['MAPBOX_STYLE']
except KeyError as e:
    # Don't expose which secret is missing in logs
    st.error("Missing required Mapbox credentials. Please configure secrets in secrets.toml")
    logger.error("Missing Mapbox secret configuration")
    st.stop()
except Exception as e:
    st.error("Error loading Mapbox credentials. Please check your configuration.")
    logger.error(f"Error loading secrets: {type(e).__name__}")
    st.stop()

# Header
st.header("True Density", divider='green')
st.markdown("Morphological density measurements using Open Street Map data")


@st.cache_data(ttl=900, max_entries=5)
def get_building_data(address: str, tags: dict, radius: int = 500) -> Optional[gpd.GeoDataFrame]:
    """
    Get building footprint data around an address using OSMnx.
    
    Args:
        address: Location address or place name
        tags: OSM tags dictionary for filtering
        radius: Search radius in meters
        
    Returns:
        GeoDataFrame with building footprints or None if error occurs
    """
    max_retries = 2
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            # Get features using osmnx
            logger.info(f"Fetching building data for '{address}' (attempt {attempt + 1}/{max_retries})")
            
            # Get coordinates using geocoder
            latlon = getlatlon(address)
            if latlon is None:
                logger.warning(f"Could not geocode address: {address}")
                return None
            
            # Fetch buildings from OSM
            gdf = ox.features_from_point(center_point=latlon, tags=tags, dist=radius)
            
            if gdf is None or len(gdf) == 0:
                logger.warning(f"No buildings found for address: {address}")
                return None
            
            # Project the GeoDataFrame
            fp_proj = ox.projection.project_gdf(gdf)
            
            # Select only polygons (buildings)
            fp_proj = fp_proj[fp_proj.geometry.type == 'Polygon']
            
            if len(fp_proj) == 0:
                logger.warning(f"No polygon buildings found for address: {address}")
                return None
            
            # Select desired columns, handling missing columns gracefully
            cols = ["geometry", "building", "building:levels", "addr:street"]
            existing_cols = [col for col in cols if col in fp_proj.columns]
            fp_poly = fp_proj[existing_cols].copy()
            
            # Add missing columns if needed
            if "building:levels" not in fp_poly.columns:
                fp_poly["building:levels"] = None
                
            # Calculate areas
            fp_poly["area"] = fp_poly.geometry.area
            
            # Filter by minimum area
            fp_poly = fp_poly[fp_poly["area"] > 50]
            
            if len(fp_poly) == 0:
                logger.warning(f"No buildings larger than 50 sqm found for address: {address}")
                return None
            
            # Convert levels to numeric
            fp_poly["building:levels"] = pd.to_numeric(
                fp_poly["building:levels"], 
                errors="coerce", 
                downcast="float"
            )
            
            return fp_poly
                
        except Exception as e:
            logger.error(f"Error fetching building data for {address} (attempt {attempt + 1}): {type(e).__name__}")
            # Retry on certain errors
            if attempt < max_retries - 1:
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ['timeout', 'connection', 'network', 'request']):
                    logger.info(f"Retrying after {retry_delay}s delay...")
                    time.sleep(retry_delay)
                    continue
            return None
    
    return None

# USER INPUT
user_input = st.text_input('Type address or place on earth', placeholder='e.g., Times Square, New York')

if not user_input:
    st.info("👆 Enter a location to analyze urban density")
    st.stop()

# Clean the address input
import re
add = re.sub(' +', ' ', user_input.strip())

# Fetch building data
with st.spinner(f'Fetching building data from OpenStreetMap for "{add}"... Please wait.'):
    tags = {'building': True}
    radius = 500
    
    # Show progress indicator
    progress_text = st.empty()
    progress_text.text("⏳ Geocoding address and connecting to OpenStreetMap...")
    
    start_time = time.time()
    buildings = get_building_data(add, tags, radius)
    elapsed_time = time.time() - start_time
    
    progress_text.empty()  # Clear progress text

# Error handling for building data
if buildings is None:
    st.error(f"❌ Could not fetch building data for '{add}'.")
    st.info("""💡 **Possible reasons:**
    - The location name is not recognized by OpenStreetMap
    - The request timed out (server may be busy)
    - No building data available for this area
    
    **Try:**
    - Use a more specific address (e.g., "Times Square, New York, NY, USA")
    - Try a well-known landmark name
    - Wait a moment and try again if the server was busy
    """)
    st.stop()
    
logger.info(f"Successfully fetched {len(buildings)} buildings in {elapsed_time:.1f}s")

if len(buildings) == 0:
    st.warning(f"⚠️ No buildings found within {radius}m of '{add}'")
    st.stop()

# Prepare focus area and cut edge footprints
try:
    with st.spinner('Preparing map...'):
        union = buildings.union_all()
        env = union.envelope
        focus = gpd.GeoSeries(env)
        focus_area = gpd.GeoSeries(focus)
        focus_circle = focus_area.centroid.buffer(radius)
        focus_gdf = gpd.GeoDataFrame(focus_circle, geometry=0, crs=buildings.crs)
        
        # Cut buildings to focus circle
        fp_cut = gpd.overlay(buildings, focus_gdf, how='intersection')
        
        if len(fp_cut) == 0:
            st.error("No buildings found in the analysis area after filtering")
            st.stop()
            
except Exception as e:
    st.error("Error processing building geometries. Please try a different location.")
    logger.error(f"Geometry processing error: {type(e).__name__}")
    st.stop()

# Create map
try:
    plot = fp_cut.to_crs(4326)
    tag_order = plot.building.value_counts().index.tolist()
    centroid = plot.union_all().centroid
    lat = centroid.y
    lon = centroid.x
    
    # Use mapbox-based choropleth for correct basemap rendering
    mymap = px.choropleth_mapbox(
        plot,
        geojson=plot.geometry,
        locations=plot.index,
        title=f'{add}',
        color="building",
        hover_name='building',
        hover_data=['building:levels', 'addr:street'],
        labels={"building": 'Building tags in use sorted by count'},
        category_orders={"building": tag_order},
        mapbox_style=MAPBOX_STYLE,
        color_discrete_sequence=px.colors.qualitative.D3,
        center={"lat": lat, "lon": lon},
        zoom=14,
        opacity=0.8,
        width=1200,
        height=700
    )
    
    with st.expander("Map", expanded=True):
        st.plotly_chart(mymap, width='stretch')
        
except Exception as e:
    st.error("Error creating map. Please try a different location.")
    logger.error(f"Map creation error: {type(e).__name__}")

# Floor information
try:
    flr_rate = 100 - round(buildings['building:levels'].isna().sum() / len(buildings.index) * 100, 0)
    floor_med = buildings['building:levels'].median()
    st.caption(
        f'Floor number information in {flr_rate}% of buildings with median value of {floor_med}. '
        'The rest will be estimated using nearby medians.'
    )
except Exception as e:
    logger.warning(f"Could not calculate floor statistics: {type(e).__name__}")
    st.caption("Floor information is incomplete for this area.")

# -------------------------------------------------------------------
import numpy as np
import geopandas as gpd
from libpysal import graph
import momepy
import streamlit as st
from typing import Optional

#@st.cache_data(ttl=120)
def osm_densities(_buildings: gpd.GeoDataFrame) -> Optional[gpd.GeoDataFrame]:
    """
    Calculate morphological density measures for buildings.

    Parameters
    ----------
    _buildings : GeoDataFrame
        GeoDataFrame with building footprints (EPSG:4326 or any projected CRS).

    Returns
    -------
    GeoDataFrame
        Input buildings with added density metrics in EPSG:4326.
        Returns None if something goes wrong.
    """

    if _buildings.empty:
        return None

    # ------------------------------------------------------------------
    # 1. Project to UTM for accurate area calculations
    # ------------------------------------------------------------------
    try:
        utm = _buildings.estimate_utm_crs()
        gdf = _buildings.to_crs(utm).copy()
    except Exception as e:
        st.error(f"CRS / projection error: {e}")
        return None

    # Make sure we have a clean geometry column
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()

    # Ensure 'building:levels' column exists
    if "building:levels" not in gdf.columns:
        gdf["building:levels"] = np.nan

    # ------------------------------------------------------------------
    # 2. Preprocess and tessellate
    # ------------------------------------------------------------------
    try:
        # Preprocess (fixes typical OSM weirdness)
        gdf = momepy.preprocess(gdf)

        # You can also define a limit if you want to clip (optional)
        # limit = momepy.buffered_limit(gdf, buffer="adaptive")
        # tessellation = momepy.morphological_tessellation(gdf, clip=limit)

        tessellation = momepy.morphological_tessellation(gdf)

        # Verify that tessellation matches buildings – drop collapsed ones
        collapsed, multipolygons = momepy.verify_tessellation(tessellation, gdf)
        if len(collapsed) > 0:
            # Drop problem buildings & corresponding tessellation cells
            gdf = gdf.drop(index=collapsed)
            tessellation = tessellation.drop(index=collapsed)
            st.info(f"Dropped {len(collapsed)} collapsed buildings from tessellation.")

        # Align indices strictly (momepy uses original index for tessellation)
        gdf = gdf.loc[tessellation.index].copy()

    except Exception as e:
        st.error(f"Tessellation error: {e}")
        return None

    st.toast(f"Generated {len(tessellation)} morphological plots for {len(gdf)} buildings.")

    # ------------------------------------------------------------------
    # 3. Build contiguity graph (queen by default) and higher-order graph
    # ------------------------------------------------------------------
    try:
        contiguity_graph = graph.Graph.build_contiguity(tessellation, rook=False)
        higher_order_graph = (
            contiguity_graph
            .higher_order(k=2, lower_order=True)
            .assign_self_weight()  # include focal cell in its own neighborhood
        )
    except Exception as e:
        st.error(f"Graph building error: {e}")
        return None

    # ------------------------------------------------------------------
    # 4. Per-building metrics: GSI, floors, GFA, FSI, OSR
    # ------------------------------------------------------------------
    tessellation_areas = tessellation.geometry.area
    building_areas = gdf.geometry.area

    # Ground Space Index (coverage ratio)
    gdf["footprint"] = building_areas
    gdf["GSI"] = (building_areas / tessellation_areas).replace([np.inf, -np.inf], np.nan)
    gdf["GSI"] = gdf["GSI"].round(3)

    # Fill missing floors using neighborhood median
    try:
        levels_series = gdf["building:levels"].astype("float64")
    except Exception:
        levels_series = pd.to_numeric(gdf["building:levels"], errors="coerce")

    neighbor_stats = higher_order_graph.describe(levels_series, statistics=["median"])
    # neighbor_stats is a DataFrame indexed by the same index as tessellation/gdf
    levels_filled = levels_series.fillna(neighbor_stats["median"]).fillna(1)
    gdf["floors"] = levels_filled.round().astype(int)

    # GFA and FSI
    gdf["GFA"] = gdf["footprint"] * gdf["floors"]
    gdf["FSI"] = (gdf["GFA"] / tessellation_areas).replace([np.inf, -np.inf], np.nan)
    gdf["FSI"] = gdf["FSI"].round(3)

    # Open Space Ratio (OSR = open space per GFA)
    # here: (1 - GSI) / FSI, safely
    with np.errstate(divide="ignore", invalid="ignore"):
        osr = (1 - gdf["GSI"]) / gdf["FSI"]
    osr = osr.replace([np.inf, -np.inf], np.nan)
    gdf["OSR"] = osr.round(3)

    # ------------------------------------------------------------------
    # 5. Neighborhood densities (_ND) using Graph.describe
    #     We follow momepy docs: sums over neighborhood / sum of areas.
    # ------------------------------------------------------------------
    try:
        # Sum of areas in each neighborhood
        areas_sum = higher_order_graph.describe(tessellation_areas, statistics=["sum"])["sum"]

        # Sum of building footprints in neighborhood
        footprint_sum = higher_order_graph.describe(gdf["footprint"], statistics=["sum"])["sum"]

        # GSI_ND = Sum(footprint) / Sum(area) in neighborhood
        gsi_nd = footprint_sum / areas_sum
        gsi_nd = gsi_nd.replace([np.inf, -np.inf], np.nan)
        gdf["GSI_ND"] = gsi_nd.round(2)

        # Sum of GFA in neighborhood
        gfa_sum = higher_order_graph.describe(gdf["GFA"], statistics=["sum"])["sum"]

        # FSI_ND = Sum(GFA) / Sum(area) in neighborhood
        fsi_nd = gfa_sum / areas_sum
        fsi_nd = fsi_nd.replace([np.inf, -np.inf], np.nan)
        gdf["FSI_ND"] = fsi_nd.round(2)

        # OSR_ND computed consistently: (1 - GSI_ND) / FSI_ND, safely
        with np.errstate(divide="ignore", invalid="ignore"):
            osr_nd = (1 - gdf["GSI_ND"]) / gdf["FSI_ND"]
        osr_nd = osr_nd.replace([np.inf, -np.inf], np.nan)
        gdf["OSR_ND"] = osr_nd.round(2)

        # Neighborhood mean of OSR (not recomputing formula)
        osr_mean = higher_order_graph.describe(gdf["OSR"], statistics=["mean"])["mean"]
        gdf["OSR_ND_mean"] = osr_mean.round(2)

    except Exception as e:
        st.error(f"Neighborhood density (_ND) calculation error: {e}")
        return None

    # ------------------------------------------------------------------
    # 6. Clip OSR values to remove extreme outliers
    # ------------------------------------------------------------------
    if gdf["OSR"].notna().any():
        osr_clip_value = gdf["OSR"].quantile(0.99)
        for col in ["OSR", "OSR_ND", "OSR_ND_mean"]:
            if col in gdf.columns:
                gdf[col] = gdf[col].clip(upper=osr_clip_value)

    # ------------------------------------------------------------------
    # 7. Reproject back to EPSG:4326
    # ------------------------------------------------------------------
    try:
        gdf_out = gdf.to_crs(epsg=4326)
    except Exception as e:
        st.error(f"Reprojection error: {e}")
        return None

    return gdf_out



def classify_density(density_data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Classify buildings by density categories based on OSR values.
    
    Args:
        density_data: GeoDataFrame with OSR values
        
    Returns:
        GeoDataFrame with density classifications
    """
    try:
        # OSR classification
        density_data['OSR_class'] = 'nan'
        density_data.loc[density_data['OSR'] > 0, 'OSR_class'] = 'close'
        density_data.loc[density_data['OSR'] > 1, 'OSR_class'] = 'dense'
        density_data.loc[density_data['OSR'] > 2, 'OSR_class'] = 'compact'
        density_data.loc[density_data['OSR'] > 4, 'OSR_class'] = 'spacious'
        density_data.loc[density_data['OSR'] > 8, 'OSR_class'] = 'airy'
        density_data.loc[density_data['OSR'] > 16, 'OSR_class'] = 'spread'
        
        # OSR_ND classification
        density_data['OSR_ND_class'] = 'nan'
        density_data.loc[density_data['OSR_ND'] > 0, 'OSR_ND_class'] = 'close'
        density_data.loc[density_data['OSR_ND'] > 1, 'OSR_ND_class'] = 'dense'
        density_data.loc[density_data['OSR_ND'] > 2, 'OSR_ND_class'] = 'compact'
        density_data.loc[density_data['OSR_ND'] > 4, 'OSR_ND_class'] = 'spacious'
        density_data.loc[density_data['OSR_ND'] > 8, 'OSR_ND_class'] = 'airy'
        density_data.loc[density_data['OSR_ND'] > 16, 'OSR_ND_class'] = 'spread'
        
        return density_data
        
    except Exception as e:
        logger.error(f"Error classifying density: {e}")
        return density_data

# CALCULATE DENSITIES
try:
    tags = buildings['building'].unique().tolist()
    tag_order = plot.building.value_counts().index.tolist()
    top_tags = tag_order[:9]  # Top 9 most common tags
    
    mytags = st.multiselect(
        'Select tags (building types) to include for density analysis',
        tags,
        default=top_tags
    )
    st.caption('Top 9 tags mostly used as a default selection set. See sorted legend of map.')
    
    if not mytags:
        st.warning("Please select at least one building type to analyze")
        st.stop()
    
    my_buildings = buildings.loc[buildings['building'].isin(mytags)]
    
    if len(my_buildings) == 0:
        st.warning("No buildings match the selected types")
        st.stop()
    
except Exception as e:
    st.error(f"Error preparing building selection: {e}")
    logger.error(f"Building selection error: {e}")
    st.stop()

run = st.checkbox('Auto-calculate densities', value=False)

if not run:
    st.info("👆 Check the box above to calculate density metrics")
    st.stop()

# Calculate densities
with st.spinner('Calculating morphological densities... This may take a minute.'):
    density_data = osm_densities(my_buildings)
    #st.data_editor(density_data, width=800)

    
if density_data is None:
    st.error("Failed to calculate density metrics. Please try a different location.")
    st.stop()

# Classify density
case_data = classify_density(density_data)

# Density color mapping
COLORMAP_OSR = {
    "close": "red",
    "dense": "darkgoldenrod",
    "compact": "darkolivegreen",
    "spacious": "lightgreen",
    "airy": "cornflowerblue",
    "spread": "lightblue",
    "nan": "grey"
}

# Create density visualizations
try:
    with st.expander(f"Density nomograms for {add}", expanded=True):
        # Calculate scale maximum
        FSI_scale_max = case_data['FSI'].quantile(0.9)
        
        # OSR plot (per plot)
        fig_OSR = px.scatter(
            case_data,
            title='Buildings colored by OSR per (morphological) plot',
            x='GSI',
            y='FSI',
            color='OSR_class',
            log_y=False,
            hover_name='building',
            hover_data=['addr:street', 'floors', 'GFA', 'OSR', 'OSR_ND'],
            labels={"OSR_class": 'Plot density'},
            category_orders={'OSR_class': ['close', 'dense', 'compact', 'spacious', 'airy', 'spread']},
            color_discrete_map=COLORMAP_OSR
        )
        fig_OSR.update_layout(xaxis_range=[0, 0.75], yaxis_range=[0, FSI_scale_max])
        fig_OSR.update_xaxes(rangeslider_visible=True)

        # OSR_ND plot (per neighborhood)
        fig_OSR_ND = px.scatter(
            case_data,
            title='Buildings colored by OSR per neighbourhood',
            x='GSI',
            y='FSI',
            color='OSR_ND_class',
            log_y=False,
            hover_name='building',
            hover_data=['addr:street', 'floors', 'GFA', 'OSR', 'OSR_ND'],
            labels={"OSR_ND_class": 'Neighbourhood density'},
            category_orders={'OSR_ND_class': ['close', 'dense', 'compact', 'spacious', 'airy', 'spread']},
            color_discrete_map=COLORMAP_OSR
        )
        fig_OSR_ND.update_layout(xaxis_range=[0, 0.75], yaxis_range=[0, FSI_scale_max])
        fig_OSR_ND.update_xaxes(rangeslider_visible=True)
        
        # Display charts
        col1, col2 = st.columns(2)
        col1.plotly_chart(fig_OSR, width='stretch')
        col2.plotly_chart(fig_OSR_ND, width='stretch')
        
        # Summary metrics
        bu_count = len(case_data)
        tot_gfa = round(case_data['GFA'].sum(), -2)
        e_area = tot_gfa / 785375
        
        m1, m2 = st.columns(2)
        m1.metric(
            label=f"Total GFA in {add} in {radius}m radius ({bu_count} buildings)",
            value=f"{tot_gfa:,.0f} sqm",
            delta=f"Density (FSI/FAR) = {e_area:.2f}"
        )
        st.caption('Values are based on footprints and floor number information. Underground GFA is excluded.')

        # Prepare download data
        try:
            save_data = gpd.overlay(
                density_data.to_crs(3067),
                focus_gdf.set_crs(3067),
                how='intersection'
            ).to_crs(4326)
            save_data.insert(0, 'TimeStamp', pd.to_datetime('now').replace(microsecond=0))
            save_data['date'] = pd.to_datetime(save_data['TimeStamp']).dt.date
            save_me = save_data.drop(
                columns=['uID', 'TimeStamp', 'OSR_class', 'OSR_ND_class']
            ).assign(location=add)
            save_me = save_me.assign(flr_rate=flr_rate if 'flr_rate' in locals() else 0)
            save_me['wkt'] = save_me.geometry.to_wkt()
            save_as_wkt = save_me.drop(columns="geometry")
            raks = save_as_wkt.to_csv().encode('utf-8')
            
            st.download_button(
                label="💾 Save density data as CSV",
                data=raks,
                file_name=f'buildings_{add}.csv',
                mime='text/csv'
            )
        except Exception as e:
            logger.warning(f"Could not prepare download data: {e}")
            st.caption("Download option unavailable for this dataset")
            
except Exception as e:
    st.error("Error creating density visualizations. Please try again.")
    logger.error(f"Visualization error: {type(e).__name__}")

# expl container
with st.expander("What is this?", expanded=False):
    st.markdown('Density measures in the nomogram above are derived from the latest density research by'
                ' Meta Berghouser Pont and Per Haupt (2021), Kim Dowey and Elek Pafka (2014) as well as'
                ' from Finnish seminal work by O-I Meurman in 1947.')
    # expl
    selite = '''
    **Density measures**<br>
    **GFA** = Gross Floor Area = Total area of in-door space in building including all floors<br>
    **FSI** = Floor Space Index = FAR = Floor Area Ratio = Ratio of floor area per total area of _morphological plot_<br>
    **GSI** = Ground Space Index = Coverage = Ratio of building footprint per total area of _morphological plot_<br>
    **OSR** = Open Space Ratio = Ratio of non-build space per square meter of gross floor area<br>
    **OSR_ND** = Average OSR of plots in nearby neighborhood<br>
    
    Density classification is based on OSR-values:<br>
    <i>
    Close: OSR < 1 <br>
    Dense: OSR 1-2 <br>
    Compact: OSR 2-4 <br>
    Spacious: OSR 4-8 <br>
    Airy: OSR 8-16 <br>
    Spread: OSR > 16 <br>
    </i>
    <br>
    _Morfological plot_ is a plot generated using polygonal tessellation around buildings using 
    <a href="http://docs.momepy.org/en/stable/user_guide/elements/tessellation.html" target="_blank">Momepy</a>.<br>
    Nearby neighborhood in OSR_ND calculation is based on queen contiguity for 2 degree neighbours 
    (border neighbors and their neighbours as an "experienced neighborhood").<br>
    <br>
    Average OSR values of morphological plots classify urban density well as they combine both
    the volume of architecture (FSI) and the compactness of urban planning (GSI).
    '''
    soveltaen = '''
    <p style="font-family:sans-serif; color:Dimgrey; font-size: 12px;">
    References:<br><i>
    Berghauser Pont, Meta, and Per Haupt. 2021. Spacematrix: Space, Density and Urban Form. Rotterdam: nai010 publishers.<br>
    Dovey, Kim, Pafka, Elek. 2014. The urban density assemblage: Modelling multiple measures. Urban Des Int 19, 66–76<br>
    Meurman, Otto-I. 1947. Asemakaavaoppi. Helsinki: Rakennuskirja.<br>
    Fleischmann, Martin. 2019. momepy: Urban Morphology Measuring Toolkit. Journal of Open Source Software, 4(43), 1807<br>
    Boeing, G. 2017. “OSMnx: New Methods for Acquiring, Constructing, Analyzing, and Visualizing Complex Street Networks.” Computers, Environment and Urban Systems. 65, 126-139.<br>
    </i>
    </p>
    '''
    st.markdown(selite, unsafe_allow_html=True)
    cs1, cs2, cs3 = st.columns(3)
    cs1.latex(r'''
            OSR = \frac {1-GSI} {FSI}
            ''')  # https://katex.org/docs/supported.html

    st.markdown(soveltaen, unsafe_allow_html=True)


# Footer
st.markdown('---')
st.markdown('''
<a href="https://share.streamlit.io/user/teemuja" target="_blank">
    <img src="https://img.shields.io/badge/&copy;-teemuja-fab43a" alt="teemuja" title="Teemu Jama">
</a>
''', unsafe_allow_html=True)