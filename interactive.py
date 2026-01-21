import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import random


# --- CONFIGURATION ---
NUM_STATIONS = 100  # Number of simulated weather stations (reduced from 1000 for demo performance)
OUTPUT_HTML = "climate_glyph.html"
OUTPUT_LEGEND = "exploded_legend.png"


# --- 1. GEOMETRY DEFINITIONS ---
# Define vertices and faces for standard polyhedra
def get_polyhedron(type='octahedron'):
   if type == 'tetrahedron':
       # 4 faces, 4 vertices
       verts = np.array([[1,1,1], [1,-1,-1], [-1,1,-1], [-1,-1,1]]) * 0.7
       faces = np.array([[0,1,2], [0,2,3], [0,3,1], [1,3,2]])
   elif type == 'cube':
       # 6 faces, 8 vertices (Triangulated for Mesh3d -> 12 triangular faces)
       # Simplified: We will use a dedicated Cube mesh or standard box
       # For this demo, we'll use Octahedron as the primary glyph as requested in the specific mapping
       return get_polyhedron('octahedron')
   elif type == 'octahedron':
       # 8 faces, 6 vertices
       verts = np.array([[1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]])
       # Face indices
       faces = np.array([
           [0,2,4], [2,1,4], [1,3,4], [3,0,4], # Top pyramid
           [0,3,5], [3,1,5], [1,2,5], [2,0,5]  # Bottom pyramid
       ])
   else:
       return get_polyhedron('octahedron')
   return verts, faces


# --- 2. DATA SIMULATION ---
def generate_climate_data(n):
   data = []
   for _ in range(n):
       # Position
       lat = np.random.uniform(-90, 90)
       lon = np.random.uniform(-180, 180)
       alt = np.random.uniform(0, 5000)
      
       # Attributes
       temp_dev = np.random.normal(0, 2)     # Face 1: Diverging (Blue-White-Red)
       humidity = np.random.uniform(0, 100)  # Face 2: Sequential (Light-Dark Blue)
       precip = np.random.uniform(0, 30)     # Face 3: Binned (Green-Yellow-Red)
       wind_speed = np.random.uniform(0, 50) # Face 4: Saturation
       cloud_cover = np.random.uniform(0, 1) # Face 5: Opacity
       pressure = np.random.uniform(900, 1050) # Face 6: Lightness
       zone = np.random.choice(['Arid', 'Temperate', 'Polar']) # Face 7: Categorical
       uv_index = np.random.uniform(0, 12)   # Face 8: Angular Tilt Mapping
      
       # Angular
       wind_dir = np.random.uniform(0, 360)  # Rotation Z
      
       data.append({
           'pos': [lon, lat, alt/1000], # Scale altitude for viz
           'attributes': [temp_dev, humidity, precip, wind_speed, cloud_cover, pressure, zone, uv_index],
           'angle': wind_dir
       })
   return data


# --- 3. COLOR MAPPING HELPERS ---
def map_temp_color(val):
   # Diverging Blue (-5) -> White (0) -> Red (+5)
   if val < -2: return 'rgb(50,50,255)'
   elif val < 2: return 'rgb(240,240,240)'
   else: return 'rgb(255,50,50)'


def map_humidity_color(val):
   # Sequential Light Blue -> Dark Blue
   intensity = int(val * 2.55)
   return f'rgb({255-intensity}, {255-intensity}, 255)' # Simple cyanish fade


def map_precip_color(val):
   # Binned Green (<5), Yellow (5-20), Red (>20)
   if val < 5: return 'rgb(50,200,50)'
   elif val < 20: return 'rgb(255,255,50)'
   else: return 'rgb(200,50,50)'


def map_zone_color(val):
   if val == 'Arid': return 'rgb(255, 200, 100)' # Yellow/Orange
   if val == 'Temperate': return 'rgb(100, 200, 100)' # Green
   return 'rgb(150, 200, 255)' # Polar Blue


def get_face_colors(attrs):
   # Map the 8 attributes to the 8 faces of an Octahedron
   # attrs: [temp, humid, precip, wind, cloud, press, zone, uv]
  
   # Face 1: Temp
   c1 = map_temp_color(attrs[0])
   # Face 2: Humidity
   c2 = map_humidity_color(attrs[1])
   # Face 3: Precip
   c3 = map_precip_color(attrs[2])
   # Face 4: Wind Speed (Simulated as purple saturation)
   c4 = f'rgb({int(attrs[3]*5)}, 0, {int(attrs[3]*5)})'
   # Face 5: Cloud Cover (Simulated as Gray scale)
   c5 = f'rgba(100,100,100, {attrs[4]})' # Note: Mesh3d alpha is global usually, but we try per face
   # Face 6: Pressure (Darkness)
   press_norm = (attrs[5] - 900)/150
   c6 = f'rgb({int(press_norm*255)}, {int(press_norm*255)}, {int(press_norm*255)})'
   # Face 7: Zone
   c7 = map_zone_color(attrs[6])
   # Face 8: UV (Red scale)
   c8 = f'rgb({int(attrs[7]*20)}, 0, 0)'
  
   return [c1, c2, c3, c4, c5, c6, c7, c8]


# --- 4. VISUALIZATION GENERATION (PLOTLY) ---
def create_interactive_chart(data):
   print("Generating 3D meshes...")
  
   mesh_x, mesh_y, mesh_z = [], [], []
   mesh_i, mesh_j, mesh_k = [], [], []
   mesh_facecolor = []
  
   base_verts, base_faces = get_polyhedron('octahedron')
  
   # We combine all glyphs into a single Mesh3d for performance
   current_vert_idx = 0
  
   for d in data:
       x, y, z = d['pos']
       rot_z = np.radians(d['angle'])
      
       # Rotate vertices
       cos_r, sin_r = np.cos(rot_z), np.sin(rot_z)
       # Simple rotation matrix around Z
       rotated_verts = base_verts.copy()
       rotated_verts[:,0] = base_verts[:,0]*cos_r - base_verts[:,1]*sin_r
       rotated_verts[:,1] = base_verts[:,0]*sin_r + base_verts[:,1]*cos_r
      
       # Translate
       final_verts = rotated_verts + [x, y, z]
      
       # Append Vertices
       mesh_x.extend(final_verts[:,0])
       mesh_y.extend(final_verts[:,1])
       mesh_z.extend(final_verts[:,2])
      
       # Append Faces (adjusted for current index)
       new_faces = base_faces + current_vert_idx
       mesh_i.extend(new_faces[:,0])
       mesh_j.extend(new_faces[:,1])
       mesh_k.extend(new_faces[:,2])
      
       # Append Colors
       f_colors = get_face_colors(d['attributes'])
       mesh_facecolor.extend(f_colors)
      
       current_vert_idx += len(base_verts)


   # Construct Figure
   fig = go.Figure(data=[
       go.Mesh3d(
           x=mesh_x, y=mesh_y, z=mesh_z,
           i=mesh_i, j=mesh_j, k=mesh_k,
           facecolor=mesh_facecolor,
           opacity=1.0,
           hoverinfo='text',
           text=[f"Station {i}" for i in range(len(data)) for _ in range(8)] # Tooltip per face (approx)
       )
   ])


   fig.update_layout(
       title="Comprehensive Polyhedral Climate Glyph (12 Dimensions)",
       scene=dict(
           xaxis_title='Longitude',
           yaxis_title='Latitude',
           zaxis_title='Altitude (km)',
           aspectmode='manual',
           aspectratio=dict(x=2, y=1, z=0.5)
       ),
       margin=dict(l=0, r=0, b=0, t=40)
   )
  
   fig.write_html(OUTPUT_HTML)
   print(f"Interactive chart saved to {OUTPUT_HTML}")


# --- 5. LEGEND GENERATION (MATPLOTLIB) ---
def create_exploded_legend():
   print("Generating static legend inset...")
   fig, ax = plt.subplots(figsize=(6, 6))
  
   # Define "exploded" polygon faces manually for 2D representation
   # Center is 0,0
   polys = [
       # Top half
       [[0,0], [1,1], [-1,1]],    # Face 1 (Top Back)
       [[1,1], [2,0], [1,-1]],    # Face 2 (Right)
       [[1,-1], [0,-2], [-1,-1]], # Face 3 (Bottom)
       [[-1,-1], [-2,0], [-1,1]], # Face 4 (Left)
       # Inner/Other faces represented offset
       [[0,0.5], [0.5,1.5], [-0.5,1.5]], # Face 5
       [[1.5,0.5], [2.5,0], [1.5,-0.5]], # Face 6
       [[0,-1.5], [0.5,-2.5], [-0.5,-2.5]], # Face 7
       [[-1.5,0.5], [-2.5,0], [-1.5,-0.5]], # Face 8
   ]
  
   colors = [
       'blue',         # 1: Temp (Cold)
       'lightblue',    # 2: Humidity (Low)
       'green',        # 3: Precip (Low)
       'purple',       # 4: Wind
       'gray',         # 5: Cloud
       'black',        # 6: Pressure
       'yellow',       # 7: Zone (Arid)
       'red'           # 8: UV
   ]
  
   labels = [
       "1: Temp Dev", "2: Humidity", "3: Precip", "4: Wind Spd",
       "5: Cloud Cover", "6: Pressure", "7: Clim. Zone", "8: UV Index"
   ]


   for i, p in enumerate(polys):
       poly = Polygon(p, facecolor=colors[i], edgecolor='white', alpha=0.8)
       ax.add_patch(poly)
       # Label position approx center of poly
       cx = sum([pt[0] for pt in p])/3
       cy = sum([pt[1] for pt in p])/3
       ax.text(cx, cy, str(i+1), ha='center', va='center', color='white', fontweight='bold')


   # Add text list
   ax.text(0, -3.5, "\n".join(labels), ha='center', fontsize=10,
           bbox=dict(facecolor='white', alpha=0.9))


   ax.set_xlim(-3, 3)
   ax.set_ylim(-4, 3)
   ax.axis('off')
   ax.set_title("Exploded Glyph Legend", fontsize=14)
  
   plt.savefig(OUTPUT_LEGEND, dpi=100)
   print(f"Legend inset saved to {OUTPUT_LEGEND}")


# --- MAIN EXECUTION ---
if __name__ == "__main__":
   print("--- Polyhedral Framework: Climate Example Generator ---")
   data = generate_climate_data(NUM_STATIONS)
   create_interactive_chart(data)
   create_exploded_legend()
   print("Done.")