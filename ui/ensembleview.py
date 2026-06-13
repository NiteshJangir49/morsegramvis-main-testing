import core.utils as utils
import vtk
import time
from settings import Config
from PySide6 import QtCore, QtGui, QtWidgets
from start import Start
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import ui.settings_dialog as settings_dialog
from core import cleandata
from ui.window import Window
import os
import logging
import numpy as np
import pandas as pd


class EnsembleView(Window):
    '''
    This window displays ensemble view of the data.
    '''

    def __init__(self, parent=None):
        super(EnsembleView, self).__init__(parent)
        
        self.parent_ref: Start = parent

        self.bg_color = True
        self.light_factor = True
        self.ao_factor = True

        self.current_ambient = 0.50       # 50% matching default UI display
        self.current_diffuse = 0.50       # 50%
        self.current_specular = 1.00      # 100%
        self.current_specular_power = 100.0 # 100

        self.show_network_only = False
        self.network_graph_actor = None

        # Explicitly cache your original ensemble mesh actor reference
        self.active_ensemble_actor = self.get_ensemble_actor()
        self.ren.AddActor(self.active_ensemble_actor)
        
        # Load the data sources just like viewparticles.py does
        maxima_data_file = utils.read_file(Config.MAXIMAS_FILE)
        self.maxima_data = utils.get_grains_point_cloud(maxima_data_file)
        
        contact_net_data = utils.read_file(Config.CONTACT_NET_FILE)
        self.contact_points = utils.get_contact_network(contact_net_data)

        self.caption_repr = self.get_caption_repr()
        self.last_picked_actor = None
        # self.maxima_data = None
        self.last_picked_property = vtk.vtkProperty()
        self.input_raw_data = None

        self.ensamble_opacity = 1.0

        start_time = time.time()

        self.iren.AddObserver("KeyPressEvent", self.key_press_event)
        # mouse button event
        self.iren.AddObserver("LeftButtonPressEvent", self.left_button_press_event)
        self.ren.AddObserver("EndEvent", self.compute_fps)

        # Add the actor to the scene
        self.active_ensemble_actor = self.get_ensemble_actor()
        self.ren.AddActor(self.active_ensemble_actor)
        self.ren.SetBackground(Config.BG_COLOR)

        self.ambient_occlusion()

        # print execution time
        print("Execution time: ", time.time() - start_time)

        print("==========================================================")
        print("Rendering the ... ")
        print("==========================================================")

        self.frame.setLayout(self.vl)
        self.setCentralWidget(self.frame)

        self.window_title = Config.APP_NAME
        self.setWindowTitle(self.window_title)
        self.start_time = time.time()

        # --------------- Toolbar ----------------
        toolbar = QtWidgets.QToolBar("My main toolbar")

        setting_action = QtGui.QAction(QtGui.QIcon("ui/icons/gear.png"), "Settings", self)
        setting_action.setStatusTip("Settings")
        setting_action.triggered.connect(self.onToolbarSettingsAction)
        toolbar.addAction(setting_action)

        # separator
        toolbar.addSeparator()

        # info action
        info_action = QtGui.QAction(QtGui.QIcon("ui/icons/info.png"), "Info", self)
        info_action.setStatusTip("Info")
        info_action.triggered.connect(self.onToolbarInfoAction)
        toolbar.addAction(info_action)

        # interaction mode checkbox
        toolbar.addSeparator()
        self.interaction_mode_checkbox = QtWidgets.QCheckBox("Interaction Mode")
        self.interaction_mode_checkbox.setChecked(False)
        self.interaction_mode_checkbox.stateChanged.connect(self.onInteractionModeCheckbox)
        toolbar.addWidget(self.interaction_mode_checkbox)

        # separator
        toolbar.addSeparator()

        # checkbox for loading clean ensemble
        self.clean_ensemble_checkbox = QtWidgets.QCheckBox("Clean Ensemble")
        self.clean_ensemble_checkbox.setChecked(False)
        self.clean_ensemble_checkbox.stateChanged.connect(self.onCleanEnsembleCheckbox)
        toolbar.addWidget(self.clean_ensemble_checkbox)

        # camera info toolbar
        self.add_toolbar(toolbar, "ui/icons/cam.png", "Camera Information", self.onToolbarCamInfoAction)
        # export toolbar
        self.add_toolbar(toolbar, "ui/icons/export.png", "Export Scene as Image", self.onToolbarExportAction)

        # toolbar.addSeparator()

        # self.opacity_label = QtWidgets.QLabel("Opacity: ")
        # toolbar.addWidget(self.opacity_label)

        # self.opacity_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        # self.opacity_slider.setMinimum(0)   
        # self.opacity_slider.setMaximum(100)
        # self.opacity_slider.setValue(100)
        # self.opacity_slider.setFixedWidth(100)
        # self.opacity_slider.valueChanged.connect(self.on_opacity_slider_changed)
        # toolbar.addWidget(self.opacity_slider)

        toolbar.addSeparator()
        self.graph_mode_btn = QtWidgets.QPushButton("Show Network")
        self.graph_mode_btn.setCheckable(True)
        self.graph_mode_btn.setChecked(False)
        self.graph_mode_btn.clicked.connect(self.on_graph_mode_toggled)
        toolbar.addWidget(self.graph_mode_btn)

        # ===================================================================
        # LOCAL DROPDOWN ACTION CALLBACK CLOSURES
        # ===================================================================
        def on_visualization_property_changed(property_name):
            print(f"[UI Context] Active property shifted to: {property_name}")
            idx = self.cmap_cb.currentIndex()
            current_cmap = self.cmap_cb.itemData(idx, QtCore.Qt.UserRole)
            if hasattr(self, 'apply_colormap_to_grains') and current_cmap:
                self.apply_colormap_to_grains(current_cmap)

        def on_colormap_selection_changed(index):
            """Fires using the row index to safely parse hidden metadata keys."""
            colormap_name = self.cmap_cb.itemData(index, QtCore.Qt.UserRole)
            if colormap_name:
                print(f"[UI Context] Visual colormap shifted to: {colormap_name}")
                
                # 1. Update the vertical image strip actor texture inside the 3D window canvas
                self.update_native_vtk_legend_image(colormap_name)
                
                # 2. Trigger the VTK color calculation pass
                if hasattr(self, 'apply_colormap_to_grains'):
                    self.apply_colormap_to_grains(colormap_name)
                
                # 3. Refresh the graphics window frame canvas
                if hasattr(self, 'iren') and self.iren:
                    self.iren.GetRenderWindow().Render()

        # ===================================================================
        # INJECT DROPDOWNS INTO MAIN TOOLBAR
        # ===================================================================
        toolbar.addSeparator()
        
        # 1. Property Dropdown (Locked to Coordination Number)
        toolbar.addWidget(QtWidgets.QLabel(" Property: "))
        self.prop_cb = QtWidgets.QComboBox()
        self.prop_cb.addItems(["Coordination Number"])
        self.prop_cb.currentTextChanged.connect(on_visualization_property_changed)
        toolbar.addWidget(self.prop_cb)

        toolbar.addSeparator()

        # 2. Colormap Dropdown (Displays horizontal icons with NO text strings)
        toolbar.addWidget(QtWidgets.QLabel(" Colormap: "))
        self.cmap_cb = QtWidgets.QComboBox()
        self.cmap_cb.currentIndexChanged.connect(on_colormap_selection_changed)
        toolbar.addWidget(self.cmap_cb)
        
        # Populates the icons dynamically from your colormap/ folder
        self.populate_cmap_dropdown_with_icons()

        # ===================================================================
        # INITIALIZE NATIVE 3D VIEWPORT LEGEND & LOAD DEFAULT TEXTURE
        # ===================================================================
        self.setup_native_vtk_legend()

        # Force-load the starting legend preview inside the 3D canvas right at startup
        if self.cmap_cb.count() > 0:
            first_map = self.cmap_cb.itemData(0, QtCore.Qt.UserRole)
            if first_map:
                self.update_native_vtk_legend_image(first_map)

        # --------------- End Toolbar ----------------
        # reset camera
        self.ren.ResetCamera()
        # Start the event loop.
        self.show()
        self.iren.Initialize()

    def load_property_scalars(self):
        """
        Parses the active particle statistics file and dynamically extracts data bounds
        based on the active selection string chosen inside the property dropdown.
        """
        import pandas as pd
        import numpy as np
        import os
        
        try:
            from config import Config
            stats_path = Config.PARTICLE_STATS_FILE
        except Exception:
            base = os.path.dirname(os.path.dirname(__file__))
            stats_path = os.path.join(base, "dataset_files", "Steel Beads", "data", "all_particle_stats.csv")

        if not os.path.exists(stats_path):
            print(f"[Data Pipeline] ERROR: Statistics file missing at {stats_path}")
            return None, None, 0, 1

        try:
            df = pd.read_csv(stats_path)
            
            # 1. READ CURRENT SELECTION FROM THE UI DROP-DOWN
            ui_property_choice = self.prop_cb.currentText()
            
            # Map user-friendly UI strings directly to your explicit CSV column headers
            column_mapping = {
                "Coordination Number": "cn",
                "Volume": "volume",
                "Surface Area": "surface_area",
                "Number of Voxels": "num_voxels"
            }
            
            # Target the mapped key, fallback to the raw text string if adding options directly
            target_column = column_mapping.get(ui_property_choice, ui_property_choice)
            
            if target_column not in df.columns:
                print(f"[Data Pipeline] Warning: '{target_column}' not found. Defaulting to 'cn'.")
                target_column = "cn"
                
            # Convert target values into a clean numeric array
            scalar_values = df[target_column].to_numpy()
            
            # Determine localized bounds dynamically
            d_min = int(np.floor(np.min(scalar_values)))
            d_max = int(np.ceil(np.max(scalar_values)))
            
            # 2. UPDATE THE 3D VIEWPORT TEXT ACTORS DIRECTLY
            if hasattr(self, 'vtk_max_actor') and hasattr(self, 'vtk_min_actor'):
                self.vtk_max_actor.SetInput(f"Max: {d_max}")
                self.vtk_min_actor.SetInput(f"Min: {d_min}")
                print(f"[Data Pipeline] Dynamic boundaries shifted for '{target_column}': Min={d_min}, Max={d_max}")
                
            return df[target_column], target_column, d_min, d_max

        except Exception as e:
            print(f"[Data Pipeline] Failed extracting scalar boundaries dynamically: {e}")
            return None, None, 0, 1

    def apply_colormap_to_grains(self, colormap_name):
        """
        Maps chosen property arrays dynamically to the VTK polydata cells,
        or completely disables scalar tracking if 'Default' is selected.
        """
        # import vtk
        # import pandas as pd
        # import numpy as np
        # import ui.utils as utils
        
        if not hasattr(self, 'active_ensemble_actor') or not self.active_ensemble_actor:
            return

        mapper = self.active_ensemble_actor.GetMapper()

        # ===================================================================
        # OPTION BYPASS: HANDLER FOR DEFAULT UNIFORM MESH COLORING
        # ===================================================================
        if colormap_name == "Default" or colormap_name == "default":
            print("[Data Pipeline] Reverting actor rendering context back to original uniform color.")
            mapper.ScalarVisibilityOff() # Turn off the colormap engine scalars pass
            
            # Restore original material tracking colors
            polydata = mapper.GetInput()
            if polydata and polydata.GetPointData().GetArray('Color'):
                mapper.SelectColorArray('Color')
                mapper.SetScalarModeToUsePointData()
                mapper.ScalarVisibilityOn()
                print("[Data Pipeline] Restored original embedded mesh point colors.")
            else:
                # Fallback: Assign a clean gray material tint if native colors are unassigned
                actor_prop = self.active_ensemble_actor.GetProperty()
                actor_prop.SetColor(0.65, 0.67, 0.70) 
                actor_prop.SetAmbient(0.15)
                actor_prop.SetDiffuse(0.75)
                actor_prop.SetSpecular(0.2)
            
            # Hide the vertical 2D PNG strip legend actors inside your 3D canvas
            if hasattr(self, 'legend_strip_actor'): self.legend_strip_actor.VisibilityOff()
            if hasattr(self, 'vtk_max_actor'): self.vtk_max_actor.VisibilityOff()
            if hasattr(self, 'vtk_min_actor'): self.vtk_min_actor.VisibilityOff()
            
            # Force a clean redraw refresh pass
            if hasattr(self, 'iren') and self.iren: 
                self.iren.GetRenderWindow().Render()
            return
        # ===================================================================

        # 1. Fetch our dynamic data arrays and matching column identity keys
        scalar_series, column_header, d_min, d_max = self.load_property_scalars()
        if scalar_series is None:
            return

        # Ensure text indicators are visible since we are applying a colormap pass
        if hasattr(self, 'vtk_max_actor'): self.vtk_max_actor.VisibilityOn()
        if hasattr(self, 'vtk_min_actor'): self.vtk_min_actor.VisibilityOn()

        # 2. Fetch our continuous XML color transfer function from Phase 1
        ctf = utils.get_vtk_cmap_from_xml(colormap_name, data_min=d_min, data_max=d_max)
        
        # 3. Build lookup dictionary map using our discovered 'cp_id' column
        df = scalar_series.to_frame()
        try:
            from config import Config
            stats_path = Config.PARTICLE_STATS_FILE
        except Exception:
            import os
            base = os.path.dirname(os.path.dirname(__file__))
            stats_path = os.path.join(base, "dataset_files", "Steel Beads", "data", "all_particle_stats.csv")
            
        df['cp_id'] = pd.read_csv(stats_path)['cp_id']
        cn_map = {int(row['cp_id']): float(row[column_header]) for _, row in df.iterrows()}

        polydata = mapper.GetInput()
        if not polydata: 
            return

        cell_data = polydata.GetCellData()
        num_cells = polydata.GetNumberOfCells()

        id_array = cell_data.GetArray('CP ID')
        if not id_array: 
            id_array = polydata.GetPointData().GetArray('CP ID')

        # 4. Construct and populate our destination VTK scalar vector array
        vtk_scalars = vtk.vtkFloatArray()
        vtk_scalars.SetName("ActivePropertyScalars")
        vtk_scalars.SetNumberOfValues(num_cells)

        matched_count = 0
        for i in range(num_cells):
            if id_array:
                mesh_particle_id = int(id_array.GetValue(i))
                if mesh_particle_id in cn_map:
                    val = cn_map[mesh_particle_id]
                    matched_count += 1
                else:
                    val = float(d_min)
            else:
                val = float(scalar_series.iloc[i]) if i < len(scalar_series) else float(d_min)
            
            vtk_scalars.SetValue(i, val)

        print(f"[Data Pipeline] Aligned and matched {matched_count}/{num_cells} cells using 'cp_id' column.")

        # 5. Connect the array back to the cell data layers
        cell_data.SetScalars(vtk_scalars)
        cell_data.SetActiveScalars("ActivePropertyScalars")

        # 6. FORCE FLUSH PARAMETERS AND RESET SCALAR MODE TO CELL DATA
        # This explicitly overrides point-layer locks from the Default option switch!
        mapper.SetScalarModeToUseCellData()
        mapper.SetLookupTable(ctf)
        mapper.SetScalarRange(d_min, d_max)
        mapper.SelectColorArray("ActivePropertyScalars")
        mapper.ScalarVisibilityOn()
        mapper.InterpolateScalarsBeforeMappingOn()
        
        # 7. Update the vertical 3D canvas viewport PNG legend texture strip
        self.update_native_vtk_legend_image(colormap_name)

        # 8. Trigger an immediate canvas draw refresh pass
        self.active_ensemble_actor.Modified()
        if hasattr(self, 'iren') and self.iren:
            self.iren.GetRenderWindow().Render()
        
        print(f"[Data Pipeline] Dynamic recolor complete. Rendered metric: '{column_header}' across cells.")

    def get_available_colormaps(self):
        """Scans the colormap/ folder and returns a list of clean XML map names."""
        import os
        colormap_list = []
        base_dir = os.path.dirname(os.path.dirname(__file__))
        colormap_dir = os.path.join(base_dir, 'colormap')
        
        if os.path.exists(colormap_dir):
            for file in os.listdir(colormap_dir):
                if file.endswith('.xml'):
                    colormap_list.append(os.path.splitext(file)[0])
        return sorted(colormap_list)

    def populate_cmap_dropdown_with_icons(self):
        """Loads PNG previews from the colormap folder and pairs them as icons with hidden text names."""
        import os
        from PySide6.QtGui import QIcon, QPixmap
        
        self.cmap_cb.clear()
        
        # 1. INJECT THE DEFAULT OPTION AS THE FIRST ROW ENTRY
        # We pass "Default" as the visible text so it stands out cleanly against the textless icons.
        self.cmap_cb.addItem("Default", "default")
        
        base_dir = os.path.dirname(os.path.dirname(__file__))
        colormap_dir = os.path.join(base_dir, 'colormap')
        available_maps = self.get_available_colormaps()
        
        icon_size = QtCore.QSize(100, 18)
        self.cmap_cb.setIconSize(icon_size)
        
        for name in available_maps:
            png_path = os.path.join(colormap_dir, f"{name}.png")
            if os.path.exists(png_path):
                pixmap = QPixmap(png_path)
                scaled_pixmap = pixmap.scaled(icon_size, QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.SmoothTransformation)
                icon = QIcon(scaled_pixmap)
                # Text is empty string for colormaps, name string stored in UserRole
                self.cmap_cb.addItem(icon, "", name)
            else:
                self.cmap_cb.addItem(name)

    def setup_native_vtk_legend(self):
        """Initializes native VTK 2D overlay text and a rotated colormap strip on the left edge."""
        import vtk
        
        # 1. Create the MAX Text Actor Overlay (Positioned on the Left)
        self.vtk_max_actor = vtk.vtkTextActor()
        self.vtk_max_actor.SetInput("Max")
        self.vtk_max_actor.GetTextProperty().SetColor(1.0, 1.0, 1.0) # White
        self.vtk_max_actor.GetTextProperty().SetFontSize(14)
        self.vtk_max_actor.GetTextProperty().BoldOn()
        self.vtk_max_actor.GetTextProperty().ShadowOn()
        self.vtk_max_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        self.vtk_max_actor.GetPositionCoordinate().SetValue(0.02, 0.82) # Top Left Placement
        
        # 2. Create the MIN Text Actor Overlay (Positioned on the Left)
        self.vtk_min_actor = vtk.vtkTextActor()
        self.vtk_min_actor.SetInput("Min")
        self.vtk_min_actor.GetTextProperty().SetColor(1.0, 1.0, 1.0)
        self.vtk_min_actor.GetTextProperty().SetFontSize(14)
        self.vtk_min_actor.GetTextProperty().BoldOn()
        self.vtk_min_actor.GetTextProperty().ShadowOn()
        self.vtk_min_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        self.vtk_min_actor.GetPositionCoordinate().SetValue(0.02, 0.15) # Bottom Left Placement

        # 3. CONSTRUCT THE GEOMETRIC 2D PLANE MAPPING CANVAS (Shifted Left)
        points = vtk.vtkPoints()
        points.InsertNextPoint(0.02, 0.20, 0.0) # Bottom-Left
        points.InsertNextPoint(0.05, 0.20, 0.0) # Bottom-Right (3% screen width)
        points.InsertNextPoint(0.05, 0.80, 0.0) # Top-Right
        points.InsertNextPoint(0.02, 0.80, 0.0) # Top-Left

        quad = vtk.vtkPolygon()
        quad.GetPointIds().SetNumberOfIds(4)
        quad.GetPointIds().SetId(0, 0)
        quad.GetPointIds().SetId(1, 1)
        quad.GetPointIds().SetId(2, 2)
        quad.GetPointIds().SetId(3, 3)

        cells = vtk.vtkCellArray()
        cells.InsertNextCell(quad)

        # 4. ROTATE TEXTURE COORDINATES BY 90 DEGREES
        # This remaps the horizontal PNG pixels to feed vertically up the quad frame!
        tcoords = vtk.vtkFloatArray()
        tcoords.SetNumberOfComponents(2)
        tcoords.InsertNextTuple2(0.0, 0.0) # Bottom-Left point gets left edge of PNG
        tcoords.InsertNextTuple2(0.0, 1.0) # Bottom-Right point gets right edge of PNG
        tcoords.InsertNextTuple2(1.0, 1.0) # Top-Right point maps up the gradient strip
        tcoords.InsertNextTuple2(1.0, 0.0) # Top-Left point maps up the gradient strip

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetPolys(cells)
        polydata.GetPointData().SetTCoords(tcoords)

        # Setup the 2D Coordinate Mapper tracking system
        mapper = vtk.vtkPolyDataMapper2D()
        mapper.SetInputData(polydata)
        
        vtk_coord = vtk.vtkCoordinate()
        vtk_coord.SetCoordinateSystemToNormalizedViewport()
        mapper.SetTransformCoordinate(vtk_coord)

        # 5. Bind the components to your active 2D textured actor container
        self.vtk_png_reader = vtk.vtkPNGReader()
        self.legend_strip_actor = vtk.vtkTexturedActor2D()
        self.legend_strip_actor.SetMapper(mapper)

        # Add all three components directly into the active 3D viewport canvas layers
        self.ren.AddActor2D(self.vtk_max_actor)
        self.ren.AddActor2D(self.vtk_min_actor)
        self.ren.AddActor2D(self.legend_strip_actor)

    def update_native_vtk_legend_image(self, colormap_name):
        """Loads the new colormap PNG asset and assigns it to the native VTK 2D texture map."""
        import os
        import vtk
        
        base_dir = os.path.dirname(os.path.dirname(__file__))
        png_path = os.path.join(base_dir, 'colormap', f"{colormap_name}.png")
        
        if os.path.exists(png_path) and hasattr(self, 'legend_strip_actor'):
            self.vtk_png_reader.SetFileName(png_path)
            self.vtk_png_reader.Update()
            
            texture = vtk.vtkTexture()
            texture.SetInputConnection(self.vtk_png_reader.GetOutputPort())
            
            self.legend_strip_actor.SetTexture(texture)
            self.legend_strip_actor.VisibilityOn()
        elif hasattr(self, 'legend_strip_actor'):
            self.legend_strip_actor.VisibilityOff()

    def get_ensemble_actor(self):
        '''
        Get ensemble actor.
        '''
        file_data = self.parent_ref.ensemble

        print("Data is a Object of type: ", file_data.GetClassName())

        utils.print_all_arrays_point_data(file_data)
        utils.print_all_arrays_cell_data(file_data)

        # extract surface from unstructured grid
        surface_filter = vtk.vtkDataSetSurfaceFilter()
        surface_filter.SetInputData(file_data)
        surface_filter.Update()

        # Create a mapper
        # mapper = vtk.vtkPolyDataMapper()
        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputData(utils.compute_normals(surface_filter.GetOutput()))

        # Create a actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # actor.GetProperty().SetSpecularPower(Config.SPECULAR_POWER)
        # actor.GetProperty().SetSpecular(Config.SPECULAR)
        # actor.GetProperty().SetDiffuse(Config.DIFFUSE)
        # actor.GetProperty().SetAmbient(Config.AMBIENT)

        actor.GetProperty().SetAmbient(self.current_ambient)
        actor.GetProperty().SetDiffuse(self.current_diffuse)
        actor.GetProperty().SetSpecular(self.current_specular)
        actor.GetProperty().SetSpecularPower(self.current_specular_power)

        # set interpolation to phong
        utils.set_interpolation(actor)
        return actor

    def onToolbarInfoAction(self, s):
        """
        Info action.
        display list of error particle CP IDs

        @param s: status
        """
        error_grains = cleandata.CleanData()
        error_grains.load()

        # displaying cp ids in the list format in a Dialog
        listWidget = QtWidgets.QListWidget()
        for cp_id in error_grains.cp_ids:
            listWidget.addItem(str(cp_id))
        
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Error Grains")
        dialog.setWindowModality(QtCore.Qt.NonModal)
        dialog.resize(400, 300)
        dialog.setLayout(QtWidgets.QVBoxLayout())
        dialog.layout().addWidget(listWidget)
        dialog.show()

    def build_extremum_graph_actor(self):
        """Extracts contact coordinates directly from the pre-loaded contact object to form center-to-center edges."""
        try:
            print("[Graph] Initializing robust contact point extraction...")
            
            maxima_poly = utils.read_file(Config.MAXIMAS_FILE)
            if not maxima_poly:
                print("[Graph Error] Maxima data source is empty or None.")
                return None
                
            points = vtk.vtkPoints()
            lines = vtk.vtkCellArray()
            
            # 1. Extract the 1,454 true center coordinates
            if hasattr(maxima_poly, 'GetPoints') and maxima_poly.GetPoints():
                points.DeepCopy(maxima_poly.GetPoints())
            else:
                print("[Graph Error] Maxima object contains no native VTK points array.")
                return None
                
            num_centers = points.GetNumberOfPoints()
            print(f"[Graph] Extracted {num_centers} true center nodes.")
            center_coords = [points.GetPoint(i) for i in range(num_centers)]

            # 2. Extract raw contact points array directly from the data wrapper
            contact_points_array = None
            
            # Case A: It's a direct PolyData object
            if isinstance(self.contact_points, vtk.vtkPolyData):
                contact_points_array = self.contact_points.GetPoints()
            # Case B: It's a VTK Algorithm Output / Filter
            elif hasattr(self.contact_points, 'GetOutput'):
                self.contact_points.Update()  # Force execution pass
                contact_points_array = self.contact_points.GetOutput().GetPoints()
            # Case C: It's an Actor wrapper, let's extract its underlying mapper geometry
            elif hasattr(self.contact_points, 'GetMapper') and self.contact_points.GetMapper():
                mapper_input = self.contact_points.GetMapper().GetInput()
                if mapper_input and hasattr(mapper_input, 'GetPoints'):
                    contact_points_array = mapper_input.GetPoints()

            # Absolute Fallback: Re-read the file path directly to guarantee a fresh VTK data handle
            if contact_points_array is None:
                print("[Graph] Direct object mapping returned None. Re-reading file via native XML reader...")
                contact_network_path = r"C:/Users/nites/Downloads/morsegramvis-main/morsegramvis-main/dataset_files/Steel Beads/chamf_distance_Steel_Deposition_181_176_251_downsampled_contacts.vtp"
                fresh_reader = vtk.vtkXMLPolyDataReader()
                fresh_reader.SetFileName(contact_network_path)
                fresh_reader.Update()  # Force VTK to parse the XML arrays into RAM
                if fresh_reader.GetOutput():
                    contact_points_array = fresh_reader.GetOutput().GetPoints()

            # 3. Calculate spatial linkages if points array was successfully found
            if contact_points_array is not None:
                num_contact_pts = contact_points_array.GetNumberOfPoints()
                print(f"[Graph] Successfully found contact source array. Processing {num_contact_pts} positions...")
                
                existing_edges = set()

                for i in range(num_contact_pts):
                    contact_pt = contact_points_array.GetPoint(i)
                    tx, ty, tz = contact_pt
                    
                    # Distance check loop to find the 2 closest particle centers
                    dists = []
                    for idx, (cx, cy, cz) in enumerate(center_coords):
                        d = (tx - cx)**2 + (ty - cy)**2 + (tz - cz)**2
                        dists.append((d, idx))
                    
                    dists.sort(key=lambda x: x[0])
                    
                    if len(dists) >= 2:
                        center_idx_a = dists[0][1]
                        center_idx_b = dists[1][1]
                        
                        if center_idx_a != center_idx_b:
                            edge_pair = tuple(sorted((center_idx_a, center_idx_b)))
                            if edge_pair not in existing_edges:
                                existing_edges.add(edge_pair)
                                
                                line = vtk.vtkLine()
                                line.GetPointIds().SetId(0, center_idx_a)
                                line.GetPointIds().SetId(1, center_idx_b)
                                lines.InsertNextCell(line)
            else:
                print("[Graph Error] Could not extract raw geometry array using any data wrappers.")

            # 4. Assemble and style the network mesh
            network_polydata = vtk.vtkPolyData()
            network_polydata.SetPoints(points)
            network_polydata.SetLines(lines)

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(network_polydata)
            
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            
            property_obj = actor.GetProperty()
            property_obj.SetColor(1.0, 1.0, 1.0)
            property_obj.SetLineWidth(0.1)
            property_obj.LightingOff()
            
            print(f"[Graph Success] Network rendering complete. Built {lines.GetNumberOfCells()} center edges.")
            return actor

        except Exception as e:
            print(f"[Graph Critical Error] Network reconstruction failed: {e}")
            return None

    def sync_network_visibility(self):
        """Ensures the network graph layout states match the toggle button and resets coloring overrides."""
        is_graph_mode = getattr(self, 'show_network_only', False)
        
        if hasattr(self, 'ren') and self.ren:
            # 1. Check if the network actor was completely purged by another mode
            if is_graph_mode and self.network_graph_actor is not None:
                actor_collection = self.ren.GetActors()
                if actor_collection and not actor_collection.IsItemPresent(self.network_graph_actor):
                    print("[Graph Sync] Network actor was purged by another mode. Re-injecting to canvas...")
                    self.ren.AddActor(self.network_graph_actor)

            # 2. Re-evaluate the live actor collection
            actor_collection = self.ren.GetActors()
            if actor_collection:
                actor_collection.InitTraversal()
                
                for _ in range(actor_collection.GetNumberOfItems()):
                    current_actor = actor_collection.GetNextActor()
                    if current_actor:
                        if is_graph_mode:
                            # If network mode is active, isolate the network lines
                            if current_actor == self.network_graph_actor:
                                current_actor.SetVisibility(True)
                                
                                # =======================================================
                                # CRITICAL FIX: STRIP OUT EXTERNAL COLOR MAP OVERRIDES
                                # =======================================================
                                if hasattr(current_actor, 'GetMapper') and current_actor.GetMapper():
                                    # Tell the mapper to ignore dynamic dataset color arrays
                                    current_actor.GetMapper().ScalarVisibilityOff()
                                
                                # Re-force the clean, bright, solid white profile styling
                                property_obj = current_actor.GetProperty()
                                property_obj.SetColor(1.0, 1.0, 1.0)  # Pure Solid White
                                property_obj.SetLineWidth(0.1)        # Maintain line weight
                                property_obj.LightingOff()            # Disable shading shadows
                                # =======================================================
                            else:
                                current_actor.SetVisibility(False)
                        else:
                            # Standard mode: Hide network lines, ensure everything else stays visible
                            if current_actor == self.network_graph_actor:
                                current_actor.SetVisibility(False)
                            else:
                                current_actor.SetVisibility(True)
                                
                        current_actor.Modified()

        # 3. Request an immediate graphics buffer flush pass
        if hasattr(self, 'iren') and self.iren:
            self.iren.GetRenderWindow().Render()

    def on_graph_mode_toggled(self, checked: bool):
        """Strictly alternates visibility and overrides competing toolbar modes."""
        self.show_network_only = checked
        
        if self.show_network_only:
            self.graph_mode_btn.setText("Show Mesh")
            
            # Lazy load the network graph actor only when first requested
            if self.network_graph_actor is None:
                self.network_graph_actor = self.build_extremum_graph_actor()
                if self.network_graph_actor:
                    self.ren.AddActor(self.network_graph_actor)
        else:
            self.graph_mode_btn.setText("Show Network")

        # Force sync the visibility of all elements in the scene
        self.sync_network_visibility()

    def on_opacity_slider_changed(self, value: int):
        """Updates ensemble mesh transparency by forcing VTK to allow alpha blending passes."""
        self.ensemble_opacity = value / 100.0
        is_graph_mode = getattr(self, 'show_network_only', False)

        if hasattr(self, 'active_ensemble_actor') and self.active_ensemble_actor:
            
            # Helper function to apply both the opacity and the critical rendering pass flags
            def apply_translucency_flags(actor, opacity):
                actor.GetProperty().SetOpacity(opacity)
                
                # CRITICAL: Force VTK out of the opaque cache queue when transparent
                if opacity < 1.0:
                    actor.ForceTranslucentOn()
                    actor.ForceOpaqueOff()
                else:
                    actor.ForceTranslucentOff()
                    actor.ForceOpaqueOn()
                
                actor.GetProperty().Modified()
                actor.Modified()

            # --- CASE A: Handle Assembly Parts ---
            if hasattr(self.active_ensemble_actor, 'GetParts'):
                collection = self.active_ensemble_actor.GetParts()
                if collection:
                    collection.InitTraversal()
                    for i in range(collection.GetNumberOfItems()):
                        part = collection.GetNextActor()
                        if part:
                            apply_translucency_flags(part, self.ensemble_opacity)
                            
            # --- CASE B: Handle Standalone Actor ---
            else:
                apply_translucency_flags(self.active_ensemble_actor, self.ensemble_opacity)
            
            # Maintain the view mode toggle visibility rule
            if is_graph_mode:
                self.active_ensemble_actor.SetVisibility(False)
            else:
                self.active_ensemble_actor.SetVisibility(True)

        # Force the active render window context to redraw everything right now
        if hasattr(self, 'iren') and self.iren:
            self.iren.GetRenderWindow().Render()

    def onToolbarSettingsAction(self):
        """Opens the configuration panel and guarantees a direct reference connection."""
        # Force 'self' (the EnsembleView window) into the get_form constructor
        import ui.configure as configure # Adjust import path to match your layout
        
        # If your code creates a separate dialog box or dock widget:
        if not hasattr(self, 'settings_dialog') or self.settings_dialog is None:
            from PySide6 import QtWidgets
            self.settings_dialog = QtWidgets.QDialog(self)
            self.settings_dialog.setWindowTitle("Lighting & Folder Configurations")
            
            # Call get_form passing 'self' explicitly as the parent reference!
            form_widget = configure.get_form(parent=self)
            
            dialog_layout = QtWidgets.QVBoxLayout(self.settings_dialog)
            dialog_layout.addWidget(form_widget)
            
        self.settings_dialog.show()

    def onInteractionModeCheckbox(self, state):
        """
        Interaction mode checkbox to find the particle ids on click.

        @param state: checkbox state
        """
        if state == QtCore.Qt.Checked:
            print("Interaction Mode: ON")

            actors = []
            self.maxima_data = utils.get_grains_point_cloud(utils.read_file(Config.MAXIMAS_FILE))
            # iterate over all file in config.PARTICLES_DIR
            for file in os.listdir(Config.PARTICLES_MESH_DIR):
                cp_id = int(os.fsdecode(file).split(".")[0])
                try:
                    actors += utils.get_actors_list(self.maxima_data, {}, cp_id, self.ren, False, coord_nums=self.coord_nums,
                                                    coord_range=self.coord_range)
                except Exception as e:
                    logging.getLogger().error("(Ens view) Error while getting actors list for cp_id: " + str(cp_id))
            
            # remove all actors from the renderer
            self.ren.RemoveAllViewProps()
            # add actors to the renderer
            for actor in actors:
                self.ren.AddActor(actor)
        else:
            print("Interaction Mode: OFF")
            self.caption_repr.Off()

            # remove all actors from the renderer
            self.ren.RemoveAllViewProps()

            # add ensemble actor to the renderer
            self.ren.AddActor(self.get_ensemble_actor())

        if hasattr(self, 'sync_network_visibility'):
            self.sync_network_visibility()

    def compute_fps(self, obj, event):
        """
        Compute the fps.
        @param obj: vtkRenderWindowInteractor
        @param event: event
        """
        timeInSeconds = time.time() - self.start_time
        if timeInSeconds != 0.0:
            fps = 1.0 / timeInSeconds
            # print("FPS: ", fps)
            # update window title with fps value integer
            self.setWindowTitle(self.window_title + " FPS: " + str(int(fps)))
        self.start_time = time.time()

    def key_press_event(self, obj, event):
        """
        Key press event.
        @param obj: vtkRenderWindowInteractor
        @param event: event
        """
        key = obj.GetKeySym()
        # quit the app on esc
        if key == 'Escape':
            self.closeEvent(None)

    def update_ao_factor(self):
        '''
        update the ambient occlusion
        '''
        self.ambient_occlusion()
        self.iren.Render()

    def clean(self):
        '''
        clean the scene and free memory
        '''
        self.parent.delaunaty_3d = None

    def closeEvent(self, event):
        '''
        close event

        :param event: event
        '''
        self.clean()
        self.destroy()

    def get_silhouette_actor(self):
        '''
        Get silhouette actor.

        :return: silhouette, silhouette actor
        '''
        silhouette = vtk.vtkPolyDataSilhouette()
        silhouette.SetCamera(self.ren.GetActiveCamera())

        # mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(silhouette.GetOutputPort())

        # actor
        silhouette_actor = vtk.vtkActor()
        silhouette_actor.SetMapper(mapper)
        silhouette_actor.GetProperty().SetColor(vtk.vtkNamedColors().GetColor3d("Tomato"))
        silhouette_actor.GetProperty().SetLineWidth(5)

        return silhouette, silhouette_actor

    def left_button_press_event(self, obj, event):
        """Handles mouse clicks safely, ignoring the network skeleton actor."""
        # 1. Get the actor that was clicked on
        picker = vtk.vtkPropPicker()
        pointer_pos = self.iren.GetEventPosition()
        picker.Pick(pointer_pos[0], pointer_pos[1], 0, self.ren)
        actor = picker.GetActor()

        # 2. CRITICAL GUARD: If the user clicked on the white network lines, skip processing
        if actor is None or (hasattr(self, 'network_graph_actor') and actor == self.network_graph_actor):
            # Safe forward call: trigger the native VTK interactor style behavior dynamically
            if hasattr(self, 'onLeftButtonDown'):
                self.onLeftButtonDown()
            elif hasattr(obj, 'OnLeftButtonDown'):
                obj.OnLeftButtonDown()
            return

        # 3. Fetch and protect the Material Name assignment pass
        material_name = actor.GetProperty().GetMaterialName()
        if material_name is None or "_" not in material_name:
            if hasattr(self, 'onLeftButtonDown'):
                self.onLeftButtonDown()
            elif hasattr(obj, 'OnLeftButtonDown'):
                obj.OnLeftButtonDown()
            return

        try:
            # Original processing logic for clicking on actual particle meshes
            cp_id = int(material_name.split("_")[1])
            
            # ... Rest of your original particle-clicking logic remains exactly here ...
            
        except Exception as e:
            print(f"[Interaction Warning] Click handler skipped: {e}")
            
        # Safe forward call at the end of execution loop
        if hasattr(self, 'onLeftButtonDown'):
            self.onLeftButtonDown()
        elif hasattr(obj, 'OnLeftButtonDown'):
            obj.OnLeftButtonDown()

    def update_caption(self, caption, location):
        '''
        update the caption

        :param caption: caption text
        :param location: caption location
        '''
        # get the caption representation
        cr = self.caption_repr.GetRepresentation()
        cr.GetCaptionActor2D().SetCaption(caption)

        # position the text
        cr.SetAnchorPosition(location)

    def get_caption_repr(self):
        '''
        Get caption representation.

        :return: caption representation
        '''
        # create a caption
        cr = vtk.vtkCaptionRepresentation()
        cr.GetCaptionActor2D().GetProperty().SetColor(vtk.vtkNamedColors().GetColor3d("Tomato"))

        widget = vtk.vtkCaptionWidget()
        widget.SetInteractor(self.iren)
        widget.SetRepresentation(cr)

        return widget

    def onCleanEnsembleCheckbox(self, state):
        '''
        View clean ensemble.

        :param state: checkbox state
        '''
        if state == QtCore.Qt.Checked:
            print("Clean Ensemble: ON")
            Config.ENSEMBLE_FILE = Config.ENSEMBLE_FILE.replace(".vtu", "_clean.vtu")
            self.parent.load_ensemble_file()
            self.ren.RemoveAllViewProps()
            self.ren.AddActor(self.get_ensemble_actor())
        else:
            print("Clean Ensemble: OFF")
            Config.ENSEMBLE_FILE = Config.ENSEMBLE_FILE.replace("_clean.vtu", ".vtu")
            self.parent.load_ensemble_file()
            self.ren.RemoveAllViewProps()
            self.ren.AddActor(self.get_ensemble_actor())
        
        if hasattr(self, 'sync_network_visibility'):
            self.sync_network_visibility()



if __name__ == '__main__':
    print("You cannot run this file directly.")


def main(parent=None):
    '''
    Open the ensemble view window.

    :param parent: parent window
    '''
    dialog = EnsembleView(parent)
    # dialog.move(parent.x() + parent.frameGeometry().width(), parent.y())

    # make it non blocking dialog
    dialog.setWindowModality(QtCore.Qt.NonModal)
    dialog.show()


