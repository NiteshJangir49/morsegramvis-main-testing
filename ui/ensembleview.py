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

        # --------------- End Toolbar ----------------
        # reset camera
        self.ren.ResetCamera()
        # Start the event loop.
        self.show()
        self.iren.Initialize()

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

        actor.GetProperty().SetSpecularPower(Config.SPECULAR_POWER)
        actor.GetProperty().SetSpecular(Config.SPECULAR)
        actor.GetProperty().SetDiffuse(Config.DIFFUSE)
        actor.GetProperty().SetAmbient(Config.AMBIENT)

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
            property_obj.SetLineWidth(0.3)
            property_obj.LightingOff()
            
            print(f"[Graph Success] Network rendering complete. Built {lines.GetNumberOfCells()} center edges.")
            return actor

        except Exception as e:
            print(f"[Graph Critical Error] Network reconstruction failed: {e}")
            return None

    def on_graph_mode_toggled(self, checked: bool):
        """Strictly alternates visibility and prevents black-screen locking on error."""
        self.show_network_only = checked
        
        if self.show_network_only:
            self.graph_mode_btn.setText("Show Mesh")
            
            # Lazy load the network graph actor only when first requested
            if self.network_graph_actor is None:
                self.network_graph_actor = self.build_extremum_graph_actor()
                
                # RECOVERY CHECK: If building failed, abort immediately and keep mesh on screen
                if self.network_graph_actor is None:
                    print("[UI Error] Aborting network view pass: Could not process network coordinates.")
                    self.show_network_only = False
                    self.graph_mode_btn.setChecked(False)
                    self.graph_mode_btn.setText("Show Network")
                else:
                    self.ren.AddActor(self.network_graph_actor)
        else:
            self.graph_mode_btn.setText("Show Network")

        # ===================================================================
        # VISIBILITY MANAGER WITH FALLBACK DEFENSE
        # ===================================================================
        if hasattr(self, 'ren') and self.ren:
            actor_collection = self.ren.GetActors()
            if actor_collection:
                actor_collection.InitTraversal()
                
                for _ in range(actor_collection.GetNumberOfItems()):
                    current_actor = actor_collection.GetNextActor()
                    if current_actor:
                        if self.show_network_only:
                            if current_actor == self.network_graph_actor:
                                current_actor.SetVisibility(True)
                            else:
                                current_actor.SetVisibility(False)
                        else:
                            # Safely restore everything back to visible when toggle is off
                            if current_actor == self.network_graph_actor:
                                current_actor.SetVisibility(False)
                            else:
                                current_actor.SetVisibility(True)
                                
                        current_actor.Modified()

        # Render the fresh view state updates
        if hasattr(self, 'iren') and self.iren:
            self.iren.GetRenderWindow().Render()

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

    def onToolbarSettingsAction(self, s):
        """
        Open the settings dialog.

        @param s: status
        """
        settings_dialog.main(self)

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
        '''
        Find the particle id on click.

        :param obj: vtkRenderWindowInteractor
        :param event: event
        '''
        if self.interaction_mode_checkbox.isChecked():
            click_pos = self.iren.GetEventPosition()

            # pick the actor
            picker = vtk.vtkPropPicker()
            picker.Pick(click_pos[0], click_pos[1], 0, self.ren)

            # get the actor
            actor = picker.GetActor()

            if actor:
                
                if self.last_picked_actor:
                    self.last_picked_actor.GetProperty().DeepCopy(self.last_picked_property)

                self.last_picked_property.DeepCopy(actor.GetProperty())
                actor.GetProperty().SetColor(vtk.vtkNamedColors().GetColor3d("Tomato"))
                actor.GetProperty().SetDiffuse(1.0)
                actor.GetProperty().SetSpecular(0.0)
                actor.GetProperty().EdgeVisibilityOn()

                self.last_picked_actor = actor

                cp_id = int(actor.GetProperty().GetMaterialName().split("_")[1])

                # update the caption
                self.update_caption(actor.GetProperty().GetMaterialName(), self.maxima_data[cp_id][0])

                self.caption_repr.On()

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


