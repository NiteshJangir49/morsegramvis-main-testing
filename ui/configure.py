from PySide6 import QtCore, QtWidgets
from settings import Config
import core.xmlutils as xmlutils
import os

MATERIAL_PRESETS = {
    "Custom": None,
    "Metal": {
        "ambient": 35,
        "specular": 90,
        "diffuse": 60,
        # "ambient_power": 40,
        "specular power": 80
    },
    "Sand": {
        "ambient": 35,
        "specular": 10,
        "diffuse": 85,
        # "ambient_power": 30,
        "specular power": 5
    },
    "Wood": {
        "ambient": 35,
        "specular": 20,
        "diffuse": 70,
        # "ambient_power": 50,
        "specular power": 15
    }
}

def get_HLine():
    '''
    get a horizontal line

    @return: the horizontal line
    '''
    frame = QtWidgets.QFrame()
    frame.setFrameShape(QtWidgets.QFrame.HLine)
    frame.setFrameShadow(QtWidgets.QFrame.Sunken)
    return frame


def selectFolder():
    '''
    Open a file dialog to select a folder
    @return: the selected folder
    '''
    # Open a file dialog to select a folder
    folder = QtWidgets.QFileDialog.getExistingDirectory()
    return folder


def get_form(parent=None):
    '''
    get the form

    @param parent: the parent widget
    @return: the form
    '''
    # ===================================================================
    # 1. CENTRAL ENGINE CALLBACKS & STATE SYNC HELPERS
    # ===================================================================
    parent._updating_from_preset = False

    def get_real_3d_window():
        """Finds the active 'Start' window wrapper and extracts its inner viewport context."""
        app = QtWidgets.QApplication.instance()
        if app:
            for widget in app.topLevelWidgets():
                # 1. Target our main application window class container
                if widget.__class__.__name__ == 'Start':
                    # If the window itself holds a direct link to the view object
                    if hasattr(widget, 'view'):
                        return getattr(widget, 'view')
                    if hasattr(widget, 'ensemble_view'):
                        return getattr(widget, 'ensemble_view')
                    
                    # 2. Deep structural tree lookup: find an embedded EnsembleView class inside the window frame
                    if hasattr(widget, 'findChild'):
                        # Dynamically seek any child component matching our target rendering view signature
                        for child in widget.findChildren(QtWidgets.QWidget):
                            if child.__class__.__name__ == 'EnsembleView':
                                return child
                    
                    # 3. Flat attribute search: look inside the object's variables for an active view context
                    for attr_name in dir(widget):
                        try:
                            attr_val = getattr(widget, attr_name)
                            if attr_val.__class__.__name__ == 'EnsembleView':
                                return attr_val
                        except Exception:
                            continue
                            
                    # Fallback directly to the main 'Start' object container itself
                    return widget
        return None

    def sync_background_config_state(prop_name, integer_value):
        """Updates persistent state tracking variables on the rendering engine window target."""
        float_val = integer_value / 100.0 if prop_name != 'specular power' else float(integer_value)
        
        view_window = get_real_3d_window()
        if view_window:
            # Overwrite both standard properties and underscore variations seamlessly
            if prop_name == 'ambient': view_window.current_ambient = float_val
            elif prop_name == 'diffuse': view_window.current_diffuse = float_val
            elif prop_name == 'specular': view_window.current_specular = float_val
            elif prop_name == 'specular power': view_window.current_specular_power = float_val

    def update_live_vtk_property(prop_name, integer_value):
        """Finds active VTK canvas pipelines inside the view window context and pushes live adjustments."""
        if prop_name != 'refresh':
            sync_background_config_state(prop_name, integer_value)
        
        view_window = get_real_3d_window()
        if view_window is None:
            return

        renderer = None
        interactor = None
        
        # Scan both the extracted view widget and the parent application window for active VTK contexts
        search_targets = [view_window]
        if hasattr(view_window, 'parentWidget') and view_window.parentWidget():
            search_targets.append(view_window.parentWidget())
            
        for target in search_targets:
            for attr in ['ren', 'renderer']:
                if hasattr(target, attr) and getattr(target, attr) is not None:
                    renderer = getattr(target, attr)
                    break
            for attr in ['iren', 'interactor']:
                if hasattr(target, attr) and getattr(target, attr) is not None:
                    obj = getattr(target, attr)
                    interactor = obj.GetRenderWindow() if hasattr(obj, 'GetRenderWindow') else obj
                    break

        # Fallback deep lookup search for VTK contexts hidden inside generic embedded widgets
        if renderer is None and hasattr(view_window, 'findChildren'):
            for child in view_window.findChildren(QtWidgets.QWidget):
                if hasattr(child, 'ren') and getattr(child, 'ren') is not None:
                    renderer = child.ren
                    if hasattr(child, 'iren'): interactor = child.iren.GetRenderWindow() if hasattr(child.iren, 'GetRenderWindow') else child.iren
                    break

        # Apply parameters onto live graphics actors pipelines
        if renderer is not None:
            actor_collection = renderer.GetActors()
            if actor_collection:
                actor_collection.InitTraversal()
                for _ in range(actor_collection.GetNumberOfItems()):
                    actor = actor_collection.GetNextActor()
                    if actor and hasattr(actor, 'GetProperty'):
                        prop = actor.GetProperty()
                        prop.SetAmbient(getattr(view_window, 'current_ambient', 0.5))
                        prop.SetDiffuse(getattr(view_window, 'current_diffuse', 0.5))
                        prop.SetSpecular(getattr(view_window, 'current_specular', 1.0))
                        prop.SetSpecularPower(getattr(view_window, 'current_specular_power', 100.0))
            
        if interactor is not None:
            interactor.Render()

    def check_for_custom_override():
        """Bounces the dropdown menu state to 'Custom' if a slider handle is moved manually."""
        if not getattr(parent, '_updating_from_preset', False):
            if hasattr(parent, 'material_dropdown'):
                parent.material_dropdown.blockSignals(True)
                parent.material_dropdown.setCurrentText("Custom")
                parent.material_dropdown.blockSignals(False)

    def on_material_changed(material_name):
        """Processes preset choices, moves slider positions, and forces a scene redraw."""
        preset = MATERIAL_PRESETS.get(material_name)
        if preset is None: 
            return  # Custom chosen: leave parameters as-is
            
        parent._updating_from_preset = True
        try:
            # Shift the UI handles programmatically
            for key, value in preset.items():
                if key in parent.lighting and parent.lighting[key] is not None:
                    parent.lighting[key].setValue(value)
                    sync_background_config_state(key, value)
            
            # Fire a direct scene update sweep
            update_live_vtk_property('refresh', 0)
        finally:
            parent._updating_from_preset = False

    # ===================================================================
    # 2. STANDALONE UI LAYOUT & WIDGET ASSEMBLY
    # ===================================================================
    window = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout()
    label = QtWidgets.QLabel("Select a folder")
    button = QtWidgets.QPushButton("Select Folder")
    parent.lineEdit = QtWidgets.QLineEdit()

    button.clicked.connect(lambda: parent.lineEdit.setText(selectFolder()))
    parent.lineEdit.setAlignment(QtCore.Qt.AlignCenter)

    folder_layout = QtWidgets.QHBoxLayout()
    folder_layout.addWidget(label)
    folder_layout.addWidget(parent.lineEdit)
    folder_layout.addWidget(button)

    layout.addLayout(folder_layout)
    layout.addWidget(get_HLine())

    # --- RGB Sliders Block ---
    rgb_layout = QtWidgets.QVBoxLayout()
    parent.rgb = {'red' : None, 'green' : None, 'blue' : None}
    for c in parent.rgb.keys():
        h_layout = QtWidgets.QHBoxLayout()
        c_label = QtWidgets.QLabel(c)
        c_label.setFixedSize(50, 20)
        h_layout.addWidget(c_label)
        
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setRange(0, 255)
        
        slider_label = QtWidgets.QLabel("0")
        slider_label.setFixedSize(50, 20)
        slider.valueChanged.connect(lambda value, label=slider_label: label.setText(str(value)))
        
        h_layout.addWidget(slider)
        h_layout.addWidget(slider_label)
        rgb_layout.addLayout(h_layout)
        parent.rgb[c] = slider

    layout.addLayout(rgb_layout)
    layout.addWidget(get_HLine())

    # ===================================================================
    # 3. MATERIAL PRESETS DROPDOWN COMPONENT INJECTION
    # ===================================================================
    preset_h_layout = QtWidgets.QHBoxLayout()
    preset_label = QtWidgets.QLabel("Material Preset:")
    preset_label.setFixedSize(120, 20)
    preset_h_layout.addWidget(preset_label)
    
    parent.material_dropdown = QtWidgets.QComboBox()
    parent.material_dropdown.addItems(["Custom", "Metal", "Sand", "Wood"])
    parent.material_dropdown.currentTextChanged.connect(on_material_changed)
    preset_h_layout.addWidget(parent.material_dropdown)
    
    spacer_lbl = QtWidgets.QLabel("")
    spacer_lbl.setFixedSize(50, 20)
    preset_h_layout.addWidget(spacer_lbl)
    
    layout.addLayout(preset_h_layout)

    # --- Lighting Sliders Setup Loop ---
    lighting_dict = {   'ambient': [0, 100], 
                        'diffuse': [0, 100],
                        'specular': [0, 100],
                        'specular power': [0, 100]}
    
    lighting_layout = QtWidgets.QVBoxLayout()
    parent.lighting = {k : None for k in lighting_dict.keys()}

    for k, v in lighting_dict.items():
        h_layout = QtWidgets.QHBoxLayout()
        c_label = QtWidgets.QLabel(k)
        c_label.setFixedSize(120, 20)
        h_layout.addWidget(c_label)
        
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setRange(v[0], v[1])

        label_text = ""
        if k in ['ambient', 'diffuse']:
            slider.setValue(50)
            label_text = "50"
        elif k in ['specular', 'specular power']:
            slider.setValue(100)
            label_text = "100"
            
        slider_label = QtWidgets.QLabel(label_text)
        slider_label.setFixedSize(50, 20)
        
        # --- BIND INTERACTION SIGNALS SEQUENTIALLY ---
        # 1. Sync numerical text label
        slider.valueChanged.connect(lambda value, label=slider_label: label.setText(str(value)))
        # 2. Track custom override modifications
        slider.valueChanged.connect(check_for_custom_override)
        # 3. Stream adjustments directly back to shaders & graphics buffers
        slider.valueChanged.connect(lambda value, prop_name=k: update_live_vtk_property(prop_name, value))
        
        h_layout.addWidget(slider)
        h_layout.addWidget(slider_label)
        lighting_layout.addLayout(h_layout)
        parent.lighting[k] = slider

    layout.addLayout(lighting_layout)
    layout.addWidget(get_HLine())

    # --- Ambient Occlusion Block ---
    ao_dict = {'ambient occlusion factor' : 0, 'kernel size' : 128}
    ao_layout = QtWidgets.QVBoxLayout()
    parent.ao = {k : None for k in ao_dict.keys()}
    for k, v in ao_dict.items():
        h_layout = QtWidgets.QHBoxLayout()
        c_label = QtWidgets.QLabel(k)
        c_label.setFixedSize(200, 20)
        h_layout.addWidget(c_label)
        
        input_text = QtWidgets.QLineEdit()
        input_text.setFixedSize(50, 20)
        input_text.setText(str(v))
        
        h_layout.addWidget(input_text)
        ao_layout.addLayout(h_layout)
        parent.ao[k] = input_text

    layout.addLayout(ao_layout)
    layout.addWidget(get_HLine())

    window.setLayout(layout)
    return window


def get_proj_card(data, parent):
    '''
    Card that contains the project name and the load button

    @param data: the project data
    @type data: dict
    @param parent: the parent widget
    @type parent: QtWidgets.QWidget
    @return: the card
    '''
    # create card
    card = QtWidgets.QFrame()
    card.setFrameShape(QtWidgets.QFrame.StyledPanel)
    # create layout
    layout = QtWidgets.QHBoxLayout()
    # add stretch factor to make the layout resizable
    layout.addStretch()
    # create vertical layout
    v_layout = QtWidgets.QVBoxLayout()
    # create horizontal layout
    h_layout = QtWidgets.QHBoxLayout()
    # create label
    label = QtWidgets.QLabel(data['settings']['base_folder'])
    # set word wrap to True to allow automatic line wrapping
    label.setWordWrap(True)

    # create button
    button = QtWidgets.QPushButton("Load Project")
    button.clicked.connect(lambda: parent.load_settings(data))

    # add to layout
    h_layout.addWidget(label)

    # add to layout
    v_layout.addLayout(h_layout)
    v_layout.addWidget(button)

    # add to layout
    layout.addLayout(v_layout)

    # add stretch factor to make the layout resizable
    layout.addStretch()

    # add to card
    card.setLayout(layout)

    return card


# dialog class
class Configure(QtWidgets.QDialog):
    """
    Initialize the dataset for the first time and configure it
    for subsequent use.
    """

    def __init__(self, parent=None):
        super(Configure, self).__init__(parent)

        # window buttons
        self.setWindowFlags(QtCore.Qt.Window
                            | QtCore.Qt.WindowMinimizeButtonHint
                            | QtCore.Qt.WindowMaximizeButtonHint
                            | QtCore.Qt.WindowCloseButtonHint)

        # ref to parent
        self.parent = parent
        self.setWindowTitle("Configure Settings")

        # create layout
        self.layout = QtWidgets.QHBoxLayout()

        # add frame to layout
        self.get_existing_config_frame()

        config_form_frame = QtWidgets.QFrame()
        config_form_frame.setLayout(QtWidgets.QVBoxLayout())
        # border
        config_form_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)

        # add form to frame
        config_form_frame.layout().addWidget(get_form(self))

        # add form to layout
        self.layout.addWidget(config_form_frame)

        # stretch layout
        self.layout.setStretch(0, 2)
        self.layout.setStretch(1, 3)

        # horizontal layout
        h_layout = QtWidgets.QHBoxLayout()

        # update settings
        self.update_button = QtWidgets.QPushButton("Done")
        # add icon to button
        self.update_button.setIcon(
            QtWidgets.QApplication.style().standardIcon(
                QtWidgets.QStyle.SP_DialogApplyButton))
        self.update_button.clicked.connect(self.update_settings)
        h_layout.addWidget(self.update_button)

        # close button
        self.close_button = QtWidgets.QPushButton("Close")
        # add icon to button
        self.close_button.setIcon(
            QtWidgets.QApplication.style().standardIcon(
                QtWidgets.QStyle.SP_DialogCloseButton))
        self.close_button.clicked.connect(self.close_app)
        h_layout.addWidget(self.close_button)

        # add to form frame
        config_form_frame.layout().addLayout(h_layout)

        # set layout
        self.setLayout(self.layout)

    def update_settings(self):
        '''
        Update the settings for the dataset
        '''
        # ==== input folder check =====
        check = True
        # validate input
        if self.lineEdit.text() == "":
            check = False
        elif os.path.exists(self.lineEdit.text()) == False:
            check = False
        
        if check == False:
            QtWidgets.QMessageBox.warning(self, "Error", "Invalid input for folder")
            return
        # =============================

        # ==== check for ao ===========
        check = True
        for k, v in self.ao.items():
            try:
                int(v.text())
            except:
                check = False
                break
        
        if check == False:
            QtWidgets.QMessageBox.warning(self, "Error", "Invalid input for ambient occlusion")
            return
        # ==============================

        xmlutils.save_settings(self)
        self.close_app()

    def close_app(self):
        '''
        Close dialog window
        '''
        self.close()
        self.destroy()

    def load_settings(self, data):
        '''
        Load the settings from the given data

        @param data: the project data
        @type data: dict
        '''
        try:
            Config.set_base_folder(data)
            self.close_app()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", 
                "There was an error loading the settings: {}".format(e))
            xmlutils.remove_settings(data['settings']['base_folder'])

            # reload settings
            self.get_existing_config_frame()

    def get_existing_config_frame(self):
        '''
        this function creates a frame that contains a scroll area
        that contains a list of project cards

        @param parent: the parent widget
        @type parent: QtWidgets.QWidget
        '''

        # empty frame
        existing_config_frame = QtWidgets.QFrame()

        # add scroll area to frame
        existing_config_frame.setLayout(QtWidgets.QVBoxLayout())

        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)

        # add widget to scroll area
        list_widget = QtWidgets.QWidget()
        scroll_area.setWidget(list_widget)

        list_layout = QtWidgets.QVBoxLayout()
        list_widget.setLayout(list_layout)

        for proj_setting in xmlutils.get_settings():
            list_layout.addWidget(get_proj_card(proj_setting, self))
        
        list_layout.setAlignment(QtCore.Qt.AlignCenter)
        # make list start at top
        list_layout.addStretch()

        # add scroll area to frame
        existing_config_frame.layout().addWidget(scroll_area)

        # add to layout in the first position
        if self.layout.count() > 1:
            self.layout.removeWidget(self.layout.itemAt(0).widget())
        self.layout.insertWidget(0, existing_config_frame)

        # stretch layout
        self.layout.setStretch(0, 2)
        self.layout.setStretch(1, 3)
    def on_material_changed(material_name):
        """Updates dictionary slider items simultaneously and forces a live VTK scene update."""
        preset = MATERIAL_PRESETS.get(material_name)
        if preset is None:
            return  # Custom option clicked: leave values as-is
            
        parent._updating_from_preset = True
        try:
            # 1. Update visual slider handle positions in the UI panel
            for key, value in preset.items():
                if key in parent.lighting and parent.lighting[key] is not None:
                    parent.lighting[key].setValue(value)
            
            # 2. Extract values converted back to VTK's 0.0 to 1.0 expected range
            amb = preset["ambient"] / 100.0
            dif = preset["diffuse"] / 100.0
            spec = preset["specular"] / 100.0
            spec_p = float(preset["specular power"])

            # 3. Direct Pipeline Overhaul: push values directly into the active render pipeline
            if hasattr(parent, 'ren') and parent.ren:
                actor_collection = parent.ren.GetActors()
                if actor_collection:
                    actor_collection.InitTraversal()
                    for _ in range(actor_collection.GetNumberOfItems()):
                        actor = actor_collection.GetNextActor()
                        if actor and hasattr(actor, 'GetProperty'):
                            prop = actor.GetProperty()
                            # Apply the fresh material preset weights directly to the shaders
                            prop.SetAmbient(amb)
                            prop.SetDiffuse(dif)
                            prop.SetSpecular(spec)
                            prop.SetSpecularPower(spec_p)
                
                # 4. Force a hard graphics buffer frame refresh pass
                if hasattr(parent, 'iren') and parent.iren:
                    parent.iren.GetRenderWindow().Render()
                    
        except Exception as e:
            print(f"[Preset Error] Direct pipeline material assignment failed: {e}")
        finally:
            parent._updating_from_preset = False

    def check_for_custom_override(self):
        """Flips the dropdown selection to 'Custom' if the user manually drags a slider."""
        if not getattr(self, '_updating_from_preset', False):
            if hasattr(self, 'material_dropdown'):
                # Block signals temporarily so setting 'Custom' doesn't alter values
                self.material_dropdown.blockSignals(True)
                self.material_dropdown.setCurrentText("Custom")
                self.material_dropdown.blockSignals(False)

def main(parent=None):
    '''
    Open the configure settings dialog to load the new dataset
    and configure it for subsequent use.
    '''
    dialog = Configure(parent)
    # dialog.move(parent.x() + parent.frameGeometry().width(), parent.y())
    # make it non blocking dialog
    dialog.setWindowModality(QtCore.Qt.NonModal)
    dialog.show()