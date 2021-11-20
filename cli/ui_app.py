import numpy as np
from PIL import Image, ImageTk
from tkinter import filedialog, ttk, font
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import os
import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # shut tensorflow up

import app_support as app
from codebase import model, features

# %% Constants/Formatting
WINDOW_HEIGHT = 800
WINDOW_WIDTH = 1500

# Where to put each group in the UI
GROUP_GRID_PARAM = {
    'model': {'row': 0, 'column': 0, 'columnspan': 2},
    'timeline': {'row': 1, 'column': 0, 'columnspan': 2},
    'region_tab': {'row': 2, 'column': 0, 'columnspan': 8},
    'location_tab': {'row': 2, 'column': 0, 'columnspan': 8},
}

# %% UI
class Ui():
    """Main application UI management"""

    def __init__(self):
        """Construct application"""
        # Stored data
        self.Model = model.WrapperModel()  # to be loaded later
        self.sim_results_region = {}  # stores all region simulation results
        self.sim_results_location = {}  # stores all region simulation results

        self.window = None  # overall window widget
        self.groups = {}  # groups belonging to UI

        self.window = tk.Tk()
        self.window.title('Predictive Carbon Modeling')
        self.window.geometry(f'{WINDOW_WIDTH}x{WINDOW_HEIGHT}')

        # Add groups
        self.__add_model_group(widget=self.window, grid_param=GROUP_GRID_PARAM['model'])
        self.__add_timeline_group(widget=self.window, grid_param=GROUP_GRID_PARAM['timeline'])

        # Tabs for each mode
        tab_control = ttk.Notebook(self.window)
        region_tab = ttk.Frame(tab_control)
        location_tab = ttk.Frame(tab_control)
        tab_control.add(region_tab, text='Region')
        tab_control.add(location_tab, text='Location')
        tab_control.grid(row=5, column=0)

        # Add region-based to region tab
        self.__add_region_group(
            widget=region_tab,
            grid_param={'row': 0, 'column': 0, 'columnspan': 2}
        )
        self.__add_region_plot_group(
            widget=region_tab,
            grid_param={'row': 0, 'column': 2, 'columnspan': 2}
        )
        self.__add_region_render_group(
            widget=region_tab,
            grid_param={'row': 1, 'column': 0, 'columnspan': 8}
        )

        # Add location-based to location tab
        self.__add_location_group(
            widget=location_tab,
            grid_param={'row': 0, 'column': 0, 'columnspan': 2}
        )
        # Halt execution
        self.window.mainloop()

    # %% Groups
    def __add_model_group(self, widget, grid_param):
        self.groups['model'] = ttk.LabelFrame(widget, text='Model')
        self.groups['model'].grid(**grid_param)

        load_model_button = tk.Button(
            self.groups['model'],
            text='Load Model',
            command=self.__load_model_pressed
        )
        self.model_dir = tk.StringVar()
        self.model_dir.set('No Model Selected')
        self.load_model_label = tk.Label(
            self.groups['model'], textvariable=self.model_dir, width=50)
        load_model_button.grid(row=0, column=0)  # position within group
        self.load_model_label.grid(row=0, column=1, columnspan=3)

    def __add_timeline_group(self, widget, grid_param):
        """Group for simulation parameters"""
        self.groups['timeline'] = ttk.LabelFrame(widget, text='Timeline')
        self.groups['timeline'].grid(**grid_param)

        start_date_label = tk.Label(self.groups['timeline'], text='Start Date (DD/MM/YYYY)')
        stop_date_label = tk.Label(self.groups['timeline'], text='Stop Date (DD/MM/YYYY)')
        sim_step_label = tk.Label(self.groups['timeline'], text='Simulation Step')

        self.start_date_str = tk.StringVar()
        self.stop_date_str = tk.StringVar()
        self.sim_step = tk.StringVar()

        start_date_entry = tk.Entry(self.groups['timeline'], textvariable=self.start_date_str)
        stop_date_entry = tk.Entry(self.groups['timeline'], textvariable=self.stop_date_str)

        sim_options = ['daily', 'weekly', 'monthly', 'annually']
        self.sim_step.set(sim_options[0])  # default
        sim_step_dropdown = tk.OptionMenu(
            self.groups['timeline'],
            self.sim_step,
            *sim_options
        )
        sim_step_dropdown.config(width=10)

        simulate_button = tk.Button(
            self.groups['timeline'],
            text='Simulate',
            bg='green',
            command=self.__simulate_pressed
        )
        self.sim_status = tk.StringVar()
        sim_status_label = tk.Label(self.groups['timeline'], textvariable=self.sim_status)

        self.mode = tk.StringVar()
        self.mode.set('region')

        start_date_label.grid(row=0, column=0)
        start_date_entry.grid(row=0, column=1)
        stop_date_label.grid(row=1, column=0)
        stop_date_entry.grid(row=1, column=1)
        sim_step_label.grid(row=0, column=2)
        sim_step_dropdown.grid(row=0, column=3)
        simulate_button.grid(row=1, column=2)
        sim_status_label.grid(row=1, column=3)

    def __add_region_group(self, widget, grid_param):
        """Group for simulation parameters"""
        self.groups['region'] = ttk.LabelFrame(widget, text='Region')
        self.groups['region'].grid(**grid_param)

        min_lat_label = tk.Label(self.groups['region'], text='Latitude Min. (Deg)')
        max_lat_label = tk.Label(self.groups['region'], text='Latitude Max. (Deg)')
        step_lat_label = tk.Label(self.groups['region'], text='Latitude Step (Deg)')
        min_lon_label = tk.Label(self.groups['region'], text='Longitude Min. (Deg)')
        max_lon_label = tk.Label(self.groups['region'], text='Longitude Max. (Deg)')
        step_lon_label = tk.Label(self.groups['region'], text='Longitude Step (Deg)')

        self.min_lat_str = tk.StringVar()
        self.max_lat_str = tk.StringVar()
        self.step_lat_str = tk.StringVar()
        self.min_lon_str = tk.StringVar()
        self.max_lon_str = tk.StringVar()
        self.step_lon_str = tk.StringVar()

        min_lat_entry = tk.Entry(self.groups['region'], textvariable=self.min_lat_str)
        max_lat_entry = tk.Entry(self.groups['region'], textvariable=self.max_lat_str)
        step_lat_entry = tk.Entry(self.groups['region'], textvariable=self.step_lat_str)
        min_lon_entry = tk.Entry(self.groups['region'], textvariable=self.min_lon_str)
        max_lon_entry = tk.Entry(self.groups['region'], textvariable=self.max_lon_str)
        step_lon_entry = tk.Entry(self.groups['region'], textvariable=self.step_lon_str)

        min_lat_label.grid(row=0, column=3)
        max_lat_label.grid(row=1, column=3)
        step_lat_label.grid(row=2, column=3)
        min_lat_entry.grid(row=0, column=4)
        max_lat_entry.grid(row=1, column=4)
        step_lat_entry.grid(row=2, column=4)

        min_lon_label.grid(row=0, column=5)
        max_lon_label.grid(row=1, column=5)
        step_lon_label.grid(row=2, column=5)
        min_lon_entry.grid(row=0, column=6)
        max_lon_entry.grid(row=1, column=6)
        step_lon_entry.grid(row=2, column=6)

    def __add_location_group(self, widget, grid_param):
        """Group for location parameters"""
        self.groups['location'] = ttk.LabelFrame(widget, text='Location')
        self.groups['location'].grid(**grid_param)

        lats_label = tk.Label(self.groups['location'], text='Latitudes (comma-separated)')
        lons_label = tk.Label(self.groups['location'], text='Longitudes (comma-separated)')

        self.lats_str = tk.StringVar()
        self.lons_str = tk.StringVar()
        lats_entry = tk.Entry(self.groups['location'], textvariable=self.lats_str)
        lons_entry = tk.Entry(self.groups['location'], textvariable=self.lons_str)

        lats_label.grid(row=0, column=0)
        lats_entry.grid(row=0, column=1)
        lons_label.grid(row=1, column=0)
        lons_entry.grid(row=1, column=1)

    def __add_region_plot_group(self, widget, grid_param):
        """Group for all plot configuration"""
        self.groups['region_plot'] = ttk.LabelFrame(widget, text='Plot Configuration')
        self.groups['region_plot'].grid(**grid_param)

        # Interpolation
        self.interp_label = tk.Label(self.groups['region_plot'], text='Interpolation Type')
        interp_options = ['none', 'bilinear', 'bicubic', 'hanning', 'sinc']
        self.interp_type = tk.StringVar()
        self.interp_type.set(interp_options[0])  # default
        self.interp_dropdown = tk.OptionMenu(
            self.groups['region_plot'],
            self.interp_type,
            *interp_options,
            command=self.__interp_changed,  # replot when interp changes
        )

        # Colormap
        self.cmap_label = tk.Label(self.groups['region_plot'], text='Colormap')
        cmap_options = ['jet', 'magma', 'gray', 'hot', 'cool', 'seismic']
        self.cmap_type = tk.StringVar()
        self.cmap_type.set(cmap_options[0])  # default
        self.cmap_dropdown = tk.OptionMenu(
            self.groups['region_plot'],
            self.cmap_type,
            *cmap_options,
            command=self.__cmap_changed,  # re-plot when cmap changes
        )

        # Simulation timescale slider
        self.time_slider = tk.Scale(
            self.groups['region_plot'],
            from_=0.0,
            to=100.0,
            sliderlength=20,  # size of slider
            orient='horizontal',
            length=300
        )

        # Run command on slider release
        self.time_slider.bind('<ButtonRelease-1>', self.__time_slider_changed)

        self.interp_label.grid(row=0, column=0)
        self.interp_dropdown.grid(row=0, column=1)
        self.cmap_label.grid(row=0, column=2)
        self.cmap_dropdown.grid(row=0, column=3)
        self.time_slider.grid(row=1, column=0, columnspan=2)

    def __add_region_render_group(self, widget, grid_param):
        """Group where figure(s) will be rendered"""
        self.groups['region_render'] = ttk.LabelFrame(widget)
        self.groups['region_render'].grid(**grid_param)

        self.fig, ax = plt.subplots(nrows=1, ncols=2)
        self.fig.set_figheight(5)
        self.fig.set_figwidth(10)

        canvas = FigureCanvasTkAgg(self.fig, master=self.groups['region_render'])
        self.plot_widget = canvas.get_tk_widget()
        self.plot_widget.grid(row=0, column=0)

    # %% Callbacks
    # -- Sliders
    def __interp_changed(self, interp):
        """Interpolation method changed"""
        # Ignore interp since already stored in variable
        self.__update_region_figure()

    def __cmap_changed(self, cmap):
        """Colormap method changed"""
        # Ignore cmap since already stored in variable
        self.__update_region_figure()

    def __time_slider_changed(self, percent):
        """Time slider changed"""
        self.__update_region_figure()

    # -- Buttons
    def __load_model_pressed(self):
        """Load TensorFlow given a directory"""
        dir = filedialog.askdirectory(initialdir=self.model_dir)
        # to update label to show directory
        self.model_dir.set('Loading Model...')

        try:
            self.Model.init_from_file(model_type='tf', path=dir)
            self.model_dir.set(dir)
        except:
            self.model_dir.set('Invalid TensorFlow Model Directory')

    def __simulate_pressed(self):
        self.sim_status.set('Simulating...')

        if not self.Model.ready():
            self.sim_status.set('Must load valid model first')
            return

        # Check that UI inputs are in proper format
        try:
            start_date_str = self.start_date_str.get()
            start_day, start_month, start_year \
                = [int(val) for val in start_date_str.split('/')]

            # This will fail if data is invalid
            start_datetime = datetime.datetime(year=start_year, month=start_month, day=start_day)
        except:
            self.sim_status.set('Problem in start date; may not exist')
            return

        try:
            stop_date_str = self.stop_date_str.get()
            stop_day, stop_month, stop_year \
                = [int(val) for val in stop_date_str.split('/')]
            stop_datetime = datetime.datetime(year=stop_year, month=stop_month, day=stop_day)
        except:
            self.sim_status.set('Problem in end date; may not exist')
            return

        # Setup timeline
        try:
            sim_times = app.setup_timeline(
                start_date=(start_year, start_month, start_day),
                stop_date=(stop_year, stop_month, stop_day),
                sim_step=self.sim_step.get(),
            )
            n_time = sim_times.size
        except ValueError as err:
            self.sim_status.set(str(err))
            return

        sim_mode = self.mode.get().lower()
        if sim_mode == 'location':
            lat_str = self.lats_str.get()
            lon_str = self.lons_str.get()
            if lat_str.empty() or lon_str.empty():
                self.sim_status.set('Either lats or lons empty')
                return

            # Store locations as lat/lon
            lats = np.array([float(val) for val in self.lats_str.get().split(',')])
            lons = np.array([float(val) for val in self.lons_str.get().split(',')])
            if len(lats) != len(lons):
                self.sim_status.set('Lats/Lons lengths must match')
                return

            # (n_day, n_loc)
            n_loc = lats.size
            lat_grid = np.repeat(lats[np.newaxis, :], axis=0, repeats=n_time)
            lon_grid = np.repeat(lons[np.newaxis, :], axis=0, repeats=n_time)
            time_grid = np.repeat(sim_times[:, np.newaxis], axis=-1, repeats=n_loc)

        else:  # region mode
            try:
                min_lat = float(self.min_lat_str.get())
                max_lat = float(self.max_lat_str.get())
                step_lat = float(self.step_lat_str.get())
                min_lon = float(self.min_lon_str.get())
                max_lon = float(self.max_lon_str.get())
                step_lon = float(self.step_lon_str.get())
            except:
                self.sim_status.set('Problem with lat/lon config')
                return

            try:
                lats, lons = app.setup_spatial_support(
                    lat_bounds=(min_lat, max_lat),
                    lon_bounds=(min_lon, max_lon),
                    lat_res=step_lat,
                    lon_res=step_lon,
                )
                # Grid all inputs
                time_grid, lat_grid, lon_grid = np.meshgrid(
                    sim_times, lats, lons, indexing='ij')
            except ValueError as err:
                self.sim_status.set(str(err))
                return

        grid_shape = time_grid.shape
        x = features.preprocess(lat=lat_grid, lon=lon_grid, epoch_time=time_grid)
        carbon = self.Model.predict(x)

        # At this point, simulation was successful. Store all necessary results
        if sim_mode == 'region':
            self.sim_results_region = {
                'lats': lats,  # (n_loc)
                'lons': lons,  # (n_loc)
                'sim_times': sim_times,  # (n_time)
                'carbon': carbon,  # (n_time, n_loc)
            }
            self.__update_region_figure()
        else:
            self.sim_results_location = {
                'lats': lats,  # (n_loc)
                'lons': lons,  # (n_loc)
                'sim_times': sim_times,  # (n_time)
                'carbon': carbon,  # (n_time, n_loc)
            }
            self.__update_location_figure()

        self.sim_status.set('Simulation Complete')

    # Plotting
    def __update_region_figure(self):
        """Update figure(s) for region-based simulation"""
        # Show the Mean Plot
        sim_results = self.sim_results_region
        lat_axis = sim_results['lats']
        lon_axis = sim_results['lons']
        carbon = sim_results['carbon']
        sim_times = sim_results['sim_times']
        mean_carbon = np.mean(carbon, axis=0)

        ax0 = self.fig.axes[0]
        ax0.clear()
        min_lat, max_lat = lat_axis.min(), lat_axis.max()
        min_lon, max_lon = lon_axis.min(), lon_axis.max()
        im = ax0.imshow(
            mean_carbon,
            extent=[min_lon, max_lon, max_lat, min_lat],
            aspect='auto',
            interpolation=self.interp_type.get(),
            cmap=self.cmap_type.get(),
        )
        ax0.set_xlabel('Longitude')
        ax0.set_ylabel('Latitude')

        start_datetime = datetime.datetime.fromtimestamp(sim_times.min())
        stop_datetime = datetime.datetime.fromtimestamp(sim_times.max())
        fmt = '%m/%d/%Y'
        ax0.set_title('Mean Carbon - {} to {}'.format(
            start_datetime.strftime(fmt), stop_datetime.strftime(fmt)
        ))
        ax0.grid(True)
        # self.fig.colorbar(im, ax=ax0)

        # Show the instantaneous plot
        sim_percent = self.time_slider.get()
        n_times = sim_times.size
        idx = int(np.round(sim_percent / 100 * n_times))
        inst_time = sim_times[idx]
        inst_date = datetime.datetime.fromtimestamp(inst_time).strftime(fmt)

        ax1 = self.fig.axes[1]
        ax1.clear()
        im = ax1.imshow(
            carbon[idx, :, :],
            extent=[min_lon, max_lon, max_lat, min_lat],
            aspect='auto',
            interpolation=self.interp_type.get(),
            cmap=self.cmap_type.get(),
        )
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.set_title('Carbon Concentration on {}'.format(inst_date))
        ax1.grid(True)

        self.fig.canvas.draw_idle()

# %%
if __name__ == '__main__':
    App = Ui()
