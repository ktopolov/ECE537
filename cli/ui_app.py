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

FIG_HEIGHT = 5
FIG_WIDTH = 10

# %% UI
class Ui():
    """Main application UI management"""

    def __init__(self):
        """Construct application"""
        # Stored data
        self.Model = model.WrapperModel()  # to be loaded later
        self.sim_results_region = {}  # stores all region simulation results
        self.sim_results_location = {}  # stores all region simulation results

        self.groups = {}  # groups belonging to UI
        self.window = tk.Tk()
        self.window.title('Predictive Carbon Modeling')
        self.window.geometry(f'{WINDOW_WIDTH}x{WINDOW_HEIGHT}')

        # Add groups
        self.__add_model_group(
            widget=self.window,
            grid_param={'row': 0, 'column': 0, 'columnspan': 2}
        )
        self.__add_timeline_group(
            widget=self.window,
            grid_param={'row': 1, 'column': 0, 'columnspan': 2}
        )

        # Tabs for each mode
        tab_control = ttk.Notebook(self.window)
        region_tab = ttk.Frame(tab_control)
        location_tab = ttk.Frame(tab_control)
        tab_control.add(region_tab, text='Region')
        tab_control.add(location_tab, text='Location')
        tab_control.grid(row=2, column=0)

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
        self.__add_location_render_group(
            widget=location_tab,
            grid_param={'row': 1, 'column': 0, 'columnspan': 8}
        )

        # Halt execution
        self.window.mainloop()

    # %% Groups
    def __add_model_group(self, widget, grid_param):
        group = ttk.LabelFrame(widget, text='Model')
        group.grid(**grid_param)

        load_model_button = tk.Button(
            group,
            text='Load Model',
            command=self.__load_model_pressed
        )
        model_dir = tk.StringVar()
        model_dir.set('No Model Selected')
        load_model_label = tk.Label(
            group, textvariable=model_dir, width=50)
        load_model_button.grid(row=0, column=0)  # position within group
        load_model_label.grid(row=0, column=1, columnspan=3)

        self.groups['model'] = {
            'group': group,
            'vars': {'model_dir': model_dir}
        }

    def __add_timeline_group(self, widget, grid_param):
        """Group for simulation parameters"""
        group = ttk.LabelFrame(widget, text='Timeline')
        group.grid(**grid_param)

        start_date_label = tk.Label(group, text='Start Date (DD/MM/YYYY)')
        stop_date_label = tk.Label(group, text='Stop Date (DD/MM/YYYY)')
        sim_step_label = tk.Label(group, text='Simulation Step')

        start_date_str = tk.StringVar()
        stop_date_str = tk.StringVar()
        sim_step = tk.StringVar()

        start_date_entry = tk.Entry(group, textvariable=start_date_str)
        stop_date_entry = tk.Entry(group, textvariable=stop_date_str)

        sim_options = ['daily', 'weekly', 'monthly', 'annually']
        sim_step.set(sim_options[0])  # default
        sim_step_dropdown = tk.OptionMenu(group, sim_step, *sim_options)
        sim_step_dropdown.config(width=10)

        simulate_button = tk.Button(
            group,
            text='Simulate',
            bg='green',
            command=self.__simulate_pressed
        )
        sim_status = tk.StringVar()
        sim_status_label = tk.Label(group, textvariable=sim_status)

        start_date_label.grid(row=0, column=0)
        start_date_entry.grid(row=0, column=1)
        stop_date_label.grid(row=1, column=0)
        stop_date_entry.grid(row=1, column=1)
        sim_step_label.grid(row=0, column=2)
        sim_step_dropdown.grid(row=0, column=3)
        simulate_button.grid(row=1, column=2)
        sim_status_label.grid(row=1, column=3)

        self.groups['timeline'] = {
            'group': group,
            'vars': {
                'sim_status': sim_status,
                'start_date_str': start_date_str,
                'stop_date_str': stop_date_str,
                'sim_step': sim_step,
            }
        }

    def __add_region_group(self, widget, grid_param):
        """Group for simulation parameters"""
        group = ttk.LabelFrame(widget, text='Region')
        group.grid(**grid_param)

        min_lat_label = tk.Label(group, text='Latitude Min. (Deg)')
        max_lat_label = tk.Label(group, text='Latitude Max. (Deg)')
        step_lat_label = tk.Label(group, text='Latitude Step (Deg)')
        min_lon_label = tk.Label(group, text='Longitude Min. (Deg)')
        max_lon_label = tk.Label(group, text='Longitude Max. (Deg)')
        step_lon_label = tk.Label(group, text='Longitude Step (Deg)')

        min_lat_str, max_lat_str = tk.StringVar(), tk.StringVar()
        step_lat_str, step_lon_str = tk.StringVar(), tk.StringVar()
        min_lon_str, max_lon_str = tk.StringVar(), tk.StringVar()

        min_lat_entry = tk.Entry(group, textvariable=min_lat_str)
        max_lat_entry = tk.Entry(group, textvariable=max_lat_str)
        step_lat_entry = tk.Entry(group, textvariable=step_lat_str)
        min_lon_entry = tk.Entry(group, textvariable=min_lon_str)
        max_lon_entry = tk.Entry(group, textvariable=max_lon_str)
        step_lon_entry = tk.Entry(group, textvariable=step_lon_str)

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

        self.groups['region'] = {
            'group': group,
            'vars': {
                'min_lat_str': min_lat_str,
                'max_lat_str': max_lat_str,
                'min_lon_str': min_lon_str,
                'max_lon_str': max_lon_str,
                'step_lat_str': step_lat_str,
                'step_lon_str': step_lon_str,
            }
        }

    def __add_location_group(self, widget, grid_param):
        """Group for location parameters"""
        group = ttk.LabelFrame(widget, text='Location')
        group.grid(**grid_param)

        lats_label = tk.Label(group, text='Latitudes (comma-separated)')
        lons_label = tk.Label(group, text='Longitudes (comma-separated)')

        lats_str = tk.StringVar()
        lons_str = tk.StringVar()
        lats_entry = tk.Entry(group, textvariable=lats_str)
        lons_entry = tk.Entry(group, textvariable=lons_str)

        lats_label.grid(row=0, column=0)
        lats_entry.grid(row=0, column=1)
        lons_label.grid(row=1, column=0)
        lons_entry.grid(row=1, column=1)

        self.groups['location'] = {
            'group': group,
            'vars': {
                'lats_str': lats_str,
                'lons_str': lons_str
            }
        }

    def __add_region_plot_group(self, widget, grid_param):
        """Group for all plot configuration"""
        group = ttk.LabelFrame(widget, text='Plot Configuration')
        group.grid(**grid_param)

        # Interpolation
        interp_label = tk.Label(group, text='Interpolation Type')
        interp_options = ['none', 'bilinear', 'bicubic', 'hanning', 'sinc']
        interp_type = tk.StringVar()
        interp_type.set(interp_options[0])  # default
        interp_dropdown = tk.OptionMenu(
            group,
            interp_type,
            *interp_options,
            command=self.__interp_changed,  # replot when interp changes
        )

        # Colormap
        cmap_label = tk.Label(group, text='Colormap')
        cmap_options = ['jet', 'magma', 'gray', 'hot', 'cool', 'seismic']
        cmap_type = tk.StringVar()
        cmap_type.set(cmap_options[0])  # default
        cmap_dropdown = tk.OptionMenu(
            group,
            cmap_type,
            *cmap_options,
            command=self.__cmap_changed,  # re-plot when cmap changes
        )

        # Simulation timescale slider
        time_slider = tk.Scale(
            group,
            from_=0.0,
            to=100.0,
            sliderlength=20,  # size of slider
            orient='horizontal',
            length=300
        )

        # Run command on slider release
        time_slider.bind('<ButtonRelease-1>', self.__time_slider_changed)

        interp_label.grid(row=0, column=0)
        interp_dropdown.grid(row=0, column=1)
        cmap_label.grid(row=0, column=2)
        cmap_dropdown.grid(row=0, column=3)
        time_slider.grid(row=1, column=0, columnspan=2)

        self.groups['region_plot'] = {
            'group': group,
            'vars': {
                'interp_type': interp_type,
                'cmap_type': cmap_type,
                'time_slider': time_slider
            }
        }

    def __add_region_render_group(self, widget, grid_param):
        """Group where figure(s) will be rendered"""
        group = ttk.LabelFrame(widget)
        group.grid(**grid_param)

        fig, _ = plt.subplots(nrows=1, ncols=2)
        fig.set_figheight(FIG_HEIGHT)
        fig.set_figwidth(FIG_WIDTH)

        canvas = FigureCanvasTkAgg(fig, master=group)
        plot_widget = canvas.get_tk_widget()
        plot_widget.grid(row=0, column=0)

        self.groups['region_render'] = {
            'group': group,
            'vars': {'fig': fig}
        }

    def __add_location_render_group(self, widget, grid_param):
        """Group where figure(s) will be rendered"""
        group = ttk.LabelFrame(widget)
        group.grid(**grid_param)

        fig, _ = plt.subplots(nrows=1, ncols=2)
        fig.set_figheight(FIG_HEIGHT)
        fig.set_figwidth(FIG_WIDTH)

        canvas = FigureCanvasTkAgg(fig, master=group)
        plot_widget = canvas.get_tk_widget()
        plot_widget.grid(row=0, column=0)

        self.groups['location_render'] = {
            'group': group,
            'vars': {'fig': fig}
        }

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
        # Ignore percent since slider can be read in function
        self.__update_region_figure()

    # -- Buttons
    def __load_model_pressed(self):
        """Load TensorFlow given a directory"""
        model_group_vars = self.groups['model']['vars']
        model_dir = model_group_vars['model_dir']
        dir = filedialog.askdirectory(initialdir=model_dir)
        # to update label to show directory
        model_dir.set('Loading Model...')

        try:
            self.Model.init_from_file(model_type='tf', path=dir)
            model_dir.set(dir)
        except:
            model_dir.set('Invalid TensorFlow Model Directory')

    def __simulate_pressed(self):
        timeline_group_vars = self.groups['timeline']['vars']
        sim_status = timeline_group_vars['sim_status']
        sim_status.set('Simulating...')

        if not self.Model.ready():
            sim_status.set('Must load valid model first')
            return

        # Check that UI inputs are in proper format
        start_date_str = timeline_group_vars['start_date_str']
        try:
            start_day, start_month, start_year \
                = [int(val) for val in start_date_str.get().split('/')]

            # This will fail if data is invalid
            start_datetime = datetime.datetime(year=start_year, month=start_month, day=start_day)
        except:
            sim_status.set('Problem in start date; may not exist')
            return

        stop_date_str = timeline_group_vars['stop_date_str']
        try:
            stop_day, stop_month, stop_year \
                = [int(val) for val in stop_date_str.get().split('/')]
            stop_datetime = datetime.datetime(year=stop_year, month=stop_month, day=stop_day)
        except:
            sim_status.set('Problem in end date; may not exist')
            return

        # Setup timeline
        sim_step = timeline_group_vars['sim_step']
        try:
            sim_times = app.setup_timeline(
                start_date=(start_year, start_month, start_day),
                stop_date=(stop_year, stop_month, stop_day),
                sim_step=sim_step.get(),
            )
            n_time = sim_times.size
        except ValueError as err:
            sim_status.set(str(err))
            return

        # FIXME-KT: Set this based on which tab is open
        sim_mode = 'location'

        if sim_mode == 'location':
            location_group_vars = self.groups['location']['vars']
            lats_str = location_group_vars['lats_str']
            lons_str = location_group_vars['lons_str']

            if not lats_str.get() or not lons_str.get():
                sim_status.set('Either lats or lons empty')
                return

            # Store locations as lat/lon
            lats = np.array([float(val) for val in lats_str.get().split(',')])
            lons = np.array([float(val) for val in lons_str.get().split(',')])
            if len(lats) != len(lons):
                sim_status.set('Lats/Lons lengths must match')
                return

            # (n_day, n_loc)
            n_loc = lats.size
            lat_grid = np.repeat(lats[np.newaxis, :], axis=0, repeats=n_time)
            lon_grid = np.repeat(lons[np.newaxis, :], axis=0, repeats=n_time)
            time_grid = np.repeat(sim_times[:, np.newaxis], axis=-1, repeats=n_loc)

        else:  # region mode
            region_group_vars = self.groups['region']['vars']

            try:
                min_lat = float(region_group_vars['min_lat_str'].get())
                max_lat = float(region_group_vars['max_lat_str'].get())
                step_lat = float(region_group_vars['step_lat_str'].get())
                min_lon = float(region_group_vars['min_lon_str'].get())
                max_lon = float(region_group_vars['max_lon_str'].get())
                step_lon = float(region_group_vars['step_lon_str'].get())
            except:
                sim_status.set('Problem with lat/lon config')
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
                sim_status.set(str(err))
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

        sim_status.set('Simulation Complete')

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

        region_render_group_vars = self.groups['region_render']['vars']
        fig = region_render_group_vars['fig']

        region_plot_group_vars = self.groups['region_plot']['vars']
        interp_type = region_render_group_vars['interp_type']
        cmap_type = region_render_group_vars['cmap_type']
        time_slider = region_render_group_vars['time_slider']

        ax0 = fig.axes[0]
        ax0.clear()
        min_lat, max_lat = lat_axis.min(), lat_axis.max()
        min_lon, max_lon = lon_axis.min(), lon_axis.max()

        # FIXME-Make slider for these
        vmin, vmax = carbon.min(), carbon.max()

        im = ax0.imshow(
            mean_carbon,
            extent=[min_lon, max_lon, max_lat, min_lat],
            aspect='auto',
            interpolation=interp_type.get(),
            cmap=cmap_type.get(),
            vmin=vmin,
            vmax=vmax
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
        sim_percent = time_slider.get()
        n_times = sim_times.size
        idx = int(np.round(sim_percent / 100 * n_times))
        inst_time = sim_times[idx]
        inst_date = datetime.datetime.fromtimestamp(inst_time).strftime(fmt)

        ax1 = fig.axes[1]
        ax1.clear()
        im = ax1.imshow(
            carbon[idx, :, :],
            extent=[min_lon, max_lon, max_lat, min_lat],
            aspect='auto',
            interpolation=interp_type.get(),
            cmap=cmap_type.get(),
            vmin=vmin,
            vmax=vmax
        )
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.set_title('Carbon Concentration on {}'.format(inst_date))
        ax1.grid(True)

        fig.canvas.draw_idle()

    def __update_location_figure(self):
        """Update figure(s) for location-based simulation"""
        sim_results = self.sim_results_location
        lats = sim_results['lats']
        lons = sim_results['lons']
        carbon = sim_results['carbon']
        sim_times = sim_results['sim_times']
        mean_carbon = np.mean(carbon, axis=0)

        # Show carbon for all locations across time
        location_render_group_vars = self.groups['location_render']['vars']
        fig = location_render_group_vars['fig']

        ax0 = fig.axes[0]
        ax0.clear()

        n_loc = lats.size
        for i_loc in range(n_loc):
            lat, lon = lats[i_loc], lons[i_loc]
            # Convert sim times to date
            ax0.plot(sim_times, carbon[:, i_loc], label=f'Lat: {lat}, Lon: {lon}')

        ax0.set_xlabel('Epoch Time (TODO-Use Date)')
        ax0.set_ylabel('Carbon')
        ax0.set_title('Carbon per Location')
        ax0.legend()
        ax0.grid(True)

        mean_carbon = carbon.mean(axis=0)
        ax1 = fig.axes[1]
        ax1.clear()
        ax1.scatter(lons, lats, c=mean_carbon)
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.set_title('Carbon Concentration per Location')
        ax1.legend()
        ax1.grid(True)
        # colorbar?

        fig.canvas.draw_idle()

# %%
if __name__ == '__main__':
    App = Ui()

# FIXME: Add these
# Button to write to KML
# Button to write to CSV
# Slider for vmax/vmin for plotting images
# Tab for location-based simulation
