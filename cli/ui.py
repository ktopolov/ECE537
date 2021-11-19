from codebase import model
import numpy as np
from PIL import Image, ImageTk
from tkinter import filedialog, ttk, font
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # shut tensorflow up


# %% Callbacks
model_dir = None


class Ui():
    """Main application UI management"""

    def __init__(self):
        """Construct application"""
        # Stored data
        self.Model = model.WrapperModel()  # to be loaded later

        n_lat, n_lon = 16, 32
        self.lat_axis = np.linspace(-90.0, 90.0, num=n_lat)
        self.lon_axis = np.linspace(-180.0, 180.0, num=n_lon)
        self.predict_grid = np.sinc(self.lat_axis / 10.0)[:, np.newaxis] \
            + np.sinc(self.lon_axis / 10.0) \
            + 0.01 * np.random.randn(n_lat, n_lon)  # np.zeros((n_lat, n_lon))

        # Window setup parameters
        self.__setup_window()
        self.__setup_tab1()
        self.__setup_tab2()
        self.window.mainloop()

    # UI Setup
    def __setup_window(self):
        """Create the window to render"""
        # Main window
        width = 1500
        height = 800
        self.window = tk.Tk()
        self.window.title('Predictive Carbon Modeling')
        self.window.geometry(f'{width}x{height}')

        # Add multiple tabs
        self.tab_control = ttk.Notebook(self.window)
        self.tab1 = ttk.Frame(self.tab_control)
        self.tab2 = ttk.Frame(self.tab_control)

        self.tab_control.add(self.tab1, text='Tab 1')
        self.tab_control.add(self.tab2, text='Tab 2')
        self.tab_control.pack(expand=1, fill='both')

    # %% Tabs
    def __setup_tab1(self):
        """Setup the first tab"""
        model_param = {'row': 0, 'column': 0, 'columnspan': 2}
        sim_param = {'row': 0, 'column': 2, 'columnspan': 2}
        image_param = {'row': 1, 'column': 0, 'columnspan': 4}
        plot_param = {'row': 2, 'column': 0, 'columnspan': 4}

        self.__setup_model_group(widget=self.tab1, grid_param=model_param)
        self.__setup_simulation_group(widget=self.tab1, grid_param=sim_param)
        self.__setup_plot_group(widget=self.tab1, grid_param=plot_param)
        self.__setup_render_group(widget=self.tab1, grid_param=image_param)

    def __setup_tab2(self):
        """Setup the second tab"""
        self.label2 = tk.Label(self.tab2, text='label2')
        self.label2.grid(column=0, row=0)

    # %% Groups
    def __setup_model_group(self, widget, grid_param):
        model_group = ttk.LabelFrame(widget, text='Model')
        model_group.grid(**grid_param)

        load_model_button = tk.Button(
            model_group,
            text='Load Model',
            command=self.__load_model_pressed
        )
        self.model_dir = tk.StringVar()
        self.model_dir.set('No Model Selected')
        self.load_model_label = tk.Label(
            model_group, textvariable=self.model_dir, width=30)
        load_model_button.grid(row=0, column=0)  # position within group
        self.load_model_label.grid(row=0, column=1)

    def __setup_simulation_group(self, widget, grid_param):
        """Group for simulation parameters"""
        sim_group = ttk.LabelFrame(widget, text='Simulation')
        sim_group.grid(**grid_param)

        self.mode_label = tk.Label(sim_group, text='Select Mode')
        modes = ['location', 'grid']
        self.mode = tk.StringVar()
        self.mode_dropdown = tk.OptionMenu(sim_group, self.mode, *modes)
        self.mode_dropdown.config(width=40)

        simulate_button = tk.Button(
            sim_group,
            text='Simulate',
            bg='green',
            command=self.__simulate_pressed
        )

        self.mode_label.grid(row=0, column=0)
        self.mode_dropdown.grid(row=0, column=1)
        simulate_button.grid(row=1)

    def __setup_plot_group(self, widget, grid_param):
        """Group for all plot configuration"""
        plot_group = ttk.LabelFrame(widget, text='Plot Configuration')
        plot_group.grid(**grid_param)

        self.interp_label = tk.Label(plot_group, text='Interpolation Type')
        interp_options = [
            'none', 'antialiased', 'nearest', 'bilinear', 'bicubic',
            'spline16', 'spline36', 'hanning', 'hamming', 'hermite',
            'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell',
            'sinc', 'lanczos', 'blackman'
        ]
        self.interp_type = interp_options[0]
        interp_type = tk.StringVar()  # no need to store in self since command passes value
        self.interp_dropdown = tk.OptionMenu(
            plot_group,
            interp_type,
            *interp_options,
            command=self.__interp_changed,
        )
        self.interp_dropdown.config(width=40)

        # Simulation timescale slider
        self.time_slider = tk.Scale(
            plot_group,
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
        self.time_slider.grid(row=1, column=0, columnspan=2)

    def __setup_render_group(self, widget, grid_param):
        """Group where figure(s) will be rendered"""
        render_group = ttk.LabelFrame(widget)
        render_group.grid(**grid_param)

        self.fig, ax = plt.subplots(nrows=1, ncols=2)
        self.fig.set_figheight(5)
        self.fig.set_figwidth(10)
        self.__update_figure()

        canvas = FigureCanvasTkAgg(self.fig, master=render_group)
        self.plot_widget = canvas.get_tk_widget()
        self.plot_widget.grid(row=0, column=0)

    # %% Callbacks
    # -- Sliders
    def __time_slider_changed(self, percent):
        """Time slider changed"""
        sim_percent = self.time_slider.get()
        print(sim_percent)

    # -- Dropdowns
    def __interp_changed(self, interp_type):
        """Interpolation type changed"""
        self.interp_type = interp_type
        self.__update_figure()

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
        print('Simulating')
        print('Done')

    # Plotting
    def __update_figure(self):
        """Update figure"""
        ax0 = self.fig.axes[0]
        ax0.clear()
        min_lat, max_lat = self.lat_axis.min(), self.lat_axis.max()
        min_lon, max_lon = self.lon_axis.min(), self.lon_axis.max()
        im = ax0.imshow(
            self.predict_grid,
            extent=[min_lon, max_lon, max_lat, min_lat],
            aspect='auto',
            interpolation=self.interp_type,
        )
        ax0.set_xlabel('Longitude')
        ax0.set_ylabel('Latitude')
        ax0.set_title('Predicted Carbon Concentration')
        ax0.grid(True)
        # self.fig.colorbar(im, ax=ax0)
        self.fig.canvas.draw_idle()


# %%
if __name__ == '__main__':
    App = Ui()
