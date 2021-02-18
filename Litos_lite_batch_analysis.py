import numpy as np
import csv
import os
import re
from os import listdir
from os.path import isfile, isdir, join
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
import seaborn as sns
import tkinter as tk
import datetime
import tkinter.messagebox as msgb
# np.set_printoptions(threshold=np.inf)
pd.set_option("display.max_rows", None, "display.max_columns", None)
# set colormap for the graphs
step = 20
color = [plt.cm.tab20(i) for i in np.linspace(0, 1, step)]
preconditioning_protocol = False

# TODO: sinthesize code
# TODO: ADD working pixel selection based on dark JV
# TODO: extract shunt and series resistance from JV scan (light and dark or only light?)
# TODO: Add temperature and moisture extraction (during measurement)
# TODO: grey out JVs of poorly working pixels
# TODO: add ppt reporting of data
# TODO: add interaction with recipe data


# --------------- GUI functions -------------


def JV_loader():
    filepath = insert.get()
    status = Importer(filepath, presentation=True)
    msgb.showinfo(title=None, message=status)


def center(win):
    win.update_idletasks()
    width = win.winfo_width()
    height = win.winfo_height()
    x = (win.winfo_screenwidth() - width) // 2
    y = (win.winfo_screenheight() - height) // 3
    win.geometry('{}x{}+{}+{}'.format(width, height, x, y))


def SetWinActive(win):
    """
    SetWinActive(window)
    To get the window passed as parameter as topmost in the screen.
    """
    win.lift()
    win.attributes('-topmost', True)
    win.after_idle(win.attributes, '-topmost', False)

# ---------------- Analysis functions --------------------------


def import_file(path, files, type):
    """ Import the specified file. Returns light condition, JVdata.
        Returns the output in seconds, volts, milliamps"""
    # Initialize variables
    measurement_type = 'type:	Perform parallel JV' if type == 'JV' else 'type:	MPP tracking' if type == 'MPP' else None
    Suns, dataset, sweep, head, scans, sweep_speeds, measurement_start_dates = [
    ], [], [], [], [], [], []
    all_light_scans = {}
    match = re.search(r'(.+) - (.+)\[([0-8])\]_', files[0])
    # Begin analysis
    for measurement in files:
        filepath = join(path, measurement)
        with open(filepath) as f:
            lines = list(f)
        if measurement_type not in lines[1]:
            pass
            # parts data from descriptors (which specify what column is what
        else:
            flag_normalize = False
            data = []
            for line in lines:
                try:
                    data.append([float(item) for item in line.split(',')])
                    # if lines[1] == 'type:	MPP tracking':
                    #     print(line)
                except ValueError as e:
                    if 'Light intensity' in line:
                        sun = float(line.split(',')[1])
                    elif 'Sample area' in line:
                        Area = float(re.search(r'\d+\.\d+', line).group())
                    elif '(mA)' in line:
                        flag_normalize = True
                    elif 'direction' in line:
                        d = int(re.search(r'(\d+)\.\d+', line).group(1))
                        swp = ['Fw'] if d == 0 else ['Rv'] if d == 1 else [
                            'Fw', 'Rv'] if d == 2 else ['Rv', 'Fw']
                    elif 'sweep_speed' in line:
                        speed = float(
                            re.search(r'\d+\.\d+', line).group())
                    elif 'start_time' in line:
                        line = line.replace('#start_time:\t', '')
                        line = line.replace('\n', '')
                        start_date = datetime.datetime.strptime(
                            line, '%m/%d/%Y %H:%M:%S %p')
                        discard = datetime.timedelta(minutes=start_date.minute,
                                                     seconds=start_date.second)
                        start_date -= discard
                        if discard >= datetime.timedelta(minutes=30):
                            start_date += datetime.timedelta(hours=1)
                    else:
                        pass
            data = np.array(data)
            # counter for the number of same scans performed on a device
            if any(sun == x for x in all_light_scans):
                all_light_scans[sun] += 1
            if all(sun != x for x in all_light_scans):
                all_light_scans[sun] = 0
            if type == 'MPP' and 'type:	MPP tracking' in lines[1]:
                if flag_normalize:
                    data[:, 2] /= Area
                else:
                    data[:, 2] *= 1000 * 1000
                Suns.append(sun)
                dataset.append(data)
                head.append('{}__{}__p{}__{}__scan{}'.format(match.group(1), match.group(2), match.group(3),
                                                             str(sun / 100) + 'Sun', all_light_scans[sun]))
                scans.append(all_light_scans[sun])
                measurement_start_dates.append(start_date)
            elif type == 'JV' and 'type:	Perform parallel JV' in lines[1]:
                if flag_normalize:
                    data[:, 1] /= Area
                else:
                    data[:, 1] *= 1000
                if swp == ['Fw'] or swp == ['Rv']:
                    Suns.append(sun)
                    dataset.append(data)
                    sweep.append(swp[0])
                    head.append('{}__{}__p{}__{}__{}__scan{}'.format(match.group(1), match.group(2), match.group(3),
                                                                     'Dark' if sun == 0 else str(sun / 100) + 'Sun', swp[0], all_light_scans[sun]))
                    scans.append(all_light_scans[sun])
                    sweep_speeds.append(speed)
                    measurement_start_dates.append(start_date)
                elif swp == ['Fw', 'Rv'] or swp == ['Rv', 'Fw']:
                    Suns.append(sun)
                    Suns.append(sun)
                    n = next(i for i, x in enumerate(
                        data[:, 0]) if np.isnan(x))
                    dataset.append(data[:n, :])
                    dataset.append(data[n + 1:-1, :])
                    sweep.append(swp[0])
                    sweep.append(swp[1])
                    head.append('{}__{}__p{}__{}__{}__scan{}'.format(match.group(1), match.group(2), match.group(3),
                                                                     'Dark' if sun == 0 else str(sun / 100) + 'Sun', swp[0], all_light_scans[sun]))
                    head.append('{}__{}__p{}__{}__{}__scan{}'.format(match.group(1), match.group(2), match.group(3),
                                                                     'Dark' if sun == 0 else str(sun / 100) + 'Sun', swp[1], all_light_scans[sun]))
                    scans.append(all_light_scans[sun])
                    scans.append(all_light_scans[sun])
                    sweep_speeds.append(speed)
                    sweep_speeds.append(speed)
                    measurement_start_dates.append(start_date)
                    measurement_start_dates.append(start_date)

    return (Suns, dataset, head, scans, measurement_start_dates) if type == 'MPP' else (Suns, dataset, sweep, head, scans, sweep_speeds, measurement_start_dates) if type == 'JV' else []


def create_folder(filepath, foldername):
    """
    Create a new folder if not existing.
    Return Folderpath.
    """
    newfolder = join(filepath, foldername)
    try:
        os.mkdir(newfolder)
    except OSError:
        pass
    return newfolder


def JV_parameters(JV, Pin):
    # JV_smooth = signal.savgol_filter(JV[:,1], 5,3)
    # I actually removed the smoothing to avoid problems with changes in FF, Jsc,Voc....
    JV_smooth = JV[:, 1]
    # the filter is at the line before, if we want to reintroduce it. Overall it was not super effective.
    try:
        # np.around approximate to .2 digits
        Jsc = np.around(JV_smooth[np.nanargmin(np.abs(JV[:, 0]))], 2)
        # np.nanargmin return position in the array whwere | V | is minimum
    except ValueError:
        return [np.nan, np.nan, np.nan, np.nan]
    Voc = np.around(JV[np.nanargmin(np.abs(JV_smooth)), 0], 2)
    JV_smoothed = np.hstack(
        [JV[:, 0][:, np.newaxis], JV_smooth[:, np.newaxis]])
    try:
        Pout = np.nanmax(JV_smoothed[:, 0] * JV_smoothed[:, 1])
        PCE = np.around(Pout / Pin * 100, 2)
        FF = np.around(Pout * 100 / (Voc * Jsc), 2)
    except RuntimeWarning:
        return [np.nan, np.nan, np.nan, np.nan]
        # condition over which collected parameters are not considered trustable.
    if Voc > 0.1 and Jsc > 1 and PCE > 1 and FF < 100:
        return [Voc, Jsc, FF, PCE]
    else:
        return [np.nan, np.nan, np.nan, np.nan]


# NOTE: here i need a class to be able return the input values. If I insert a function to create a child window
# I will have no way to store the data I saved (except by creating a global variable)


class ask_batch_structures(tk.Toplevel):
    """

    """

    def __init__(self, master, device_list):
        tk.Toplevel.__init__(self, master)
        self.resizable(True, True)
        self.architectures = []
        self.device_list = device_list
        self.device_list.sort_values(['Variable', 'Device'], inplace=True)
        self.device_list.reset_index(drop=True, inplace=True)
        # print(self.device_list)
        entry_names = []
        entry_variable = []
        entry_architectures = []
        sample_stack = tk.Label(
            self, text=r'eg.  Glass | FTO | TiO2-c | TiO2-mp | Cs0.05FA0.79MA0.16PbI2.49Br0.51 | spiro-OMeTAD | Au', foreground='gray')
        sample_stack.grid(column=2, columnspan=2, padx=10, sticky='w')
        for i in range(len(self.device_list.index)):
            self.architectures.append(tk.StringVar())
            entry_name = tk.Label(
                self, text=self.device_list.loc[i, 'Device'],  width=5)
            entry_variable = tk.Label(
                self, text=self.device_list.loc[i, 'Variable'], width=int(1.2 * len(self.device_list.loc[i, 'Variable'])))
            entry_architecture = tk.Entry(
                self, textvariable=self.architectures[i],  width=100)
            if "PEAI" not in self.device_list.loc[i, 'Variable']:
                entry_architecture.insert(
                    0, r'Glass | FTO | TiO2-c | TiO2-mp | Cs0.05FA0.79MA0.16PbI2.49Br0.51 | spiro-OMeTAD | Au')
            else:
                entry_architecture.insert(
                    0, r'Glass | FTO | TiO2-c | TiO2-mp | Cs0.05FA0.79MA0.16PbI2.49Br0.51 | PEAI | spiro-OMeTAD | Au')
            entry_name.grid(row=i + 1, sticky='E')
            entry_variable.grid(row=i + 1, column=1, sticky='WE')
            entry_architecture.grid(
                row=i + 1, column=2, columnspan=2, sticky='WE')
            entry_names.append(entry_name)
            entry_architectures.append(entry_architecture)
        entry_architectures[0].focus_set()
        ok = tk.Button(self, text='Ok', padx=15)
        cancel = tk.Button(self, text='Cancel')
        ok.grid(row=i + 2, column=2, sticky='E')
        ok.bind('<Button-1>', self.getStacks)
        ok.bind('<Return>', self.getStacks)
        cancel.grid(row=i + 2, column=3, sticky='W')

        center(self)
        SetWinActive(self)
        # self.transient(master)
        self.grab_set()
        master.wait_window(self)
        # self.title('Insert Device Stacks')
        # self.mainloop()

    def getStacks(self, event=None):
        self.architectures = [self.architectures[i].get()
                              for i in range(len(self.device_list.index))]
        self.stack_df = pd.DataFrame(np.hstack([np.array(self.device_list['Device'])[:, np.newaxis], np.array(self.architectures)[:, np.newaxis]
                                                ]), columns=['Device', 'Cell Stack'])
        self.destroy()


def Importer(analysis_path, presentation=False, **extra_params):
    """Accepts the extra variable 'only_sweep'='Fw' or ='Rv'. Enable to restrict box plotting to only one of the scans"""
    # Set path as working directory
    os.chdir(analysis_path)
    devices_data = join(analysis_path, 'devices')
    root = os.path.abspath(join(analysis_path, os.pardir))
    experiment = os.path.basename(analysis_path)
    # all paios files are stored in different folders, this adds all folders as paths and explore them
    folders = [f for f in os.listdir(devices_data) if isdir(
        join(devices_data, f)) and f != 'Analysis']
    # RETRIEVE NAME for each device of the experiment
    names = set([re.sub('\[[1-8]\]$', '', f) for f in folders])
    # stores in this folder the elaborated data
    result_dir = create_folder(create_folder(root, 'Analyzed'), experiment)
    print(
        f'---------------------------------{experiment} Analysis Begun---------------------------------')

    # mW/cm2 - I assume AM1.5 impinging intensity. To be updated once the impinging power is known and calibrated.
    Pin = 100
    JVLog, MPPTLog = [], []
    for name in names:
        print('{}    --Started--'.format(name))
        device = [f for f in folders if re.sub('\[[1-8]\]$', '', f) == name]
        # print(device)
        # device_dir = create_folder(result_dir, name)
        # loop iterates over all pixels and measurements done on a device.
        Illuminations, JVs, scan_directions, header, scan_number, sweep_speed, measurement_start_date = [
        ], [], [], [], [], [], []
        Light_mppt, P_mppt, head_mppt, scan_number_mppt = [], [], [], []
        stabilized_mpp, time, measurement_start_date_MPP = [], [], []
        working_JV, working_mpp = [], []
        for pixel in device:
            path = join(devices_data, pixel)
            files = [f for f in listdir(
                path) if re.search('0.csv$', f) != None]
            # type:	Perform parallel JV
            # keeps track of program progress
            # JV
            I, J, s, h, sc, sp, msd = import_file(path, files, 'JV')
            Illuminations.extend(I)
            JVs.extend(J)
            scan_directions.extend(s)
            header.extend(h)
            scan_number.extend(sc)
            sweep_speed.extend(sp)
            measurement_start_date.extend(msd)
            # MPPT Plotting
            L, mppt, H, sc_mppt, msd_mppt = import_file(path, files, 'MPP')
            Light_mppt.extend(L)
            P_mppt.extend(mppt)
            head_mppt.extend(H)
            scan_number_mppt.extend(sc_mppt)
            measurement_start_date_MPP.extend(msd_mppt)
            # print(mppt)
            # print(mppt[0][-5:, 0])
            time.extend([np.mean(x[-5:, 0]) for x in mppt])
            stabilized_mpp.extend(
                [np.mean(x[-5:, 1] * x[-5:, 2]) if np.mean(x[-5:, 1] * x[-5:, 2]) > 0 else 0 for x in mppt])
        # print('step 0')
        # print(len(Illuminations), len(JVs), len(
        #     scan_directions), len(header), len(scan_number))
        # print(len(Light_mppt), len(P_mppt), len(
        #     head_mppt), len(scan_number_mppt), len(time), len(stabilized_mpp))
        # print(H)
        MPPTLog.extend([np.append([item for k, item in enumerate(head_mppt[w].split('__')) if k != 3 and k != 4],
                                  [scan_number_mppt[w], Light_mppt[w], time[w], stabilized_mpp[w], measurement_start_date_MPP[w]]) for w in range(len(head_mppt))])
        lights = set(Illuminations)
        Illuminations = np.array(Illuminations)
        for light in lights:
            hed0 = [header[i]
                    for i, x in enumerate(Illuminations) if x == light]
            JV0 = [JVs[i] for i, x in enumerate(Illuminations) if x == light]
            # JV = JVs)
            sp0 = [scan_directions[i]
                   for i, x in enumerate(Illuminations) if x == light]
            sc_num0 = [scan_number[i]
                       for i, x in enumerate(Illuminations) if x == light]
            swp_spd0 = [sweep_speed[i]
                        for i, x in enumerate(Illuminations) if x == light]
            ms_st_date0 = [measurement_start_date[i]
                           for i, x in enumerate(Illuminations) if x == light]
            # triplicate headers for mppt
            PCE_mppt0 = [P_mppt[i]
                         for i, x in enumerate(Light_mppt) if x == light]
            head_mpp0 = [head_mppt[i]
                         for i, x in enumerate(Light_mppt) if x == light]
            sc_num_mpp = [scan_number_mppt[i]
                          for i, x in enumerate(Light_mppt) if x == light]
            # print('step 1')
            # print(len(hed0), len(JV0), len(
            #     sp0))
            # print(len(PCE_mppt0), len(head_mpp0), len(sc_num_mpp))
            for scan in set(scan_number):
                hed = [hed0[i]
                       for i, x in enumerate(sc_num0) if x == scan for _ in (0, 1)]
                JV = [JV0[i] for i, x in enumerate(sc_num0) if x == scan]
                # JV = JVs
                try:
                    short = min([len(x) for x in JV])
                except ValueError:
                    pass
                JV = [i[:short, :] for i in JV]
                sp = [sp0[i]
                      for i, x in enumerate(sc_num0) if x == scan]
                swp_spd = [swp_spd0[i]
                           for i, x in enumerate(sc_num0) if x == scan]
                ms_st_date = [ms_st_date0[i]
                              for i, x in enumerate(sc_num0) if x == scan]
                # triplicate headers for mppt
                PCE_mppt = [PCE_mppt0[i]
                            for i, x in enumerate(sc_num_mpp) if x == scan]
                hed_mpp = [head_mpp0[i]
                           for i, x in enumerate(sc_num_mpp) if x == scan for _ in (0, 1, 2)]
                # print('step 2')
                # print(len(hed), len(JV), len(
                #     sp))
                # print(len(PCE_mppt), len(hed_mpp), len(sc_num_mpp))
                # print(np.array(Illuminations) == light)
                # duplicates headers to account for V and J columns in JV scan
                savetxt_dir = create_folder(result_dir, 'Data')
                if len(hed) > 0:
                    try:
                        np.savetxt(join(savetxt_dir, f'{name}_{light/100}Sun_Scan{scan}.txt'), np.vstack((
                            hed, np.hstack((JV[:])))), fmt='%s', newline='\n', delimiter='\t')
                    except ValueError:
                        print(
                            'You should use the same measurement parameters on different pixels')
                if len(hed_mpp) > 0:
                    try:
                        np.savetxt(join(savetxt_dir, f'{name}_{light/100}Sun_Scan{scan}_mppt.txt'), np.vstack(
                            (hed_mpp, np.hstack((PCE_mppt[:])))), fmt='%s', newline='\n', delimiter='\t')
                    except ValueError:
                        print(
                            'You should use the same measurement parameters on different pixels')

                # ------------------------------------
                #             JV plotting
                # ------------------------------------
                if JV != []:
                    fig, ax = plt.subplots()
                    for i in range(len(sp)):
                        # hed format: '{label}_{variable}_p{pixel}_{light}_{direction}_scan{scan}'
                        a = [item for k, item in enumerate(
                            hed[2 * i].split('__')) if k == 2 or k == 4]
                        p = int(a[0].strip('p'))
                        if sp[i] == 'Fw':
                            if light != 0:
                                ax.plot(JV[i][:, 0], JV[i][:, 1],
                                        label=f'{a[0]}_{a[1]}', color=color[(p - 1) * 2])
                            else:
                                ax.semilogy(JV[i][:, 0], np.abs(JV[i]
                                                                [:, 1]), label=f'{a[0]}_{a[1]}', color=color[(p - 1) * 2])
                        else:
                            if light != 0:
                                ax.plot(JV[i][:, 0], JV[i][:, 1],
                                        '--', label=f'{a[0]}_{a[1]}', color=color[p * 2 - 1])
                            else:
                                ax.semilogy(JV[i][:, 0], np.abs(JV[i][:, 1]),
                                            '--', label=f'{a[0]}_{a[1]}', color=color[p * 2 - 1])
                    # print(hed[2 * i])
                    # print(a)
                    ax.set_xlabel('Bias [V]')
                    ax.set_ylabel('Jmeas [' + '$mA cm^{-2}$' + ']')
                    ax.set_xlim([-0.5, 1.5])
                    flag = [-50, 30] if light != 0 else [1e-4, 30]
                    ax.set_ylim(flag)
                    ax.grid(True, linestyle='--')
                    ax.tick_params(direction='in')
                    lg = ax.legend(loc=2, bbox_to_anchor=(
                        1.05, 1), borderaxespad=0.)
                    ax.set_title(
                        f'Dev.{name}  {light/100} Sun Scan {scan}')
                    plt.savefig(
                        join(result_dir, f'{name}_{light/100}Sun.png'), bbox_extra_artists=(lg,), bbox_inches='tight', dpi=800)
                    plt.close()
                if light != 0:
                    # ------------------------------------
                    #   MPPT plotting & stab_mpp extraction
                    # ------------------------------------
                    if PCE_mppt != []:
                        # print(hed_mpp[3 * i])
                        # print(hed_mpp)
                        fig, ax = plt.subplots()
                        for i in range(len(PCE_mppt)):
                            a = [item for k, item in enumerate(
                                hed_mpp[3 * i].split('__')) if k == 2]
                            p = int(a[0].strip('p'))
                            ax.plot(PCE_mppt[i][:, 0], PCE_mppt[i][:, 1] * PCE_mppt[i]
                                    [:, 2], label=f'{a[0]}', color=color[(p - 1) * 2])
                        ax.set_xlabel('Time [s]')
                        ax.set_ylabel('PCE [%]')
                        ax.set_xlim([0, np.amax([np.amax(PCE_mppt[i][:, 0])
                                                 for i in range(len(PCE_mppt))])])
                        ax.set_ylim(
                            [0, np.amax([np.amax(PCE_mppt[i][:, 1] * PCE_mppt[i][:, 2]) for i in range(len(PCE_mppt))]) * (1.2)])
                        ax.grid(True, linestyle='--')
                        ax.tick_params(direction='in')
                        lg = ax.legend(loc=2, bbox_to_anchor=(
                            1.05, 1), borderaxespad=0.)
                        ax.set_title(
                            f'Dev.{name}  {light/100} Sun Scan {scan}')
                        plt.savefig(
                            join(result_dir, f'{name}_{light/100}Sun_mppt.png'), bbox_extra_artists=(lg,), bbox_inches='tight', dpi=800)
                        plt.close()
                    # ------------------------------------
                    #         JV_Log.txt creation
                    # ------------------------------------
                    for i in range(len(sp)):
                        measurement_parameters = np.append([re.search(r'^(.+?)_', hed[2 * i]).group(1),
                                                            'p{}'.format(re.search(r'_p([0-8])_',
                                                                                   hed[2 * i]).group(1)), sp[i], scan,
                                                            re.search(r'.+ - (.+)\[[0-8]\]',
                                                                      device[-1]).group(1), '{:.2f}'.format(light)],
                                                           JV_parameters(JV[i], light))
                        measurement_parameters = np.append(
                            measurement_parameters, [swp_spd[i], ms_st_date[i]])
                        JVLog.append(measurement_parameters)

        print('{}    --Completed--'.format(name))


# --------------------------
# Dataframe creation
# ADDED v.2.7 - Sweep speed and measurement start date parameters
# --------------------------

    col_names = ['Device', 'Pixel', 'Sweep Direction', 'Scan number', 'Variable', 'Pin[mW/cm2]',
                 'Voc[V]', 'Jsc[mA/cm2]', 'FF[%]', 'PCE[%]', 'Sweep Speed', 'Measurement Start Date']
    df0 = pd.DataFrame(JVLog, columns=col_names)
    # df0.sort_values(['Variable', 'Device'], inplace=True)
    if 'only_sweep' in extra_params:
        df0 = df0[df0['Sweep Direction'] == extra_params['only_sweep']]

    df0 = df0.astype(
        {'Scan number': 'int64', 'Pin[mW/cm2]': 'float64', 'Voc[V]': 'float64', 'Jsc[mA/cm2]': 'float64', 'FF[%]': 'float64', 'PCE[%]': 'float64'})

# ------------------------------------
# ADDED v2.5 - request user to input device structures
# ADDED v2.6 - variable in device name and widened stack field
# ------------------------------------

    print(root_win)
    list_of_devices = df0[['Device', 'Variable']
                          ].drop_duplicates(subset='Device')
    batch_params = ask_batch_structures(root_win, list_of_devices)
    # print(batch_params.architectures)
    # print(batch_params.stack_df)
# ------------------------------------
#         Boxplots and Yield
# ------------------------------------

    df = df0.copy()
    # Error!!! the line below takes the max of Fw and Rv, but they can be max in different lines!
    df.loc[:, 'maxpce'] = df[['Device', 'Pixel', 'Sweep Direction',  'Pin[mW/cm2]', 'PCE[%]']].groupby(
        ['Device', 'Pixel', 'Sweep Direction', 'Pin[mW/cm2]'])['PCE[%]'].transform(max)
    df = df[df['PCE[%]'] == df['maxpce']]
    del df['maxpce']
    # del df['maxpce']
    df.sort_values(['Variable', 'Device'], inplace=True)
    # df = df0.loc[pd.to_numeric(df0['PCE[%]'], errors='coerce').notnull(), :]
    variable_yeld = df.groupby('Variable').size().div(
        df0[df0['Scan number'] == 0].groupby('Variable').size()).mul(100)
    device_yeld = df.groupby('Device').size().div(
        df0[df0['Scan number'] == 0].groupby('Device').size()).mul(100)
    variable_yeld.plot.bar()
    plt.ylabel('Yield [%]')
    plt.title('Yield per variable')
    plt.tick_params(direction='in')
    lg = plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(
        join(result_dir, 'Yield_per_variable.png'),  dpi=600)
    plt.close()
    device_yeld.plot.bar()
    plt.ylabel('Yield [%]')
    plt.title('Yield per device')
    plt.tick_params(direction='in')
    lg = plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(
        join(result_dir, 'Yield_per_device.png'),  dpi=600)
    plt.close()
    # plt.show()
    # triggers an error due to the conversion to numeric type of these columns
    # df.iloc[:, -4:] = df.iloc[:, -4:].apply(pd.to_numeric)
    directions = df['Sweep Direction'].unique()
    # print(directions)
    if directions.size > 0:
        df.loc[:, 'Device+Variable'] = df.loc[:, ['Variable', 'Device']
                                              ].agg(' -- '.join, axis=1)  # to correct
        # df = df[df['PCE[%]']>=2.5]
        # print(JVLog)
        hue = 'Sweep Direction' if len(directions) > 1 else None
        col = ['palegreen', 'peachpuff'] if len(
            directions) > 1 else 'palegreen' if directions[0] == 'Rv' else 'peachpuff'
        for item in ['Voc[V]', 'Jsc[mA/cm2]', 'FF[%]',
                     'PCE[%]']:

            ax = sns.boxplot(x='Variable', y=item,
                             hue=hue, data=df, palette=col,  showfliers=False)
            ax = sns.swarmplot(x='Variable', y=item,
                               hue=hue, data=df, palette="Set2", dodge=True, size=5)
            plt.tick_params(direction='in')
            lg = plt.xticks(rotation=30, ha='right')
            plt.legend(loc='upper left',
                       bbox_to_anchor=(-0.2, 1), borderaxespad=0.)
            plt.tight_layout()
            # plt.show()
            plt.savefig(join(result_dir, 'Boxplot_{}_variables.png'.format(
                item.split('[')[0])),  dpi=600)
            plt.close()
            # plt.show()
            ax = sns.boxplot(x='Device+Variable', y=item,
                             hue=hue, data=df, palette=col,  showfliers=False)
            ax = sns.swarmplot(x='Device+Variable', y=item,
                               hue=hue, data=df, palette="Set2", dodge=True, size=5)
            plt.tick_params(direction='in')
            lg = plt.xticks(rotation=30, ha='right')
            plt.legend(loc='upper left',
                       bbox_to_anchor=(-0.2, 1), borderaxespad=0.)
            plt.tight_layout()

            figure = plt.gcf()
            figure.set_size_inches(10, 5)
            # plt.savefig("sample.png", dpi=100)

            # plt.show()
            plt.savefig(
                join(result_dir, 'Boxplot_{}_devices.png'.format(item.split('[')[0])),  dpi=600)
            plt.close()
        del df['Device+Variable']
        np.savetxt(join(result_dir, f'{experiment}_Log.txt'), np.vstack([np.array(
            col_names), JVLog]), fmt='%s', newline='\n', delimiter='\t')
        df_Rv = df0[df0['Sweep Direction'] == 'Rv'].reset_index()
        df_Fw = df0[df0['Sweep Direction'] == 'Fw'].reset_index()
        del df_Rv['Sweep Direction']
        del df_Fw['Sweep Direction']
        df_JV = df_Rv.merge(df_Fw, how='outer', on=[
            'Device', 'Pixel', 'Scan number', 'Variable', 'Pin[mW/cm2]', 'Sweep Speed', 'Measurement Start Date'], suffixes=[' Rv', ' Fw'])
        del df_JV['index Fw']
        del df_JV['index Rv']
        df_JV = df_JV.astype(
            {'Pin[mW/cm2]': 'float64', 'Scan number': 'int64'})

        df_mpp = pd.DataFrame(MPPTLog,
                              columns=['Device', 'Variable', 'Pixel', 'Scan number',  'Pin[mW/cm2]', 'Time_MPP [s]', 'Stabilized_MPP [%]', 'Measurement Start Date'])

        df_mpp[['Pin[mW/cm2]', 'Scan number', 'Stabilized_MPP [%]']] = df_mpp[['Pin[mW/cm2]', 'Scan number', 'Stabilized_MPP [%]']].astype(
            {'Pin[mW/cm2]': 'float64', 'Scan number': 'float64', 'Stabilized_MPP [%]': 'float64'}, copy=False)
        df_mpp['Scan number'] = df_mpp['Scan number'].astype(
            {'Scan number': 'int64'}, copy=False)
        # df_mpp = df_mpp[df_mpp['Stabilized_MPP [%]']>=2.5]
        df_mpp.sort_values(['Variable', 'Device'], inplace=True)
        try:
            ax = sns.boxplot(x='Variable', y='Stabilized_MPP [%]',
                             data=df_mpp, palette=col,  showfliers=False)
            ax = sns.swarmplot(x='Variable', y='Stabilized_MPP [%]',
                               data=df_mpp, palette="Set2", dodge=True, size=5)
            plt.tick_params(direction='in')
            lg = plt.xticks(rotation=30, ha='right')
            # plt.legend(loc='upper left',
            #            bbox_to_anchor=(-0.2, 1), borderaxespad=0.)
            plt.tight_layout()
            # plt.show()
            plt.savefig(
                join(result_dir, 'Boxplot_stabilizedPCE.png'),  dpi=600)
            plt.close()
        except ValueError as e:
            pass
        dff = df_JV.merge(df_mpp, how='outer', on=[
            'Device', 'Pixel', 'Scan number', 'Variable', 'Pin[mW/cm2]', 'Measurement Start Date'])
        dff.insert(0, 'Batch', experiment)
        dff.insert(3, 'Architecture', 'nip')
        for key, value in extra_params.items():
            if key == 'user':
                dff.insert(3, 'User', value)
        dff = dff.merge(batch_params.stack_df, how='outer', on='Device')
        stack_col = dff.pop('Cell Stack')
        dff.insert(5, 'Cell Stack', stack_col)
        dff.insert(19, 'Preconditioning Protocol', preconditioning_protocol)
        dff.sort_values(['Variable', 'Device', 'Pixel',
                         'Scan number'], inplace=True)
        # print(dff)
        dff = dff[['Batch', 'Device', 'Pixel', 'Variable', 'User', 'Cell Stack', 'Architecture',  'Scan number',
                   'Pin[mW/cm2]', 'Voc[V] Rv', 'Jsc[mA/cm2] Rv', 'FF[%] Rv', 'PCE[%] Rv', 'Voc[V] Fw',
                   'Jsc[mA/cm2] Fw', 'FF[%] Fw', 'PCE[%] Fw', 'Time_MPP [s]', 'Stabilized_MPP [%]', 'Preconditioning Protocol', 'Sweep Speed', 'Measurement Start Date']]
        dff.to_csv(join(result_dir, f'{experiment}_Log.csv'), index=False)

        # save_ppt_plot(result_dir)
        # columns = [Batch,Device,Pixel,User,Architecture,Cell Stack,Scan number,Variable,Pin[mW/cm2],Voc[V] Rv,Jsc[mA/cm2] Rv,FF[%] Rv,PCE[%] Rv,Voc[V] Fw,Jsc[mA/cm2] Fw	FF[%] Fw	PCE[%] Fw	Time_MPP [s]	Stabilized_MPP [%]

    else:
        return 'Sorry, no working device in this batch.....boxplots skipped'
    print(f'{experiment} Analysis Completed')
    return f'{experiment} Analysis Completed'


def run_importer(event):
    status = Importer(entry_path.get(), user=entry_name.get())
    msgb.showinfo('Analysis', status)


def Quit_app(event):
    closureCheck = msgb.askyesno(title='Exit',
                                 message='Quit application?',
                                 icon='question')
    if closureCheck == True:
        root_win.destroy()


if __name__ == '__main__':
    root_win = tk.Tk()
    root_win.title('CP - Lithos Lite Batch Analysis')

    menu = tk.Menu(root_win)
    root_win.config(menu=menu)

    subMenu = tk.Menu(menu)
    menu.add_cascade(label='File', menu=subMenu)
    subMenu.add_command(label='Quit', command=root_win.destroy)
    root_win.resizable(False, False)

    user = tk.StringVar()
    # layer_stack = tk.StringVar()
    path = tk.StringVar()

    name = tk.Label(root_win, text='Username:')
    # stack = tk.Label(root_win, text='Device Stack:')
    filepath = tk.Label(root_win, text='Analysis Path:')

    sample_label = tk.Label(
        root_win, text=r'eg.    C:\Users\cperini8\OD\Work\Post-Doc\Projects\Device Optimization\JVs\Batch 3', foreground='gray')
    # sample_stack = tk.Label(
    #     root_win, text=r'eg.  Glass | FTO | TiO2-c | TiO2-mp | Cs0.05FA0.79MA0.16PbI2.49Br0.51 | spiro-OMeTAD | Au', foreground='gray')
    sample_user = tk.Label(
        root_win, text=r'eg.  Franklin', foreground='gray')

    entry_name = tk.Entry(root_win, textvariable=user, width=10)
    # entry_stack = tk.Entry(root_win, textvariable=layer_stack, width=20)
    # entry_stack.insert(0,
    #                    r'Glass | FTO | TiO2-c | TiO2-mp | Cs0.05FA0.79MA0.16PbI2.49Br0.51 | spiro-OMeTAD | Au')
    entry_path = tk.Entry(root_win, textvariable=path, width=20)

    ok = tk.Button(root_win, text='OK', padx=15)
    cancel = tk.Button(root_win, text='Quit')

    ok.grid(row=4, column=1, padx=4, pady=3, sticky='se')

    cancel.grid(row=4, column=2, padx=2, pady=3, sticky='sw')
    sample_label.grid(row=3, column=1, columnspan=2,
                      padx=10, sticky='w')
    # sample_stack.grid(row=3, column=1, columnspan=2, padx=10, sticky='w')
    sample_user.grid(row=1, column=1, columnspan=2,
                     padx=10, sticky='w')
    name.grid(row=0, column=0, sticky='ne')
    # stack.grid(row=2, column=0, sticky='ne')
    filepath.grid(row=2, column=0, sticky='ne')
    entry_name.grid(row=0, column=1,
                    sticky='we', padx=10, pady=1)
    entry_name.focus_set()
    # entry_stack.grid(row=2, column=1, columnspan=2,
    #                  sticky='we', padx=10, pady=1)
    entry_path.grid(row=2, column=1, columnspan=2,
                    sticky='we', padx=10, pady=(5, 1))

    # cancel.bind('<Button-1>', )
    center(root_win)
    SetWinActive(root_win)
    ok.bind('<Button-1>', run_importer)
    root_win.bind('<Return>', run_importer)
    cancel.bind('<Button-1>', Quit_app)
    root_win.mainloop()

    # path = r'Insert path Here'
    # path = r'C:\Users\cperini8\OD\Work\Post-Doc\Projects\Surface passivation\Data\JV data'
    # Importer(path, user='Carlo')
    # If you want to run the analysis on multiple batches at the same time uncomment the lines below and comment the two above.
    # path = r'C:\Users\cperini8\OD\Work\Post-Doc\Projects\Surface passivation\Data\JV data'
    # experiments = [exp for exp in os.listdir(path) if isdir(
    #     join(path, exp)) and exp != 'Analysed']
    # for exp in experiments:
    #     try:
    #         print(Importer(join(path, exp),  user='Carlo'))
    #     except FileNotFoundError as e:
    #         print(
    #             f'-----------------------------{exp} Not Found--------------------------------')
