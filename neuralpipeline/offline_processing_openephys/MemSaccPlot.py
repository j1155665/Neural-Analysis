import os
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

class NeuralDataLoader:
    def __init__(self, base_dir=None, electrode_id = None):
        if base_dir is None:
            try:
                base_dir = self.select_folder()
            except:
                raise RuntimeError("No base_dir provided and GUI folder picker failed in Jupyter. Please pass a folder path.")
        self.base_dir = base_dir
        print(f"Selected base directory: {self.base_dir}")

        self.continuous_data = None
        self.continuous_timestamps = None
        self.ttl_data = None
        self.ttl_timestamps = None
        self.channel_positions = None

        self.load_all(electrode_id)

    def select_folder(self):
        """Open a dialog to select a folder"""
        root = Tk()
        root.withdraw()  # Hide main window
        folder = filedialog.askdirectory(title="Select experiment folder")
        if not folder:
            raise ValueError("No folder selected")
        return folder

    def load_continuous_data(self, num_channels = 384):
        """Load continuous.dat and timestamps.npy"""
        cont_dir = os.path.join(self.base_dir, "continuous", "Neuropix-PXI-100.ProbeA-AP")
        cont_file = os.path.join(cont_dir, "continuous.dat")
        ts_file = os.path.join(cont_dir, "timestamps.npy")

        if not (os.path.exists(cont_file) and os.path.exists(ts_file)):
            raise FileNotFoundError("Continuous data or timestamps not found")

        print("Loading continuous data...")
        self.continuous_data = np.fromfile(cont_file, dtype=np.int16).reshape((-1, num_channels))
        self.continuous_timestamps = np.load(ts_file)

    def load_spike_data(self, electrode_id):
        """
        Load spike timestamps for a specific electrode.

        """
        spike_dir = os.path.join(
            self.base_dir,
            "spikes",
            "Spike_Detector-110.ProbeA-AP",
            f"Electrode {electrode_id}"
        )
        print(spike_dir)
        timestamps_file = os.path.join(spike_dir, "timestamps.npy")
        cluster_file = os.path.join(spike_dir, "clusters.npy")

        if not os.path.exists(timestamps_file):
            raise FileNotFoundError(f"No spike timestamps found for Electrode {electrode_id}")

        print(f"Loading spikes from Electrode {electrode_id}...")
        self.spike_times = np.load(timestamps_file)
        self.cluster_id = np.load(cluster_file)
       

    def load_ttl_data(self):
        """Load TTL full_words.npy and timestamps.npy"""
        ttl_dir = os.path.join(self.base_dir, "events", "NI-DAQmx-107.PXIe-6341", "TTL")
        ttl_file = os.path.join(ttl_dir, "full_words.npy")

        ts_file = os.path.join(ttl_dir, "timestamps.npy")

        if not (os.path.exists(ttl_file) and os.path.exists(ts_file)):
            raise FileNotFoundError("TTL data or timestamps not found")

        print("Loading TTL data...")
        self.ttl_data = np.load(ttl_file)
        self.ttl_timestamps = np.load(ts_file)

    def load_probe_config(self):
        """Parse settings.xml to get channel positions as a matrix"""
        parent_dir = os.path.dirname(os.path.dirname(self.base_dir))
        settings_file = os.path.join(parent_dir, "settings.xml")
        if not os.path.exists(settings_file):
            raise FileNotFoundError(f"settings.xml not found in {parent_dir}")
        
        print("Parsing probe configuration...")
        tree = ET.parse(settings_file)
        root = tree.getroot()
        probe_node = root.find(".//NP_PROBE")

        xpos_node = probe_node.find("ELECTRODE_XPOS")
        ypos_node = probe_node.find("ELECTRODE_YPOS")

        xpos = {int(k[2:]): int(v) for k, v in xpos_node.attrib.items() if k.startswith("CH")}
        ypos = {int(k[2:]): int(v) for k, v in ypos_node.attrib.items() if k.startswith("CH")}

        ch_list = sorted(set(xpos) | set(ypos))
        pos_matrix = np.array([[ch, xpos.get(ch, np.nan), ypos.get(ch, np.nan)] for ch in ch_list])

        self.channel_positions = pos_matrix
        print("Channel positions loaded as matrix.")

    def load_all(self, electrode_id):
        """Load all data components"""
        if electrode_id is None:
            self.load_continuous_data()
        else:
            self.load_spike_data(electrode_id)

        self.load_ttl_data()
        self.load_probe_config()

    def get_spike_times(self, channel_idx, threshold=-50, refractory_ms=1):
        """
        Quick spike detection on one channel using threshold crossing + refractory window.
        Returns spike times in seconds.
        """
        fs = 1 / np.mean(np.diff(self.continuous_timestamps))  # Sampling frequency in Hz
        signal = self.continuous_data[:, channel_idx]

        # Detect negative threshold crossings
        crossings = np.where((signal[:-1] > threshold) & (signal[1:] <= threshold))[0] + 1

        refractory_samples = int(refractory_ms * fs / 1000)
        spike_times = []
        last_spike = -np.inf

        for t in crossings:
            if t - last_spike > refractory_samples:
                spike_times.append(t)
                last_spike = t

        spike_times = np.array(spike_times)
        # Convert sample indices to time in seconds using timestamps
        spike_times_sec = self.continuous_timestamps[spike_times]
        self.spike_times = spike_times_sec
        print(f"Detected {len(spike_times_sec)} spikes on channel {channel_idx}")
        return self.spike_times
 
    
    def parse_ttl_events(self):
        """
        Parse TTL signals to extract trial events and mark good trials.
        """
        # Event codes
        TRIAL, FIX, FIXATION, STIMON, STIMOFF, SACC, REWARD, BREAKFIX = 1, 2, 3, 4, 4, 6, 9, 10
        event_codes = [TRIAL, FIX, FIXATION, STIMON, STIMOFF, SACC, REWARD, BREAKFIX]
        event_names = ['TRIAL', 'FIX', 'FIXATION', 'STIMON', 'STIMOFF', 'SACC', 'REWARD', 'BREAKFIX']

        # Clean up TTL values
        ttl_values = np.where(self.ttl_data >= 256, self.ttl_data - 256, self.ttl_data)
        ttl_times = self.ttl_timestamps

        # Find trial start indices (TRIAL events)
        trial_indices = np.where(ttl_values == TRIAL)[0]
        fix_indices = np.where(ttl_values == FIX)[0]
        idx_diffs = np.diff(trial_indices) > 10
        self.trial_indices = np.concatenate(([trial_indices[0]], trial_indices[1:][idx_diffs]))

        # Initialize event data dictionary
        event_data = {name: [] for name in event_names}
        event_data['condition'] = []
        event_data['goodtrial'] = []

        # Loop through each trial
        for i, trial_idx in enumerate(trial_indices):
            if i != 0:
                trial_start = trial_indices[i-1]
            else:
                trial_start = fix_indices[0] if fix_indices.size > 0 else 0

            # Get TTLs and times for this trial
            trial_ttls = ttl_values[trial_start:trial_idx]
            trial_times = ttl_times[trial_start:trial_idx]

            # Store timestamps for each event
            trial_events = {name: np.nan for name in event_names}
            for name, code in zip(event_names, event_codes):
                matching = np.where(trial_ttls == code)[0]
                if matching.size > 0:
                    if name == 'STIMOFF':
                        trial_events[name] = trial_times[matching[-1]]
                    else:
                        trial_events[name] = trial_times[matching[0]]

                event_data[name].append(trial_events[name])

            # Condition code (e.g., 101–120 minus offset)
            cond_code = np.nan
            cond_idx = np.where(trial_ttls > 100)[0]
            if cond_idx.size > 0:
                cond_code = trial_ttls[cond_idx[0]] - 100
            event_data['condition'].append(cond_code)

            # Mark good trial
            breakfix = trial_events['BREAKFIX']
            if np.isnan(breakfix):
                event_data['goodtrial'].append(1)
            else:
                event_data['goodtrial'].append(0)

        # Convert lists to numpy arrays
        for key in event_data:
            event_data[key] = np.array(event_data[key])

        self.event_data = event_data
        print(f"\nParsed {len(trial_indices)} trials, {np.sum(event_data['goodtrial'])} good trials.")
        return  self.event_data

    def avg_firing_rate(self, align_info):

        event_name = align_info['event']
        pre = align_info['pre']
        post = align_info['post']

        conditions = self.event_data['condition']
        align_event = self.event_data[event_name]
        good_trials = self.event_data['goodtrial']
        spike_times = self.spike_times

        unique_conditions = np.unique(conditions[~np.isnan(conditions)]).astype(int)
        avg_firing_rates = {}

        for cond in unique_conditions:
            # Find indices of trials with this condition AND marked as good
            trial_inds = np.where((conditions == cond) & (good_trials == 1))[0]
            rates = []
            for idx in trial_inds:
                start = align_event[idx] + pre
                end = align_event[idx] + post

                if end <= start:
                    print('check your time')
                    continue


                # Skip if any event missing (NaN) or invalid window
                if np.isnan(start) or np.isnan(end) or end <= start:
                    continue

                # Count spikes in interval
                spikes_in_interval = np.sum((spike_times >= start) & (spike_times <= end))

                duration = end - start
                rate = spikes_in_interval / duration  # spikes per unit time
                rates.append(rate)

            if rates:
                avg_firing_rates[cond] = np.mean(rates)
            else:
                avg_firing_rates[cond] = np.nan  # No good trials for this condition

        return avg_firing_rates
    
    def plot_avg_firing_rate_heatmap(self, align_info, avg_firing_rates, eccs, angles, vmin=0, vmax=1.5):


        event_name = align_info['event']
        rates = np.array(list(avg_firing_rates.values()))

        if len(rates) == 0 or np.all(np.isnan(rates)):
            print("No valid firing rates to plot.")
            return

        z_scores = (rates - np.nanmean(rates)) / np.nanstd(rates)

        x = eccs * np.cos(np.deg2rad(angles))
        y = eccs * np.sin(np.deg2rad(angles))

        background_size = 12

        plt.figure(figsize=(6, 6))
        plt.xlim(-background_size, background_size)
        plt.ylim(-background_size, background_size)
        plt.gca().set_facecolor(plt.cm.viridis(0.0))

        scatter = plt.scatter(
            x, y,
            c=z_scores,
            cmap='viridis',
            s=100,
            edgecolor=None,
            marker='s',
            vmin=vmin,
            vmax=vmax
        )

        plt.colorbar(scatter, label='Z-scored Firing Rate')
        plt.xlabel('X position (deg)')
        plt.ylabel('Y position (deg)')
        plt.title(f"Memory Saccade Heatmap, {event_name}")
        plt.grid(False)
        plt.axis('equal')
        plt.show()







if __name__ == "__main__":
    loader = NeuralDataLoader()
    print("\n--- Data Loaded ---")
    print(f"Continuous data shape: {loader.continuous_data.shape}")
    print(f"TTL data shape: {loader.ttl_data.shape}")
    print(f"Channel positions shape: {loader.channel_positions.shape}")
