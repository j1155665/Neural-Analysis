import os
import re
import numpy as np
import pandas as pd
import scipy.io
from scipy.io import loadmat, savemat
import types 
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

class MergeRecordingFile:
    def __init__(self, directory, subject, date):
        self.directory = directory
        self.subject = subject
        self.date = date
        self.filedate = f"{date[:4]}-{date[4:6]}-{date[6:]}"
        self.pattern = re.compile(f"{self.filedate}_\\d{{2}}-\\d{{2}}-\\d{{2}}")

        self.matching_files = self.find_matching_files()
        self.matching_data = self.find_matching_data()
        self.time_adjustment = self.check_time_adjustment()
        self.ttl_unique = self.check_unique_block()

    def find_matching_files(self):
        """Finds recording session folders based on date pattern."""
        files_path = os.path.join(self.directory, f"{self.date}")
        files = os.listdir(files_path)
        matched_files = [f for f in files if self.pattern.match(f)]
        if not matched_files:
            raise ValueError(f"No files found for subject {self.subject} on {self.date}.")
        return matched_files

    def find_matching_data(self):
        """Finds the paths to recording data inside matched session folders."""
        matching_data = []
        for file in self.matching_files:
            data_directory = os.path.join(self.directory, self.date, file, "Record Node 101")
            if not os.path.exists(data_directory):
                continue
            for root, dirs, files in os.walk(data_directory):
                path_parts = root.split(os.sep)
                if len(path_parts) >= 2 and "recording" in path_parts[-1].lower():
                    matching_data.append(root)

        # Sort using experiment and recording number
        def extract_exp_and_rec(path):
            exp_match = re.search(r'experiment(\d+)', path, re.IGNORECASE)
            rec_match = re.search(r'recording(\d+)', path, re.IGNORECASE)
            exp_num = int(exp_match.group(1)) if exp_match else -1
            rec_num = int(rec_match.group(1)) if rec_match else -1
            return (exp_num, rec_num)

        matching_data.sort(key=extract_exp_and_rec)
        return matching_data

    def get_electrode_configuration(self, filepath, attribute="electrodeConfigurationPreset"):
        """Extracts the electrode configuration from settings.xml."""
        settings_path = os.path.join(filepath, 'settings.xml')
        if not os.path.exists(settings_path):
            return None
        try:
            tree = ET.parse(settings_path)
            root = tree.getroot()
            for elem in root.iter():
                if attribute in elem.attrib:
                    return elem.attrib[attribute]
        except ET.ParseError:
            return None
        return None
    
    def extract_channel_positions(self):
        """Get the configuration of the probe as ProbeConfig.txt"""

        xml_file = os.path.join(self.directory, self.date , self.matching_files[0], 'Record Node 101', 'settings.xml')
        tree = ET.parse(xml_file)
        root = tree.getroot()

        probe_node = root.find(".//NP_PROBE")

        xpos_node = probe_node.find("ELECTRODE_XPOS")
        ypos_node = probe_node.find("ELECTRODE_YPOS")

        xpos = {int(k[2:]): int(v) for k, v in xpos_node.attrib.items() if k.startswith("CH")}
        ypos = {int(k[2:]): int(v) for k, v in ypos_node.attrib.items() if k.startswith("CH")}

        ch_positions = [(ch, xpos.get(ch, None), ypos.get(ch, None)) for ch in sorted(set(xpos) | set(ypos))]

        x_pos = []
        y_pos = []
        for ch, x, y in ch_positions:
            x_pos.append(x)
            y_pos.append(y)

        file_path = os.path.join(self.directory, self.date ,"ProbeConfig.txt")
        with open(file_path, 'w') as f:
            f.write("X-pos\n")
            f.write(f"[{', '.join(str(x) for x in x_pos)}]\n\n")
            f.write("Y-pos\n")
            f.write(f"[{', '.join(str(y) for y in y_pos)}]\n\n")

        print(f"Channel configuration saved to '{file_path}'.")

        return

    def check_electrode_consistency(self):
        """Checks if electrode configuration is consistent across all recordings."""
        consistent_preset = None
        for file in self.matching_files:
            recording_dir = os.path.join(self.directory, file, 'Record Node 101')
            preset = self.get_electrode_configuration(recording_dir)
            if preset is None:
                continue
            if consistent_preset is None:
                consistent_preset = preset
            elif consistent_preset != preset:
                raise ValueError(f"Inconsistent electrodeConfigurationPreset in {file}: {preset} (expected {consistent_preset}).")
        return consistent_preset
    
    def check_time_adjustment(self):
        """Check if we need to adjust the timestamp across trials."""
        start_time = []
        stop_time = []
        delta_time_sum = 0
        delta_time = []
        time_adjustment = []
    
        for i, file in enumerate(self.matching_data):
            # Load AP timestamps and add block info
            apt_filepath = os.path.join(file, 'continuous', 'Neuropix-PXI-100.ProbeA-AP', 'timestamps.npy')
            if os.path.exists(apt_filepath):

                ap_timestamps = np.load(apt_filepath)
                start_time.append(ap_timestamps[0])
                stop_time.append(ap_timestamps[-1])
                delta_time.append(ap_timestamps[-1] - ap_timestamps[0] + delta_time_sum)
                delta_time_sum += (ap_timestamps[-1] - ap_timestamps[0])
            else:
                print(f"Warning: AP timestamps file not found at {file}")

        time_adjustment.append(-start_time[0])
        for i, diff in enumerate(delta_time[0:-1]):
            time_adjustment.append(-start_time[i+1] + diff + 1)

        return time_adjustment
    
    def check_unique_block(self):
        """Check the unique number (time) for the file name"""

        ttl_unique_list = []
        for i, file in enumerate(self.matching_data):

            ttl_filepath = os.path.join(file, 'events', 'NI-DAQmx-104.PXIe-6341', 'TTL', 'full_words.npy')

            # Load TTL data and add block markers
            if os.path.exists(ttl_filepath):

                ttl_data = np.load(ttl_filepath)

                ttl_data[ttl_data >= 256] -= 256
                ttl_data = ttl_data[(ttl_data > 0) & (ttl_data < 256)]
                ordered_unique = np.unique(ttl_data, return_index=True)
                ordered_unique = ordered_unique[0][np.argsort(ordered_unique[1])]
                
                ttl_unique = ordered_unique[1] * 100 + ordered_unique[2]
                ttl_unique_list.append(ttl_unique)

            else:
                print(f"Warning: TTL file not found at {file}")

        print("valid PDS filetime:", [int(x) for x in ttl_unique_list])

        return ttl_unique_list

    def merge_ap_data(self, num_channels=385):
        """Merges AP (action potential) data from multiple recording files and includes block information."""
        output_path = os.path.join(self.directory, f"{self.date}", f"{self.subject}{self.date}dots3DMP_AP.dat")
        aptime_out_path = os.path.join(self.directory, f"{self.date}", f"{self.subject}{self.date}dots3DMP_APtimestamps.npy")
        aptime_block_path = os.path.join(self.directory, f"{self.date}", f"{self.subject}{self.date}dots3DMP_APblocks.npy")

        # Lists to store AP timestamps and block markers
        ap_timestamps_list = []
        ap_block_list = []
        
        final_original_size = 0

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        for i, file in enumerate(self.matching_data):
            print(f"Processing file {i+1}/{len(self.matching_data)}: {file}")
            
            # Path to continuous data
            filepath = os.path.join(file, 'continuous', 'Neuropix-PXI-100.ProbeA-AP', 'continuous.dat')

            # Check if the file exists
            if os.path.exists(filepath):
                size_in_gb = os.path.getsize(filepath) / (1024 ** 3)
            else:
                print(f"File not found at {filepath}")
                continue 
            
            # Load data using memmap
            data = np.memmap(filepath, dtype='int16')
            data_size = data.size * 2 / (1024 ** 3)  # Size in GB
            
            # Check if the loaded data size matches the file size
            if abs(size_in_gb - data_size) < 1e-6:  
                print(f"Loaded data size matches: {data_size:.2f} GB")
            else:
                print(f"Size mismatch: {size_in_gb:.2f} GB (file) vs {data_size:.2f} GB (loaded). Stopping merging.")
                break
            
            # Accumulate total original size
            final_original_size += data_size
            
            # Reshape the data
            data = np.reshape(data, (data.size // num_channels, num_channels))

            # Determine the mode for file opening (write or append)
            mode = 'wb' if i == 0 else 'ab'
            with open(output_path, mode) as f:
                start_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
                data.tofile(f)  # Write or append data
                end_size = os.path.getsize(output_path)
                print(f"Merged file size: {end_size / (1024 ** 3):.2f} GB")

            # Load AP timestamps and add block info
            apt_filepath = os.path.join(file, 'continuous', 'Neuropix-PXI-100.ProbeA-AP', 'timestamps.npy')
            if os.path.exists(apt_filepath):
                ap_timestamps = np.load(apt_filepath)
                ap_timestamps += np.ones(ap_timestamps.shape) * self.time_adjustment[i]
                ap_timestamps_list.append(ap_timestamps)

                ap_block = np.ones(ap_timestamps.shape) * self.ttl_unique[i]  # Add block number
                ap_block_list.append(ap_block)
            else:
                print(f"Warning: AP timestamps file not found at {file}")
        
        # Merge AP timestamps and save
        if ap_timestamps_list:
            merged_ap_timestamps = np.concatenate(ap_timestamps_list)
            np.save(aptime_out_path, merged_ap_timestamps)
            np.save(aptime_block_path, np.concatenate(ap_block_list))
            print(f"Merged AP timestamps saved")
        else:
            print("No AP timestamps to merge.")  
        
        # Final file size and size verification
        final_size = os.path.getsize(output_path) / (1024 ** 3)  # Size in GB
        print(f"Final merged file size: {final_size:.2f} GB")
        
        if abs(final_size - final_original_size) < 1e-6:  
            print("Final size matches the total original size.")
        else:
            print(f"WARNING: Size mismatch! Difference: {final_size - final_original_size:.2f} GB")

        return 


    def merge_ttl_data(self):
        """Merges TTL data and timestamps from multiple recordings, adds block markers, and saves the merged data."""
        ttl_out_path = os.path.join(self.directory, f"{self.date}", f"{self.subject}{self.date}dots3DMP_ttl.npy")
        ttl_block_path = os.path.join(self.directory, f"{self.date}", f"{self.subject}{self.date}dots3DMP_ttlblocks.npy")
        ttltime_out_path = os.path.join(self.directory, f"{self.date}", f"{self.subject}{self.date}dots3DMP_ttltimestamps.npy")

        ttl_list = []
        ttl_timestamps_list = []
        ttl_block_list = []

        os.makedirs(os.path.dirname(ttl_out_path), exist_ok=True)

        for i, file in enumerate(self.matching_data):
            # print(f"Processing file {i+1}/{len(self.matching_data)}: {file}")

            ttl_filepath = os.path.join(file, 'events', 'NI-DAQmx-104.PXIe-6341', 'TTL', 'full_words.npy')
            ttlt_filepath = os.path.join(file, 'events', 'NI-DAQmx-104.PXIe-6341', 'TTL', 'timestamps.npy')

            # Load TTL data and add block markers
            if os.path.exists(ttl_filepath):

                ttl_data = np.load(ttl_filepath)
                ttl_list.append(ttl_data)

                ttl_block = np.ones(ttl_data.shape)
                ttl_block *= self.ttl_unique[i]  # Add block number for each file
                ttl_block_list.append(ttl_block)

            else:
                print(f"Warning: TTL file not found at {file}")

            # Load TTL timestamps
            if os.path.exists(ttlt_filepath):
                ttl_timestamps = np.load(ttlt_filepath)
                ttl_timestamps += np.ones(ttl_timestamps.shape) * self.time_adjustment[i]
                ttl_timestamps_list.append(ttl_timestamps)
            else:
                print(f"Warning: TTL timestamps file not found at {file}")

        # Merge TTL data and save
        if ttl_list:
            merged_ttl = np.concatenate(ttl_list)
            np.save(ttl_out_path, merged_ttl)
            np.save(ttl_block_path, np.concatenate(ttl_block_list))
            print(f"Merged TTL saved")
        else:
            print("No TTL data to merge.")

        # Merge TTL timestamps and save
        if ttl_timestamps_list:
            merged_ttl_timestamps = np.concatenate(ttl_timestamps_list)
            np.save(ttltime_out_path, merged_ttl_timestamps)
            print(f"Merged TTL timestamps saved")
        else:
            print("No TTL timestamps to merge.")

        return 

    def merge_eye_data(self,num_channels = 4):
        """Merges eye data from multiple recording files and includes block information."""
        output_path = os.path.join(self.directory, f"{self.date}", f"{self.subject}{self.date}dots3DMP_eyeXY.dat")
        eyetime_out_path = os.path.join(self.directory, f"{self.date}", f"{self.subject}{self.date}dots3DMP_eyeXYtimestamps.npy")
        eyetime_block_path = os.path.join(self.directory, f"{self.date}", f"{self.subject}{self.date}dots3DMP_eyeXYblocks.npy")

        # Lists to store AP timestamps and block markers
        eye_timestamps_list = []
        eye_block_list = []

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        for i, file in enumerate(self.matching_data):
            print(f"Processing eyeXY file {i+1}/{len(self.matching_data)}: {file}")
            
            # Path to continuous data
            filepath = os.path.join(file, 'continuous', 'NI-DAQmx-104.PXIe-6341', 'continuous.dat')

            # Check if the file exists
            if not os.path.exists(filepath):
                print(f"File not found at {filepath}")
                continue 
            
            # Load data using memmap
            data = np.memmap(filepath, dtype='int16')
            
            # Reshape the data
            data = np.reshape(data, (data.size // num_channels, num_channels))
            print(data.shape)

            # Determine the mode for file opening (write or append)
            mode = 'wb' if i == 0 else 'ab'
            with open(output_path, mode) as f:
                data.tofile(f)  # Write or append data

            # Load eyeXY timestamps and add block info
            eyet_filepath = os.path.join(file, 'continuous', 'NI-DAQmx-104.PXIe-6341', 'timestamps.npy')
            if os.path.exists(eyet_filepath):
                eye_timestamps = np.load(eyet_filepath)
                eye_timestamps += np.ones(eye_timestamps.shape) * self.time_adjustment[i]
                eye_timestamps_list.append(eye_timestamps)

                eye_block = np.ones(eye_timestamps.shape) * self.ttl_unique[i]  # Add block number
                eye_block_list.append(eye_block)
            else:
                print(f"Warning: eyeXY timestamps file not found at {file}")
        
        # Merge AP timestamps and save
        if eye_timestamps_list:
            merged_eye_timestamps = np.concatenate(eye_timestamps_list)
            np.save(eyetime_out_path, merged_eye_timestamps)
            np.save(eyetime_block_path, np.concatenate(eye_block_list))
            print(f"Merged eyeXY timestamps saved")
        else:
            print("No eyeXY timestamps to merge.")  

        return 


class CreateEventStruct:
    def __init__(self, directory, subject, date):
        self.directory = directory
        self.subject = subject
        self.date = date
        self.data_path = os.path.join(directory, date)
        
        self.full_words = np.load(os.path.join(self.data_path, f"{subject}{date}dots3DMP_ttl.npy"))
        self.timestamps = np.load(os.path.join(self.data_path, f"{subject}{date}dots3DMP_ttltimestamps.npy"))
        self.ttl_blocks = np.load(os.path.join(self.data_path, f"{subject}{date}dots3DMP_ttlblocks.npy"))
        
        self.full_words[self.full_words >= 256] -= 256
        self.block_indices = self.ttl_blocks >= 0  # Include all blocks
        self.block_type = ['dots3DMP', 'dots3DMPtuning']
        self.event_data = None

    def filter_events(self):
        """ Filters event timestamps and words based on valid indices. """
        valid_indices = (self.full_words > 0) & (self.full_words != 13) & (self.full_words <= 256)

        if np.any(self.block_indices):
            self.filtered_full_words = self.full_words[valid_indices & self.block_indices]
            self.filtered_timestamps = self.timestamps[valid_indices & self.block_indices]
            self.filtered_ttl_blocks = self.ttl_blocks[valid_indices & self.block_indices]
        else:
            self.filtered_full_words = self.full_words[valid_indices]
            self.filtered_timestamps = self.timestamps[valid_indices]
            self.filtered_ttl_blocks = self.ttl_blocks[valid_indices]

    def load_info_data(self):
        """ Loads MATLAB `.mat` file and extracts necessary parameters. """
        mat_data = scipy.io.loadmat(os.path.join(self.data_path, f"{self.subject}{self.date}dots3DMP_info.mat"))
        info = mat_data['info']

        self.pldaps_filetimes = info['pldaps_filetimes'][0][0].flatten()
        self.par = info['par'][0, 0]

    def process_trials(self):
        """Processes trials, extracts event timestamps, and assigns behavioral indices."""
        self.data = self.filtered_full_words.copy()
        self.timestamps = self.filtered_timestamps.copy()

        # Define event codes
        TRIAL, FIX, FIXATION, STIMONOFF, SACC, TARGHOLD, POSTTARGHOLD,REWARD, BREAKFIX = 1, 2, 3, 5, 6, 7, 8, 9, 10

        # Define BLOCK and event indices
        total_blocks = np.unique(self.filtered_ttl_blocks)
        self.block_type = np.full(self.filtered_ttl_blocks.shape, '', dtype='O')

        for i, block in enumerate(total_blocks):

            if not np.isin(block, self.pldaps_filetimes).any():
                continue

            matched_idx = np.where(block == self.pldaps_filetimes)[0]
            corresponding_par = self.par[matched_idx[0]]

            block_indices = np.where(self.filtered_ttl_blocks == block)[0]
            data_block = self.data[block_indices[0]:block_indices[-1] + 1]
            
            self.block_type[block_indices[0]:block_indices[-1] + 1] = corresponding_par
            
            first_fix_idx = np.where(data_block == FIX)[0]

            if first_fix_idx.size == 0:
                print("no fixation found!")
                continue

            
        # Define TRIAL
        trial_indices = np.where(self.data == TRIAL)[0]
        idx_diffs = np.diff(trial_indices) > 10
        self.trial_indices = np.concatenate(([trial_indices[0]], trial_indices[1:][idx_diffs]))
        self.trial_type = np.array([b.strip() for b in self.block_type[self.trial_indices]])
        self.trial_size = len(self.trial_indices)


        # Define Behavior Indices
        event_indices = {
            'fpOn_idx': self.timestamps[np.where(self.data == FIX)[0]],
            'fixation_idx': self.timestamps[np.where(self.data == FIXATION)[0]],
            'stimOn_idx': self.timestamps[np.where(self.data == STIMONOFF)[0]],
            'stimOff_idx': self.timestamps[np.where(self.data == STIMONOFF)[-0]],
            'saccOnset_idx': self.timestamps[np.where(self.data == SACC)[0]],
            'targHold_idx': self.timestamps[np.where(self.data == TARGHOLD)[0]],
            'postTargHold_idx': self.timestamps[np.where(self.data == POSTTARGHOLD)[0]],
            'reward_idx': self.timestamps[np.where(self.data == REWARD)[0]],
            'breakFix_idx': self.timestamps[np.where(self.data == BREAKFIX)[0]]
        }

        # Initialize event_data dictionary
        self.event_data = {key: np.full(self.trial_size, np.nan) for key in [
            'fpOn', 'fixation', 'stimOn', 'stimOff',
            'saccOnset', 'targHold', 'postTargHold','reward', 'breakFix',
            'goodtrial', 'headingInd', 'modality', 'coherenceInd',
            'deltaInd', 'choice', 'correct', 'PDW','block'
        ]}

        vars_to_process = [
            ('deltaInd', 'PDW'),
            ('coherenceInd', 'correct'),
            ('modality', 'choice'),
            ('headingInd',)
        ]

        fpOn_times = event_indices['fpOn_idx']

        # Process each trial
        for j in range(self.trial_size):
            current_trial = self.timestamps[self.trial_indices[j]]
            previous_trial = self.timestamps[0] if j == 0 else self.timestamps[self.trial_indices[j - 1]]

            
            fpOn_valid = fpOn_times[(fpOn_times > previous_trial) & (fpOn_times < current_trial)]
            self.event_data['fpOn'][j] = fpOn_valid[0] if fpOn_valid.size > 0 else np.nan
            
            
            refined_start = self.event_data['fpOn'][j] if not np.isnan(self.event_data['fpOn'][j]) else previous_trial

            for key, idx in event_indices.items():
                if key == 'fpOn':
                    continue
                valid_idx = idx[(idx > refined_start) & (idx < current_trial)]
                if key == 'stimOff_idx':
                     self.event_data[key.replace("_idx", "")][j] = valid_idx[-1] if valid_idx.size > 0 else np.nan
                else:
                        self.event_data[key.replace("_idx", "")][j] = valid_idx[0] if valid_idx.size > 0 else np.nan

            breakfix = self.event_data['breakFix'][j]

            required_events = ['fixation', 'stimOn', 'stimOff']
            

            self.event_data['block'][j] = self.filtered_ttl_blocks[trial_indices[j]]

            if np.isnan(breakfix) and all(not np.isnan(self.event_data[event][j]) for event in required_events):
                self.event_data['goodtrial'][j] = 1
            else:
                self.event_data['goodtrial'][j] = 0

                for var_tuple in vars_to_process:
                    for var in var_tuple:
                        self.event_data[var][j] = np.nan
                continue

            if j == self.trial_size - 1:
                event_info = self.data[self.trial_indices[j]:]
            else:
                event_info = self.data[self.trial_indices[j]:self.trial_indices[j + 1]]

            event_info -= 70

            for idx, (var1, *var2) in enumerate(vars_to_process):
                if idx > 0:
                    event_info -= 10
                if "headingInd" in var1 or any("headingInd" in v for v in var2):
                    new_idx = (event_info <= 20) & (event_info >= 0)
                else:
                    new_idx = (event_info <= 10) & (event_info >= 0)

                new_info = event_info[new_idx]

                if new_info.size > 0:
                    self.event_data[var1][j] = new_info[0]
                    if var2:
                        self.event_data[var2[0]][j] = new_info[-1]
                        if self.event_data['breakFix'][j] == 0:
                            self.event_data[var2[0]][j] = np.nan
                else:
                    self.event_data[var1][j] = np.nan
                    if var2:
                        self.event_data[var2[0]][j] = np.nan
        

    def save_to_mat(self):
        """ Saves event data to a MATLAB `.mat` file and reports trial counts. """
        save_path = os.path.join(self.data_path, f"{self.subject}{self.date}dots3DMP.mat")
        par_type = ['dots3DMP', 'dots3DMPtuning']


        if os.path.exists(save_path):
            mat_contents = scipy.io.loadmat(save_path, simplify_cells=True)
            data = mat_contents['data']
        else:
            data = {}
            print("Generating a new file.")

        for p_type in par_type:
            trial_2_save = self.trial_type == p_type
            trial_data = {}
            
            for key in self.event_data:
                if p_type == 'dots3DMPtuning' and key in ['choice', 'correct', 'PDW']:
                    continue
                trial_data[key] = self.event_data[key][trial_2_save]

            total_trials = np.size(self.event_data['goodtrial'][trial_2_save])
            good_trials = int(np.sum(self.event_data['goodtrial'][trial_2_save]))

            data[p_type] = {'events': trial_data}
            
            print(f"{p_type}: Total Trials = {total_trials}, Good Trials = {good_trials}")

        scipy.io.savemat(save_path, {'data': data})
        print(f"Data saved at '{save_path}'")


class CreateUnitStruct:
    def __init__(self, directory, subject, date, kilosort):
        self.directory = directory
        self.subject = subject
        self.date = date
        self.kilosort = kilosort
        self.data_path = os.path.join(directory, date)
        self.kilosort_path = os.path.join(directory, date, kilosort)
        
        self.spike_time = np.load(os.path.join(self.kilosort_path, f"spike_times.npy"))
        self.spike_clusters = np.load(os.path.join(self.kilosort_path, f"spike_clusters.npy"))
        # self.channel_positions = np.load(os.path.join(self.kilosort_path, f"channel_positions.npy"))
        self.cluster_group = pd.read_csv(os.path.join(self.kilosort_path, "cluster_info.tsv"), sep='\t')

        self.timestamps = np.load(os.path.join(self.data_path, f"{subject}{date}dots3DMP_APtimestamps.npy"))
        self.AP_blocks = np.load(os.path.join(self.data_path, f"{subject}{date}dots3DMP_APblocks.npy"))

        self.pldaps_filetimes, self.par, self.par_type = self.check_trial_par()

    def check_trial_par(self):
        """Get the trial type"""
        print("Extracting unit from kilosort...")
        mat_file = os.path.join(self.data_path, f"{self.subject}{self.date}dots3DMP_info.mat")

        mat_contents = scipy.io.loadmat(mat_file, simplify_cells=True)
        info = mat_contents['info']
        par_raw = info['par']
        par = np.array([p.strip() if isinstance(p, str) else p for p in par_raw])
        par_type = np.unique(par)

        return info['pldaps_filetimes'], par, par_type 

    def build_unit_structure(self):
        uniq_spike_clusters = np.unique(self.spike_clusters)

        unit_struct = {
            par_type: {
                'depth': [], 'cluster_id': [], 'groups': [], 'spiketimes': []
            } for par_type in self.par_type
        }


        for par_type in self.par_type:

            trial_indices = [i for i, p in enumerate(self.par) if p == par_type]

            if len(trial_indices) == 0:
                print("No trial type", par_type)
                continue

            filetime_set = set(self.pldaps_filetimes[i] for i in trial_indices)
            blk_idx = np.isin(self.AP_blocks, list(filetime_set))

            spiketimes = []
            cluster_id = []
            depth = []
            group = []

   
            for i, cluster in enumerate(uniq_spike_clusters):
                idx = np.where((self.spike_clusters == cluster) )[0]
                if len(idx) == 0:
                    continue
                unit_kilo_frames = self.spike_time[idx]
                keep_mask = blk_idx[unit_kilo_frames]
                unit_kilo_frames = unit_kilo_frames[keep_mask]

                unit_spike_time = self.timestamps[unit_kilo_frames]

                cg_row = self.cluster_group.loc[self.cluster_group["cluster_id"] == cluster]
                if cg_row.empty:
                    continue

                cluster_id.append(cluster)
                depth.append(cg_row["depth"].values[0])
                group.append(int(cg_row["group"].values[0] == "good"))
                spiketimes.append(unit_spike_time)


            unit_struct[par_type] = {
                'depth': np.array(depth),
                'cluster_id': np.array(cluster_id),
                'groups': np.array(group),
                'spiketimes': np.array(spiketimes, dtype=object),
                'cluster_group': self.cluster_group.to_dict(orient='list')
            }

        self.unit_struct = unit_struct

        

    def save_units_to_mat(self):
        """ Saves unit data to a MATLAB `.mat` file and reports trial counts. """
        save_path = os.path.join(self.data_path, f"{self.subject}{self.date}dots3DMP.mat")

        if os.path.exists(save_path):
            mat_contents = scipy.io.loadmat(save_path, simplify_cells=True)
            data_dict = mat_contents['data']
        else:
            data_dict = {}
            print("Generating a new file.")


    
        for p_type in self.par_type:
            if p_type not in data_dict or not isinstance(data_dict[p_type], dict):
                data_dict[p_type] = {}

            data_dict[p_type]['unit'] = self.unit_struct[p_type]

        scipy.io.savemat(save_path, {'data': data_dict})
        print(f"Data saved at '{save_path}'")

class CreateEyeXYStruct:
    def __init__(self, directory, subject, date):
        self.directory = directory
        self.subject = subject
        self.date = date
        self.data_path = os.path.join(directory, date)

        self.eyeXY = np.fromfile(os.path.join(self.data_path, f"{subject}{date}dots3DMP_eyeXY.dat"), dtype='int16')
        self.eyeXY = np.reshape(self.eyeXY, (self.eyeXY.size // 4, 4))

        self.eyeXY_timestamps = np.load(os.path.join(self.data_path, f"{subject}{date}dots3DMP_eyeXYtimestamps.npy"))
        self.eyeXY_blocks = np.load(os.path.join(self.data_path, f"{subject}{date}dots3DMP_eyeXYblocks.npy"))

        self.pldaps_filetimes, self.par, self.par_type = self.check_trial_par()

    def check_trial_par(self):
        """Get the trial type"""
        mat_file = os.path.join(self.data_path, f"{self.subject}{self.date}dots3DMP_info.mat")

        mat_contents = scipy.io.loadmat(mat_file, simplify_cells=True)
        info = mat_contents['info']
        par_raw = info['par']
        par = np.array([p.strip() if isinstance(p, str) else p for p in par_raw])
        par_type = np.unique(par)

        return info['pldaps_filetimes'], par, par_type 

    def build_eyeXY_structure(self):

        eyeXY_struct = {
            par_type: {
                'eyeXY': [], 'timestamps': []
            } for par_type in self.par_type
        }


        for par_type in self.par_type:

            trial_indices = [i for i, p in enumerate(self.par) if p == par_type]

            if len(trial_indices) == 0:
                print("No trial type", par_type)
                continue

            filetime_set = set(self.pldaps_filetimes[i] for i in trial_indices)
            blk_idx = np.isin(self.eyeXY_blocks, list(filetime_set))

            if np.sum(blk_idx) == 0:
                print(f"No matching eyeXY data for trial type {par_type}")
                continue

            eyeXY_data = self.eyeXY[blk_idx, :2]           
            eyeXY_timestamps = self.eyeXY_timestamps[blk_idx]

            eyeXY_struct[par_type]['eyeXY'] = eyeXY_data
            eyeXY_struct[par_type]['timestamps'] = eyeXY_timestamps


        self.eyeXY_struct = eyeXY_struct

        

    def save_eyeXY_to_mat(self):
        """ Saves eyeXY data to a MATLAB `.mat` file and reports trial counts. """
        save_path = os.path.join(self.data_path, f"{self.subject}{self.date}dots3DMP.mat")

        if os.path.exists(save_path):
            mat_contents = scipy.io.loadmat(save_path, simplify_cells=True)
            data_dict = mat_contents['data']
        else:
            data_dict = {}
            print("Generating a new file.")


    
        for p_type in self.par_type:
            if p_type not in data_dict or not isinstance(data_dict[p_type], dict):
                data_dict[p_type] = {}

            data_dict[p_type]['eyelink'] = self.eyeXY_struct[p_type]

        scipy.io.savemat(save_path, {'data': data_dict})
        print(f"Data saved at '{save_path}'")



if __name__ == "__main__":
    directory = "D:\\"
    subject = "zarya"
    date = "20250306"
    kilo = "kilosort4_phy"

    # Run the MergeRecordingFile class
    processor = MergeRecordingFile(directory, subject, date)
    processor.check_electrode_consistency()
    processor.merge_ap_data()
    processor.merge_ttl_data()

    # Run the new CreateEventStruct class
    event_processor = CreateEventStruct(directory, subject, date)
    event_processor.filter_events()
    event_processor.load_info_data()
    event_processor.process_trials()
    event_processor.save_to_mat()

    # Run the new CreateUnitStruct class
    unit_processor = CreateUnitStruct(directory, subject, date, kilo)
    unit_processor.build_unit_structure()
    unit_processor.save_units_to_mat()

