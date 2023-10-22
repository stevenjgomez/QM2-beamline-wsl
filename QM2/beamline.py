import re

import fabio
import numpy as np
from nexusformat.nexus import (NeXusError, NXdata, NXentry, NXfield,
                               NXgoniometer, NXinstrument, NXlink, NXmonitor,
                               NXsample, NXsource, NXsubentry, nxopen)
from nexusformat.nexus.tree import natural_sort
from nxrefine.nxbeamline import NXBeamLine
from nxrefine.nxutils import SpecParser

prefix_pattern = re.compile(r'^([^.]+)(?:(?<!\d)|(?=_))')
file_index_pattern = re.compile(r'^(.*?)([0-9]*)[.](.*)$')
directory_index_pattern = re.compile(r'^(.*?)([0-9]*)$')


class QM2BeamLine(NXBeamLine):

    name = 'QM2'
    source_name = 'Cornell High-Energy Synchrotron'
    make_scans_enabled = False
    import_data_enabled = True

    def __init__(self, reduce=None, directory=None):
        super().__init__(reduce=reduce, directory=directory)

    def import_data(self, config_file, overwrite=False):
        self.config_file = nxopen(config_file)
        scans = self.raw_directory / self.sample / self.label
        y_size, x_size = self.config_file['f1/instrument/detector/shape']
        for scan in [s for s in scans.iterdir() if s.is_dir()]:
            scan_name = self.sample+'_'+scan.name+'.nxs'
            scan_file = self.base_directory / scan_name
            if scan_file.exists() and not overwrite:
                continue
            scan_directories = [s.name for s in scan.glob(f'{self.sample}_*')
                                if s.is_dir()]
            if len(scan_directories) == 0:
                continue
            scan_directory = self.base_directory / scan.name
            scan_directory.mkdir(exist_ok=True)
            if overwrite:
                mode = 'w'
            else:
                mode = 'a'
            with nxopen(scan_file, mode) as root:
                if 'entry' not in root:
                    root['entry'] = self.config_file['entry']
                i = 0
                for s in scan_directories:
                    scan_number = self.get_index(s, directory=True)
                    if scan_number:
                        i += 1
                        entry_name = f"f{i}"
                        if entry_name in root and not overwrite:
                            continue
                        root[entry_name] = self.config_file['f1']
                        entry = root[entry_name]
                        entry['scan_number'] = scan_number
                        entry['data'] = NXdata()
                        linkpath = '/entry/data/data'
                        linkfile = scan_directory / f'f{i:d}.h5'
                        entry['data'].nxsignal = NXlink(linkpath, linkfile)
                        entry['data/x_pixel'] = np.arange(x_size, dtype=int)
                        entry['data/y_pixel'] = np.arange(y_size, dtype=int)
                        self.image_directory = scan / s
                        frame_number = len(self.get_files())
                        entry['data/frame_number'] = np.arange(frame_number,
                                                               dtype=int)
                        entry['data'].nxaxes = [entry['data/frame_number'],
                                                entry['data/y_pixel'],
                                                entry['data/x_pixel']]

    def load_data(self, overwrite=False):
        if self.reduce.raw_data_exists() and not overwrite:
            return True
        try:
            self.scan_number = self.entry['scan_number'].nxvalue
            scan_directory = f"{self.sample}_{self.scan_number:03d}"
            self.image_directory = (self.raw_directory /
                                    self.sample / self.label /
                                    self.scan / scan_directory)
            entry_file = self.entry.nxname+'.h5temp'
            self.raw_file = self.directory / entry_file
            self.write_data()
            self.raw_file.rename(self.raw_file.with_suffix('.h5'))
            return True
        except NeXusError:
            return False

    def get_prefix(self):
        prefixes = []
        for filename in self.image_directory.iterdir():
            match = prefix_pattern.match(filename.stem)
            if match and filename.suffix in ['.cbf', '.tif', '.tiff']:
                prefixes.append(match.group(1).strip('-').strip('_'))
        return max(prefixes, key=prefixes.count)

    def get_index(self, name, directory=False):
        try:
            if directory:
                return int(directory_index_pattern.match(str(name)).group(2))
            else:
                return int(file_index_pattern.match(str(name)).group(2))
        except Exception:
            return None

    def get_files(self):
        prefix = self.get_prefix()
        return sorted(
            [str(f) for f in self.image_directory.glob(prefix+'*')
             if f.suffix in ['.cbf', '.tif', '.tiff']],
            key=natural_sort)

    def read_image(self, filename):
        im = fabio.open(str(filename))
        return im.data

    def read_images(self, filenames, shape):
        good_files = [str(f) for f in filenames if f is not None]
        if good_files:
            v0 = self.read_image(good_files[0])
            if v0.shape != shape:
                raise NeXusError(
                    f'Image shape of {good_files[0]} not consistent')
            v = np.empty([len(filenames), v0.shape[0], v0.shape[1]],
                         dtype=np.float32)
        else:
            v = np.empty([len(filenames), shape[0], shape[1]],
                         dtype=np.float32)
        v.fill(np.nan)
        for i, filename in enumerate(filenames):
            if filename:
                v[i] = self.read_image(filename)
        return v

    def initialize_entry(self, filenames):
        z_size = len(filenames)
        v0 = self.read_image(filenames[0])
        x = NXfield(range(v0.shape[1]), dtype=np.uint16, name='x_pixel')
        y = NXfield(range(v0.shape[0]), dtype=np.uint16, name='y_pixel')
        z = NXfield(np.arange(z_size), dtype=np.uint16, name='frame_number',
                    maxshape=(5000,))
        v = NXfield(name='data', shape=(z_size, v0.shape[0], v0.shape[1]),
                    dtype=np.float32,
                    maxshape=(5000, v0.shape[0], v0.shape[1]))
        return NXentry(NXdata(v, (z, y, x)))

    def write_data(self):
        filenames = self.get_files()
        with nxopen(self.raw_file, 'w') as root:
            root['entry'] = self.initialize_entry(filenames)
            z_size = root['entry/data/data'].shape[0]
            image_shape = root['entry/data/data'].shape[1:3]
            chunk_size = root['entry/data/data'].chunks[0]
            k = 0
            for i in range(0, z_size, chunk_size):
                files = []
                for j in range(i, min(i+chunk_size, z_size)):
                    if j == self.get_index(filenames[k]):
                        print('Processing', filenames[k])
                        files.append(filenames[k])
                        k += 1
                    elif k < len(filenames):
                        files.append(None)
                    else:
                        break
                root['entry/data/data'][i:i+len(files), :, :] = (
                    self.read_images(files, image_shape))

    def read_logs(self):
        spec_file = self.raw_directory / self.sample
        if not spec_file.exists():
            self.reduce.logger.info(f"'{spec_file}' does not exist")
            raise NeXusError('SPEC file not found')

        with self.reduce:
            scan_number = self.entry['scan_number'].nxvalue
            logs = SpecParser(spec_file).read(scan_number).NXentry[0]
            logs.nxclass = NXsubentry
            if 'logs' in self.entry:
                del self.entry['logs']
            self.entry['logs'] = logs
            frame_number = self.entry['data/frame_number']
            frames = frame_number.size
            if 'date' in logs:
                self.entry['start_time'] = logs['date']
                self.entry['data/frame_time'].attrs['start'] = logs['date']
            if 'flyc1' in logs['data']:
                if 'monitor1' in self.entry:
                    del self.entry['monitor1']
                data = logs['data/flyc1'][:frames]
                # Remove outliers at beginning and end of frames
                data[0:2] = data[2]
                data[-2:] = data[-3]
                self.entry['monitor1'] = NXmonitor(NXfield(data, name='flyc1'),
                                                   frame_number)
                if 'data/frame_time' in self.entry:
                    self.entry['monitor1/frame_time'] = (
                        self.entry['data/frame_time'])
            if 'flyc2' in logs['data']:
                if 'monitor2' in self.entry:
                    del self.entry['monitor2']
                data = logs['data/flyc2'][:frames]
                # Remove outliers at beginning and end of frames
                data[0:2] = data[2]
                data[-2:] = data[-3]
                self.entry['monitor2'] = NXmonitor(NXfield(data, name='flyc2'),
                                                   frame_number)
                if 'data/frame_time' in self.entry:
                    self.entry['monitor2/frame_time'] = (
                        self.entry['data/frame_time'])
            if 'instrument' not in self.entry:
                self.entry['instrument'] = NXinstrument()
            if 'source' not in self.entry['instrument']:
                self.entry['instrument/source'] = NXsource()
            self.entry['instrument/source/name'] = self.source_name
            self.entry['instrument/source/type'] = self.source_type
            self.entry['instrument/source/probe'] = 'x-ray'
            if 'goniometer' not in self.entry['instrument']:
                self.entry['instrument/goniometer'] = NXgoniometer()
            if 'phi' in logs['data']:
                phi = self.entry['instrument/goniometer/phi'] = (
                    logs['data/phi'][0])
                phi.attrs['end'] = logs['data/phi'][-1]
                phi.attrs['step'] = logs['data/phi'][1] - logs['data/phi'][0]
            if 'chi' in logs['positioners']:
                self.entry['instrument/goniometer/chi'] = (
                    90.0 - logs['positioners/chi'])
            if 'th' in logs['positioners']:
                self.entry['instrument/goniometer/theta'] = (
                    logs['positioners/th'])
            if 'sample' not in self.root['entry']:
                self.root['entry/sample'] = NXsample()
            self.root['entry/sample/name'] = self.sample
            self.root['entry/sample/label'] = self.label
            if 'sampleT' in logs['data']:
                self.root['entry/sample/temperature'] = (
                    logs['data/sampleT'].average())
                self.root['entry/sample/temperature'].attrs['units'] = 'K'
            if 'sample' not in self.entry:
                self.entry.makelink(self.root['entry/sample'])
