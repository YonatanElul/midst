from scipy.io import loadmat
from scipy.interpolate import interp1d
from wfdb.processing.qrs import xqrs_detect
from typing import Union, Dict, Optional, cast

import os
import abc
import glob
import wfdb
import numpy as np
import pandas as pd

FLOATS = [np.float, np.float32, np.float64, float]


class BasePhysioReader(abc.ABC):
    """
    Base class for all WFDB readers classes.
    """

    PHYSIONET_RHYTHMS_CODINGS = {
        '': 0,  # Saved for Normal Sinus Rhythm from healthy patient
        'N': 1,  # Normal Sinus Rhythm from sick patients
        '(N': 1,  # Normal Sinus Rhythm from sick patients
        '(NSR': 1,  # Normal Sinus Rhythm from sick patients
        'NSR': 1,  # Normal Sinus Rhythm from sick patients
        '(AFIB': 3,  # Atrial Fibrillation
        'AFIB': 3,  # Atrial Fibrillation
        'SVTA': 4,  # Supra-Ventricular Tachycardia
        '(SVTA': 4,  # Supra-Ventricular Tachycardia
        'VT': 5,  # Ventricular Tachycardia
        '(VT': 5,  # Ventricular Tachycardia
        'B': 6,  # Ventricular Bigeminy
        '(B': 6,  # Ventricular Bigeminy
        '(T': 7,  # Ventricular Trigeminy
        '(IVR': 8,  # Idioventricular Rhythm
        '(AB': 9,  # Atrial Bigeminy
        'SBR': 10,  # Sinus Bradycardia
        '(SBR': 10,  # Sinus Bradycardia
        '(BII': 11,  # Second Degree Heart Block
        '(PREX': 12,  # Pre-excitation (WPW)
        '(AFL': 13,  # Atrial Flutter
        '(P': 14,  # Paced rhythm
        'PM': 14,  # Pacemaker (pace rhythm)
        'NOD': 15,  # Nodal (A-V junctional) rhythm
        '(NOD': 15,  # Nodal (A-V junctional) rhythm
        '(J': 16,  # Junctional rhythm
        'VFL': 17,  # Ventricular Flutter
        '(VFL': 17,  # Ventricular Flutter
        'VF': 18,  # Ventricular Fibrillation
        'VFIB': 18,  # Ventricular Fibrillation
        '(ST0+': 19,  # Ventricular Fibrillation - Sudden Cardiac Death Database
        '(ST0-': 19,  # Ventricular Fibrillation - Sudden Cardiac Death Database
        '(ST1+': 19,  # Ventricular Fibrillation - Sudden Cardiac Death Database
        '(ST1-': 19,  # Ventricular Fibrillation - Sudden Cardiac Death Database
        'ST0+)': 19,  # Ventricular Fibrillation - Sudden Cardiac Death Database
        'ST0-)': 19,  # Ventricular Fibrillation - Sudden Cardiac Death Database
        'ST1+)': 19,  # Ventricular Fibrillation - Sudden Cardiac Death Database
        'ST1-)': 19,  # Ventricular Fibrillation - Sudden Cardiac Death Database
        'ASYS': 20,  # Asystole
        'BI': 21,  # First Degree Heart Block
        'HGEA': 22,  # High Grade Ventricular Ectopic Activity
        'VER': 23,  # Ventricular Escape Rhythm
        'NOISE': 24,  # Noise
    }

    def __init__(
            self,
            db_path: str = None,
            data_extenstion: str = 'dat',
            annotation_extension: Union[str, dict] = 'atr',
            detect_qrs_locations: bool = False,
            cache_dir: Optional[str] = None,
            qrs_detection_replacement_gap: float = 0.05,
            verbose: bool = True,
    ):
        """
        Constructor for 'BasePhysioReader'.

        :param db_path: (str) Path to the directory containing all of the raw
        database _files.
        :param data_extenstion: (str) The file extension which denotes _files with the
        physiological recordings.
        :param annotation_extension: (str) The file extension which denotes _files with
        the physiological recordings annotations and labels.
        """

        assert (isinstance(annotation_extension, str) or
                isinstance(annotation_extension, dict)), \
            f"Cant' specify 'annotation_extension' of type " \
            f"{type(annotation_extension)}. Must be either a string for a single " \
            f"annotation file, or a dictionary with the keys: 'anno' and 'qrs' for" \
            f" multiple annotation files cases, where the labels and QRS annotations" \
            f"are separated."

        # Setup
        self._db_path = db_path
        self._data_extenstion = data_extenstion
        self._annotation_extension = annotation_extension
        self._verbose = verbose
        self._detect_qrs_locations = detect_qrs_locations
        self._qrs_detection_replacement_gap = 1 + qrs_detection_replacement_gap

        self.frequency = 0
        self.iterator = 0

        self.files = sorted(glob.glob(os.path.join(db_path, f'*.{data_extenstion}')))
        self.records = [f.split(os.sep)[-1] for f in self.files]

        self._len = len(self.files)

        if detect_qrs_locations:
            if cache_dir is None:
                cache_dir = os.path.join(db_path, "cache")

            os.makedirs(cache_dir, exist_ok=True)
            self._cache_dir = cache_dir

    def __getitem__(self, item: int) -> str:
        # Check boundary case
        if item >= self.__len__():
            raise ValueError(f'Index {item} is out of bounds '
                             f'in an array with length {self.__len__()}.')

        return self.files[item]

    def __next__(self) -> str:
        # Check boundary case
        if self.iterator == self.__len__():
            raise StopIteration

        self.iterator += 1

        return self.files[self.iterator - 1]

    def __len__(self) -> int:
        return self._len

    def read_record(
            self,
            index: int = 0,
    ) -> Dict:
        """
        A method for reading the data & labels of a record at location 'index'.

        :param index: (int) The index of the recording to read. Recordings are index
        based on the sorted paths to each recording,

        :return: (dict) A dictionary with the following structure:
        {
            'signal': (NumPy array) Containing the ECG recordings
            'qrs': (NumPy vector) Containing the QRS annotations
            'rhythms': (NumPy vector) Containing the rhythms annotations
            'comments': (str) Conatins any additional comments regarding the recording
        }
        """

        if self._verbose:
            print(f"Reading record #{index + 1}: {self.files[index]}")

        # Read record
        record_name = self.files[index][:-4]
        record = wfdb.rdsamp(record_name)

        # Extract record annotations & info
        if isinstance(self._annotation_extension, dict):
            anno = wfdb.rdann(
                self.files[index][:-4],
                self._annotation_extension['anno'],
            )
            try:
                qrs_anno = wfdb.rdann(
                    self.files[index][:-4],
                    self._annotation_extension['qrs'],
                )

            except FileNotFoundError:
                qrs_anno = anno

            qrs_annotations = qrs_anno.sample
            annotations = anno.aux_note
            annotations = [beat.rstrip() for beat in annotations]
            annotations = [beat.rstrip('\x00') for beat in annotations]

        else:
            anno = wfdb.rdann(
                self.files[index][:-4],
                cast(str, self._annotation_extension),
            )
            qrs_annotations = anno.sample
            annotations = anno.aux_note
            annotations = [beat.rstrip() for beat in annotations]
            annotations = [beat.rstrip('\x00') for beat in annotations]

        leads = record[0]
        comments = record[1]['comments']

        # Remove unclear annotations
        annotations = [
            ''
            if ann not in self.PHYSIONET_RHYTHMS_CODINGS else
            ann for ann in annotations
        ]

        # Convert the rhythm annotations from strings to integers
        annotations = list(
            map(lambda beat: self.PHYSIONET_RHYTHMS_CODINGS[beat], annotations)
        )
        annotations = np.array(annotations)

        # Match annotations to qrs_annotations and auto-complete the rhythm annotations
        rhythm_annotations = np.zeros_like(qrs_annotations)
        anno_inds = np.where(annotations > 0)[0]

        # If there are rhythm annotations
        if len(anno_inds) == 1:
            rhythm_annotations[:] = annotations[anno_inds[0]]

        elif len(anno_inds) > 1:
            start = 0
            end = anno_inds[1]
            current_rhythm = annotations[anno_inds[0]]
            for i, ind in enumerate(anno_inds):
                rhythm_annotations[start:end] = current_rhythm

                start = end

                if i == (len(anno_inds) - 2):
                    end = len(rhythm_annotations)
                    current_rhythm = annotations[anno_inds[-1]]

                elif i < (len(anno_inds) - 2):
                    end = anno_inds[i + 1]
                    current_rhythm = annotations[anno_inds[i + 1]]

        if self._detect_qrs_locations:
            qrs_path = os.path.join(self._cache_dir, f"{record_name.split(os.sep)[-1]}_qrs.npy")
            anno_path = os.path.join(self._cache_dir, f"{record_name.split(os.sep)[-1]}_anno.npy")
            if os.path.isfile(qrs_path):
                detected_qrs_annotations = np.load(file=qrs_path, allow_pickle=True)
                interpolated_labels = np.load(file=anno_path, allow_pickle=True)

            else:
                # Compute the QRS annotations
                detected_qrs_annotations = [
                    xqrs_detect(
                        sig=leads[:, l],
                        fs=self.frequency,
                        sampfrom=0,
                        sampto='end',
                        conf=None,
                        learn=True,
                        verbose=True
                    ).astype(int)
                    for l in range(min([leads.shape[1], 3]))
                ]
                detected_qrs_annotations = detected_qrs_annotations[
                    int(np.argmax([len(l) for l in detected_qrs_annotations]))
                ]

                # Cache the QRS annotations
                np.save(file=qrs_path, arr=detected_qrs_annotations)

                # Pick the QRS detection method which picked more R-Peaks &
                # Potentially interpolate the labels according to the new QRS
                if len(detected_qrs_annotations) > (len(qrs_annotations) * self._qrs_detection_replacement_gap):
                    if len(qrs_annotations) == 0:
                        interpolated_labels = np.zeros_like(detected_qrs_annotations)

                    else:
                        interpolator = interp1d(
                            x=qrs_annotations,
                            y=rhythm_annotations,
                            kind='zero',
                            bounds_error=False,
                            fill_value=(rhythm_annotations[0], rhythm_annotations[-1]),
                            assume_sorted=True,
                        )
                        interpolated_labels = interpolator(detected_qrs_annotations)

                else:
                    detected_qrs_annotations = qrs_annotations
                    interpolated_labels = rhythm_annotations

                # Cache the labels annotations
                np.save(file=anno_path, arr=interpolated_labels)

            qrs_annotations = detected_qrs_annotations
            rhythm_annotations = interpolated_labels

        record_data = {
            'signal': leads,
            'qrs': qrs_annotations,
            'rhythms': rhythm_annotations,
            'comments': comments,
            'file_name': record_name.split(os.sep)[-1]
        }

        return record_data


class LongTermReader(BasePhysioReader):
    """
    A reader for PhysioNet's Long Term AF Database.
    https://www.physionet.org/content/ltafdb/1.0.0/
    """

    def __init__(
            self,
            db_path: str,
            detect_qrs_locations: bool = False,
            cache_dir: Optional[str] = None,
            qrs_detection_replacement_gap: float = 0.05,
    ):
        """
        Constructor method for the LongTermReader class.

        :param db_path: (str) Path to the directory containing all of the raw
        database _files.
        """

        # Initialize the super class
        super().__init__(
            db_path=db_path,
            data_extenstion='dat',
            annotation_extension='atr',
            detect_qrs_locations=detect_qrs_locations,
            cache_dir=cache_dir,
            qrs_detection_replacement_gap=qrs_detection_replacement_gap
        )

        # Database specific parameters
        self.frequency = 128  # In [Hz], From PhysioNet


class NormalSinusRhythmReader(BasePhysioReader):
    """
    A reader for PhysioNet's Normal Sinus Rhythm Database.
    https://www.physionet.org/content/nsrdb/1.0.0/
    """

    def __init__(
            self,
            db_path: str,
            detect_qrs_locations: bool = False,
            cache_dir: Optional[str] = None,
            qrs_detection_replacement_gap: float = 0.05,
            data_extenstion: str = 'dat',
            annotation_extension: str = 'atr',
    ):
        """
        Constructor method for the NormalSinusRhythmReader class

        :param db_path: (str) Path to the directory containing all of the raw
        database _files.
        """

        # Initialize the super class
        super().__init__(
            db_path=db_path,
            data_extenstion=data_extenstion,
            annotation_extension=annotation_extension,
            detect_qrs_locations=detect_qrs_locations,
            cache_dir=cache_dir,
            qrs_detection_replacement_gap=qrs_detection_replacement_gap,
        )

        # Database specific parameters
        self.frequency = 128  # In [Hz], From PhysioNet

    def read_record(
            self,
            index: int = 0,
    ) -> Dict:
        """
        A method for reading the data & labels of a record at location 'index'.

        :param index: (int) The index of the recording to read. Recordings are index
        based on the sorted paths to each recording,

        :return: (dict) A dictionary with the following structure:
        {
            'signal': (NumPy array) Containing the ECG recordings
            'qrs': (NumPy vector) Containing the QRS annotations
            'rhythms': (NumPy vector) Containing the rhythms annotations
            'comments': (str) Conatins any additional comments regarding the recording
        }
        """

        if self._verbose:
            print(f"Reading record #{index + 1}: {self.files[index]}")

        # Read record
        record_name = self.files[index][:-4]
        record = wfdb.rdsamp(record_name)

        # Extract record annotations & info
        if isinstance(self._annotation_extension, dict):
            anno = wfdb.rdann(
                self.files[index][:-4],
                self._annotation_extension['anno'],
            )
            try:
                qrs_anno = wfdb.rdann(
                    self.files[index][:-4],
                    self._annotation_extension['qrs'],
                )

            except FileNotFoundError:
                qrs_anno = anno

            qrs_annotations = qrs_anno.sample

        else:
            anno = wfdb.rdann(
                self.files[index][:-4],
                cast(str, self._annotation_extension),
            )
            qrs_annotations = anno.sample

        leads = record[0]
        comments = record[1]['comments']
        rhythm_annotations = np.zeros_like(qrs_annotations)

        if self._detect_qrs_locations:
            qrs_path = os.path.join(self._cache_dir, f"{record_name.split(os.sep)[-1]}_qrs.npy")
            anno_path = os.path.join(self._cache_dir, f"{record_name.split(os.sep)[-1]}_anno.npy")
            if os.path.isfile(qrs_path):
                detected_qrs_annotations = np.load(file=qrs_path, allow_pickle=True)
                interpolated_labels = np.load(file=anno_path, allow_pickle=True)

            else:
                # Compute the QRS annotations
                detected_qrs_annotations = [
                    xqrs_detect(
                        sig=leads[:, l],
                        fs=self.frequency,
                        sampfrom=0,
                        sampto='end',
                        conf=None,
                        learn=True,
                        verbose=True
                    ).astype(int)
                    for l in range(min([leads.shape[1], 3]))
                ]
                detected_qrs_annotations = detected_qrs_annotations[
                    int(np.argmax([len(l) for l in detected_qrs_annotations]))
                ]

                # Cache the QRS annotations
                np.save(file=qrs_path, arr=detected_qrs_annotations)

                # Pick the QRS detection method which picked more R-Peaks &
                # Potentially interpolate the labels according to the new QRS
                if len(detected_qrs_annotations) > (len(qrs_annotations) * self._qrs_detection_replacement_gap):
                    if len(qrs_annotations) == 0:
                        interpolated_labels = np.zeros_like(detected_qrs_annotations)

                    else:
                        interpolator = interp1d(
                            x=qrs_annotations,
                            y=rhythm_annotations,
                            kind='zero',
                            bounds_error=False,
                            fill_value=(rhythm_annotations[0], rhythm_annotations[-1]),
                            assume_sorted=True,
                        )
                        interpolated_labels = interpolator(detected_qrs_annotations)

                else:
                    detected_qrs_annotations = qrs_annotations
                    interpolated_labels = rhythm_annotations

                # Cache the labels annotations
                np.save(file=anno_path, arr=interpolated_labels)

            qrs_annotations = detected_qrs_annotations
            rhythm_annotations = interpolated_labels

        record_data = {
            'signal': leads,
            'qrs': qrs_annotations,
            'rhythms': rhythm_annotations,
            'comments': comments,
            'file_name': record_name.split(os.sep)[-1]
        }

        return record_data


class NormalSinusRhythmRRReader(BasePhysioReader):
    """
    A reader for PhysioNet's Normal Sinus Rhythm Database.
    https://www.physionet.org/content/nsrdb/1.0.0/
    """

    def __init__(
            self,
            db_path: str,
            detect_qrs_locations: bool = False,
            cache_dir: Optional[str] = None,
            qrs_detection_replacement_gap: float = 0.05,
            data_extenstion: str = 'ecg',
            annotation_extension: str = 'ecg',
    ):
        """
        Constructor method for the NormalSinusRhythmReader class

        :param db_path: (str) Path to the directory containing all of the raw
        database _files.
        """

        # Initialize the super class
        super().__init__(
            db_path=db_path,
            data_extenstion=data_extenstion,
            annotation_extension=annotation_extension,
        )

        # Database specific parameters
        self.frequency = 128  # In [Hz], From PhysioNet

    def read_record(
            self,
            index: int = 0,
    ) -> Dict:
        """
        A method for reading the data & labels of a record at location 'index'.

        :param index: (int) The index of the recording to read. Recordings are index
        based on the sorted paths to each recording,

        :return: (dict) A dictionary with the following structure:
        {
            'signal': (NumPy array) Containing the ECG recordings
            'qrs': (NumPy vector) Containing the QRS annotations
            'rhythms': (NumPy vector) Containing the rhythms annotations
            'comments': (str) Conatins any additional comments regarding the recording
        }
        """

        if self._verbose:
            print(f"Reading record #{index + 1}: {self.files[index]}")

        # Read record
        record_name = self.files[index][:-4]

        # Extract record annotations & info
        if isinstance(self._annotation_extension, dict):
            anno = wfdb.rdann(
                self.files[index][:-4],
                self._annotation_extension['anno'],
            )
            try:
                qrs_anno = wfdb.rdann(
                    self.files[index][:-4],
                    self._annotation_extension['qrs'],
                )

            except FileNotFoundError:
                qrs_anno = anno

            qrs_annotations = qrs_anno.sample

        else:
            anno = wfdb.rdann(
                self.files[index][:-4],
                cast(str, self._annotation_extension),
            )
            qrs_annotations = anno.sample

        leads = np.array([])
        comments = []
        rhythm_annotations = np.zeros_like(qrs_annotations)

        record_data = {
            'signal': leads,
            'qrs': qrs_annotations,
            'rhythms': rhythm_annotations,
            'comments': comments,
            'file_name': record_name.split(os.sep)[-1]
        }

        return record_data


class AFReader(BasePhysioReader):
    """
    A reader for PhysioNet's AF Database.
    https://www.physionet.org/content/afdb/1.0.0/
    """

    def __init__(
            self,
            db_path: str,
            detect_qrs_locations: bool = False,
            cache_dir: Optional[str] = None,
            qrs_detection_replacement_gap: float = 0.05,
    ):
        """
        Constructor method for the AFReader class

        :param db_path: (str) Path to the directory containing all of the raw
        database _files.
        """

        # Initialize the super class
        super().__init__(
            db_path=db_path,
            data_extenstion='dat',
            annotation_extension={
                'qrs': 'qrs',
                'anno': 'atr',
            },
            detect_qrs_locations=detect_qrs_locations,
            cache_dir=cache_dir,
            qrs_detection_replacement_gap=qrs_detection_replacement_gap,
        )

        # Database specific parameters
        self.frequency = 250  # In [Hz], From PhysioNet


class ArrhythmiaReader(BasePhysioReader):
    """
    A reader for PhysioNet's Arrhythmia Database.
    https://www.physionet.org/content/mitdb/1.0.0/
    """

    def __init__(
            self,
            db_path: str,
            detect_qrs_locations: bool = False,
            cache_dir: Optional[str] = None,
            qrs_detection_replacement_gap: float = 0.05,
    ):
        """
        Constructor method for the ArrhythmiaReader class

        :param db_path: (str) Path to the directory containing all of the raw
        database _files.
        """

        # Initialize the super class
        super().__init__(
            db_path=db_path,
            data_extenstion='dat',
            annotation_extension='atr',
            detect_qrs_locations=detect_qrs_locations,
            cache_dir=cache_dir,
            qrs_detection_replacement_gap=qrs_detection_replacement_gap,
        )

        # Database specific parameters
        self.frequency = 360  # In [Hz], From PhysioNet


class SVTArrhythmiaReader(BasePhysioReader):
    """
    A reader for PhysioNet's Supraventricular Arrhythmia Database.
    https://physionet.org/content/svdb/1.0.0/
    """

    def __init__(
            self,
            db_path: str,
            detect_qrs_locations: bool = False,
            cache_dir: Optional[str] = None,
            qrs_detection_replacement_gap: float = 0.05,
    ):
        """
        Constructor method for the SVTArrhythmiaReader class

        :param db_path: (str) Path to the directory containing all of the raw
        database _files.
        """

        # Initialize the super class
        super().__init__(
            db_path=db_path,
            data_extenstion='dat',
            annotation_extension='atr',
            detect_qrs_locations=detect_qrs_locations,
            cache_dir=cache_dir,
            qrs_detection_replacement_gap=qrs_detection_replacement_gap,
        )

        # Database specific parameters
        self.frequency = 360  # In [Hz], From PhysioNet


class MVArrhythmiaReader(BasePhysioReader):
    """
    A reader for PhysioNet's Malignant Ventricular Ectopy Database.
    https://physionet.org/content/vfdb/1.0.0/
    """

    def __init__(
            self,
            db_path: str,
            detect_qrs_locations: bool = False,
            cache_dir: Optional[str] = None,
            qrs_detection_replacement_gap: float = 0.05,
    ):
        """
        Constructor method for the MVArrhythmiaReader class

        :param db_path: (str) Path to the directory containing all of the raw
        database _files.
        """

        # Initialize the super class
        super().__init__(
            db_path=db_path,
            data_extenstion='dat',
            annotation_extension='atr',
            detect_qrs_locations=detect_qrs_locations,
            cache_dir=cache_dir,
            qrs_detection_replacement_gap=qrs_detection_replacement_gap,
        )

        # Database specific parameters
        self.frequency = 250  # In [Hz], From PhysioNet


class MGHMFArrhythmiaReader(BasePhysioReader):
    """
    A reader for PhysioNet's Malignant Ventricular Ectopy Database.
    https://physionet.org/content/vfdb/1.0.0/
    """

    INDEX_TO_LABEL = {
        0: 1,
        2: 9,
        3: 3,
        4: 13,
        5: 9,
        6: 4,
        7: 15,
        8: 16,
        9: 8,
        11: 6,
        12: 7,
        13: 5,
        14: 17,
        15: 18,
        16: 20,
        17: 12,
        18: 14,
    }

    def __init__(
            self,
            db_path: str,
            detect_qrs_locations: bool = False,
            cache_dir: Optional[str] = None,
            qrs_detection_replacement_gap: float = 0.05,
    ):
        """
        Constructor method for the MGHMFArrhythmiaReader class

        :param db_path: (str) Path to the directory containing all of the raw
        database _files.
        """

        # Initialize the super class
        super().__init__(
            db_path=db_path,
            data_extenstion='dat',
            annotation_extension='ari',
            detect_qrs_locations=detect_qrs_locations,
            cache_dir=cache_dir,
            qrs_detection_replacement_gap=qrs_detection_replacement_gap,
        )

        # Database specific parameters
        self.frequency = 360  # In [Hz], From PhysioNet

        # Read labels
        with open(os.path.join(db_path, 'labels.csv')) as f:
            labels = f.readlines()

        rhythms_indices = sorted(self.INDEX_TO_LABEL.keys())
        label_per_record = [
            label.strip().split(',')
            for label in labels[1:]
        ]
        label_per_record = [
            (
                label[0],
                [
                    self.INDEX_TO_LABEL[ind]
                    for ind in rhythms_indices
                    if label[1:][ind] == '1'
                ],
            )
            for label in label_per_record
        ]
        self.labels = {
            i: label_per_record[i]
            for i in range(len(label_per_record))
        }

    def read_record(
            self,
            index: int = 0,
    ) -> Dict:
        # Read the record
        record = super().read_record(index)
        record_name = record['file_name']

        # Get rhythm annotations from the csv labels files
        if self.labels[index][0] == record_name:
            rhythm_annotations = np.array(self.labels[index][1])

        else:
            raise ValueError(
                f"Index {index} resulted with ecg from the record: {record_name}, "
                f"and in a label from the record: {self.labels[index][0]}."
            )

        # Insert the appropriate labels
        record['rhythms'] = rhythm_annotations

        return record


class ECGDMMLDReader(BasePhysioReader):
    """
    A reader for PhysioNet's ECG Effects of Dofetilide, Moxifloxacin,
    Dofetilide+Mexiletine, Dofetilide+Lidocaine and Moxifloxacin+Diltiazem Database.
    https://www.physionet.org/content/ecgdmmld/1.0.0/
    """

    def __init__(
            self,
            db_path: str,
            detect_qrs_locations: bool = False,
            cache_dir: Optional[str] = None,
            qrs_detection_replacement_gap: float = 0.05,
    ):
        """
        Constructor method for the ECGDMMLDReader class

        :param db_path: (str) Path to the directory containing all of the raw
        database _files.
        :param detect_r_peaks: (bool) Whether to detect R-peaks in each recording using WFDB's 'xqrs_detect' algorithm.
        """

        # Initialize the super class
        super().__init__(
            db_path=db_path,
            data_extenstion='dat',
            annotation_extension='hea',
            detect_qrs_locations=detect_qrs_locations,
            cache_dir=cache_dir,
            qrs_detection_replacement_gap=qrs_detection_replacement_gap,
        )

        # Database specific parameters
        self.frequency = 1000  # In [Hz], From PhysioNet

    def read_record(
            self,
            index: int = 0,
    ) -> Dict:
        print(f"Reading record #{index + 1}: {self.files[index]}")

        # Read record
        record = wfdb.rdsamp(self.files[index][:-4])

        leads = record[0]
        description_csv = pd.read_csv(os.path.join(self._db_path,
                                                   "SCR-003.Clinical.Data.csv"))

        current_file = self.files[index].split(os.sep)[-1][:-4]
        file_index = description_csv.index[description_csv['EGREFID'] ==
                                           current_file][0]
        description = dict(description_csv.iloc[file_index])
        for key in description:
            if type(description[key]) in FLOATS:
                if np.isnan(description[key]).item():
                    description[key] = 0.

        comments = [description, record[1]['sig_name']]

        if self._detect_qrs_locations:
            qrs_annotations = [
                xqrs_detect(
                    sig=leads[:, i],
                    fs=self.frequency,
                    sampfrom=0,
                    sampto='end',
                    conf=None,
                    learn=True,
                    verbose=True,
                )
                for i in range(leads.shape[1])
            ]
            max_beats = int(np.argmax([len(qrs) for qrs in qrs_annotations]))
            qrs_annotations = qrs_annotations[max_beats]

        else:
            qrs_annotations = np.ndarray([])

        record_data = {
            'signal': leads,
            'qrs': qrs_annotations,
            'rhythms': np.ndarray([]),
            'comments': comments,
        }

        return record_data


class ECGRDVQReader(BasePhysioReader):
    """
    A reader for PhysioNet's ECG Effects of Dofetilide, Moxifloxacin,
    Dofetilide+Mexiletine, Dofetilide+Lidocaine and Moxifloxacin+Diltiazem Database.
    https://www.physionet.org/content/ecgdmmld/1.0.0/
    """

    def __init__(
            self,
            db_path: str,
            csv_only: bool = False,
            detect_qrs_locations: bool = False,
            cache_dir: Optional[str] = None,
            qrs_detection_replacement_gap: float = 0.05,
    ):
        """
        Constructor method for the ECGRDVQReader class

        :param db_path: (str) Path to the directory containing all of the raw
        database _files.
        :param detect_r_peaks: (bool) Whether to detect R-peaks in each recording using WFDB's 'xqrs_detect' algorithm.
        :param csv_only: (bool) Whether to use data from the csv file only, i.e. without ECG signals information.
        """

        # Initialize the super class
        super().__init__(
            db_path=db_path,
            data_extenstion='dat',
            annotation_extension='hea',
            detect_qrs_locations=detect_qrs_locations,
            cache_dir=cache_dir,
            qrs_detection_replacement_gap=qrs_detection_replacement_gap,
        )

        # Database specific parameters
        self.frequency = 1000  # In [Hz], From PhysioNet
        self._csv_only = csv_only

        if self._csv_only:
            description_csv = pd.read_csv(
                os.path.join(
                    self._db_path,
                    "SCR-002.Clinical.Data.Imputed.csv",
                )
            )
            patients_ids = description_csv['RANDID'].values
            self._files = list(description_csv['EGREFID'].values)
            self._len = len(self._files)

    def read_record(
            self,
            index: int = 0,
    ) -> Dict:
        print(f"Reading record #{index + 1}: {self._files[index]}")

        description_csv = pd.read_csv(
            os.path.join(
                self._db_path,
                "SCR-002.Clinical.Data.Imputed.csv",
            )
        )
        if self._csv_only:
            current_file = self._files[index]

        else:
            current_file = self._files[index].split(os.sep)[-1][:-4]

        file_index = description_csv.index[description_csv['EGREFID'] == current_file][0]
        description = dict(description_csv.iloc[file_index])

        for key in description:
            if type(description[key]) in FLOATS:
                if np.isnan(description[key]).item():
                    description[key] = 0.

        if self._csv_only:
            leads = np.ndarray([])
            comments = [description, ]
            qrs_annotations = np.ndarray([])
            waves_annotations = np.ndarray([])

        else:
            # Read record
            record = wfdb.rdsamp(self._files[index][:-4])
            leads = record[0]
            comments = [description, record[1]['sig_name']]

            if self._detect_qrs_locations:
                qrs_annotations = [
                    xqrs_detect(
                        sig=leads[:, i],
                        fs=self.frequency,
                        sampfrom=0,
                        sampto='end',
                        conf=None,
                        learn=True,
                        verbose=True,
                    )
                    for i in range(leads.shape[1])
                ]
                max_beats = int(np.argmax([len(qrs) for qrs in qrs_annotations]))
                qrs_annotations = qrs_annotations[max_beats]
                waves_annotations = np.ndarray([])

            elif len(glob.glob(self._files[index][:-4] + '*' + '.mat')):
                waves_annotations_file = glob.glob(self._files[index][:-4] + '*' + '.mat')[0]
                waves_annotations = loadmat(waves_annotations_file)
                waves_annotations = waves_annotations['waves']
                qrs_annotations = waves_annotations[3, :]

            else:
                qrs_annotations = np.ndarray([])
                waves_annotations = np.ndarray([])

        record_data = {
            'signal': leads,
            'qrs': qrs_annotations,
            'waves': waves_annotations,
            'rhythms': np.ndarray([]),
            'comments': comments,
        }

        return record_data


class FantasiaReader(NormalSinusRhythmReader):
    """
    A reader for PhysioNet's Fantasia Database.
    https://physionet.org/content/fantasia/1.0.0/
    """

    def __init__(
            self,
            db_path: str,
            detect_qrs_locations: bool = False,
            cache_dir: Optional[str] = None,
            qrs_detection_replacement_gap: float = 0.05,
    ):
        """
        Constructor method for the ArrhythmiaReader class

        :param db_path: (str) Path to the directory containing all of the raw
        database _files.
        """

        # Initialize the super class
        super().__init__(
            db_path=db_path,
            detect_qrs_locations=detect_qrs_locations,
            cache_dir=cache_dir,
            qrs_detection_replacement_gap=qrs_detection_replacement_gap,
            data_extenstion='dat',
            annotation_extension='ecg',
        )

        # Database specific parameters
        self.frequency = 250  # In [Hz], From PhysioNet


class SCDReader(BasePhysioReader):
    def __init__(
            self,
            db_path: str,
            detect_qrs_locations: bool = False,
            cache_dir: Optional[str] = None,
            qrs_detection_replacement_gap: float = 0.05,
    ):
        # Initialize the super class
        super().__init__(
            db_path=db_path,
            data_extenstion='dat',
            annotation_extension={
                'qrs': 'atr',
                'anno': 'ari',
            },
            detect_qrs_locations=detect_qrs_locations,
            cache_dir=cache_dir,
            qrs_detection_replacement_gap=qrs_detection_replacement_gap,
        )

        self.frequency = 250


class INCARTReader(BasePhysioReader):
    def __init__(
            self,
            db_path: str,
            detect_qrs_locations: bool = False,
            cache_dir: Optional[str] = None,
            qrs_detection_replacement_gap: float = 0.05,
    ):
        # Initialize the super class
        super().__init__(
            db_path=db_path,
            data_extenstion='dat',
            annotation_extension='atr',
            detect_qrs_locations=detect_qrs_locations,
            cache_dir=cache_dir,
            qrs_detection_replacement_gap=qrs_detection_replacement_gap,
        )

        self.frequency = 257


class PhysioReader(abc.ABC):
    """
    A general purpose reader which wraps around all other existing readers.
    This reader is the one users should use.
    """

    DB_NAMES = (
        'ltafdb',
        'nsrdb',
        'nsrdbrr',
        'afdb',
        'mitdb',
        'mitsvtdb',
        'mitmvedb',
        'ecgdmmld',
        'ecgrdvq',
        'mghmfdb',
        'fantasia',
        'svtdb',
        'incart',
    )

    READERS_DICT = {
        'ltafdb': LongTermReader,
        'nsrdb': NormalSinusRhythmReader,
        'nsrdbrr': NormalSinusRhythmRRReader,
        'afdb': AFReader,
        'mitdb': ArrhythmiaReader,
        'mitmvedb': MVArrhythmiaReader,
        'ecgdmmld': ECGDMMLDReader,
        'ecgrdvq': ECGRDVQReader,
        'mghmfdb': MGHMFArrhythmiaReader,
        'mitsvtdb': SVTArrhythmiaReader,
        'fantasia': FantasiaReader,
        'svtdb': SVTArrhythmiaReader,
        'incart': INCARTReader,
    }

    def __init__(
            self,
            db_path: str,
            db_name: str,
            detect_qrs_locations: bool = False,
            cache_dir: Optional[str] = None,
            qrs_detection_replacement_gap: float = 0.05,
    ):
        """
        Constructor for the PhysioReader.

        :param db_path: (str) Path to the directory containing all of the database's
        raw files.
        :param db_name: (str) The name of the database to load. This is crucial since
        different databases are saved under different formats and structures.
        """

        assert db_name in self.DB_NAMES, \
            f"{db_name} is not a supported database type. " \
            "The currently supported databases are: " \
            "\n'ltafdb': Long-Term AF DB,\n'nsrdb': Normal Sinus Rhythm DB,\n" \
            "'afdb': MIT-BIH Fibrillation DB,\n'mitdb': MIT-BIH Arrhythmia DB,\n" \
            "'ecgdmmld': ECG Effects of Dofetilide, Moxifloxacin, " \
            "Dofetilide+Mexiletine, Dofetilide+Lidocaine and " \
            "Moxifloxacin+Diltiazem DB,\n " \
            "ecgrdvq: ECG Effects of Ranolazine, Dofetilide, Verapamil, and Quinidine\n."

        self.db_name = db_name
        self.reader = self.READERS_DICT[db_name](
            db_path=db_path,
            detect_qrs_locations=detect_qrs_locations,
            cache_dir=cache_dir,
            qrs_detection_replacement_gap=qrs_detection_replacement_gap,
        )
        self.iterator = 0
        self._frequency = self.reader.frequency

    @property
    def frequency(self) -> float:
        """
        Class property, returns the sampling frequency of records in the database.
        """

        return self._frequency

    def __getitem__(self, item: int) -> str:
        # Check boundary case
        if item >= self.__len__():
            raise ValueError(f'Index {item} is out of bounds '
                             f'in an array with length {self.__len__()}.')

        return self.reader[item]

    def __next__(self) -> str:
        # Check boundary case
        if self.iterator == self.__len__():
            raise StopIteration

        self.iterator += 1

        return self.reader[self.iterator - 1]

    def __len__(self):
        """
        :return:
        """

        return len(self.reader)

    def read_record(
            self,
            index: int,
    ) -> Dict:
        """
        This method read a record at index 'index' from the database through calling
        the appropriate reader's 'read_record' method

        :param index: (int) Index of the record to read

        :return: (dict) A dictionary with the following structure:
        {
            'ecg': (NumPy array) Containing the ECG recordings
            'qrs': (NumPy vector) Containing the QRS annotations
            'rhythms': (NumPy vector) Containing the rhythms annotations
            'comments': (str) Conatins any additional comments regarding the recording
        }
        """

        return self.reader.read_record(index=index)
