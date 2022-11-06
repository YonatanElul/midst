from midst import DATA_DIR
from midst.data.physionet.physionet_readers import ECGRDVQReader
from midst.data.physionet.physionet_writers import PhysioRecorder

import os
import numpy as np

SEX = {
    'M': 0,
    'F': 1,
}
TREATMENTS = {
    'Placebo': 0,
    'Ranolazine': 1,
    'Verapamil HCL': 2,
    'Quinidine Sulph': 3,
    'Dofetilide': 4,
}
frequency = 1000
save_dir = os.path.join(DATA_DIR, 'ECGRDVQ_DS')
os.makedirs(save_dir, exist_ok=True)
if __name__ == '__main__':
    rdr = ECGRDVQReader(db_path=os.path.join(DATA_DIR, 'ECGRDVQ_CSV'), detect_qrs_locations=False, csv_only=True)
    nan_conversion = lambda x: x if (
            isinstance(x, int) or isinstance(x, float)
    ) else (
        0 if isinstance(x, str) and x == 'NA' else float(x)
    )
    units_conversion = {
        'mg': 1,
        'ug': (1 / 1000),
        'ng': (1 / 1000000),
        'pg': (1 / 1000000000),
        'mg/mL': 1,
        'ug/mL': (1 / 1000),
        'ng/mL': (1 / 1000000),
        'pg/mL': (1 / 1000000000),
        'NA': 0,
        0: 0,
        0.0: 0,
    }
    records_per_id = {}
    for i in range(len(rdr)):
        record = rdr.read_record(i)
        record_data = {
            'signal': record['signal'],
            'qrs': record['qrs'],
            'waves': record['waves'],
            'rdr_index': np.array([i, ]),
            'sex': np.array([SEX[record['comments'][0]['SEX']], ]),
            'age': np.array([record['comments'][0]['AGE'], ]),
            'height': np.array([record['comments'][0]['HGHT'], ]),
            'weight': np.array([record['comments'][0]['WGHT'], ]),
            'race': record['comments'][0]['RACE'],
            'sysbp': record['comments'][0]['SYSBP'],
            'diabp': record['comments'][0]['DIABP'],
            'treatment': record['comments'][0]['EXTRT'],
            'treatment_key': np.array([TREATMENTS[record['comments'][0]['EXTRT']]]),
            'time_post_treatment': np.array([record['comments'][0]['TPT'], ]),
            'raw_file': record['comments'][0]['EGREFID'],
            'avg_rr': np.array([record['comments'][0]['RR'], ]),
            'avg_pr': np.array([record['comments'][0]['PR'], ]),
            'avg_qt': np.array([record['comments'][0]['QT'], ]),
            'avg_qrs': np.array([record['comments'][0]['QRS'], ]),
            'dosage': (
                    np.array([nan_conversion(record['comments'][0]['EXDOSE']), ]) *
                    units_conversion[record['comments'][0]['EXDOSU']]
            ),
            'erd_30': np.array([record['comments'][0]['ERD_30'], ]),
            'lrd_30': np.array([record['comments'][0]['LRD_30'], ]),
            't_amp': np.array([record['comments'][0]['Twave_amplitude'], ]),
            't_asym': np.array([record['comments'][0]['Twave_asymmetry'], ]),
            't_flat': np.array([record['comments'][0]['Twave_flatness'], ]),
        }

        if record['comments'][0]['RANDID'] not in records_per_id:
            records_per_id[record['comments'][0]['RANDID']] = [
                record_data,
            ]

        else:
            records_per_id[record['comments'][0]['RANDID']].append(
                record_data,
            )

    processed_records_per_id = records_per_id

    recorder = PhysioRecorder(
        save_dir=save_dir,
        db_name="ecgrdvq",
    )
    additional_arrays_keys = (
        'rdr_index',
        'sex',
        'age',
        'height',
        'weight',
        'time_post_treatment',
        'sysbp',
        'diabp',
        'treatment_key',
        'avg_rr',
        'avg_pr',
        'avg_qt',
        'avg_qrs',
        'dosage',
        'erd_30',
        'lrd_30',
        't_amp',
        't_asym',
        't_flat',
    )

    additional_attributes_keys = (
        'race',
        'raw_file',
        'treatment',
    )
    for key, recordings in processed_records_per_id.items():
        os.makedirs(os.path.join(save_dir, str(key)), exist_ok=True)

        for r, record in enumerate(recordings):
            additional_attributes = {
                additional_key: record[additional_key]
                for additional_key in additional_attributes_keys
            }
            additional_arrays = {
                additional_key: record[additional_key]
                for additional_key in additional_arrays_keys
            }

            if record['time_post_treatment'] < 0:
                sub_dir = 'pre_treatment'

            else:
                sub_dir = 'post_treatment'

            sub_dir = os.path.join(sub_dir, record['treatment'])
            current_save_dir = os.path.join(save_dir, str(key), sub_dir)
            os.makedirs(current_save_dir, exist_ok=True)
            raw_file_name = f"time_{record['time_post_treatment'].item()}_1.h5"

            if os.path.isfile(os.path.join(current_save_dir, raw_file_name)):
                raw_file_name = f"time_{record['time_post_treatment'].item()}_2.h5"

                if os.path.isfile(os.path.join(current_save_dir, raw_file_name)):
                    raw_file_name = f"time_{record['time_post_treatment'].item()}_3.h5"

            recorder(
                x=record['signal'],
                raw_file_name=raw_file_name,
                y=None,
                record_id=r,
                raw_qrs_inds=record['qrs'],
                additional_arrays=additional_arrays,
                additional_attributes=additional_attributes,
                save_dir=current_save_dir
            )
