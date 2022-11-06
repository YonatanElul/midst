from midst import DATA_DIR
from midst.data.datasets import ECGRDVQDataOrganizer

import os


if __name__ == '__main__':
    data_dir = os.path.join(DATA_DIR, "ECGRDVQ_DS")
    cache_path = os.path.join(DATA_DIR, "ECGRDVQ_CSV")
    csv_only = True
    arrays_keys = (
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
    )
    attributes_keys = (
        'db_name',
        'race',
    )

    ECGRDVQDataOrganizer(
        dir_path=data_dir,
        cache_path=cache_path,
        arrays_keys=arrays_keys,
        attributes_keys=attributes_keys,
        csv_only=csv_only,
    )


