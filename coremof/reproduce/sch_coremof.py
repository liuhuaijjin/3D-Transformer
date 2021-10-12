import os


def load_coremof():
    import pandas as pd
    import numpy as np
    from ase.io import read
    from schnetpack import AtomsData
    try:
        os.remove('../../coremof.db')
    except FileNotFoundError:
        pass
    new_dataset = AtomsData('../../coremof.db', available_properties=['LCD', 'PLD', 'D', 'ASA', 'NASA', 'AV'])
    y = pd.read_csv('../../core-mof.csv')
    cif_path = '../../coremof_cif/'
    file_list = os.listdir(cif_path)

    for n, i in enumerate(file_list):
        property_list = []
        prop = y[y['filename'] == i.split('.')[0]]

        LCD = np.array(prop['LCD'], dtype=np.float32)
        PLD = np.array(prop['PLD'], dtype=np.float32)
        D = np.array(prop['D'], dtype=np.float32)
        ASA = np.array(prop['ASA'], dtype=np.float32)
        NASA = np.array(prop['NASA'], dtype=np.float32)
        AV = np.array(prop['AV'], dtype=np.float32)
        property_list.append({'LCD': LCD, 'PLD': PLD, 'D': D, 'ASA': ASA, 'NASA': NASA, 'AV': AV})

        atoms = read(cif_path + i, index=':')
        new_dataset.add_systems(atoms, property_list)
        if n > 0 and n % 1000 == 0:
            print(n)

    example = new_dataset[0]
    print('Properties of molecule with id 0:')

    for k, v in example.items():
        print('-', k, ':', v.shape)


if __name__ == '__main__':
    load_coremof()