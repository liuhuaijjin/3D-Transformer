import os
import pandas as pd
from time import time
from mendeleev import element
import torch


def read_coremof(path_x='../../core-mof/', path_y='../../core-mof.csv', path_gas='../../Core-MOF_CO2_all.xlsx'):
    """ 1w MOF data """
    label = pd.read_csv(path_y, usecols=range(1, 13))
    label_gas = pd.read_excel(path_gas, engine='openpyxl')
    column_name = label.columns
    gas_name = label_gas.columns

    data_x, data_y, data_c, data_a, data_gas, data_name, miss = [], [], [], [], [], [], []
    file_list = os.listdir(path_x)
    start, tmp = time(), time()
    for n, i in enumerate(file_list):
        df = pd.read_csv(path_x + i, delimiter='\s+', header=None, skiprows=5, skipfooter=2, engine='python',
                         converters={7: lambda x: element(x).atomic_number})
        df_c = pd.read_csv(path_x + i, delimiter='\s+', header=None, skiprows=4, nrows=1, engine='python')

        # 以numpy还是torch保存更好没有定论 https://discuss.pytorch.org/t/save-a-tensor-to-file/37136/6
        x = torch.tensor(df[[1, 2, 3, 7]].to_numpy())
        c = torch.tensor(df_c[[0, 1, 2]].to_numpy())
        a = torch.tensor(df_c[[3, 4, 5]].to_numpy())
        y = torch.tensor(label[label['filename'] == i.split('.')[0]][column_name[1:]].to_numpy())
        if torch.min(y) < 0:
            continue
        if sum(label_gas['Framework name'] == i.split('.')[0]) > 0:
            gas = torch.tensor(label_gas[label_gas['Framework name'] == i.split('.')[0]][gas_name[1:]].to_numpy())
            if torch.min(gas) < 0:
                continue
            data_gas.append(gas)
        else:
            miss.append(n)
            data_gas.append(torch.tensor(0.0))
        data_x.append(x)
        data_y.append(y)
        data_c.append(c)
        data_a.append(a)
        data_name.append(i)

        if len(data_a) % 1000 == 0:
            print('Currently processing {}th sample using {:.2f}s.'.format(len(data_a), time() - tmp))
            tmp = time()

    # 将miss的值有均值替代
    data_gas = torch.cat(data_gas)
    for i in miss:
        data_gas[i] = torch.mean(data_gas, dim=0)

    # 将非气体标注与气体标注cat
    data_y = torch.cat((torch.cat(data_y), data_gas), dim=1)
    torch.save([data_x, data_y, torch.cat(data_c), torch.cat(data_a)], '../data/coremof.pt')
    torch.save(data_name, '../data/coremof_name.pt')
    print('Data loading finished with {} samples using {:.2f}s in total.'.format(len(data_a), time() - start))


def strut_complex_measure():
    x, y = torch.load('../data/coremof.pt')
    x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=0)
    data_x, data_pos = x[..., 3], x[..., :3]
    res = []
    for i in range(len(data_x)):
        mask = (data_x[i] != 0)
        pos = data_pos[i][mask]
        xyz_len = torch.max(pos, dim=0)[0] - torch.min(pos, dim=0)[0]

        # 将晶胞扩展10倍
        xyz_sort = torch.sort(xyz_len)[0]
        com = (xyz_sort[0] / xyz_sort[1] + xyz_sort[1] / xyz_sort[2] - 1) * torch.tanh(torch.tensor(len(pos) * 10).float())
        res.append(com.item())

    torch.save(res, '../../coremof.pt')
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.displot(res, kind="kde", bw_adjust=.5)
    plt.show()
    print(f'Average Structure Complexity is {sum(res) / len(data_x)}')


strut_complex_measure()
