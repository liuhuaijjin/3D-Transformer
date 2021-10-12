import os
from time import time, strftime, localtime

import torch
import torch.optim as opt
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import mse_loss
import sys

sys.path.append("..")
from model.tr_spe import build_model
from utils.utils import parse_args, Logger, set_seed


def main():
    args = parse_args()
    log = Logger('../' + args.save_path + 'coremof/', f'base_{strftime("%Y-%m-%d_%H-%M-%S", localtime())}.log')
    set_seed(args.seed)
    args.bs = 8 * len(args.gpu_id.split(','))
    args.lr = 1e-4 * len(args.gpu_id.split(','))

    x, y = torch.load('../data/coremof.pt')
    x = pad_sequence([i[:args.max_len - 1] for i in x], batch_first=True, padding_value=0)
    x, pos = x[..., 3], x[..., :3]
    x = torch.cat((torch.ones(x.shape[0]).unsqueeze(-1) * args.atom_class, x), dim=1)
    pos = torch.cat((torch.mean(pos, dim=-2, keepdim=True), pos), dim=1)
    scales = [[y[:, i].mean().item(), y[:, i].std().item()] for i in range(y.shape[-1])]
    for i in range(y.shape[-1]):
        y[:, i] = (y[:, i] - scales[i][0]) / scales[i][1]
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    x, pos, y = x.long().cuda(), pos.cuda(), y.cuda()

    train_size = int(0.05 * len(x))
    test_size = len(x) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(TensorDataset(x, pos, y), [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.bs * 2)

    # 打印数据集信息和部分训练信息，计算unique原子时去掉占位原子
    targets = ['LCD', 'PLD', 'D', 'ASA', 'NASA', 'AV']
    log.logger.info(f'{"=" * 20} CORE-MOF {"=" * 20}\n Encoder: {args.n_encoder}; Head: {args.head}; '
                    f'Embed_dim: {args.embed_dim}; Max_len: {x.shape[1]}\nTrain: {train_size}; Test: {test_size}; '
                    f'Unique_class: {len(torch.unique(x[..., 3])) - 2}\nTarget Label: {targets}\n'
                    f'GPU: {args.gpu_id}; Batch_size: {args.bs} \n{"=" * 20} Start Training {"=" * 20}')
    criterion = torch.nn.MSELoss()
    best_mse = [1e9] * len(targets)

    for i in range(5, len(targets)):
        t0, early_stop = time(), 0
        best_loss = 1e9
        log.logger.info(f'Training {targets[i]}')
        model = build_model(args.atom_class + 1, 1, dropout=args.dropout).cuda()
        if len(args.gpu_id) > 1: model = torch.nn.DataParallel(model)
        optimizer = opt.Adam(model.parameters(), lr=args.lr)
        lr_scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.6, patience=10, min_lr=5e-7)

        for epoch in range(300):
            model.train()
            loss = 0.0
            t1 = time()
            for x, pos, y in train_loader:
                mask = (x != 0).unsqueeze(1)
                pred = model(x.long(), mask, pos)
                loss_batch = criterion(pred[..., 0], y[:, i])
                loss += loss_batch.item() / (train_size * args.bs)
                optimizer.zero_grad()
                loss_batch.backward()
                optimizer.step()

            # 每个轮次都测试模型，不能简单的整个test进行预测，会内存爆炸
            model.eval()
            mse = 0
            for x, pos, y in test_loader:
                mask = (x != 0).unsqueeze(1)
                with torch.no_grad():
                    pred = model(x.long(), mask, pos)
                mse += mse_loss(pred[..., 0], y[:, i], reduction='sum').item() / test_size * scales[i][1]

            if mse < best_mse[i]:
                best_mse[i] = mse
                best_epoch = epoch + 1
            if loss < best_loss:
                best_loss = loss
                early_stop = 0
            else:
                early_stop += 1

            lr_scheduler.step(mse)
            loss *= args.bs / train_size
            log.logger.info('Epoch: {} | Time: {:.1f}s | Loss: {:.2f} | MSE: {:.3f} | Lr: {:.3f}'.
                            format(epoch + 1, time() - t1, loss, mse, optimizer.param_groups[0]['lr'] * 1e5))
            if early_stop >= 30:
                log.logger.info(f'Early Stopping!!! No Improvement on Loss for 30 Epochs.')
                break
        log.logger.info('{} End Training (Time: {:.2f}h) {}'.format("=" * 20, (time() - t0) / 3600, "=" * 20))
        log.logger.info('Best Epoch: {} | MSE: {}'.format(best_epoch, [round(i, 3) for i in best_mse]))


if __name__ == '__main__':
    main()
