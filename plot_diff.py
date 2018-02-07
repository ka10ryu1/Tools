#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = 'logファイルの複数比較'
#

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

from func import argsPrint, getFilePath, sortTimeStamp


def command():
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument('log_dir', nargs='+',
                        help='入力データセットのフォルダ')
    parser.add_argument('--auto_ylim', action='store_true',
                        help='ylim自動設定')
    parser.add_argument('-o', '--out_path', default='./result/',
                        help='生成物の保存先(default: ./result/)')

    return parser.parse_args()


def jsonRead(path):
    """
    chainerのextensionで出力されたlogをjsonで読み込む
    [in]  path: logのパス
    [out] d:    読み込んだ辞書データ
    """

    try:
        with open(path, 'r') as f:
            d = json.load(f)

    except json.JSONDecodeError as e:
        print('JSONDecodeError: ', e)
        exit()

    return d


def main(args):
    loss_str = 'validation/main/loss'
    acc_str = 'validation/main/accuracy'

    vml = []
    vma = []
    log_file = []
    for l in sortTimeStamp(args.log_dir, '.log'):
        log_file.append(l)
        print(log_file[-1])
        data = jsonRead(log_file[-1])
        vml.append([i[loss_str] for i in data if(loss_str in i.keys())])
        vma.append([i[acc_str] for i in data if(acc_str in i.keys())])

    # logファイルが見つからなかった場合、ここで終了
    if not vml:
        print('[Error] .log not found')
        exit()

    if not vma[0]:
        sub = 111
        loc = 'upper left'
    else:
        sub = 121
        loc = 'lower right'

    # 対数グラフの設定
    f = plt.figure(figsize=(10, 6))
    a = f.add_subplot(sub)
    a.grid(which='major', color='gray', linestyle=':')
    a.grid(which='minor', color='gray', linestyle=':')
    a.set_yscale("log")
    # args.auto_ylimが設定された場合、ylimを設定する
    # ymax: 各データの1/8番目（400個データがあれば50番目）のうち最小の数を最大値とする
    # ymin: 各データのうち最小の数X0.98を最小値とする
    if args.auto_ylim:
        ymax = np.min([i[int(len(i) / 8)] for i in vml])
        ymin = np.min([np.min(i)for i in vml]) * 0.98
        a.set_ylim([ymin, ymax])
        print('loss ymin:{0:.4f}, ymax:{1:.4f}'.format(ymin, ymax))

    # 数値のプロット
    [a.plot(np.array(v), label=d) for v, d in zip(vml, log_file)]

    if vma[0]:
        b = f.add_subplot(sub + 1)
        b.grid(which='major', color='black', linestyle='-')
        b.grid(which='minor', color='black', linestyle='-')
        # args.auto_ylimが設定された場合、ylimを設定する
        if args.auto_ylim:
            b.set_ylim([0.7, 1])
            print('acc ymin:{0:.4f}, ymax:{1:.4f}'.format(ymin, ymax))

        # 数値のプロット
        [b.plot(np.array(v), label=d) for v, d in zip(vma, log_file)]

    # グラフの保存と表示
    plt.legend(loc=loc)
    plt.savefig(
        getFilePath(args.out_path, 'plot_diff', '.png'),
        dpi=200
    )
    plt.show()


if __name__ == '__main__':
    args = command()
    argsPrint(args)
    main(args)
