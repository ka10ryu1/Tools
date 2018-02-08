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
    parser.add_argument('-l', '--label', default='loss',
                        help='取得するラベル(default: loss, other: lr)')
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


def subplot(sub, val, log, ylim):
    """
    subplotを自動化
    [in] sub:  subplotオブジェクト
    [in] val:  入力する値のリスト
    [in] log:  入力するラベルのリスト
    [in] ylim: auto_ylimを使用する場合はTrue
    """

    # グリッドを灰色の点線で描画する
    sub.grid(which='major', color='gray', linestyle=':')
    sub.grid(which='minor', color='gray', linestyle=':')
    sub.set_yscale("log")
    # args.auto_ylimが設定された場合、ylimを設定する
    # ymax: 各データの1/8番目（400個データがあれば50番目）のうち最小の数を最大値とする
    # ymin: 各データのうち最小の数X0.98を最小値とする
    if ylim:
        ymax = np.min([i[int(len(i) / 8)] for i in val])
        ymin = np.min([np.min(i)for i in val]) * 0.98
        sub.set_ylim([ymin, ymax])
        print('ymin:{0:.4f}, ymax:{1:.4f}'.format(ymin, ymax))

    # プロット
    [sub.plot(np.array(v), label=d) for v, d in zip(val, log)]


def savePNG(plt, loc, name, dpi=200):
    """
    png形式での保存を自動化
    [in] plt:  pltオブジェクト
    [in] loc:  ラベルの位置
    [in] name: 保存するファイル名
    [in] dpi:  保存時の解像度
    """

    plt.legend(loc=loc)
    plt.savefig(getFilePath(args.out_path, name, '.png'), dpi=dpi)


def plot(args, search, loc, name):
    """
    プロットメイン部
    [in] args:   オプション引数
    [in] search: 探索ラベル
    [in] loc:    ラベルの位置
    [in] name:   保存するファイル名
    """

    val = []
    log_file = []
    for l in sortTimeStamp(args.log_dir, '.log'):
        log_file.append(l)
        print(log_file[-1])
        data = jsonRead(log_file[-1])
        val.append([i[search] for i in data if(search in i.keys())])

    # logファイルが見つからなかった場合、ここで終了
    if not val:
        print('[Error] .log not found')
        exit()

    # 対数グラフの設定
    f = plt.figure(figsize=(10, 6))
    a = f.add_subplot(111)
    subplot(a, val, log_file, args.auto_ylim)

    # グラフの保存と表示
    savePNG(plt, loc, name)
    plt.show()


def main(args):
    if(args.label == 'loss'):
        plot(args, 'validation/main/loss', 'upper right', 'plot_diff_loss')
    elif(args.label == 'acc'):
        plot(args, 'validation/main/accuracy', 'lower right', 'plot_diff_acc')
    elif(args.label == 'lr'):
        plot(args, 'lr', 'lower right', 'plot_diff_lr')


if __name__ == '__main__':
    args = command()
    argsPrint(args)
    main(args)
