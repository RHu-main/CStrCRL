import argparse
import sys
import os
import shutil
import zipfile
import time

# torchlight
import torchlight as torchlight
from torchlight import import_class

from processor.processor import init_seed

init_seed(0)


def save_src(target_path):
    code_root = os.getcwd()
    srczip = zipfile.ZipFile('./src.zip', 'w')
    for root, dirnames, filenames in os.walk(code_root):
        for filename in filenames:
            if filename.split('\n')[0].split('.')[-1] == 'py':
                srczip.write(os.path.join(root, filename).replace(code_root, '.'))
            if filename.split('\n')[0].split('.')[-1] == 'yaml':
                srczip.write(os.path.join(root, filename).replace(code_root, '.'))
            if filename.split('\n')[0].split('.')[-1] == 'ipynb':
                srczip.write(os.path.join(root, filename).replace(code_root, '.'))
    srczip.close()
    save_path = os.path.join(target_path, 'src_%s.zip' % time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime()))
    shutil.copy('./src.zip', save_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Processor collection')
    processors = dict()

    processors['pretrain_CStrCRL_ST'] = import_class('processor.pretrain_CStrCRL_ST.CStrCRL_Processor')
    processors['linear_evaluation_ST'] = import_class('processor.linear_evaluation.LE_Processor_ST')

    # add sub-parser
    subparsers = parser.add_subparsers(dest='processor')
    for k, p in processors.items():
        subparsers.add_parser(k, parents=[p.get_parser()])

    # read arguments
    arg = parser.parse_args()

    # start
    Processor = processors[arg.processor]
    p = Processor(sys.argv[2:])

    if p.arg.phase == 'train':
        # save src
        save_src(p.arg.work_dir)

    p.start()
