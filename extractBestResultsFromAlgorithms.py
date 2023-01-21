from setting import config
from utils import read_information_from_result_from_models

import warnings
from glob import glob

warnings.simplefilter(action='ignore', category=FutureWarning)


def main():

    path = config['apksResultJsonPath']
    resultApksPath = config['resultApksPath']
    resultModelsPath = f'{resultApksPath}/resultModels'
    files = glob(resultModelsPath + '/*.json')

    data = read_information_from_result_from_models(files, resultModelsPath, resultApksPath)


if __name__ == '__main__':
    main()
