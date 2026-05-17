import os
import sys
import argparse
from tqdm import tqdm
from glob import glob


def get_args():
    parser = argparse.ArgumentParser(
        description='Symlink silhouette data and parsing data into the same folder for PAGait training.'
    )

    parser.add_argument(
        '--parsing_data_path',
        type=str,
        required=True,
        help="Path of parsing data (absolute path)."
    )

    parser.add_argument(
        '--silhouette_data_path',
        type=str,
        required=True,
        help="Path of silhouette data (absolute path)."
    )

    parser.add_argument(
        '--dataset_pkl_ext_name',
        type=str,
        default='.pkl',
        help="Extension name for silhouette .pkl files."
    )

    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help="Path of output data."
    )

    opt = parser.parse_args()
    return opt


def main():
    opt = get_args()

    parsing_data_path = opt.parsing_data_path
    silhouette_data_path = opt.silhouette_data_path

    if not os.path.exists(parsing_data_path):
        print(f"Parsing data path {parsing_data_path} does not exist.")
        sys.exit(1)

    if not os.path.exists(silhouette_data_path):
        print(f"Silhouette data path {silhouette_data_path} does not exist.")
        sys.exit(1)

    all_parsing_files = sorted(
        glob(os.path.join(parsing_data_path, "*/*/*/*.pkl"))
    )

    all_silhouette_files = sorted(
        glob(os.path.join(
            silhouette_data_path,
            f"*/*/*/*{opt.dataset_pkl_ext_name}"
        ))
    )

    if len(all_parsing_files) >= len(all_silhouette_files):

        for parsing_file in tqdm(all_parsing_files):

            tmp_list = parsing_file.split('/')

            sil_folder = os.path.join(
                silhouette_data_path,
                *tmp_list[-4:-1]
            )

            if not os.path.exists(sil_folder):
                print(f"Silhouette folder {sil_folder} does not exist.")
                continue

            sil_files = sorted(
                glob(os.path.join(
                    sil_folder,
                    f"*{opt.dataset_pkl_ext_name}"
                ))
            )

            if len(sil_files) == 0:
                print(f"No silhouette file found in {sil_folder}")
                continue

            silhouette_file = sil_files[0]

            output_file = os.path.join(
                opt.output_path,
                *tmp_list[-4:-1]
            )

            os.makedirs(output_file, exist_ok=True)

            os.system(
                f"ln -s {silhouette_file} {output_file}/1_sil.pkl"
            )

            os.system(
                f"ln -s {parsing_file} {output_file}/0_parsing.pkl"
            )

    else:

        for silhouette_file in tqdm(all_silhouette_files):

            tmp_list = silhouette_file.split('/')

            parsing_folder = os.path.join(
                parsing_data_path,
                *tmp_list[-4:-1]
            )

            if not os.path.exists(parsing_folder):
                print(f"Parsing folder {parsing_folder} does not exist.")
                continue

            parsing_files = sorted(
                glob(os.path.join(parsing_folder, "*.pkl"))
            )

            if len(parsing_files) == 0:
                print(f"No parsing file found in {parsing_folder}")
                continue

            parsing_file = parsing_files[0]

            output_file = os.path.join(
                opt.output_path,
                *tmp_list[-4:-1]
            )

            os.makedirs(output_file, exist_ok=True)

            os.system(
                f"ln -s {silhouette_file} {output_file}/1_sil.pkl"
            )

            os.system(
                f"ln -s {parsing_file} {output_file}/0_parsing.pkl"
            )

    print("Done! Output data is in:", opt.output_path)


if __name__ == "__main__":
    main()