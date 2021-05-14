"""
Method adapted from GLAM function `run_model` by
Kangning Liu, Yiqiu Shen, Nan Wu, Jakub Chledowski, Carlos Fernandez-Granda, 
and Krzysztof J. Geras, which is licensed under a GNU Affero General Public License v3.0.
"""

import sys
import argparse

from src.scripts.run_model import start_experiment

print(sys.version, sys.platform, sys.executable)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference')
    parser.add_argument('--model-path', default='models/')
    parser.add_argument('--data-path', default='sample_data/data.pkl')
    parser.add_argument('--image-path', default='sample_data/cropped_images/')
    parser.add_argument('--segmentation-path', default='sample_data/segmentation/')
    parser.add_argument('--output-path', default='sample_output/')
    parser.add_argument('--device-type', choices=['gpu', 'cpu'], default='cpu')
    parser.add_argument('--gpu-number', default=0, type=int)
    parser.add_argument('--model-index', type=str, choices=["model_joint"], default="model_joint")
    parser.add_argument('--visualization-flag', choices=[False, True], default=True)
    args = parser.parse_args()

    parameters = {
        "device_type": args.device_type,
        "gpu_number": args.gpu_number if args.gpu_number != '' else 0,
        "max_crop_noise": (100, 100),
        "max_crop_size_noise": 100,
        "image_path": args.image_path,
        "segmentation_path": args.segmentation_path,
        "output_path": args.output_path,
        # model related hyper-parameters
        "cam_size": (184, 120),  # the saliency map size for the global module
        "top_k_per_class": 1,
        "crop_shape": (512, 512),
        'percent_k': 0.03,
    }

    start_experiment(
        model_path=args.model_path,
        data_path=args.data_path,
        output_path=args.output_path,
        model_index=args.model_index,
        parameters=parameters,
        turn_on_visualization=args.visualization_flag,
    )

    # print(f"\nFinished. You can see the predictions under {args.output_path}/predictions.csv, and the visualization under {args.output_path}/visualization.")
