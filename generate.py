import io
import os
from flux_pipeline import FluxPipeline
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Launch Flux API server")
    parser.add_argument(
        "-c",
        "--config-path",
        type=str,
        help="Path to the configuration file, if not provided, the model will be loaded from the command line arguments",
    )
    parser.add_argument(
        "-s",
        "--save-path",
        type=str,
        help="Path to the configuration file, if not provided, the model will be loaded from the command line arguments",
    )
    return parser.parse_args()



if __name__ == '__main__':
    args = parse_args()
    pipe = FluxPipeline.load_pipeline_from_config_path(
        args.config_path
    )
    os.makedirs(args.save_path, exist_ok=True)

    for i in range(3):
        output_jpeg_bytes: io.BytesIO = pipe.generate(
            # Required args:
            prompt="A beautiful asian woman in traditional clothing with golden hairpin and blue eyes, wearing a red kimono with dragon patterns",
            # Optional args:
            width=1024,
            height=1024,
            num_steps=25,
            guidance=3.5,
            num_images=32,
            seed=13456,
        )

    with open(os.path.join(args.save_path, "output.jpg"), "wb") as f:
        f.write(output_jpeg_bytes.getvalue())
