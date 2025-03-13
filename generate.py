import io
import os
from flux_pipeline import FluxPipeline
import argparse
import csv
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
    parser.add_argument(
        "--seed",
        default=13456,
        type=int,
        help="Seed",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size",
    )
    return parser.parse_args()



if __name__ == '__main__':
    args = parse_args()
    pipe = FluxPipeline.load_pipeline_from_config_path(
        args.config_path
    )
    os.makedirs(args.save_path, exist_ok=True)

    prompts =["A beautiful asian woman in traditional clothing with golden hairpin and blue eyes, wearing a red kimono with dragon patterns",
            "A beautiful asian woman in traditional clothing with golden hairpin and blue eyes, wearing a red kimono with dragon patterns",
            "A beautiful asian woman in traditional clothing with golden hairpin and blue eyes, wearing a red kimono with dragon patterns"]

    csv_file = f"./prompt_flux_seed_{str(args.seed)}_meta.csv"
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

    for i, prompt in enumerate(prompts):
        output_jpeg_bytes: io.BytesIO = pipe.generate(
            # Required args:
            prompt=prompt,
            # Optional args:
            width=1024,
            height=1024,
            num_steps=25,
            guidance=3.5,
            num_images=args.batch_size,
            seed=args.seed,
        )
        for idx in range(args.batch_size):
            save_path = os.path.join(args.save_path, f"output-{i}-{idx}.jpg")

            row = (f"{prompt}", f"{save_path}")
            with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(row)

            with open(save_path, "wb") as f:
                f.write(output_jpeg_bytes[idx].getvalue())
