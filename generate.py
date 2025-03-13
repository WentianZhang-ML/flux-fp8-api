import io
import os
from flux_pipeline import FluxPipeline
import argparse
import csv
def parse_args():
    parser = argparse.ArgumentParser(description="Launch Flux API server")
    parser.add_argument(
        "--config-path",
        type=str,
        help="Path to the configuration file, if not provided, the model will be loaded from the command line arguments",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        help="Path to the saved generated images",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        help="Path to the csv file containing <prompt, save_path> for each row",
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
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="width of images",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="height of images",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=25,
        help="number of steps for generation",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=3.5,
        help="cfg guidance weight",
    )
    return parser.parse_args()



if __name__ == '__main__':
    args = parse_args()

    pipe = FluxPipeline.load_pipeline_from_config_path(
        args.config_path
    )
    prompts =[
            "A beautiful asian woman in traditional clothing with golden hairpin and blue eyes, wearing a red kimono with dragon patterns",
            "A beautiful asian woman in traditional clothing with golden hairpin and blue eyes, wearing a red kimono with dragon patterns",
            "A beautiful asian woman in traditional clothing with golden hairpin and blue eyes, wearing a red kimono with dragon patterns"
            ]

    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.csv_path, exist_ok=True)


    csv_file = f"{args.csv_path}/prompt_flux_{args.width}x{args.height}_{args.num_steps}_{args.guidance}-seed_{str(args.seed)}_meta.csv"
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

    for i, prompt in enumerate(prompts):
        output_jpeg_bytes: io.BytesIO = pipe.generate(
            # Required args:
            prompt=prompt,
            # Optional args:
            width=args.width,
            height=args.height,
            num_steps=args.num_steps,
            guidance=args.guidance,
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
