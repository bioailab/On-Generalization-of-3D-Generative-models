import argparse
import os
from pytorch_fid.fid_score import calculate_fid_given_paths

def calculate_fid_for_subdirs(parent_dir, gt_dir, batch_size, device, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # for subdir in os.listdir(parent_dir):
    gen_dir = os.path.join(parent_dir)
    # if not os.path.isdir(gen_dir):
    #     continue
    
    fid = 0
    for view_index in range(20):
        gt_view_dir = os.path.join(gt_dir, f"view_{view_index}")
        gen_view_dir = os.path.join(gen_dir, f"view_{view_index}")
        view_score = calculate_fid_given_paths([gt_view_dir, gen_view_dir],
                                                batch_size,
                                                device,
                                                dims=2048,
                                                num_workers=8)
        fid += view_score
        
    fid /= 20
    subdir = parent_dir.split('/')[4]
    output_file = os.path.join(output_dir, f"1000_fid.txt")
    with open(output_file, 'w') as f:
        f.write(f'FID Value: {fid:.4f}')
    # break
    print(f'FID value for {subdir} saved to {output_file}')

def main(args):
    calculate_fid_for_subdirs(
        parent_dir=args.parent_dir,
        gt_dir=args.gt_dir,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-dir", type=str, default="/home/workspace/FID_eval_data/1000-homo/53", help="Parent directory containing subdirectories for gen folders")
    parser.add_argument("--gt-dir", type=str, default="/home/workspace/FID_eval_data/boat_gt", help="Ground truth directory")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for FID calculation")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for FID calculation (e.g., 'cuda' or 'cpu')")
    parser.add_argument("--output-dir", type=str, default="/home/workspace/FID/Boat", help="Directory to save FID results")

    args = parser.parse_args()

    main(args)
