# # from utils.datasets import MNISTDataset
# from utils.load_img import load_images

# # print(type(["asdf"]))

# train_ds, test_ds = load_images(['sv'])

# print(f"Data has len {len(train_ds)}")

# from pipeline import run_s2t_filtering
from pipeline import S2TFiltering
from utils.log import Log
from opts import parse_opts


EXPERIMENT_ID = "Trial"


if __name__ == "__main__":
    opts = parse_opts()
    print()
    print(opts)
    print()

    log = Log(opts.exp_id).log
    log.info("Starting pipeline.py...")
    print("Starting...")

    s2t = S2TFiltering(opts)
    s2t.run_s2t_filtering()
