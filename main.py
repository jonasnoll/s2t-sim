from opts import parse_opts
from utils.log import Log
from utils.utils import create_directory
from pipeline import S2TFiltering


# Code for KIT Hector School Master Thesis "Source to Target Similarity Filtering for Unsupervised Domain Adaptation"
# https://github.com/jonasnoll/s2t-sim
# If you consider using this code or its derivatives, please consider citing:
# @misc{
#     title = {Source to Target Similarity Filtering for Unsupervised Domain Adaptation},
#     author = {Noll, Jonas},
#     year = {2022}
# }
# Please contact me through email, if you have any questions or need any help: uguyj (at) student . kit . edu
if __name__ == "__main__":
    print("Starting...")
    opts = parse_opts()
    # Logging
    create_directory('./logs')
    log = Log(opts.exp_id).log
    log.info("Starting pipeline.py...") 
    # Pipeline
    s2t = S2TFiltering(opts)
    s2t.run_s2t_filtering()
