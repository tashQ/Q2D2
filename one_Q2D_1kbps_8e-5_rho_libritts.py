import os
from train import cli_main

def run_training():
    # Convert the config path to the format expected by Lightning CLI
    args = [
        "fit",  # Specify the subcommand (fit, validate, test, predict, or tune)
        "--config", 
        "/path/to/dir/configs/Q2D2_8e-5_1kbps_roh_3s_nq1_code777777_dim512_attn_batch16_libritts.yaml"
    ]
    
    cli_main(args=args)

if __name__ == "__main__":
    run_training()