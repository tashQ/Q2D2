import os
import glob
from periodicity import calculate_periodicity_metrics
import torchaudio
from pesq import pesq
import numpy
import torch
import math
from pystoi import stoi
import csv

eval_UTMOS = True
flac_wav = 'wav'

if eval_UTMOS:
    from UTMOS import UTMOSScore

device=torch.device('cuda:0')


def main():
    main_path = "/path/to/dir/files/after/infer/process"
    prepath=(main_path + "/out")
    rawpath=(main_path + "/in")
    preaudio = os.listdir(prepath)
    rawaudio = []

    if eval_UTMOS==True:
        UTMOS=UTMOSScore(device='cuda:0')
    
    # List of files
    for i in range(len(preaudio)):
        input_file = preaudio[i]
        if flac_wav=='wav':
            input_file = (input_file[:-7]+ "in.wav")
        else:
            input_file = (input_file[:-8]+ "in.wav")
            
        rawaudio.append(rawpath + "/" + input_file)
        print(preaudio[i])
        print(rawaudio[i])

  
    # Write results to CSV
    output_file = "evaluation_results.csv"
    output_path = (rawpath[:-2] + "results/" + output_file)

    # ensure results directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    write_header = not os.path.exists(output_path)

    
    with open(output_path, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["Raw File", "Processed File", "UTMOS_raw", "UTMOS_reconstruct", "PESQ", "F1_score", "STOI"])
        if write_header:
            writer.writeheader()
            
            
        new_sample = 16000
        utmos_sumgt=0
        utmos_sumencodec=0
        pesq_sumpre=0
        f1score_sumpre=0
        stoi_sumpre=[]
        f1score_filt=0

        for i in range(len(preaudio)):
            print(i)
            rawwav,rawwav_sr=torchaudio.load(rawaudio[i])
            prewav,prewav_sr=torchaudio.load(prepath+"/"+preaudio[i])
            rawwav=rawwav.to(device)
            prewav=prewav.to(device)
            rawwav_resampled=torchaudio.functional.resample(rawwav, orig_freq=rawwav_sr, new_freq=new_sample)  
            prewav_resampled=torchaudio.functional.resample(prewav, orig_freq=prewav_sr, new_freq=new_sample)

            ##1.UTMOS
            if eval_UTMOS==True:
                UTMOS_raw = UTMOS.score(rawwav_resampled.unsqueeze(1))[0].item()
                UTMOS_reconstruct = UTMOS.score(prewav_resampled.unsqueeze(1))[0].item()
                print("****UTMOS_raw",i,UTMOS_raw)
                print("****UTMOS_encodec",i,UTMOS_reconstruct)
                utmos_sumgt+=UTMOS_raw
                utmos_sumencodec+=UTMOS_reconstruct
            else:
                utmos_sumgt = 0 
    
            ## 2.PESQ  
            min_len=min(rawwav_resampled.size()[1],prewav_resampled.size()[1])
            rawwav_resampled_pesq=rawwav_resampled[:,:min_len].squeeze(0)
            prewav_resampled_pesq=prewav_resampled[:,:min_len].squeeze(0)
            pesq_score = pesq(new_sample, rawwav_resampled_pesq.cpu().numpy(), prewav_resampled_pesq.cpu().numpy(), "wb", on_error=1)
            print("****PESQ",i,pesq_score)
            pesq_sumpre+=pesq_score

            ## 3.F1-score
            min_len=min(rawwav_resampled.size()[1],prewav_resampled.size()[1])
            rawwav_resampled_f1score=rawwav_resampled[:,:min_len]
            prewav_resampled_f1score=prewav_resampled[:,:min_len]
            periodicity_loss, pitch_loss, f1_score = calculate_periodicity_metrics(rawwav_resampled_f1score,prewav_resampled_f1score)
            print("****f1",periodicity_loss, pitch_loss, f1_score,f1score_sumpre)
            if(math.isnan(f1_score)):
                f1score_filt+=1
                print("*****",f1score_filt)
            else:
                f1score_sumpre+=f1_score

            ## 4.STOI
            min_len=min(rawwav.size()[1],prewav.size()[1])
            rawwav_stoi=rawwav[:,:min_len].squeeze(0)
            prewav_stoi=prewav[:,:min_len].squeeze(0)
            tmp_stoi=stoi(rawwav_stoi.cpu(),prewav_stoi.cpu(),rawwav_sr,extended=False)
            print("****stoi",tmp_stoi)
            stoi_sumpre.append(tmp_stoi)
            
            # Write results to CSV immediately
            writer.writerow({
                "Raw File": rawaudio[i],
                "Processed File": preaudio[i],
                "UTMOS_raw": UTMOS_raw,
                "UTMOS_reconstruct": UTMOS_reconstruct,
                "PESQ": pesq_score,
                "F1_score": f1_score,
                "STOI": tmp_stoi
            })
            file.flush()  # Ensures data is written immediatel

        # Calculate final metrics
        pesq_avg = pesq_sumpre / len(preaudio)
        f1_score_avg = f1score_sumpre / (len(preaudio) - f1score_filt) if (len(preaudio) - f1score_filt) > 0 else "N/A"
        stoi_mean = numpy.mean(stoi_sumpre)

        print("************* Final Results *************")
        if eval_UTMOS:
            print("UTMOS_raw avg:", utmos_sumgt / len(preaudio))
            print("UTMOS_reconstruct avg:", utmos_sumencodec / len(preaudio))
        print("PESQ:", pesq_sumpre, pesq_avg)
        print("F1 Score:", f1score_sumpre, f1_score_avg, f1score_filt)
        print("STOI:", stoi_mean)
        
        # Append summary row to CSV
        writer.writerow({
            "Raw File": "AVERAGE",
            "Processed File": "",
            "UTMOS_raw": utmos_sumgt / len(preaudio) if eval_UTMOS else "",
            "UTMOS_reconstruct": utmos_sumencodec / len(preaudio) if eval_UTMOS else "",
            "PESQ": pesq_avg,
            "F1_score": f1_score_avg,
            "STOI": stoi_mean
        })


if __name__ == "__main__":
    main()