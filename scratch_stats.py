import json
import statistics
import glob
from pathlib import Path

for f in sorted(glob.glob('benchmarks/results/*.json')):
    if 'raw' in f or 'accuracy' in f or 'sweep' in f: continue
    data = json.load(open(f))
    print('---')
    print(Path(f).name)
    if 'results' in data: # phase 0,1,2,4
        ttfts = [r['ttft_ms']['mean'] for r in data['results']]
        ttft_stds = [r['ttft_ms']['std'] for r in data['results']]
        tpots = [r['tpot_ms']['mean'] for r in data['results']]
        tpot_stds = [r['tpot_ms']['std'] for r in data['results']]
        tps = [r['throughput_tok_per_sec']['mean'] for r in data['results']]
        tp_stds = [r['throughput_tok_per_sec']['std'] for r in data['results']]
        
        avg_ttft = statistics.mean(ttfts)
        avg_ttft_std = statistics.mean(ttft_stds)
        avg_tpot = statistics.mean(tpots)
        avg_tpot_std = statistics.mean(tpot_stds)
        avg_tp = statistics.mean(tps)
        avg_tp_std = statistics.mean(tp_stds)
        print(f"TTFT: {avg_ttft:.2f} \u00b1 {avg_ttft_std:.2f} ms")
        print(f"TPOT: {avg_tpot:.2f} \u00b1 {avg_tpot_std:.2f} ms/tok")
        print(f"Throughput: {avg_tp:.2f} \u00b1 {avg_tp_std:.2f} tok/s")
        print(f"VRAM: {data.get('peak_vram_gb', '')} GB")
    elif 'aggregate' in data: # phase 3, 5
        tp = data["aggregate"]["throughput_tok_per_sec"]["mean"]
        tp_std = data["aggregate"]["throughput_tok_per_sec"]["std"]
        print(f"Aggregate Throughput: {tp:.2f} \u00b1 {tp_std:.2f} tok/s")
        if 'ttft_ms' in data['aggregate']:
            t = data["aggregate"]["ttft_ms"]["mean"]
            t_std = data["aggregate"]["ttft_ms"]["std"]
            print(f"TTFT: {t:.2f} \u00b1 {t_std:.2f} ms")
        print(f"VRAM: {data.get('peak_vram_gb', '')} GB")
