"""
模型資源分析工具 — 視覺化運算資源, 評估嵌入式部署可行性

用法:
    python profile_model.py \
        --hifigan_checkpoint hifigan_finetuned/checkpoint \
        --model_path models/transduction_model/model.pt

輸出:
    output/profile_report.html   — 互動式圖表
    output/profile_summary.txt   — 文字摘要
"""

import os
import sys
import time
import numpy as np
import json
import logging

import torch
import torch.nn as nn

from architecture import Model
from data_utils import phoneme_inventory
from vocoder import Vocoder, StreamingVocoder
from read_emg import EMGDataset
from quantize_int16 import load_quantized_model

from absl import flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model_path',
                    'models/transduction_model/model.pt',
                    '模型檔案路徑')
flags.DEFINE_string('profile_output', 'output/profile_report.html',
                    '分析報告輸出路徑')
flags.DEFINE_integer('warmup_runs', 5, '暖機次數')
flags.DEFINE_integer('benchmark_runs', 20, '測量次數')

try:
    flags.DEFINE_boolean('quantized', False, '是否使用 INT16 量化模型')
except flags.DuplicateFlagError:
    pass


def count_parameters(module, name=""):
    """遞迴計算每個子模組的參數量"""
    results = []
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)

    results.append({
        'name': name or module.__class__.__name__,
        'total_params': total,
        'trainable_params': trainable,
        'size_mb': total * 4 / 1024 / 1024,  # float32
    })

    for child_name, child in module.named_children():
        full_name = f"{name}.{child_name}" if name else child_name
        child_total = sum(p.numel() for p in child.parameters())
        child_trainable = sum(p.numel() for p in child.parameters()
                              if p.requires_grad)
        if child_total > 0:
            results.append({
                'name': full_name,
                'total_params': child_total,
                'trainable_params': child_trainable,
                'size_mb': child_total * 4 / 1024 / 1024,
            })

    return results


def measure_latency(func, warmup=5, runs=20):
    """測量函式延遲 (ms)"""
    # Warmup
    for _ in range(warmup):
        func()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    for _ in range(runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        func()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'p95_ms': np.percentile(times, 95),
    }


def measure_memory(device):
    """測量 GPU 記憶體使用"""
    if device == 'cpu':
        return {'allocated_mb': 0, 'reserved_mb': 0, 'peak_mb': 0}

    torch.cuda.reset_peak_memory_stats()
    return {
        'allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
        'reserved_mb': torch.cuda.memory_reserved() / 1024 / 1024,
        'peak_mb': torch.cuda.max_memory_allocated() / 1024 / 1024,
    }


def profile_streaming_components(model, vocoder, device, chunk_frames=10):
    """分別測量每個元件的延遲"""
    model.eval()
    chunk_len = chunk_frames * 8
    dummy_emg = torch.randn(1, chunk_len, 8, device=device)

    state = model.init_streaming_state(device)

    results = {}

    # 1. CNN 前端
    def run_cnn():
        nonlocal state
        x = dummy_emg.transpose(1, 2)
        st = model.init_streaming_state(device)
        for i, block in enumerate(model.conv_blocks):
            c1, c2 = st['cnn_caches'][i]
            x, c1, c2 = block.forward_causal(x, c1, c2)
        return x

    results['CNN 前端'] = measure_latency(run_cnn, FLAGS.warmup_runs,
                                          FLAGS.benchmark_runs)

    # 2. Mamba 層 (逐層)
    cnn_out = run_cnn()
    x_mamba = cnn_out.transpose(1, 2)
    x_mamba = model.w_raw_in(x_mamba)

    for i, layer in enumerate(model.mamba_layers):
        d_inner = layer.mamba.d_inner
        d_conv = layer.mamba.d_conv
        d_state = layer.mamba.d_state

        def run_mamba_layer(layer=layer, x=x_mamba):
            cs = torch.zeros(1, d_inner, d_conv, device=device)
            ss = torch.zeros(1, d_inner, d_state, device=device)
            return layer.forward_streaming(x, cs, ss)

        results[f'Mamba 層 {i}'] = measure_latency(
            run_mamba_layer, FLAGS.warmup_runs, FLAGS.benchmark_runs
        )

    # 3. 完整 forward_streaming
    def run_full_streaming():
        st = model.init_streaming_state(device)
        return model.forward_streaming(dummy_emg, st)

    results['完整串流推論'] = measure_latency(
        run_full_streaming, FLAGS.warmup_runs, FLAGS.benchmark_runs
    )

    # 4. Vocoder
    dummy_mel = torch.randn(chunk_frames, 80, device=device)

    def run_vocoder():
        return vocoder(dummy_mel)

    results['Vocoder (HiFi-GAN)'] = measure_latency(
        run_vocoder, FLAGS.warmup_runs, FLAGS.benchmark_runs
    )

    # 5. 完整 pipeline (streaming + vocoder)
    streaming_voc = StreamingVocoder(vocoder)

    def run_pipeline():
        st = model.init_streaming_state(device)
        mel, aux, st = model.forward_streaming(dummy_emg, st)
        mel_frames = mel.squeeze(0)
        audio = streaming_voc.synthesize_chunk(mel_frames)
        return audio

    results['完整 Pipeline'] = measure_latency(
        run_pipeline, FLAGS.warmup_runs, FLAGS.benchmark_runs
    )

    return results


def generate_html_report(param_info, latency_info, memory_info,
                         model_summary, device_info, output_path):
    """產生互動式 HTML 報告"""

    # 準備參數數據
    param_names = [p['name'] for p in param_info[1:]]  # skip total
    param_counts = [p['total_params'] for p in param_info[1:]]
    param_sizes = [p['size_mb'] for p in param_info[1:]]

    # 準備延遲數據
    lat_names = list(latency_info.keys())
    lat_means = [latency_info[k]['mean_ms'] for k in lat_names]
    lat_stds = [latency_info[k]['std_ms'] for k in lat_names]

    total_params = param_info[0]['total_params']
    total_size_mb = param_info[0]['size_mb']

    # 嵌入式評估
    embed_targets = [
        {'name': 'Raspberry Pi 4', 'ram_mb': 4096, 'freq_ghz': 1.5,
         'flops_gops': 13.5, 'has_gpu': False},
        {'name': 'Jetson Nano', 'ram_mb': 4096, 'freq_ghz': 1.43,
         'flops_gops': 472, 'has_gpu': True},
        {'name': 'STM32H7', 'ram_mb': 1, 'freq_ghz': 0.48,
         'flops_gops': 0.48, 'has_gpu': False},
        {'name': 'ESP32-S3', 'ram_mb': 0.5, 'freq_ghz': 0.24,
         'flops_gops': 0.24, 'has_gpu': False},
    ]

    embed_rows = ""
    for t in embed_targets:
        name = t['name']
        ram = t['ram_mb']
        freq = t['freq_ghz']
        fits_ram = "✅" if total_size_mb < ram * 0.5 else "❌"
        if total_size_mb < ram * 0.1:
            fits_ram = "✅ 輕鬆"
        estimated_rt = "需量化" if not t['has_gpu'] and total_params > 1_000_000 else "可能"
        if t['has_gpu']:
            estimated_rt = "✅ 可行"
        if ram < total_size_mb:
            estimated_rt = "❌ 記憶體不足"
        embed_rows += (f"<tr><td>{name}</td><td>{ram} MB</td>"
                       f"<td>{freq} GHz</td><td>{fits_ram}</td>"
                       f"<td>{estimated_rt}</td></tr>\n")

    # 預先組裝參數表
    param_rows = ""
    for p in param_info[1:]:
        pname = p['name']
        pcount = f"{p['total_params']:,}"
        psize = f"{p['size_mb']:.2f}"
        ppct = f"{p['total_params']/total_params*100:.1f}%"
        param_rows += (f"<tr><td>{pname}</td><td>{pcount}</td>"
                       f"<td>{psize}</td><td>{ppct}</td></tr>\n")

    # 預先組裝延遲表
    latency_rows = ""
    for lname, info in latency_info.items():
        lmean = f"{info['mean_ms']:.2f}"
        lp95 = f"{info['p95_ms']:.2f}"
        lstd = f"±{info['std_ms']:.2f}"
        latency_rows += (f"<tr><td>{lname}</td><td>{lmean}</td>"
                         f"<td>{lp95}</td><td>{lstd}</td></tr>\n")

    peak_mb = f"{memory_info.get('peak_mb', 0):.0f}"
    fp16_mb = f"{total_size_mb/2:.1f}"
    int16_mb = f"{total_size_mb/2:.1f}"
    int8_mb = f"{total_size_mb/4:.1f}"

    html = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<title>EMG-to-Speech 模型資源分析</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #e0e0e0;
    min-height: 100vh;
    padding: 20px;
  }}
  h1 {{
    text-align: center;
    font-size: 2em;
    margin: 20px 0;
    background: linear-gradient(90deg, #00d2ff, #3a7bd5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }}
  h2 {{
    color: #3a7bd5;
    margin: 20px 0 10px;
    font-size: 1.3em;
  }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 20px;
    max-width: 1400px;
    margin: 0 auto;
  }}
  .card {{
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 20px;
    backdrop-filter: blur(10px);
  }}
  .card.full {{ grid-column: 1 / -1; }}
  .stat-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px;
    margin: 10px 0;
  }}
  .stat {{
    background: rgba(58,123,213,0.15);
    border-radius: 12px;
    padding: 12px;
    text-align: center;
  }}
  .stat .value {{
    font-size: 1.6em;
    font-weight: bold;
    color: #00d2ff;
  }}
  .stat .label {{
    font-size: 0.8em;
    color: #aaa;
    margin-top: 4px;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 10px 0;
  }}
  th, td {{
    padding: 8px 12px;
    text-align: left;
    border-bottom: 1px solid rgba(255,255,255,0.1);
  }}
  th {{ color: #3a7bd5; font-weight: 600; }}
  canvas {{ max-height: 350px; }}
  .tag {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 0.8em;
    margin: 2px;
  }}
  .tag.ok {{ background: rgba(0,200,100,0.2); color: #0c6; }}
  .tag.warn {{ background: rgba(255,200,0,0.2); color: #fa0; }}
  .tag.bad {{ background: rgba(255,60,60,0.2); color: #f44; }}
</style>
</head>
<body>
<h1>🧠 EMG-to-Speech 模型資源分析</h1>

<div class="grid">
  <div class="card full">
    <h2>📊 模型總覽</h2>
    <div class="stat-grid">
      <div class="stat">
        <div class="value">{total_params:,}</div>
        <div class="label">總參數量</div>
      </div>
      <div class="stat">
        <div class="value">{total_size_mb:.1f} MB</div>
        <div class="label">模型大小 (FP32)</div>
      </div>
      <div class="stat">
        <div class="value">{fp16_mb} MB</div>
        <div class="label">模型大小 (FP16)</div>
      </div>
      <div class="stat">
        <div class="value">{int8_mb} MB</div>
        <div class="label">模型大小 (INT8)</div>
      </div>
      <div class="stat">
        <div class="value">{device_info}</div>
        <div class="label">運算裝置</div>
      </div>
      <div class="stat">
        <div class="value">{peak_mb} MB</div>
        <div class="label">GPU 峰值記憶體</div>
      </div>
    </div>
  </div>

  <div class="card">
    <h2>📦 參數分佈</h2>
    <canvas id="paramChart"></canvas>
  </div>

  <div class="card">
    <h2>⏱️ 推論延遲 (ms)</h2>
    <canvas id="latencyChart"></canvas>
  </div>

  <div class="card">
    <h2>📋 模組參數詳情</h2>
    <table>
      <tr><th>模組</th><th>參數量</th><th>大小 (MB)</th><th>佔比</th></tr>
      {param_rows}
    </table>
  </div>

  <div class="card">
    <h2>⏱️ 延遲詳情</h2>
    <table>
      <tr><th>元件</th><th>平均 (ms)</th><th>P95 (ms)</th><th>std</th></tr>
      {latency_rows}
    </table>
  </div>

  <div class="card full">
    <h2>🔌 嵌入式部署評估</h2>
    <table>
      <tr><th>平台</th><th>RAM</th><th>CPU 頻率</th><th>記憶體</th><th>即時推論</th></tr>
      {embed_rows}
    </table>
    <div style="margin-top:12px; font-size:0.9em; color:#aaa;">
      <b>建議:</b>
      若要部署到 MCU (STM32/ESP32)，需要:
      <span class="tag warn">INT8 量化</span>
      <span class="tag warn">模型剪枝</span>
      <span class="tag warn">知識蒸餾</span>
      <br>
      若部署到 Jetson Nano/RPi4，FP16 即可運行。
    </div>
  </div>
</div>

<script>
new Chart(document.getElementById('paramChart'), {{
  type: 'doughnut',
  data: {{
    labels: {json.dumps(param_names)},
    datasets: [{{
      data: {json.dumps(param_counts)},
      backgroundColor: [
        '#3a7bd5', '#00d2ff', '#6c5ce7', '#a29bfe',
        '#fd79a8', '#fab1a0', '#55efc4', '#81ecec',
        '#ffeaa7', '#dfe6e9', '#636e72', '#b2bec3'
      ],
      borderWidth: 0,
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{
      legend: {{
        position: 'right',
        labels: {{ color: '#ccc', font: {{ size: 11 }} }}
      }}
    }}
  }}
}});

new Chart(document.getElementById('latencyChart'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(lat_names)},
    datasets: [{{
      label: '延遲 (ms)',
      data: {json.dumps([round(v, 2) for v in lat_means])},
      backgroundColor: 'rgba(58,123,213,0.6)',
      borderColor: '#3a7bd5',
      borderWidth: 1,
      borderRadius: 6,
    }}]
  }},
  options: {{
    responsive: true,
    indexAxis: 'y',
    plugins: {{
      legend: {{ display: false }}
    }},
    scales: {{
      x: {{
        title: {{ display: true, text: 'ms', color: '#aaa' }},
        ticks: {{ color: '#ccc' }},
        grid: {{ color: 'rgba(255,255,255,0.05)' }}
      }},
      y: {{
        ticks: {{ color: '#ccc' }},
        grid: {{ color: 'rgba(255,255,255,0.05)' }}
      }}
    }}
  }}
}});
</script>
</body>
</html>"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)


def main():
    out_dir = os.path.dirname(FLAGS.profile_output) or 'output'
    os.makedirs(out_dir, exist_ok=True)

    device = 'cpu' #'cuda' if torch.cuda.is_available() else 
    device_info = device
    if device == 'cuda':
        device_info = torch.cuda.get_device_name(0)

    print(f'裝置: {device_info}')

    # 載入模型
    print('載入資料集...')
    devset = EMGDataset(dev=True)

    print(f'載入模型: {FLAGS.model_path}')
    n_phones = len(phoneme_inventory)
    model = Model(devset.num_features, devset.num_speech_features,
                  n_phones).to(device)
    if FLAGS.quantized:
        load_quantized_model(model, FLAGS.model_path, device)
    else:
        state_dict = torch.load(FLAGS.model_path, map_location=device)
        model.load_state_dict(state_dict)
    model.eval()

    print('載入 HiFi-GAN...')
    vocoder = Vocoder(device=device)

    # ── 參數分析 ──────────────────────────────────────────────
    print('\n📦 分析參數...')
    param_info = count_parameters(model, 'Model')

    # 加入 Vocoder 參數
    voc_params = sum(p.numel() for p in vocoder.generator.parameters())
    param_info.append({
        'name': 'Vocoder (HiFi-GAN)',
        'total_params': voc_params,
        'trainable_params': voc_params,
        'size_mb': voc_params * 4 / 1024 / 1024,
    })

    for p in param_info:
        print(f"  {p['name']:30s} | {p['total_params']:>10,} params | "
              f"{p['size_mb']:>6.2f} MB")

    # ── 延遲分析 ──────────────────────────────────────────────
    print(f'\n⏱️ 測量延遲 (warmup={FLAGS.warmup_runs}, '
          f'runs={FLAGS.benchmark_runs})...')
    latency_info = profile_streaming_components(
        model, vocoder, device, chunk_frames=10
    )

    for name, info in latency_info.items():
        print(f"  {name:25s} | {info['mean_ms']:>7.2f} ms "
              f"(±{info['std_ms']:.2f}, p95={info['p95_ms']:.2f})")

    # ── 記憶體分析 ─────────────────────────────────────────────
    memory_info = measure_memory(device)
    print(f"\n💾 GPU 記憶體: "
          f"分配={memory_info['allocated_mb']:.0f}MB, "
          f"峰值={memory_info['peak_mb']:.0f}MB")

    # ── 產生報告 ──────────────────────────────────────────────
    print(f'\n📄 產生報告: {FLAGS.profile_output}')
    generate_html_report(
        param_info, latency_info, memory_info,
        model_summary="", device_info=device_info,
        output_path=FLAGS.profile_output
    )

    # ── 文字摘要 ──────────────────────────────────────────────
    summary_path = FLAGS.profile_output.replace('.html', '_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("EMG-to-Speech 模型資源分析\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"裝置: {device_info}\n")
        f.write(f"總參數量: {param_info[0]['total_params']:,}\n")
        f.write(f"模型大小 (FP32): {param_info[0]['size_mb']:.1f} MB\n")
        f.write(f"模型大小 (FP16): {param_info[0]['size_mb']/2:.1f} MB\n")
        f.write(f"模型大小 (INT8): {param_info[0]['size_mb']/4:.1f} MB\n\n")

        f.write("延遲 (每 chunk):\n")
        for name, info in latency_info.items():
            f.write(f"  {name:25s} {info['mean_ms']:>7.2f} ms\n")

        f.write(f"\nGPU 峰值記憶體: {memory_info['peak_mb']:.0f} MB\n")
    print(f'📄 文字摘要: {summary_path}')
    print('\n✅ 完成！用瀏覽器開啟 HTML 報告查看互動圖表。')


if __name__ == '__main__':
    FLAGS(sys.argv)
    main()
