"""
即時 EMG → 語音 串流推論 — Web 視覺化介面
使用 Flask + Plotly.js (CDN)，在瀏覽器中即時顯示:
  1. 8 通道 EMG 輸入波形
  2. 輸出音訊波形
  3. Mel 頻譜圖
  4. 進度資訊

用法:
    python realtime_inference_ui.py \
        --hifigan_checkpoint hifigan_finetuned/checkpoint \
        --model_path models/transduction_model/model.pt \
        --sample_index 0

    然後在瀏覽器打開 http://localhost:5000
"""

import os, sys, time, threading, logging, json, io
import numpy as np
import torch
import soundfile as sf
from flask import Flask, render_template_string, jsonify, request as flask_request, send_file

from architecture import Model
from data_utils import phoneme_inventory
from vocoder import Vocoder, StreamingVocoder
from emg_simulator import EMGSimulator
from read_emg import EMGDataset
from streaming_pipeline import StreamingPipeline, AUDIO_SR
from quantize_int16 import load_quantized_model

from absl import flags
FLAGS = flags.FLAGS

# ── FLAGS ────────────────────────────────────────────────────
for name, default, help_text in [
    ('vocoder_chunk_size', 30, '每次送入 vocoder 的 mel 幀數'),
    ('emg_chunk_frames', 10, '每個 EMG chunk 對應的 mel 幀數'),
    ('sample_index', 0, '使用資料集中第幾個樣本'),
    ('streaming_output', 'output/streaming_output.wav', '串流輸出 WAV 路徑'),
    ('port', 5000, 'Web UI port'),
]:
    try:
        if isinstance(default, int):
            flags.DEFINE_integer(name, default, help_text)
        else:
            flags.DEFINE_string(name, default, help_text)
    except flags.DuplicateFlagError:
        pass

for name, default, help_text in [
    ('emg_file', None, '自訂 EMG 檔案路徑'),
    ('model_path', 'models/transduction_model/model.pt', '模型檔案路徑'),
    ('play_mode', 'none', '播放模式 (UI 模式下忽略)'),
]:
    try:
        flags.DEFINE_string(name, default, help_text)
    except flags.DuplicateFlagError:
        pass

for name, default, help_text in [
    ('realtime', True, '是否模擬真實取樣速率延遲'),
    ('quantized', False, '是否使用 INT16 量化模型'),
    ('save_wav', True, '是否儲存 WAV'),
    ('save_chunks', False, '是否儲存每段 vocoder 輸出'),
    ('compare_batch', True, '是否同時執行批次推論做比較'),
    ('list_samples', True, '列出資料集中所有可用樣本'),
]:
    try:
        flags.DEFINE_boolean(name, default, help_text)
    except flags.DuplicateFlagError:
        pass

EMG_SR = 689.06
EMG_CHANNELS = 8

# ══════════════════════════════════════════════════════════════
# Shared state (thread-safe via locks)
# ══════════════════════════════════════════════════════════════
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.emg_chunks = []       # list of (T,8) arrays
        self.audio_chunks = []     # list of 1-d arrays
        self.mel_chunks = []       # list of (T,80) arrays
        self.chunk_idx = 0
        self.total_chunks = 0
        self.status = '就緒'
        self.running = False
        self.finished = False
        self.error = None
        # version counters for efficient polling
        self.emg_version = 0
        self.audio_version = 0
        self.mel_version = 0

    def add_emg(self, chunk):
        with self.lock:
            self.emg_chunks.append(chunk.copy())
            self.emg_version += 1

    def add_audio(self, audio):
        with self.lock:
            self.audio_chunks.append(audio.copy())
            self.audio_version += 1

    def add_mel(self, mel):
        with self.lock:
            self.mel_chunks.append(mel.copy())
            self.mel_version += 1

    def set_progress(self, idx, total):
        with self.lock:
            self.chunk_idx = idx
            self.total_chunks = total

    def get_snapshot(self, emg_since=0, audio_since=0, mel_since=0):
        """Return new data since given version numbers."""
        with self.lock:
            emg_new = self.emg_chunks[emg_since:]
            audio_new = self.audio_chunks[audio_since:]
            mel_new = self.mel_chunks[mel_since:]
            return {
                'emg': emg_new,
                'audio': audio_new,
                'mel': mel_new,
                'emg_v': self.emg_version,
                'audio_v': self.audio_version,
                'mel_v': self.mel_version,
                'chunk_idx': self.chunk_idx,
                'total_chunks': self.total_chunks,
                'status': self.status,
                'running': self.running,
                'finished': self.finished,
                'error': self.error,
            }

    def reset(self):
        with self.lock:
            self.emg_chunks.clear()
            self.audio_chunks.clear()
            self.mel_chunks.clear()
            self.emg_version = 0
            self.audio_version = 0
            self.mel_version = 0
            self.chunk_idx = 0
            self.total_chunks = 0
            self.finished = False
            self.error = None

STATE = SharedState()

# ══════════════════════════════════════════════════════════════
# Pipeline runner (background thread)
# ══════════════════════════════════════════════════════════════
def run_pipeline():
    global STATE
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        STATE.status = f'裝置: {device} | 載入資料集...'

        devset = EMGDataset(dev=True)

        if FLAGS.emg_file:
            STATE.status = f'載入自訂 EMG...'
            raw_emg = np.load(FLAGS.emg_file)
            if raw_emg.ndim == 1:
                raw_emg = raw_emg.reshape(-1, 8)
            from read_emg import apply_to_all, notch_harmonics, remove_drift, subsample
            raw_emg = apply_to_all(notch_harmonics, raw_emg, 60, 1000)
            raw_emg = apply_to_all(remove_drift, raw_emg, 1000)
            raw_emg = apply_to_all(subsample, raw_emg, 689.06, 1000)
            for c in FLAGS.remove_channels:
                raw_emg[:, int(c)] = 0
            raw_emg = raw_emg.astype(np.float32) / 20
            raw_emg = 50 * np.tanh(raw_emg / 50.)
        else:
            sample = devset[FLAGS.sample_index]
            raw_emg = sample['raw_emg'].numpy()

        STATE.status = f'EMG: {raw_emg.shape} | 載入模型...'

        n_phones = len(phoneme_inventory)
        model = Model(devset.num_features, devset.num_speech_features,
                      n_phones).to(device)
        if FLAGS.quantized:
            load_quantized_model(model, FLAGS.model_path, device)
        else:
            sd = torch.load(FLAGS.model_path, map_location=device)
            model.load_state_dict(sd)
        model.eval()

        STATE.status = '載入 HiFi-GAN vocoder...'
        vocoder = Vocoder(device=device)
        streaming_vocoder = StreamingVocoder(vocoder)

        simulator = EMGSimulator(
            raw_emg=raw_emg,
            chunk_frames=FLAGS.emg_chunk_frames,
            realtime=FLAGS.realtime,
        )

        STATE.status = f'推論中... ({simulator.total_chunks} chunks)'

        pipeline = StreamingPipeline(
            model=model,
            streaming_vocoder=streaming_vocoder,
            audio_normalizer=devset.mfcc_norm,
            device=device,
            vocoder_chunk_size=FLAGS.vocoder_chunk_size,
            play_mode='none',
            on_emg_chunk=lambda c: STATE.add_emg(c),
            on_audio_chunk=lambda a: STATE.add_audio(a),
            on_mel_chunk=lambda m: STATE.add_mel(m),
            on_progress=lambda i, t: STATE.set_progress(i, t),
        )

        t0 = time.time()
        pipeline.start(emg_source=simulator)
        pipeline.wait()
        elapsed = time.time() - t0

        full_audio = pipeline.get_full_audio()
        if len(full_audio) > 0:
            dur = len(full_audio) / AUDIO_SR
            if FLAGS.save_wav:
                out_dir = os.path.dirname(FLAGS.streaming_output) or 'output'
                os.makedirs(out_dir, exist_ok=True)
                sf.write(FLAGS.streaming_output, full_audio, AUDIO_SR)
            STATE.status = f'完成! 音訊 {dur:.2f}s | 耗時 {elapsed:.2f}s | RTF {elapsed/dur:.2f}x'
        else:
            STATE.status = '完成 (無音訊)'

        STATE.finished = True
        STATE.running = False

    except Exception as e:
        STATE.error = str(e)
        STATE.status = f'錯誤: {e}'
        STATE.running = False
        logging.exception('Pipeline error')


# ══════════════════════════════════════════════════════════════
# Flask app
# ══════════════════════════════════════════════════════════════
app = Flask(__name__)
app.logger.setLevel(logging.WARNING)

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<title>EMG → Speech 即時推論</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:#111; color:#d0d0d0; font-family:'Inter','Segoe UI',system-ui,sans-serif; padding:12px 16px; }
  h1 { text-align:center; color:#d0d0d0; font-size:1.1em; font-weight:500; letter-spacing:0.5px; margin-bottom:10px; }
  .controls { text-align:center; margin-bottom:10px; }
  .controls button {
    background:#1e1e1e; color:#ccc; border:1px solid #333;
    border-radius:4px; padding:7px 18px; font-size:13px; cursor:pointer; margin:0 4px;
    transition: background 0.15s, border-color 0.15s;
  }
  .controls button:hover { background:#2a2a2a; border-color:#555; }
  .controls button:disabled { opacity:0.3; cursor:default; }
  #status { text-align:center; color:#777; margin:6px 0; font-size:12px; }
  #progress-bar {
    width:100%; height:3px; background:#222; border-radius:2px;
    overflow:hidden; margin-bottom:6px;
  }
  #progress-fill {
    height:100%; width:0%; transition:width 0.2s;
    background:#4a9eff;
  }
  #progress-text { text-align:center; font-size:11px; color:#666; margin-bottom:8px; }
  .plot-box { margin-bottom:4px; }
  .section-title { color:#888; font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:1px; margin:8px 0 3px 2px; }
</style>
</head>
<body>
<h1>EMG → Speech Streaming Inference</h1>

<div class="controls">
  <button id="btnStart" onclick="startPipeline()">▶ Start</button>
  <button id="btnPlay" onclick="playFullAudio()" disabled>▶ Replay</button>
  <button id="btnStop" onclick="stopAudio()" disabled>■ Stop</button>
  <button id="btnSave" onclick="saveWav()" disabled>Save WAV</button>
</div>
<div id="status">就緒 — 按 Start 以啟動</div>
<div id="progress-bar"><div id="progress-fill"></div></div>
<div id="progress-text"></div>

<div class="section-title">EMG Input — 8 Channels</div>
<div id="emg-container"></div>

<div class="section-title">Audio Output (realtime)</div>
<div id="audio-plot" class="plot-box"></div>

<div class="section-title">Mel Spectrogram</div>
<div id="mel-plot" class="plot-box"></div>

<script>
const EMG_SR = 689.06;
const AUDIO_SR = 22050;
const EMG_COLOR = '#999';
const WINDOW_SEC = 3;  // sliding window for all waveforms
const PLOT_BG = '#161616';
const LAYOUT_COMMON = {paper_bgcolor:'#111', plot_bgcolor:PLOT_BG, font:{color:'#999',size:10}};

let emgAll = [];
let audioAll = [];
let melAll = [];
let emgV=0, audioV=0, melV=0;
let polling = false;

// ── Web Audio API for real-time playback ──
let audioCtx = null;
let nextPlayTime = 0;
let realtimeAudio = true;

function initAudioCtx() {
  if (audioCtx) audioCtx.close();
  audioCtx = new (window.AudioContext || window.webkitAudioContext)({sampleRate: AUDIO_SR});
  nextPlayTime = 0;
}

function playChunkRealtime(samples) {
  if (!audioCtx || !realtimeAudio) return;
  let buf = audioCtx.createBuffer(1, samples.length, AUDIO_SR);
  let ch = buf.getChannelData(0);
  for (let i=0; i<samples.length; i++) ch[i] = samples[i];
  let src = audioCtx.createBufferSource();
  src.buffer = buf;
  src.connect(audioCtx.destination);
  let now = audioCtx.currentTime;
  if (nextPlayTime < now) nextPlayTime = now;
  src.start(nextPlayTime);
  nextPlayTime += buf.duration;
}

// ── Create 8 separate EMG plots ──
function initPlots() {
  let container = document.getElementById('emg-container');
  container.innerHTML = '';
  for (let ch=0; ch<8; ch++) {
    let div = document.createElement('div');
    div.id = 'emg-ch-' + ch;
    div.className = 'plot-box';
    container.appendChild(div);
    Plotly.newPlot(div.id, [{x:[], y:[], type:'scatter', mode:'lines',
      line:{color:EMG_COLOR, width:1}}],
      Object.assign({}, LAYOUT_COMMON, {
        height: 55, showlegend:false,
        margin:{l:50, r:10, t:2, b: ch==7?22:2},
        yaxis:{range:[-3,3], showticklabels:false, gridcolor:'#222',
          title:{text:'Ch'+ch, font:{size:8, color:'#666'}, standoff:4}},
        xaxis:{showticklabels: ch==7, gridcolor:'#222',
          title: ch==7 ? {text:'Time (s)', font:{size:8, color:'#666'}} : undefined}
      }), {responsive:true, displayModeBar:false});
  }

  Plotly.newPlot('audio-plot', [{x:[], y:[], type:'scatter', mode:'lines',
    line:{color:'#4a9eff', width:1}}],
    Object.assign({}, LAYOUT_COMMON, {height:110,
      margin:{l:50,r:20,t:8,b:28},
      yaxis:{range:[-0.6,0.6], gridcolor:'#222'},
      xaxis:{title:{text:'Time (s)', font:{size:8, color:'#666'}}, gridcolor:'#222'}}),
    {responsive:true, displayModeBar:false});

  Plotly.newPlot('mel-plot', [{z:[[0]], type:'heatmap', colorscale:'Greys', reversescale:true,
    zmin:-10, zmax:2, showscale:false}],
    Object.assign({}, LAYOUT_COMMON, {height:130,
      margin:{l:50,r:20,t:8,b:28},
      xaxis:{title:{text:'Frame', font:{size:8, color:'#666'}}, gridcolor:'#222'},
      yaxis:{title:{text:'Mel bin', font:{size:8, color:'#666'}}, gridcolor:'#222'}}),
    {responsive:true, displayModeBar:false});
}

function startPipeline() {
  document.getElementById('btnStart').disabled = true;
  document.getElementById('btnSave').disabled = true;
  document.getElementById('btnPlay').disabled = true;
  emgAll=[]; audioAll=[]; melAll=[];
  emgV=0; audioV=0; melV=0;
  realtimeAudio = true;
  initPlots();
  initAudioCtx();

  fetch('/api/start', {method:'POST'}).then(r=>r.json()).then(d=>{
    if(d.ok) { polling=true; pollData(); }
    else { document.getElementById('status').textContent='錯誤: '+d.error; }
  });
}

function pollData() {
  if(!polling) return;
  fetch(`/api/data?emg_v=${emgV}&audio_v=${audioV}&mel_v=${melV}`)
    .then(r=>r.json()).then(d=>{
      document.getElementById('status').textContent = d.status;

      if(d.total_chunks>0) {
        let pct = Math.round(d.chunk_idx/d.total_chunks*100);
        document.getElementById('progress-fill').style.width = pct+'%';
        document.getElementById('progress-text').textContent =
          d.chunk_idx+'/'+d.total_chunks+' chunks ('+pct+'%)';
      }

      if(d.emg && d.emg.length>0) {
        for(let chunk of d.emg) {
          for(let i=0; i<chunk.length; i++) emgAll.push(chunk[i]);
        }
        emgV = d.emg_v;
        updateEMG();
      }

      if(d.audio && d.audio.length>0) {
        for(let chunk of d.audio) {
          let flat = [];
          for(let s of chunk) { audioAll.push(s); flat.push(s); }
          // Real-time playback: play this chunk immediately
          playChunkRealtime(flat);
        }
        audioV = d.audio_v;
        updateAudio();
      }

      if(d.mel && d.mel.length>0) {
        for(let chunk of d.mel) {
          for(let frame of chunk) melAll.push(frame);
        }
        melV = d.mel_v;
        updateMel();
      }

      if(d.finished || d.error) {
        polling = false;
        document.getElementById('btnStart').disabled = false;
        document.getElementById('btnSave').disabled = false;
        document.getElementById('btnPlay').disabled = false;
        if(d.finished) {
          document.getElementById('progress-fill').style.width = '100%';
          document.getElementById('progress-text').textContent = ' 完成!';
        }
      } else {
        setTimeout(pollData, 33);
      }
    }).catch(e=>{
      console.error(e);
      setTimeout(pollData, 60);
    });
}

function updateEMG() {
  let windowSamples = Math.round(WINDOW_SEC * EMG_SR);
  let start = Math.max(0, emgAll.length - windowSamples);
  let tEnd = emgAll.length / EMG_SR;
  let tStart = Math.max(0, tEnd - WINDOW_SEC);
  for(let ch=0; ch<8; ch++) {
    let xs=[], ys=[];
    for(let i=start; i<emgAll.length; i++) {
      xs.push(i/EMG_SR);
      ys.push(emgAll[i][ch]);
    }
    Plotly.update('emg-ch-'+ch, {x:[xs], y:[ys]});
    Plotly.relayout('emg-ch-'+ch, {'xaxis.range': [tStart, tEnd]});
  }
}

function updateAudio() {
  let windowSamples = Math.round(WINDOW_SEC * AUDIO_SR);
  let start = Math.max(0, audioAll.length - windowSamples);
  let ds = Math.max(1, Math.floor(windowSamples / 6000));
  let xs=[], ys=[];
  for(let i=start; i<audioAll.length; i+=ds) {
    xs.push(i/AUDIO_SR);
    ys.push(audioAll[i]);
  }
  let tEnd = audioAll.length / AUDIO_SR;
  let tStart = Math.max(0, tEnd - WINDOW_SEC);
  Plotly.update('audio-plot', {x:[xs], y:[ys]});
  Plotly.relayout('audio-plot', {'xaxis.range': [tStart, tEnd]});
}

function updateMel() {
  if(melAll.length<2) return;
  let nMel = melAll[0].length;
  let z = [];
  for(let m=0; m<nMel; m++) {
    let row=[];
    for(let f=0; f<melAll.length; f++) row.push(melAll[f][m]);
    z.push(row);
  }
  Plotly.update('mel-plot', {z:[z]});
}

function playFullAudio() {
  realtimeAudio = false;
  initAudioCtx();
  playChunkRealtime(audioAll);
  document.getElementById('btnPlay').disabled = true;
  document.getElementById('btnStop').disabled = false;
  document.getElementById('status').textContent = ' 重播中...';
  let dur = audioAll.length / AUDIO_SR;
  setTimeout(()=>{
    document.getElementById('btnPlay').disabled = false;
    document.getElementById('btnStop').disabled = true;
    document.getElementById('status').textContent = '重播完畢';
  }, dur*1000 + 200);
}

function stopAudio() {
  if(audioCtx) {
    audioCtx.close();
    audioCtx = null;
  }
  realtimeAudio = false;
  document.getElementById('btnPlay').disabled = false;
  document.getElementById('btnStop').disabled = true;
  document.getElementById('status').textContent = '已停止播放';
}

function saveWav() {
  fetch('/api/save', {method:'POST'}).then(r=>r.json()).then(d=>{
    document.getElementById('status').textContent = d.message;
  });
}

initPlots();
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

@app.route('/api/start', methods=['POST'])
def api_start():
    global STATE
    if STATE.running:
        return jsonify(ok=False, error='already running')
    STATE.reset()
    STATE.running = True
    t = threading.Thread(target=run_pipeline, daemon=True)
    t.start()
    return jsonify(ok=True)

@app.route('/api/data')
def api_data():
    emg_v = int(flask_request.args.get('emg_v', 0))
    audio_v = int(flask_request.args.get('audio_v', 0))
    mel_v = int(flask_request.args.get('mel_v', 0))

    snap = STATE.get_snapshot(emg_v, audio_v, mel_v)

    # Convert numpy arrays to lists for JSON
    emg_list = [c.tolist() for c in snap['emg']]
    audio_list = [c.tolist() for c in snap['audio']]
    mel_list = [c.tolist() for c in snap['mel']]

    return jsonify(
        emg=emg_list, audio=audio_list, mel=mel_list,
        emg_v=snap['emg_v'], audio_v=snap['audio_v'], mel_v=snap['mel_v'],
        chunk_idx=snap['chunk_idx'], total_chunks=snap['total_chunks'],
        status=snap['status'], running=snap['running'],
        finished=snap['finished'], error=snap['error'],
    )

@app.route('/api/audio.wav')
def api_audio_wav():
    with STATE.lock:
        if not STATE.audio_chunks:
            return 'No audio', 404
        full = np.concatenate(STATE.audio_chunks)
    buf = io.BytesIO()
    sf.write(buf, full, AUDIO_SR, format='WAV')
    buf.seek(0)
    return send_file(buf, mimetype='audio/wav', download_name='output.wav')

@app.route('/api/save', methods=['POST'])
def api_save():
    with STATE.lock:
        if not STATE.audio_chunks:
            return jsonify(message='無音訊可儲存')
        full = np.concatenate(STATE.audio_chunks)
    out = FLAGS.streaming_output
    os.makedirs(os.path.dirname(out) or 'output', exist_ok=True)
    sf.write(out, full, AUDIO_SR)
    return jsonify(message=f'已儲存: {out} ({len(full)/AUDIO_SR:.2f}s)')


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    port = FLAGS.port
    print(f'\n   EMG → Speech UI 啟動: http://localhost:{port}\n')
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)


if __name__ == '__main__':
    FLAGS(sys.argv)
    main()
