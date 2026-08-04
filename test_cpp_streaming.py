#!/usr/bin/env python3
import struct
import os
import sys

def load_golden_bin(path, shape):
    with open(path, 'rb') as f:
        raw = f.read()
    numel = len(raw) // 4
    flat = list(struct.unpack(f'{numel}f', raw))
    if len(shape) == 1:
        return flat
    rows, cols = shape
    return [flat[r * cols:(r + 1) * cols] for r in range(rows)]

def max_abs_error(a_flat, b_flat):
    return max(abs(x - y) for x, y in zip(a_flat, b_flat))

def flatten_2d(mat):
    return [v for row in mat for v in row]

def main():
    weights_dir = 'exported_weights'
    if not os.path.exists(os.path.join(weights_dir, 'model.json')):
        print(f'ERROR: {weights_dir}/model.json not found')
        print('Please run export_weights.py first')
        sys.exit(1)

    try:
        from cpp_forward import CppInferenceEngine
    except ImportError:
        print('ERROR: cpp_forward module not found')
        print('Please build first: python setup_cpp.py build_ext --inplace')
        sys.exit(1)

    engine = CppInferenceEngine()
    engine.load_weights(weights_dir)
    print('C++ engine loaded')

    golden_input = load_golden_bin(
        os.path.join(weights_dir, 'golden_input.bin'), (128, 8))
    golden_mel = load_golden_bin(
        os.path.join(weights_dir, 'golden_mel_output.bin'), (16, 80))
    golden_mel_stream = load_golden_bin(
        os.path.join(weights_dir, 'golden_mel_stream_output.bin'), (16, 80))

    flat_input = flatten_2d(golden_input)
    flat_golden_mel = flatten_2d(golden_mel)
    flat_golden_mel_stream = flatten_2d(golden_mel_stream)

    # Test 1: Batch forward vs Python golden
    print('\n=== Test 1: C++ Batch Forward vs Python Golden ===')
    mel_batch = engine.forward_batch(flat_input, 128)
    flat_batch = flatten_2d(mel_batch)
    err1 = max_abs_error(flat_batch, flat_golden_mel)
    pass1 = err1 < 1e-4
    print(f'  Batch mel shape: ({len(mel_batch)}, {len(mel_batch[0])})')
    print(f'  Max abs error: {err1:.6e}  {"PASS" if pass1 else "FAIL"}')

    # Test 2: C++ Streaming (1x128) vs Python Golden Streaming
    print('\n=== Test 2: C++ Streaming (1x128) vs Python Golden Streaming ===')
    engine.init_state()
    mel_stream = engine.forward_streaming(flat_input, 128)
    flat_stream = flatten_2d(mel_stream)
    print(f'  Streaming mel shape: ({len(mel_stream)}, {len(mel_stream[0])})')

    err2 = max_abs_error(flat_stream, flat_golden_mel_stream)
    pass2 = err2 < 1e-4
    print(f'  Max abs error vs golden streaming: {err2:.6e}  {"PASS" if pass2 else "FAIL"}')

    # Print divergence vs batch as info
    err2_vs_batch = max_abs_error(flat_stream, flat_batch)
    print(f'  (Info) Max abs error vs batch: {err2_vs_batch:.6e} (Expected divergence due to causal vs symmetric padding)')

    # Test 3: Streaming 2x64 chunks vs 1x128 chunk
    print('\n=== Test 3: Streaming 2x64 vs 1x128 ===')
    engine.init_state()
    chunk1_flat = flatten_2d(golden_input[:64])
    mel_c1 = engine.forward_streaming(chunk1_flat, 64)
    chunk2_flat = flatten_2d(golden_input[64:])
    mel_c2 = engine.forward_streaming(chunk2_flat, 64)
    mel_concat = mel_c1 + mel_c2
    flat_concat = flatten_2d(mel_concat)
    print(f'  Chunk1 mel: ({len(mel_c1)}, {len(mel_c1[0]) if mel_c1 else 0})')
    print(f'  Chunk2 mel: ({len(mel_c2)}, {len(mel_c2[0]) if mel_c2 else 0})')
    if len(flat_concat) == len(flat_stream):
        err3 = max_abs_error(flat_concat, flat_stream)
        pass3 = err3 < 1e-4
        print(f'  Max abs error: {err3:.6e}  {"PASS" if pass3 else "FAIL"}')
    else:
        print(f'  Frame count mismatch: {len(mel_concat)} vs {len(mel_stream)}')

    # Test 4: Multiple small chunks
    print('\n=== Test 4: Streaming 8x16 chunks ===')
    engine.init_state()
    mel_multi = []
    for i in range(8):
        chunk = golden_input[i*16 : (i+1)*16]
        flat_chunk = flatten_2d(chunk)
        mel_i = engine.forward_streaming(flat_chunk, 16)
        mel_multi.extend(mel_i)
    print(f'  Total frames from 8x16: {len(mel_multi)}')
    print(f'  Total frames from 1x128: {len(mel_stream)}')
    if len(mel_multi) == len(mel_stream):
        flat_multi = flatten_2d(mel_multi)
        err4 = max_abs_error(flat_multi, flat_stream)
        pass4 = err4 < 1e-4
        print(f'  Max abs error: {err4:.6e}  {"PASS" if pass4 else "FAIL"}')

    # Summary
    print('\n=== Summary ===')
    print(f'  Test 1 (batch vs golden): {"PASS" if pass1 else "FAIL"}')
    print(f'  Test 2 (stream vs golden streaming): {"PASS" if pass2 else "FAIL"}')
    print('\n=== First 5 mel values ===')
    print(f'  Python golden: {flat_golden_mel[:5]}')
    print(f'  C++ batch:     {flat_batch[:5]}')
    print(f'  C++ streaming: {flat_stream[:5]}')

if __name__ == '__main__':
    main()
