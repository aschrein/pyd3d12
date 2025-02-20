/*
#  MIT License
#
#  Copyright (c) 2025 Anton Schreiner
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
*/

#define ROOT_SIGNATURE_MACRO                                                                                                                                                       \
    "DescriptorTable("                                                                                                                                                             \
    "UAV(u0, NumDescriptors = 16, space=0, flags = DESCRIPTORS_VOLATILE, offset=0) "                                                                                               \
    "), "                                                                                                                                                                          \
    "DescriptorTable("                                                                                                                                                             \
    "SRV(t0, NumDescriptors = 16, space=0, flags = DESCRIPTORS_VOLATILE, offset=0) "                                                                                               \
    "), "                                                                                                                                                                          \
    "RootConstants(b0, num32BitConstants=32, space = 0), "                                                                                                                         \
    "StaticSampler(s0, "                                                                                                                                                           \
    "Filter = FILTER_MIN_MAG_MIP_LINEAR, "                                                                                                                                         \
    "AddressU = TEXTURE_ADDRESS_WRAP, "                                                                                                                                            \
    "AddressV = TEXTURE_ADDRESS_WRAP, "                                                                                                                                            \
    "AddressW = TEXTURE_ADDRESS_WRAP), "                                                                                                                                           \
    "RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT), "

#include <utils.hlsli>

// RWTexture2D<f32x4>      g_rw_tex_0 : register(u0, space0);
// RWByteAddressBuffer     g_rw_buf_0 : register(u1, space0);

struct CBuffer {
    u32 frame_idx;
};

RWTexture2D<f32x4>      g_color : register(u0, space0);
RWStructuredBuffer<f32> g_rw_params : register(u1, space0);
RWStructuredBuffer<i32> g_rw_gradients : register(u2, space0);

Texture2D<f32x4> g_texture : register(t0, space0);

SamplerState g_sampler : register(s0, space0);

ConstantBuffer<CBuffer> b0 : register(b0, space0);

#define Kr 0.2126f
#define Kg 0.7152f
#define Kb (1.0f - Kr - Kg)

#define BATCH_SIZE (128 * 64)
#define NUM_LAYERS 7
#define MAX_NUM_NODES_PER_LAYER 36
static const uint NUM_NODES_PER_LAYER[NUM_LAYERS] = {36, 16, 16, 8, 16, 16, 3};
#define NO_RESIDUAL 0xffffffff
static const uint RESIDUAL_CONNECTIONS[NUM_LAYERS] = {NO_RESIDUAL, NO_RESIDUAL, NO_RESIDUAL, 0, 1, 2, NO_RESIDUAL};
#define NUM_ACTIVATIONS_PER_NETWORK (NUM_LAYERS * MAX_NUM_NODES_PER_LAYER)
#define NUM_GRAD_QUANTIZATION_LEVELS (1 << 13)

#define width 512
#define height 512

float3 RGB_to_YCbCr(float3 color) {
    float3x3 ycbr_matrix = float3x3(                              //
        Kr, Kg, Kb,                                               //
        -0.5f * Kr / (1.0f - Kb), -0.5f * Kg / (1.0f - Kb), 0.5f, //
        0.5f, -0.5f * Kg / (1.0f - Kr), -0.5f * Kb / (1.0f - Kr)  //
    );
    return mul(ycbr_matrix, color) + float3(0.0f, 0.5f, 0.5f);
}

float3 YCbCr_to_RGB(float3 color) {
    float3x3 rgb_matrix = float3x3(                                         //
        1.0f, 0.0f, 2.0f - 2.0f * Kr,                                       //
        1.0f, -Kb * (2.0f - 2.0f * Kb) / Kg, -Kr * (2.0f - 2.0f * Kr) / Kg, //
        1.0f, 2.0f - 2.0f * Kb, 0.0f                                        //
    );
    return mul(rgb_matrix, color - float3(0.0f, 0.5f, 0.5f));
}

struct LayerConstants {
    uint num_nodes;
    uint num_prev_nodes;
    uint num_weights;
    uint num_biases;
    uint num_adam_params;
    uint num_activations;
    uint weights_offset;
    uint biases_offset;
    uint adam_weights_offset;
    uint adam_biases_offset;
};

LayerConstants get_layer_constants(uint layer_idx) {
    if (layer_idx == 0) {
        LayerConstants o      = (LayerConstants)0;
        o.num_nodes           = NUM_NODES_PER_LAYER[0];
        o.num_prev_nodes      = 0;
        o.num_weights         = 0;
        o.num_biases          = 0;
        o.num_adam_params     = 0;
        o.num_activations     = NUM_NODES_PER_LAYER[0];
        o.weights_offset      = 0;
        o.biases_offset       = 0;
        o.adam_weights_offset = 0;
        o.adam_biases_offset  = 0;
        return o;
    } else {
        LayerConstants o      = (LayerConstants)0;
        o.num_nodes           = NUM_NODES_PER_LAYER[layer_idx];
        o.num_prev_nodes      = NUM_NODES_PER_LAYER[layer_idx - 1];
        o.num_weights         = o.num_nodes * o.num_prev_nodes;
        o.num_biases          = o.num_nodes;
        o.num_adam_params     = 4 * (o.num_weights + o.num_biases);
        o.num_activations     = o.num_nodes;
        o.weights_offset      = 0;
        o.biases_offset       = o.num_weights;
        o.adam_weights_offset = o.num_weights + o.num_biases;
        o.adam_biases_offset  = o.num_weights + o.num_biases + 2 * o.num_weights;
        return o;
    }
}

uint get_grad_offset(uint layer_idx) {
    uint offset = 0;
    for (uint i = 1; i < layer_idx; i++) {
        LayerConstants layer_constants = get_layer_constants(i);
        offset += layer_constants.num_weights + layer_constants.num_biases;
    }
    return offset;
}

uint get_layer_activations_offset(uint layer_idx) { return MAX_NUM_NODES_PER_LAYER * layer_idx; }

uint get_layer_params_offset(uint layer_idx) {
    uint offset = 0;
    for (uint i = 1; i < layer_idx; i++) {
        uint num_nodes       = MAX_NUM_NODES_PER_LAYER;
        uint num_weights     = num_nodes * num_nodes;
        uint num_biases      = num_nodes;
        uint num_adam_params = 4 * (num_weights + num_biases);
        offset += num_weights + num_biases + num_adam_params;
    }
    return offset;
}

uint lowbias32(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352d;
    x ^= x >> 15;
    x *= 0x846ca68b;
    x ^= x >> 16;
    return x;
}

float random_uniform_unit_float(inout uint rnd_state) {
    uint r    = lowbias32(rnd_state);
    rnd_state = r;
    return (float)(r & 0xffff) / (float)(0xffff);
}

float leaky_relu(float x) {
    float alpha = 0.01f;
    return max(alpha * x, x);
}

float leaky_relu_derivative(float x) {
    float alpha = 0.01f;
    return x > 0.0f ? 1.0f : alpha;
}

int quantize_grad(float grad) { return (int)round(grad * (float)NUM_GRAD_QUANTIZATION_LEVELS); }

float dequantize_grad(int quantized_grad) { return (float)quantized_grad / (float)NUM_GRAD_QUANTIZATION_LEVELS; }

void InitializeActivation(float2 uv, inout float activations[NUM_ACTIVATIONS_PER_NETWORK]) {
    uint num_frequencies = 9;
    uint num_channels    = 2;
    for (uint i = 0; i < num_frequencies; i++) {
        for (uint channel_idx = 0; channel_idx < num_channels; channel_idx++) {
            uint  activation_idx            = 2 * i * num_channels + 2 * channel_idx;
            uint  power_bias                = 0;
            float sin_val                   = sin(uv[channel_idx] * pow(2.0f, (float)(power_bias + i)) * 3.14159f);
            float cos_val                   = cos(uv[channel_idx] * pow(2.0f, (float)(power_bias + i)) * 3.14159f);
            activations[activation_idx + 0] = sin_val * pow(1.0f, (float)i);
            activations[activation_idx + 1] = cos_val * pow(1.0f, (float)i);
        }
    }
}

void Inference(uint layer_idx, uint node_idx, inout float activations[NUM_ACTIVATIONS_PER_NETWORK]) {
    LayerConstants layer_constants     = get_layer_constants(layer_idx);
    uint           layer_params_offset = get_layer_params_offset(layer_idx);
    uint           weights_offset      = layer_params_offset + layer_constants.weights_offset;
    uint           biases_offset       = layer_params_offset + layer_constants.biases_offset;
    uint           activations_offset  = get_layer_activations_offset(layer_idx - 1);
    uint           activation_idx      = get_layer_activations_offset(layer_idx) + node_idx;
    float          acc                 = 0.0f;
    for (uint i = 0; i < layer_constants.num_prev_nodes; i++) {
        uint  weight_idx = weights_offset + node_idx * layer_constants.num_prev_nodes + i;
        float weight     = g_rw_params[weight_idx];
        acc += activations[activations_offset + i] * weight;
        if (RESIDUAL_CONNECTIONS[layer_idx] != NO_RESIDUAL) {
            acc += activations[get_layer_activations_offset(RESIDUAL_CONNECTIONS[layer_idx]) + i] * weight;
        }
    }
    uint  bias_idx              = biases_offset + node_idx;
    float bias                  = g_rw_params[bias_idx];
    activations[activation_idx] = leaky_relu(acc + bias);
}

void Backprop(uint layer_idx, uint node_idx, inout float activations[NUM_ACTIVATIONS_PER_NETWORK], inout float grads[NUM_ACTIVATIONS_PER_NETWORK]) {
    LayerConstants layer_constants     = get_layer_constants(layer_idx);
    uint           layer_params_offset = get_layer_params_offset(layer_idx);
    uint           weights_offset      = layer_params_offset + layer_constants.weights_offset;
    uint           biases_offset       = layer_params_offset + layer_constants.biases_offset;
    uint           grad_offset         = get_grad_offset(layer_idx);
    uint           activations_offset  = get_layer_activations_offset(layer_idx - 1);
    uint           activation_idx      = get_layer_activations_offset(layer_idx) + node_idx;
    float          activation          = activations[activation_idx];
    float          grad                = grads[activation_idx];
    float          delta               = grad * leaky_relu_derivative(activation);

    for (uint i = 0; i < layer_constants.num_prev_nodes; i++) {
        uint  weight_idx = weights_offset + node_idx * layer_constants.num_prev_nodes + i;
        float weight     = g_rw_params[weight_idx];
        grads[activations_offset + i] += delta * weight;
        float prev_activation = activations[activations_offset + i];
        if (RESIDUAL_CONNECTIONS[layer_idx] != NO_RESIDUAL) {
            uint residual_activation_idx = get_layer_activations_offset(RESIDUAL_CONNECTIONS[layer_idx]) + i;
            prev_activation += activations[residual_activation_idx];
            grads[residual_activation_idx] += delta * weight;
        }
        uint grad_buf_idx = grad_offset + node_idx * layer_constants.num_prev_nodes + i;
        InterlockedAdd(g_rw_gradients[grad_buf_idx], quantize_grad(delta * prev_activation));
    }
    uint bias_idx = grad_offset + layer_constants.num_weights + node_idx;
    InterlockedAdd(g_rw_gradients[bias_idx], quantize_grad(delta / (float)layer_constants.num_prev_nodes));
}

float getsigneps(float x) {
    float eps = 1.0e-2f;
    return abs(x) < eps ? 0.0f : (x > 0.0f ? 1.0f : -1.0f);
}

[RootSignature(ROOT_SIGNATURE_MACRO)] //
    [numthreads(8, 8, 1)]             //
    void
    Initialize(uint3 global_id
               : SV_DispatchThreadID) {
    uint2 pos       = global_id.xy;
    uint  idx       = pos.x + pos.y * width;
    uint  layer_idx = idx / MAX_NUM_NODES_PER_LAYER;
    uint  node_idx  = idx % MAX_NUM_NODES_PER_LAYER;

    if (layer_idx == 0 || layer_idx >= NUM_LAYERS) return;
    LayerConstants layer_constants = get_layer_constants(layer_idx);
    if (node_idx >= layer_constants.num_nodes) return;

    uint  layer_params_offset = get_layer_params_offset(layer_idx);
    float normalization_const = 6.0f / sqrt((float)layer_constants.num_weights);

    uint rnd_state = idx;
    rnd_state      = lowbias32(rnd_state);
    for (uint i = 0; i < layer_constants.num_prev_nodes; i++) {
        uint weight_offset         = layer_params_offset + layer_constants.weights_offset + node_idx * layer_constants.num_prev_nodes + i;
        g_rw_params[weight_offset] = (random_uniform_unit_float(rnd_state) * 2.0f - 1.0f) * normalization_const;
    }
    uint biases_offset         = layer_params_offset + layer_constants.biases_offset + node_idx;
    g_rw_params[biases_offset] = (random_uniform_unit_float(rnd_state) * 2.0f - 1.0f) * normalization_const / 10.0f;

    for (uint i = 0; i < layer_constants.num_prev_nodes; i++) {
        uint adam_weights_offset             = layer_params_offset + layer_constants.adam_weights_offset + 2 * node_idx * layer_constants.num_prev_nodes + 2 * i;
        g_rw_params[adam_weights_offset + 0] = (random_uniform_unit_float(rnd_state) * 2.0f - 1.0f) * normalization_const / 1.0f;
        g_rw_params[adam_weights_offset + 1] = (random_uniform_unit_float(rnd_state) * 2.0f - 1.0f) * normalization_const / 1.0f;
    }
    uint adam_biases_offset             = layer_params_offset + layer_constants.adam_biases_offset + 2 * node_idx;
    g_rw_params[adam_biases_offset + 0] = 0.0f;
    g_rw_params[adam_biases_offset + 1] = 0.0f;

    uint grad_offset = get_grad_offset(layer_idx);
    for (uint i = 0; i < layer_constants.num_prev_nodes; i++) {
        uint grad_buf_idx            = grad_offset + node_idx * layer_constants.num_prev_nodes + i;
        g_rw_gradients[grad_buf_idx] = 0; // Direct assignment as each thread has unique indices
    }
    uint bias_grad_buf_idx            = grad_offset + layer_constants.num_weights + node_idx;
    g_rw_gradients[bias_grad_buf_idx] = 0;
}

[RootSignature(ROOT_SIGNATURE_MACRO)] //
    [numthreads(8, 8, 1)] void
    Main(uint3 global_id
         : SV_DispatchThreadID) {
    uint2 _pos       = global_id.xy % 8;
    uint2 _tile      = global_id.xy / 8;
    uint  num_tiles  = 64;
    uint  poffset    = lowbias32(_tile.x + lowbias32(_tile.y + lowbias32(b0.frame_idx / 1))) % (num_tiles * num_tiles);
    uint2 tile_coord = uint2(poffset % num_tiles, poffset / num_tiles);
    uint2 pos        = _pos + tile_coord * 8;
    if (pos.x >= width || pos.y >= height) return;

    float2 uv = float2((float)pos.x / (float)width, (float)pos.y / (float)height);
    float4 color = g_texture.SampleLevel(g_sampler, uv, 0);
    // float4 color = g_texture[pos.xy];
    // g_color[pos.xy] = color;

    float activations[NUM_ACTIVATIONS_PER_NETWORK];
    for (uint i = 0; i < NUM_ACTIVATIONS_PER_NETWORK; i++) activations[i] = 0.0f;
    InitializeActivation(uv, activations);

    // g_color[pos.xy] = f32x4(activations[0], activations[1], activations[2], 1.0f);

    const uint start_layer_idx = 1;
    for (uint i = start_layer_idx; i < NUM_LAYERS; i++) {
        for (uint j = 0; j < NUM_NODES_PER_LAYER[i]; j++) {
            Inference(i, j, activations);
        }
    }

    uint   final_activation_idx = get_layer_activations_offset(NUM_LAYERS - 1);
    float3 final_activation     = float3(activations[final_activation_idx + 0], activations[final_activation_idx + 1], activations[final_activation_idx + 2]);
    // g_color[pos.xy] = f32x4(final_activation[0], final_activation[1], final_activation[2], 1.0f);

    float grads[NUM_ACTIVATIONS_PER_NETWORK];
    for (uint i = 0; i < NUM_ACTIVATIONS_PER_NETWORK; i++) grads[i] = 0.0f;
    float3 target_signal            = color.xyz;
    float3 signal_diff              = final_activation - target_signal;
    grads[final_activation_idx + 0] = 0.2f * (signal_diff[0] + signal_diff[0] * signal_diff[0] * getsigneps(signal_diff[0]));
    grads[final_activation_idx + 1] = 0.2f * (signal_diff[1] + signal_diff[1] * signal_diff[1] * getsigneps(signal_diff[1]));
    grads[final_activation_idx + 2] = 0.2f * (signal_diff[2] + signal_diff[2] * signal_diff[2] * getsigneps(signal_diff[2]));

    for (uint i = NUM_LAYERS - 1; i >= start_layer_idx; i--) {
        for (uint j = 0; j < NUM_NODES_PER_LAYER[i]; j++) {
            Backprop(i, j, activations, grads);
        }
    }
}

[RootSignature(ROOT_SIGNATURE_MACRO)] //
    [numthreads(8, 8, 1)] void
    InferencePass(uint3 global_id
                  : SV_DispatchThreadID) {
    uint2 pos = global_id.xy;
    if (pos.x >= width || pos.y >= height) return;
    uint   idx = pos.x + pos.y * width;
    float2 uv  = float2((float)pos.x / (float)width, (float)pos.y / (float)height);

    float activations[NUM_ACTIVATIONS_PER_NETWORK];
    for (uint i = 0; i < NUM_ACTIVATIONS_PER_NETWORK; i++) activations[i] = 0.0f;
    InitializeActivation(uv, activations);

    const uint start_layer_idx = 1;
    for (uint i = start_layer_idx; i < NUM_LAYERS; i++) {
        for (uint j = 0; j < NUM_NODES_PER_LAYER[i]; j++) {
            Inference(i, j, activations);
        }
    }

    uint   final_activation_idx = get_layer_activations_offset(NUM_LAYERS - 1);
    float3 rgb                  = float3(activations[final_activation_idx + 0], activations[final_activation_idx + 1], activations[final_activation_idx + 2]);
    g_color[pos.xy]             = float4(rgb, 1.0f);
    // g_color[pos.xy] = g_texture[pos.xy];
}

static f32 get_learning_rate() {
    f32 t = saturate((f32)b0.frame_idx / (f32)(1 << 10));
    f32 up_slope   = 64.0f * t;
    f32 down_slope = cos(t * 3.14159f) * 0.5f + 0.5f;
    return max(min(up_slope, down_slope), 0.001f) * 0.8f / sqrt((f32)BATCH_SIZE);
}

[RootSignature(ROOT_SIGNATURE_MACRO)] //
    [numthreads(8, 8, 1)] void
    Backward(uint3 global_id
             : SV_DispatchThreadID) {
    uint2 pos       = global_id.xy;
    uint  idx       = pos.x + pos.y * width;
    uint  layer_idx = idx / MAX_NUM_NODES_PER_LAYER;
    uint  node_idx  = idx % MAX_NUM_NODES_PER_LAYER;

    if (layer_idx == 0 || layer_idx >= NUM_LAYERS) return;
    LayerConstants layer_constants = get_layer_constants(layer_idx);
    if (node_idx >= layer_constants.num_nodes) return;

    uint layer_params_offset = get_layer_params_offset(layer_idx);
    uint grad_offset         = get_grad_offset(layer_idx);
    uint adam_weights_offset = layer_params_offset + layer_constants.adam_weights_offset;
    uint adam_biases_offset  = layer_params_offset + layer_constants.adam_biases_offset;

    uint dummy; // For InterlockedExchange

    // float lr   = 0.1f / sqrt((float)BATCH_SIZE);
    float lr   = 2.0f * max(get_learning_rate(), 0.02f / sqrt((float)BATCH_SIZE));
    float t    = 1.0f + (float)b0.frame_idx;
    float rate = 1.0f / t;
    float beta = 0.97f;

    for (uint i = 0; i < layer_constants.num_prev_nodes; i++) {
        uint weight_idx   = layer_params_offset + layer_constants.weights_offset + node_idx * layer_constants.num_prev_nodes + i;
        uint grad_buf_idx = grad_offset + node_idx * layer_constants.num_prev_nodes + i;
        int  quantized_grad;
        InterlockedExchange(g_rw_gradients[grad_buf_idx], 0, quantized_grad);
        float grad = clamp(dequantize_grad(quantized_grad), -1.0, 1.0);

        uint  adam_weight_idx            = adam_weights_offset + 2 * node_idx * layer_constants.num_prev_nodes + 2 * i;
        float z                          = g_rw_params[adam_weight_idx + 0];
        float x                          = g_rw_params[adam_weight_idx + 1];
        float z_1                        = z - grad * lr;
        float x_1                        = lerp(x, z, rate);
        float y                          = lerp(z_1, x_1, beta);
        g_rw_params[weight_idx]          = y;
        g_rw_params[adam_weight_idx + 0] = z_1;
        g_rw_params[adam_weight_idx + 1] = x_1;
    }

    uint bias_idx          = layer_params_offset + layer_constants.biases_offset + node_idx;
    uint bias_grad_buf_idx = grad_offset + layer_constants.num_weights + node_idx;
    int  quantized_bias_grad;
    InterlockedExchange(g_rw_gradients[bias_grad_buf_idx], 0, quantized_bias_grad);
    float bias_grad = clamp(dequantize_grad(quantized_bias_grad), -1.0, 1.0);

    uint  adam_bias_idx            = adam_biases_offset + 2 * node_idx;
    float z                        = g_rw_params[adam_bias_idx + 0];
    float x                        = g_rw_params[adam_bias_idx + 1];
    float z_1                      = z - bias_grad * lr;
    float x_1                      = lerp(x, z, rate);
    float y                        = lerp(z_1, x_1, beta);
    g_rw_params[bias_idx]          = y;
    g_rw_params[adam_bias_idx + 0] = z_1;
    g_rw_params[adam_bias_idx + 1] = x_1;
}