// MIT License
// Copyright (c) 2025 Anton Schreiner


// Custom ML nodes for LiteGraph

// Base class helper for tensor shape propagation
function formatShape(shape) {
    if (!shape || shape.length === 0) return "?";
    return "[" + shape.join(", ") + "]";
}

// ============================================
// INPUT NODES
// ============================================

function InputTensor() {
    this.addOutput("tensor", "tensor");
    this.addProperty("shape", [1, 3, 224, 224]);
    this.addProperty("dtype", "float32");
    this.shape_widget = this.addWidget("text", "Shape", "1,3,224,224", (v) => {
        this.properties.shape = v.split(",").map(x => parseInt(x.trim()));
    });
    this.dtype_widget = this.addWidget("combo", "Dtype", "float32", (v) => {
        this.properties.dtype = v;
    }, { values: ["float32", "float16", "bfloat16", "int32", "int64"] });
    this.size = [180, 90];
}
InputTensor.title = "Input Tensor";
InputTensor.desc = "Input tensor with configurable shape";
InputTensor.prototype.onExecute = function() {
    this.setOutputData(0, { shape: this.properties.shape, dtype: this.properties.dtype });
};
InputTensor.prototype.getExtraMenuOptions = function() {
    return [{ content: "Shape: " + formatShape(this.properties.shape), disabled: true }];
};
InputTensor.prototype.onConfigure = function() {
    if (this.shape_widget) {
        this.shape_widget.value = this.properties.shape.join(", ");
    }
    if (this.dtype_widget) {
        this.dtype_widget.value = this.properties.dtype;
    }
};
LiteGraph.registerNodeType("ML/input_tensor", InputTensor);


// ============================================
// GENERIC CONVOLUTION NODE
// ============================================

function ConvolutionNode() {
    this.addInput("input", "tensor");
    this.addOutput("output", "tensor");
    
    this.addProperty("conv_type", "conv2d");
    this.addProperty("in_channels", 3);
    this.addProperty("out_channels", 64);
    this.addProperty("kernel_size", 3);
    this.addProperty("stride", 1);
    this.addProperty("padding", 1);
    this.addProperty("dilation", 1);
    this.addProperty("groups", 1);
    this.addProperty("bias", true);
    this.addProperty("output_padding", 0);  // for transpose
    this.addProperty("depth_multiplier", 1); // for depthwise
    
    this.type_widget = this.addWidget("combo", "Type", "conv2d", (v) => { 
        this.properties.conv_type = v;
        this.updateTitle();
    }, { values: ["conv2d", "conv1d", "conv3d", "conv_transpose2d", "conv_transpose1d", "conv_transpose3d", "depthwise_conv2d"] });
    
    this.addWidget("number", "In Ch", 3, (v) => { this.properties.in_channels = Math.round(v); }, { min: 1, step: 1, precision: 0 });
    this.addWidget("number", "Out Ch", 64, (v) => { this.properties.out_channels = Math.round(v); }, { min: 1, step: 1, precision: 0 });
    this.addWidget("number", "Kernel", 3, (v) => { this.properties.kernel_size = Math.round(v); }, { min: 1, step: 1, precision: 0 });
    this.addWidget("number", "Stride", 1, (v) => { this.properties.stride = Math.round(v); }, { min: 1, step: 1, precision: 0 });
    this.addWidget("number", "Padding", 1, (v) => { this.properties.padding = Math.round(v); }, { min: 0, step: 1, precision: 0 });
    this.addWidget("number", "Dilation", 1, (v) => { this.properties.dilation = Math.round(v); }, { min: 1, step: 1, precision: 0 });
    this.addWidget("number", "Groups", 1, (v) => { this.properties.groups = Math.round(v); }, { min: 1, step: 1, precision: 0 });
    this.addWidget("toggle", "Bias", true, (v) => { this.properties.bias = v; });
    
    this.size = [200, 260];
    this.updateTitle();
}
ConvolutionNode.title = "Convolution";
ConvolutionNode.desc = "Configurable convolution layer";
ConvolutionNode.prototype.updateTitle = function() {
    const typeNames = {
        "conv2d": "Conv2D",
        "conv1d": "Conv1D", 
        "conv3d": "Conv3D",
        "conv_transpose2d": "ConvT2D",
        "conv_transpose1d": "ConvT1D",
        "conv_transpose3d": "ConvT3D",
        "depthwise_conv2d": "DWConv2D"
    };
    this.title = typeNames[this.properties.conv_type] || "Convolution";
};
ConvolutionNode.prototype.onExecute = function() {
    const input = this.getInputData(0);
    if (!input || !input.shape) return;
    
    const type = this.properties.conv_type;
    const k = this.properties.kernel_size;
    const s = this.properties.stride;
    const p = this.properties.padding;
    const d = this.properties.dilation;
    const out_c = this.properties.out_channels;
    
    let shape;
    
    if (type === "conv1d") {
        const [n, c, l] = input.shape;
        const out_l = Math.floor((l + 2 * p - d * (k - 1) - 1) / s) + 1;
        shape = [n, out_c, out_l];
    } else if (type === "conv2d") {
        const [n, c, h, w] = input.shape;
        const out_h = Math.floor((h + 2 * p - d * (k - 1) - 1) / s) + 1;
        const out_w = Math.floor((w + 2 * p - d * (k - 1) - 1) / s) + 1;
        shape = [n, out_c, out_h, out_w];
    } else if (type === "conv3d") {
        const [n, c, d_in, h, w] = input.shape;
        const out_d = Math.floor((d_in + 2 * p - this.properties.dilation * (k - 1) - 1) / s) + 1;
        const out_h = Math.floor((h + 2 * p - this.properties.dilation * (k - 1) - 1) / s) + 1;
        const out_w = Math.floor((w + 2 * p - this.properties.dilation * (k - 1) - 1) / s) + 1;
        shape = [n, out_c, out_d, out_h, out_w];
    } else if (type === "conv_transpose1d") {
        const [n, c, l] = input.shape;
        const out_l = (l - 1) * s - 2 * p + d * (k - 1) + this.properties.output_padding + 1;
        shape = [n, out_c, out_l];
    } else if (type === "conv_transpose2d") {
        const [n, c, h, w] = input.shape;
        const out_h = (h - 1) * s - 2 * p + d * (k - 1) + this.properties.output_padding + 1;
        const out_w = (w - 1) * s - 2 * p + d * (k - 1) + this.properties.output_padding + 1;
        shape = [n, out_c, out_h, out_w];
    } else if (type === "conv_transpose3d") {
        const [n, c, d_in, h, w] = input.shape;
        const out_d = (d_in - 1) * s - 2 * p + d * (k - 1) + this.properties.output_padding + 1;
        const out_h = (h - 1) * s - 2 * p + d * (k - 1) + this.properties.output_padding + 1;
        const out_w = (w - 1) * s - 2 * p + d * (k - 1) + this.properties.output_padding + 1;
        shape = [n, out_c, out_d, out_h, out_w];
    } else if (type === "depthwise_conv2d") {
        const [n, c, h, w] = input.shape;
        const out_h = Math.floor((h + 2 * p - d * (k - 1) - 1) / s) + 1;
        const out_w = Math.floor((w + 2 * p - d * (k - 1) - 1) / s) + 1;
        shape = [n, c * this.properties.depth_multiplier, out_h, out_w];
    }
    
    this.setOutputData(0, { shape, dtype: input.dtype });
};
ConvolutionNode.prototype.onConfigure = function() {
    this.updateTitle();
};
LiteGraph.registerNodeType("ML/convolution", ConvolutionNode);


// ============================================
// GENERIC ACTIVATION NODE
// ============================================

function ActivationNode() {
    this.addInput("input", "tensor");
    this.addOutput("output", "tensor");
    
    this.addProperty("activation", "relu");
    this.addProperty("negative_slope", 0.01);  // for leaky_relu
    this.addProperty("alpha", 1.0);            // for elu
    this.addProperty("dim", -1);               // for softmax
    this.addProperty("approximate", "none");   // for gelu
    this.addProperty("inplace", false);
    
    this.type_widget = this.addWidget("combo", "Type", "relu", (v) => { 
        this.properties.activation = v;
        this.updateTitle();
        this.updateWidgets();
    }, { values: ["relu", "leaky_relu", "prelu", "elu", "selu", "gelu", "silu", "mish", "sigmoid", "tanh", "softmax", "log_softmax", "softplus", "softsign", "hardtanh", "hardswish", "hardsigmoid"] });
    
    // Parameter widgets (shown/hidden based on activation type)
    this.slope_widget = this.addWidget("number", "Neg Slope", 0.01, (v) => { this.properties.negative_slope = v; }, { min: 0, max: 1, step: 0.01 });
    this.alpha_widget = this.addWidget("number", "Alpha", 1.0, (v) => { this.properties.alpha = v; }, { min: 0, step: 0.1 });
    this.dim_widget = this.addWidget("number", "Dim", -1, (v) => { this.properties.dim = Math.round(v); }, { step: 1, precision: 0 });
    this.approx_widget = this.addWidget("combo", "Approx", "none", (v) => { this.properties.approximate = v; }, { values: ["none", "tanh"] });
    this.inplace_widget = this.addWidget("toggle", "Inplace", false, (v) => { this.properties.inplace = v; });
    
    this.size = [180, 100];
    this.updateTitle();
    this.updateWidgets();
}
ActivationNode.title = "Activation";
ActivationNode.desc = "Configurable activation function";
ActivationNode.prototype.updateTitle = function() {
    const names = {
        "relu": "ReLU", "leaky_relu": "LeakyReLU", "prelu": "PReLU", "elu": "ELU",
        "selu": "SELU", "gelu": "GELU", "silu": "SiLU", "mish": "Mish",
        "sigmoid": "Sigmoid", "tanh": "Tanh", "softmax": "Softmax", 
        "log_softmax": "LogSoftmax", "softplus": "Softplus", "softsign": "Softsign",
        "hardtanh": "Hardtanh", "hardswish": "Hardswish", "hardsigmoid": "Hardsigmoid"
    };
    this.title = names[this.properties.activation] || "Activation";
};
ActivationNode.prototype.updateWidgets = function() {
    const act = this.properties.activation;
    
    // Hide all optional widgets first
    if (this.slope_widget) this.slope_widget.hidden = true;
    if (this.alpha_widget) this.alpha_widget.hidden = true;
    if (this.dim_widget) this.dim_widget.hidden = true;
    if (this.approx_widget) this.approx_widget.hidden = true;
    
    // Show relevant widgets
    if (act === "leaky_relu") {
        this.slope_widget.hidden = false;
        this.size = [180, 100];
    } else if (act === "elu" || act === "selu") {
        this.alpha_widget.hidden = false;
        this.size = [180, 100];
    } else if (act === "softmax" || act === "log_softmax") {
        this.dim_widget.hidden = false;
        this.size = [180, 100];
    } else if (act === "gelu") {
        this.approx_widget.hidden = false;
        this.size = [180, 100];
    } else {
        this.size = [180, 80];
    }
    
    this.setDirtyCanvas(true, true);
};
ActivationNode.prototype.onExecute = function() {
    const input = this.getInputData(0);
    if (input) this.setOutputData(0, input);
};
ActivationNode.prototype.onConfigure = function() {
    this.updateTitle();
    this.updateWidgets();
};
LiteGraph.registerNodeType("ML/activation", ActivationNode);


// ============================================
// SPECIFIC CONVOLUTION NODES (legacy/convenience)
// ============================================

function Conv2D() {
    this.addInput("input", "tensor");
    this.addOutput("output", "tensor");
    
    this.addProperty("in_channels", 3);
    this.addProperty("out_channels", 64);
    this.addProperty("kernel_size", 3);
    this.addProperty("stride", 1);
    this.addProperty("padding", 1);
    this.addProperty("dilation", 1);
    this.addProperty("groups", 1);
    this.addProperty("bias", true);
    
    this.addWidget("number", "In Ch", 3, (v) => { this.properties.in_channels = Math.round(v); }, { min: 1, step: 1, precision: 0 });
    this.addWidget("number", "Out Ch", 64, (v) => { this.properties.out_channels = Math.round(v); }, { min: 1, step: 1, precision: 0 });
    this.addWidget("number", "Kernel", 3, (v) => { this.properties.kernel_size = Math.round(v); }, { min: 1, step: 1, precision: 0 });
    this.addWidget("number", "Stride", 1, (v) => { this.properties.stride = Math.round(v); }, { min: 1, step: 1, precision: 0 });
    this.addWidget("number", "Padding", 1, (v) => { this.properties.padding = Math.round(v); }, { min: 0, step: 1, precision: 0 });
    this.addWidget("toggle", "Bias", true, (v) => { this.properties.bias = v; });
    
    this.size = [180, 180];
}
Conv2D.title = "Conv2D";
Conv2D.desc = "2D Convolution layer";
Conv2D.prototype.onExecute = function() {
    const input = this.getInputData(0);
    if (input && input.shape) {
        const [n, c, h, w] = input.shape;
        const out_h = Math.floor((h + 2 * this.properties.padding - this.properties.kernel_size) / this.properties.stride) + 1;
        const out_w = Math.floor((w + 2 * this.properties.padding - this.properties.kernel_size) / this.properties.stride) + 1;
        this.setOutputData(0, { shape: [n, this.properties.out_channels, out_h, out_w], dtype: input.dtype });
    }
};
LiteGraph.registerNodeType("ML/conv2d", Conv2D);


function ConvTranspose2D() {
    this.addInput("input", "tensor");
    this.addOutput("output", "tensor");
    
    this.addProperty("in_channels", 64);
    this.addProperty("out_channels", 32);
    this.addProperty("kernel_size", 4);
    this.addProperty("stride", 2);
    this.addProperty("padding", 1);
    this.addProperty("output_padding", 0);
    
    this.addWidget("number", "In Ch", 64, (v) => { this.properties.in_channels = Math.round(v); }, { min: 1, step: 1, precision: 0 });
    this.addWidget("number", "Out Ch", 32, (v) => { this.properties.out_channels = Math.round(v); }, { min: 1, step: 1, precision: 0 });
    this.addWidget("number", "Kernel", 4, (v) => { this.properties.kernel_size = Math.round(v); }, { min: 1, step: 1, precision: 0 });
    this.addWidget("number", "Stride", 2, (v) => { this.properties.stride = Math.round(v); }, { min: 1, step: 1, precision: 0 });
    this.addWidget("number", "Padding", 1, (v) => { this.properties.padding = Math.round(v); }, { min: 0, step: 1, precision: 0 });
    
    this.size = [180, 160];
}
ConvTranspose2D.title = "ConvTranspose2D";
ConvTranspose2D.desc = "2D Transposed Convolution";
ConvTranspose2D.prototype.onExecute = function() {
    const input = this.getInputData(0);
    if (input && input.shape) {
        const [n, c, h, w] = input.shape;
        const out_h = (h - 1) * this.properties.stride - 2 * this.properties.padding + this.properties.kernel_size + this.properties.output_padding;
        const out_w = (w - 1) * this.properties.stride - 2 * this.properties.padding + this.properties.kernel_size + this.properties.output_padding;
        this.setOutputData(0, { shape: [n, this.properties.out_channels, out_h, out_w], dtype: input.dtype });
    }
};
LiteGraph.registerNodeType("ML/conv_transpose2d", ConvTranspose2D);


function DepthwiseConv2D() {
    this.addInput("input", "tensor");
    this.addOutput("output", "tensor");
    
    this.addProperty("kernel_size", 3);
    this.addProperty("stride", 1);
    this.addProperty("padding", 1);
    this.addProperty("depth_multiplier", 1);
    
    this.addWidget("number", "Kernel", 3, (v) => { this.properties.kernel_size = Math.round(v); }, { min: 1, step: 1, precision: 0 });
    this.addWidget("number", "Stride", 1, (v) => { this.properties.stride = Math.round(v); }, { min: 1, step: 1, precision: 0 });
    this.addWidget("number", "Padding", 1, (v) => { this.properties.padding = Math.round(v); }, { min: 0, step: 1, precision: 0 });
    this.addWidget("number", "Depth Mult", 1, (v) => { this.properties.depth_multiplier = Math.round(v); }, { min: 1, step: 1, precision: 0 });
    
    this.size = [180, 130];
}
DepthwiseConv2D.title = "DepthwiseConv2D";
DepthwiseConv2D.desc = "Depthwise Separable Convolution";
DepthwiseConv2D.prototype.onExecute = function() {
    const input = this.getInputData(0);
    if (input && input.shape) {
        const [n, c, h, w] = input.shape;
        const out_h = Math.floor((h + 2 * this.properties.padding - this.properties.kernel_size) / this.properties.stride) + 1;
        const out_w = Math.floor((w + 2 * this.properties.padding - this.properties.kernel_size) / this.properties.stride) + 1;
        this.setOutputData(0, { shape: [n, c * this.properties.depth_multiplier, out_h, out_w], dtype: input.dtype });
    }
};
LiteGraph.registerNodeType("ML/depthwise_conv2d", DepthwiseConv2D);


// ============================================
// SPECIFIC ACTIVATION NODES (legacy/convenience)
// ============================================

function ReLU() {
    this.addInput("input", "tensor");
    this.addOutput("output", "tensor");
    this.addProperty("inplace", false);
    this.addWidget("toggle", "Inplace", false, (v) => { this.properties.inplace = v; });
    this.size = [140, 50];
}
ReLU.title = "ReLU";
ReLU.desc = "Rectified Linear Unit";
ReLU.prototype.onExecute = function() {
    const input = this.getInputData(0);
    if (input) this.setOutputData(0, input);
};
LiteGraph.registerNodeType("ML/relu", ReLU);


function LeakyReLU() {
    this.addInput("input", "tensor");
    this.addOutput("output", "tensor");
    this.addProperty("negative_slope", 0.01);
    this.addWidget("number", "Neg Slope", 0.01, (v) => { this.properties.negative_slope = v; }, { min: 0, max: 1, step: 0.01 });
    this.size = [160, 50];
}
LeakyReLU.title = "LeakyReLU";
LeakyReLU.desc = "Leaky ReLU activation";
LeakyReLU.prototype.onExecute = function() {
    const input = this.getInputData(0);
    if (input) this.setOutputData(0, input);
};
LiteGraph.registerNodeType("ML/leaky_relu", LeakyReLU);


function GELU() {
    this.addInput("input", "tensor");
    this.addOutput("output", "tensor");
    this.addProperty("approximate", "none");
    this.addWidget("combo", "Approx", "none", (v) => { this.properties.approximate = v; }, { values: ["none", "tanh"] });
    this.size = [140, 50];
}
GELU.title = "GELU";
GELU.desc = "Gaussian Error Linear Unit";
GELU.prototype.onExecute = function() {
    const input = this.getInputData(0);
    if (input) this.setOutputData(0, input);
};
LiteGraph.registerNodeType("ML/gelu", GELU);


function SiLU() {
    this.addInput("input", "tensor");
    this.addOutput("output", "tensor");
    this.size = [120, 30];
}
SiLU.title = "SiLU/Swish";
SiLU.desc = "Sigmoid Linear Unit (Swish)";
SiLU.prototype.onExecute = function() {
    const input = this.getInputData(0);
    if (input) this.setOutputData(0, input);
};
LiteGraph.registerNodeType("ML/silu", SiLU);


function Sigmoid() {
    this.addInput("input", "tensor");
    this.addOutput("output", "tensor");
    this.size = [120, 30];
}
Sigmoid.title = "Sigmoid";
Sigmoid.desc = "Sigmoid activation";
Sigmoid.prototype.onExecute = function() {
    const input = this.getInputData(0);
    if (input) this.setOutputData(0, input);
};
LiteGraph.registerNodeType("ML/sigmoid", Sigmoid);


function Softmax() {
    this.addInput("input", "tensor");
    this.addOutput("output", "tensor");
    this.addProperty("dim", -1);
    this.addWidget("number", "Dim", -1, (v) => { this.properties.dim = Math.round(v); }, { step: 1, precision: 0 });
    this.size = [140, 50];
}
Softmax.title = "Softmax";
Softmax.desc = "Softmax activation";
Softmax.prototype.onExecute = function() {
    const input = this.getInputData(0);
    if (input) this.setOutputData(0, input);
};
LiteGraph.registerNodeType("ML/softmax", Softmax);


// ============================================
// NORMALIZATION NODES
// ============================================

function BatchNorm2D() {
    this.addInput("input", "tensor");
    this.addOutput("output", "tensor");
    this.addProperty("num_features", 64);
    this.addProperty("eps", 1e-5);
    this.addProperty("momentum", 0.1);
    this.addProperty("affine", true);
    
    this.addWidget("number", "Features", 64, (v) => { this.properties.num_features = Math.round(v); }, { min: 1, step: 1, precision: 0 });
    this.addWidget("toggle", "Affine", true, (v) => { this.properties.affine = v; });
    this.size = [160, 70];
}
BatchNorm2D.title = "BatchNorm2D";
BatchNorm2D.desc = "2D Batch Normalization";
BatchNorm2D.prototype.onExecute = function() {
    const input = this.getInputData(0);
    if (input) this.setOutputData(0, input);
};
LiteGraph.registerNodeType("ML/batchnorm2d", BatchNorm2D);


function LayerNorm() {
    this.addInput("input", "tensor");
    this.addOutput("output", "tensor");
    this.addProperty("normalized_shape", [64]);
    this.addProperty("eps", 1e-5);
    
    this.shape_widget = this.addWidget("text", "Norm Shape", "64", (v) => {
        this.properties.normalized_shape = v.split(",").map(x => parseInt(x.trim()));
    });
    this.size = [160, 50];
}
LayerNorm.title = "LayerNorm";
LayerNorm.desc = "Layer Normalization";
LayerNorm.prototype.onExecute = function() {
    const input = this.getInputData(0);
    if (input) this.setOutputData(0, input);
};
LayerNorm.prototype.onConfigure = function() {
    if (this.shape_widget) {
        this.shape_widget.value = this.properties.normalized_shape.join(", ");
    }
};
LiteGraph.registerNodeType("ML/layernorm", LayerNorm);


function GroupNorm() {
    this.addInput("input", "tensor");
    this.addOutput("output", "tensor");
    this.addProperty("num_groups", 32);
    this.addProperty("num_channels", 64);
    this.addProperty("eps", 1e-5);
    
    this.addWidget("number", "Groups", 32, (v) => { this.properties.num_groups = Math.round(v); }, { min: 1, step: 1, precision: 0 });
    this.addWidget("number", "Channels", 64, (v) => { this.properties.num_channels = Math.round(v); }, { min: 1, step: 1, precision: 0 });
    this.size = [160, 70];
}
GroupNorm.title = "GroupNorm";
GroupNorm.desc = "Group Normalization";
GroupNorm.prototype.onExecute = function() {
    const input = this.getInputData(0);
    if (input) this.setOutputData(0, input);
};
LiteGraph.registerNodeType("ML/groupnorm", GroupNorm);


// ============================================
// POOLING NODES
// ============================================

function MaxPool2D() {
    this.addInput("input", "tensor");
    this.addOutput("output", "tensor");
    this.addProperty("kernel_size", 2);
    this.addProperty("stride", 2);
    this.addProperty("padding", 0);
    
    this.addWidget("number", "Kernel", 2, (v) => { this.properties.kernel_size = Math.round(v); }, { min: 1, step: 1, precision: 0 });
    this.addWidget("number", "Stride", 2, (v) => { this.properties.stride = Math.round(v); }, { min: 1, step: 1, precision: 0 });
    this.addWidget("number", "Padding", 0, (v) => { this.properties.padding = Math.round(v); }, { min: 0, step: 1, precision: 0 });
    this.size = [160, 100];
}
MaxPool2D.title = "MaxPool2D";
MaxPool2D.desc = "2D Max Pooling";
MaxPool2D.prototype.onExecute = function() {
    const input = this.getInputData(0);
    if (input && input.shape) {
        const [n, c, h, w] = input.shape;
        const out_h = Math.floor((h + 2 * this.properties.padding - this.properties.kernel_size) / this.properties.stride) + 1;
        const out_w = Math.floor((w + 2 * this.properties.padding - this.properties.kernel_size) / this.properties.stride) + 1;
        this.setOutputData(0, { shape: [n, c, out_h, out_w], dtype: input.dtype });
    }
};
LiteGraph.registerNodeType("ML/maxpool2d", MaxPool2D);


function AvgPool2D() {
    this.addInput("input", "tensor");
    this.addOutput("output", "tensor");
    this.addProperty("kernel_size", 2);
    this.addProperty("stride", 2);
    this.addProperty("padding", 0);
    
    this.addWidget("number", "Kernel", 2, (v) => { this.properties.kernel_size = Math.round(v); }, { min: 1, step: 1, precision: 0 });
    this.addWidget("number", "Stride", 2, (v) => { this.properties.stride = Math.round(v); }, { min: 1, step: 1, precision: 0 });
    this.addWidget("number", "Padding", 0, (v) => { this.properties.padding = Math.round(v); }, { min: 0, step: 1, precision: 0 });
    this.size = [160, 100];
}
AvgPool2D.title = "AvgPool2D";
AvgPool2D.desc = "2D Average Pooling";
AvgPool2D.prototype.onExecute = function() {
    const input = this.getInputData(0);
    if (input && input.shape) {
        const [n, c, h, w] = input.shape;
        const out_h = Math.floor((h + 2 * this.properties.padding - this.properties.kernel_size) / this.properties.stride) + 1;
        const out_w = Math.floor((w + 2 * this.properties.padding - this.properties.kernel_size) / this.properties.stride) + 1;
        this.setOutputData(0, { shape: [n, c, out_h, out_w], dtype: input.dtype });
    }
};
LiteGraph.registerNodeType("ML/avgpool2d", AvgPool2D);


function AdaptiveAvgPool2D() {
    this.addInput("input", "tensor");
    this.addOutput("output", "tensor");
    this.addProperty("output_size", [1, 1]);
    
    this.size_widget = this.addWidget("text", "Output Size", "1,1", (v) => {
        this.properties.output_size = v.split(",").map(x => parseInt(x.trim()));
    });
    this.size = [170, 50];
}
AdaptiveAvgPool2D.title = "AdaptiveAvgPool2D";
AdaptiveAvgPool2D.desc = "2D Adaptive Average Pooling";
AdaptiveAvgPool2D.prototype.onExecute = function() {
    const input = this.getInputData(0);
    if (input && input.shape) {
        const [n, c] = input.shape;
        this.setOutputData(0, { shape: [n, c, ...this.properties.output_size], dtype: input.dtype });
    }
};
AdaptiveAvgPool2D.prototype.onConfigure = function() {
    if (this.size_widget) {
        this.size_widget.value = this.properties.output_size.join(", ");
    }
};
LiteGraph.registerNodeType("ML/adaptive_avgpool2d", AdaptiveAvgPool2D);


// ============================================
// LINEAR / DENSE NODES
// ============================================

function Linear() {
    this.addInput("input", "tensor");
    this.addOutput("output", "tensor");
    this.addProperty("in_features", 512);
    this.addProperty("out_features", 256);
    this.addProperty("bias", true);
    
    this.addWidget("number", "In", 512, (v) => { this.properties.in_features = Math.round(v); }, { min: 1, step: 1, precision: 0 });
    this.addWidget("number", "Out", 256, (v) => { this.properties.out_features = Math.round(v); }, { min: 1, step: 1, precision: 0 });
    this.addWidget("toggle", "Bias", true, (v) => { this.properties.bias = v; });
    this.size = [160, 100];
}
Linear.title = "Linear";
Linear.desc = "Fully Connected / Dense layer";
Linear.prototype.onExecute = function() {
    const input = this.getInputData(0);
    if (input && input.shape) {
        const shape = [...input.shape];
        shape[shape.length - 1] = this.properties.out_features;
        this.setOutputData(0, { shape, dtype: input.dtype });
    }
};
LiteGraph.registerNodeType("ML/linear", Linear);


// ============================================
// TENSOR OPERATIONS
// ============================================

function Reshape() {
    this.addInput("input", "tensor");
    this.addOutput("output", "tensor");
    this.addProperty("shape", [-1, 512]);
    
    this.shape_widget = this.addWidget("text", "Shape", "-1,512", (v) => {
        this.properties.shape = v.split(",").map(x => parseInt(x.trim()));
    });
    this.size = [160, 50];
}
Reshape.title = "Reshape";
Reshape.desc = "Reshape tensor";
Reshape.prototype.onExecute = function() {
    const input = this.getInputData(0);
    if (input && input.shape) {
        // Calculate -1 dimension if present
        const shape = [...this.properties.shape];
        const totalInput = input.shape.reduce((a, b) => a * b, 1);
        const negIdx = shape.indexOf(-1);
        if (negIdx !== -1) {
            const known = shape.filter(x => x > 0).reduce((a, b) => a * b, 1);
            shape[negIdx] = totalInput / known;
        }
        this.setOutputData(0, { shape, dtype: input.dtype });
    }
};
Reshape.prototype.onConfigure = function() {
    if (this.shape_widget) {
        this.shape_widget.value = this.properties.shape.join(", ");
    }
};
LiteGraph.registerNodeType("ML/reshape", Reshape);


function Flatten() {
    this.addInput("input", "tensor");
    this.addOutput("output", "tensor");
    this.addProperty("start_dim", 1);
    this.addProperty("end_dim", -1);
    
    this.addWidget("number", "Start Dim", 1, (v) => { this.properties.start_dim = Math.round(v); }, { step: 1, precision: 0 });
    this.addWidget("number", "End Dim", -1, (v) => { this.properties.end_dim = Math.round(v); }, { step: 1, precision: 0 });
    this.size = [160, 70];
}
Flatten.title = "Flatten";
Flatten.desc = "Flatten tensor dimensions";
Flatten.prototype.onExecute = function() {
    const input = this.getInputData(0);
    if (input && input.shape) {
        const shape = input.shape;
        let start = this.properties.start_dim >= 0 ? this.properties.start_dim : shape.length + this.properties.start_dim;
        let end = this.properties.end_dim >= 0 ? this.properties.end_dim : shape.length + this.properties.end_dim;
        
        const before = shape.slice(0, start);
        const middle = shape.slice(start, end + 1).reduce((a, b) => a * b, 1);
        const after = shape.slice(end + 1);
        
        this.setOutputData(0, { shape: [...before, middle, ...after], dtype: input.dtype });
    }
};
LiteGraph.registerNodeType("ML/flatten", Flatten);


function Concat() {
    this.addInput("input_a", "tensor");
    this.addInput("input_b", "tensor");
    this.addOutput("output", "tensor");
    this.addProperty("dim", 1);
    
    this.addWidget("number", "Dim", 1, (v) => { this.properties.dim = Math.round(v); }, { step: 1, precision: 0 });
    this.size = [140, 70];
}
Concat.title = "Concat";
Concat.desc = "Concatenate tensors";
Concat.prototype.onExecute = function() {
    const a = this.getInputData(0);
    const b = this.getInputData(1);
    if (a && b && a.shape && b.shape) {
        const shape = [...a.shape];
        const dim = this.properties.dim >= 0 ? this.properties.dim : shape.length + this.properties.dim;
        shape[dim] = a.shape[dim] + b.shape[dim];
        this.setOutputData(0, { shape, dtype: a.dtype });
    }
};
LiteGraph.registerNodeType("ML/concat", Concat);


function Add() {
    this.addInput("input_a", "tensor");
    this.addInput("input_b", "tensor");
    this.addOutput("output", "tensor");
    this.size = [120, 50];
}
Add.title = "Add";
Add.desc = "Element-wise addition";
Add.prototype.onExecute = function() {
    const a = this.getInputData(0);
    if (a) this.setOutputData(0, a);
};
LiteGraph.registerNodeType("ML/add", Add);


function Multiply() {
    this.addInput("input_a", "tensor");
    this.addInput("input_b", "tensor");
    this.addOutput("output", "tensor");
    this.size = [120, 50];
}
Multiply.title = "Multiply";
Multiply.desc = "Element-wise multiplication";
Multiply.prototype.onExecute = function() {
    const a = this.getInputData(0);
    if (a) this.setOutputData(0, a);
};
LiteGraph.registerNodeType("ML/multiply", Multiply);


// ============================================
// ATTENTION NODES
// ============================================

function MultiHeadAttention() {
    this.addInput("query", "tensor");
    this.addInput("key", "tensor");
    this.addInput("value", "tensor");
    this.addOutput("output", "tensor");
    this.addOutput("attn_weights", "tensor");
    
    this.addProperty("embed_dim", 512);
    this.addProperty("num_heads", 8);
    this.addProperty("dropout", 0.0);
    
    this.addWidget("number", "Embed Dim", 512, (v) => { this.properties.embed_dim = Math.round(v); }, { min: 1, step: 1, precision: 0 });
    this.addWidget("number", "Num Heads", 8, (v) => { this.properties.num_heads = Math.round(v); }, { min: 1, step: 1, precision: 0 });
    this.addWidget("number", "Dropout", 0.0, (v) => { this.properties.dropout = v; }, { min: 0, max: 1, step: 0.1 });
    this.size = [180, 130];
}
MultiHeadAttention.title = "MultiHeadAttention";
MultiHeadAttention.desc = "Multi-Head Self-Attention";
MultiHeadAttention.prototype.onExecute = function() {
    const q = this.getInputData(0);
    if (q) {
        this.setOutputData(0, q);
        if (q.shape) {
            this.setOutputData(1, { shape: [q.shape[0], this.properties.num_heads, q.shape[1], q.shape[1]], dtype: q.dtype });
        }
    }
};
LiteGraph.registerNodeType("ML/multihead_attention", MultiHeadAttention);


// ============================================
// DROPOUT / REGULARIZATION
// ============================================

function Dropout() {
    this.addInput("input", "tensor");
    this.addOutput("output", "tensor");
    this.addProperty("p", 0.5);
    this.addProperty("inplace", false);
    
    this.addWidget("number", "p", 0.5, (v) => { this.properties.p = v; }, { min: 0, max: 1, step: 0.1 });
    this.size = [140, 50];
}
Dropout.title = "Dropout";
Dropout.desc = "Dropout regularization";
Dropout.prototype.onExecute = function() {
    const input = this.getInputData(0);
    if (input) this.setOutputData(0, input);
};
LiteGraph.registerNodeType("ML/dropout", Dropout);


// ============================================
// OUTPUT NODE
// ============================================

function OutputTensor() {
    this.addInput("tensor", "tensor");
    this.addProperty("name", "output");
    this.addWidget("text", "Name", "output", { property: "name" });
    this.size = [160, 50];
}
OutputTensor.title = "Output";
OutputTensor.desc = "Output tensor";
OutputTensor.prototype.onExecute = function() {
    const input = this.getInputData(0);
    if (input && input.shape) {
        this.title = "Output: " + formatShape(input.shape);
    }
};
LiteGraph.registerNodeType("ML/output_tensor", OutputTensor);


console.log("ML nodes registered successfully!");