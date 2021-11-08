MODEL_PATH = 'model/model.bin'

frac = 1
num_word_threshold = -1
split = 0.1
Warmup_steps = 0.15

Epochs = 100
Batch_Size = 4
LR = 1e-3


transform_threshold = 0.5

skip_frame = 2
image_height = 64


vocab_size = 30522
encoder_num_res_blocks = 2
gif_channel_out = 8
gif_height = 64 
gif_width = 64
embed_out = 100
decoder_num_layers = 2
forward_expansion = 4
attention_num_heads = 5
output_max_len = 200
layer_norm_ep = 1e-6
encoder_dropout = 0.25
decoder_dropout = 0.1
