import torch 
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, input_dims, num_heads):
        super(SelfAttention, self).__init__()

        self.input_dims = input_dims
        self.num_heads = num_heads
        self.out_dims = int(self.input_dims/self.num_heads)

        self.key = nn.Linear(self.out_dims, self.out_dims)
        self.query = nn.Linear(self.out_dims, self.out_dims)
        self.value = nn.Linear(self.out_dims, self.out_dims)

        self.out  = nn.Linear(self.out_dims*self.num_heads, self.input_dims)

    def forward(self, query, key, value, mask):
        batch_size = query.shape[0]
        query_len, key_len, value_len = query.shape[1], key.shape[1], value.shape[1]

        query = query.reshape(batch_size, query_len, self.num_heads, self.out_dims)
        key = key.reshape(batch_size, key_len, self.num_heads, self.out_dims)
        value = value.reshape(batch_size, value_len, self.num_heads, self.out_dims)

        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        attention = torch.einsum("bqhd, bkhd -> bhqk", [query, key])
        if mask is not None:
            attention = attention.masked_fill(mask==0, float("-1e20"))
        
        attention_score = nn.Softmax(dim=-1)(attention/((self.num_heads)**1/2))

        out = torch.einsum("bhqv, bvhd -> bqdh", [attention_score, value])

        out = out.reshape(batch_size, query_len, self.num_heads*self.out_dims)

        out = self.out(out)

        return out


class Transformer_Block(nn.Module):
    def __init__(
        self,
        input_dims,
        forward_expansion,
        num_heads,
        dropout,
        layer_norm_ep
    ):
        super(Transformer_Block, self).__init__()
        self.input_dims = input_dims
        self.num_heads = num_heads
        
        self.attention_block = SelfAttention(self.input_dims, self.num_heads)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm_1 = nn.LayerNorm(self.input_dims, eps=layer_norm_ep)
        self.layer_norm_2 = nn.LayerNorm(self.input_dims, eps=layer_norm_ep)

        self.feed_forward = nn.Sequential(
            nn.Linear(self.input_dims, self.input_dims*forward_expansion),
            nn.GELU(),
            nn.Linear(self.input_dims*forward_expansion, self.input_dims, bias=False)
        )

    def forward(self, query, key, value, mask):
        x = self.attention_block(query, key, value, mask)
        x = x + query
        x = self.dropout(self.layer_norm_1(x))
        ff = self.feed_forward(x)
        out = self.dropout(self.layer_norm_2(ff + x))
        return out


class Decoder_Block(nn.Module):
    def __init__(
        self,
        input_dims, 
        forward_expansion,
        num_heads,
        dropout, 
        layer_norm_ep

    ):
        super(Decoder_Block, self).__init__()
        self.attention_block = SelfAttention(input_dims, num_heads)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_dims, eps=layer_norm_ep)
        self.transformer_block = Transformer_Block(
            input_dims, 
            forward_expansion, 
            num_heads, dropout, 
            layer_norm_ep
        )
    
    def forward(self, query, key, value, src_mask, casual_mask):
        x = self.attention_block(query, query, query, casual_mask)
        x = self.dropout(self.layer_norm(x+query))
        x = self.transformer_block(x, key, value, src_mask)
        return x


class ConvBlock(nn.Module):
    def __init__(self, channel_in, channel_out, dropout):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv3d(channel_in, channel_out, kernel_size=3, padding=1, bias=False)
        self.batch_norm = nn.BatchNorm3d(channel_in)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(nn.LeakyReLU()(self.batch_norm(x)))
        return x



class ResBlock(nn.Module):
    def __init__(self, channel_in, channel_out, dropout):
        super(ResBlock, self).__init__()
        self.conv_block = ConvBlock(channel_in, channel_out, dropout)
        self.skip = nn.Conv3d(channel_in, channel_out, kernel_size=1)
    
    def forward(self, x):
        skip = self.skip(x)
        x = self.conv_block(x)
        x = x + skip
        return x
        


class Encoder(nn.Module):
    def __init__(
        self,
        num_res_blocks,
        channel_out,
        out_dims,
        height, 
        width,
        dropout=0.2
    ):
        super(Encoder, self).__init__()
        self.conv1_block = nn.Sequential(
            nn.Conv3d(4, channel_out, kernel_size=3, padding=1),
            nn.BatchNorm3d(channel_out),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )
        self.batch_norm = nn.BatchNorm3d(channel_out)
        self.dropout = nn.Dropout(dropout)
        self.res_blocks = nn.Sequential(
            *[
                ResBlock(channel_out, channel_out, dropout)
                for _ in range(num_res_blocks)
            ]
        )


        self.out = nn.Linear(height, out_dims)

    def forward(self, x):
        x = self.dropout(nn.LeakyReLU()(self.batch_norm(self.conv1_block(x))))
        for block in self.res_blocks:
            x = block(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = x.mean(2).mean(2)

        out = self.out(x)

        return out




class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size, 
        embed_out,
        num_layers,
        forward_expansion,
        num_heads,
        dropout,
        max_len,
        layer_norm_ep=1e-6
        
    ):
        super(Decoder, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embed_out)
        self.positional_embeddings = nn.Parameter(torch.zeros(1, max_len, embed_out))

        self.layers_out = nn.Sequential(*[
            Decoder_Block(
                embed_out,
                forward_expansion,
                num_heads,
                dropout,
                layer_norm_ep
            )
            for _ in range(num_layers)
        ])

        self.fc = nn.Linear(embed_out,vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_out, src_mask=None, causal_mask=None):
        seq_len = x.shape[1]
        postional_embed = self.positional_embeddings[:, :seq_len, :]
        embeddings = self.word_embeddings(x) + postional_embed
        x = self.dropout(embeddings)
        for layer in self.layers_out:
            x = layer(x, encoder_out, encoder_out, src_mask, causal_mask)
        
        out = self.fc(x)
        return out



class GifToCaption(nn.Module):
    def __init__(
        self,
        encoder_num_res_blocks,
        gif_channel_out,
        gif_height, 
        gif_width,
        vocab_size, 
        embed_out,
        decoder_num_layers,
        forward_expansion,
        attention_num_heads,
        output_max_len,
        layer_norm_ep=1e-6,
        encoder_dropout=0.25,
        decoder_dropout=0.1,
        device="cpu"
    ):
        super(GifToCaption, self).__init__()
        self.device = device

        self.encoder = Encoder(
            encoder_num_res_blocks,
            gif_channel_out,
            embed_out,
            gif_height,
            gif_width,
            encoder_dropout
        )

        self.decoder = Decoder(
            vocab_size,
            embed_out,
            decoder_num_layers,
            forward_expansion,
            attention_num_heads,
            decoder_dropout,
            output_max_len,
            layer_norm_ep
        )

    def causal_mask_fn(self, seq_len):
        mask = torch.tril(torch.ones(1, seq_len, seq_len)).unsqueeze(1).to(self.device)
        return mask

    
    def forward(self, gif, target_text):
        causal_mask = self.causal_mask_fn(target_text.shape[1])
        encoder_out = self.encoder(gif)
        decoder_out = self.decoder(target_text, encoder_out, causal_mask=causal_mask)
        return decoder_out


if __name__ == "__main__":
    vocab_size = 10
    embed_out = 100
    num_layers = 2
    forward_expansion = 4
    num_heads = 5
    dropout = 0.1
    max_len = 100

    num_res_blocks = 1
    channel_out = 8
    out_dims = 100
    height = 64
    width = 64
    dropout = 0.2

    

    input_encoder = torch.randn(4, 4, height, width, 10)
    encoder_model = Encoder(
        num_res_blocks,
        channel_out,
        out_dims,
        height, 
        width,
        dropout=0.2
        )

    input_words = torch.randint(0, 9, (4, 100))
    model = Decoder(
        vocab_size, 
        embed_out,
        num_layers,
        forward_expansion,
        num_heads,
        dropout,
        max_len
        )

    encoder_out = encoder_model(input_encoder)

    out = model(input_words, encoder_out)
    

    final_model = GifToCaption(
        encoder_num_res_blocks=1,
        gif_channel_out=8,
        gif_height=64, 
        gif_width=64,
        vocab_size=10, 
        embed_out=100,
        decoder_num_layers=2,
        forward_expansion=4,
        attention_num_heads=5,
        output_max_len=100,
        layer_norm_ep=1e-6,
        encoder_dropout=0.25,
        decoder_dropout=0.1,
    )

    out = final_model(input_encoder, input_words)
    print(out.shape)