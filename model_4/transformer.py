import torch
import torch.nn as nn
import math


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.src_vocab_size = self.config.src_vocab_size
        self.trg_vocab_size = self.config.trg_vocab_size
        self.d_model = self.config.d_model

        self.src_embedding = nn.Embedding(self.src_vocab_size, self.d_model)
        self.trg_embedding = nn.Embedding(self.trg_vocab_size, self.d_model)
        self.positional_encoder = PositionalEncoder(self.config)
        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config)
        self.output_linear = nn.Linear(self.d_model, self.trg_vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.attention_pooling = AttentionPooling(self.d_model)

    def forward(self, src_input, trg_input, e_mask=None, d_mask=None):
        e_output = self.forward_encoder(src_input, e_mask)
        d_output = self.forward_decoder(trg_input, e_output, e_mask, d_mask)

        output = self.softmax(
            self.output_linear(d_output)
        )  # (B, L, d_model) => # (B, L, trg_vocab_size)
        return output

    def forward_encoder(self, src_input, e_mask=None):
        src_input = self.src_embedding(src_input)  # (B, L) => (B, L, d_model)
        src_input = self.positional_encoder(
            src_input
        )  # (B, L, d_model) => (B, L, d_model)
        e_output = self.encoder(src_input, e_mask)  # (B, L, d_model)

        # Attention pooling over the sequence dimension
        _, e_output = self.attention_pooling(
            e_output
        )  # (B, L, d_model) => (B, d_model)

        return e_output

    def forward_decoder(self, trg_input, e_output, e_mask=None, d_mask=None):
        trg_input = self.trg_embedding(trg_input)  # (B, L) => (B, L, d_model)
        trg_input = self.positional_encoder(
            trg_input
        )  # (B, L, d_model) => (B, L, d_model)

        # Expand the encoder output to match the sequence length of the decoder input
        e_output = e_output.unsqueeze(1).expand(
            -1, trg_input.size(1), -1
        )  # (B, d_model) => (B, L, d_model)

        d_output = self.decoder(trg_input, e_output, e_mask, d_mask)  # (B, L, d_model)
        return d_output


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_layers = self.config.num_layers
        self.layers = nn.ModuleList(
            [EncoderLayer(self.config) for i in range(self.num_layers)]
        )
        self.layer_norm = LayerNormalization(self.config)

    def forward(self, x, e_mask):
        for i in range(self.num_layers):
            x = self.layers[i](x, e_mask)

        return self.layer_norm(x)


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_layers = self.config.num_layers
        self.layers = nn.ModuleList(
            [DecoderLayer(self.config) for i in range(self.num_layers)]
        )
        self.layer_norm = LayerNormalization(self.config)

    def forward(self, x, e_output, e_mask, d_mask):
        for i in range(self.num_layers):
            x = self.layers[i](x, e_output, e_mask, d_mask)

        return self.layer_norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.drop_out_rate = self.config.drop_out_rate
        self.layer_norm_1 = LayerNormalization(self.config)
        self.multihead_attention = MultiheadAttention(self.config)
        self.drop_out_1 = nn.Dropout(self.drop_out_rate)

        self.layer_norm_2 = LayerNormalization(self.config)
        self.feed_forward = FeedFowardLayer(self.config)
        self.drop_out_2 = nn.Dropout(self.drop_out_rate)

    def forward(self, x, e_mask):
        x_1 = self.layer_norm_1(x)  # (B, L, d_model)
        x = x + self.drop_out_1(
            self.multihead_attention(x_1, x_1, x_1, mask=e_mask)
        )  # (B, L, d_model)
        x_2 = self.layer_norm_2(x)  # (B, L, d_model)
        x = x + self.drop_out_2(self.feed_forward(x_2))  # (B, L, d_model)

        return x  # (B, L, d_model)


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.drop_out_rate = self.config.drop_out_rate
        self.layer_norm_1 = LayerNormalization(self.config)
        self.masked_multihead_attention = MultiheadAttention(self.config)
        self.drop_out_1 = nn.Dropout(self.drop_out_rate)

        self.layer_norm_2 = LayerNormalization(self.config)
        self.multihead_attention = MultiheadAttention(self.config)
        self.drop_out_2 = nn.Dropout(self.drop_out_rate)

        self.layer_norm_3 = LayerNormalization(self.config)
        self.feed_forward = FeedFowardLayer(self.config)
        self.drop_out_3 = nn.Dropout(self.drop_out_rate)

    def forward(self, x, e_output, e_mask, d_mask):
        x_1 = self.layer_norm_1(x)  # (B, L, d_model)
        x = x + self.drop_out_1(
            self.masked_multihead_attention(x_1, x_1, x_1, mask=d_mask)
        )  # (B, L, d_model)
        x_2 = self.layer_norm_2(x)  # (B, L, d_model)
        x = x + self.drop_out_2(
            self.multihead_attention(x_2, e_output, e_output, mask=e_mask)
        )  # (B, L, d_model)
        x_3 = self.layer_norm_3(x)  # (B, L, d_model)
        x = x + self.drop_out_3(self.feed_forward(x_3))  # (B, L, d_model)

        return x  # (B, L, d_model)


class MultiheadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.inf = 1e9
        self.config = config
        self.d_model = self.config.d_model
        self.num_heads = self.config.num_heads
        self.d_k = self.d_model // self.num_heads
        self.drop_out_rate = self.config.drop_out_rate

        # W^Q, W^K, W^V in the paper
        self.w_q = nn.Linear(self.d_model, self.d_model)
        self.w_k = nn.Linear(self.d_model, self.d_model)
        self.w_v = nn.Linear(self.d_model, self.d_model)

        self.dropout = nn.Dropout(self.drop_out_rate)
        self.attn_softmax = nn.Softmax(dim=-1)

        # Final output linear transformation
        self.w_0 = nn.Linear(self.d_model, self.d_model)

    def forward(self, q, k, v, mask=None):
        input_shape = q.shape

        # Linear calculation +  split into num_heads
        q = self.w_q(q).view(
            input_shape[0], -1, self.num_heads, self.d_k
        )  # (B, L, num_heads, d_k)
        k = self.w_k(k).view(
            input_shape[0], -1, self.num_heads, self.d_k
        )  # (B, L, num_heads, d_k)
        v = self.w_v(v).view(
            input_shape[0], -1, self.num_heads, self.d_k
        )  # (B, L, num_heads, d_k)

        # For convenience, convert all tensors in size (B, num_heads, L, d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Conduct self-attention
        attn_values = self.self_attention(q, k, v, mask=mask)  # (B, num_heads, L, d_k)
        concat_output = (
            attn_values.transpose(1, 2)
            .contiguous()
            .view(input_shape[0], -1, self.d_model)
        )  # (B, L, d_model)

        return self.w_0(concat_output)

    def self_attention(self, q, k, v, mask=None):
        # Calculate attention scores with scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1))  # (B, num_heads, L, L)
        attn_scores = attn_scores / math.sqrt(self.d_k)

        # If there is a mask, make masked spots -INF
        # if mask is not None:
        #     # mask = mask.unsqueeze(1)  # (B, 1, L) => (B, 1, 1, L) or (B, L, L) => (B, 1, L, L)
        #     print("before apply unsqueeze, mask shape:", mask.shape)
        #     print(f"{attn_scores.shape=}")
        #     mask = mask.unsqueeze(1).expand_as(
        #         attn_scores
        #     )  # (B, 1, L, L) -> (B, num_heads, L, L)
        #     print("after apply unsqueeze, mask shape:", mask.shape)
        #     attn_scores = attn_scores.masked_fill_(mask == 0, -1 * self.inf)

        if mask is not None:
            # # Print the shape of attn_scores before expanding the mask
            # print("attn_scores shape:", attn_scores.shape)
            # print("mask shape:", mask.shape)
            # Check the shape of attn_scores and expand the mask accordingly
            if attn_scores.shape[-1] == 1:
                # Case: attn_scores shape is (B, num_heads, L, 1)
                mask = mask.unsqueeze(-1).expand_as(
                    attn_scores
                )  # (B, 1, L) -> (B, num_heads, L, 1)
            else:
                # Case: attn_scores shape is (B, num_heads, L, L)
                mask = mask.unsqueeze(1).expand_as(
                    attn_scores
                )  # (B, L, L) -> (B, num_heads, L, L)

            attn_scores = attn_scores.masked_fill_(mask == 0, -1 * self.inf)

        # Softmax and multiplying K to calculate attention value
        attn_distribs = self.attn_softmax(attn_scores)

        attn_distribs = self.dropout(attn_distribs)
        attn_values = torch.matmul(attn_distribs, v)  # (B, num_heads, L, d_k)

        return attn_values


class FeedFowardLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = self.config.d_model
        self.d_ff = self.config.d_ff
        self.drop_out_rate = self.config.drop_out_rate
        self.linear_1 = nn.Linear(self.d_model, self.d_ff, bias=True)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(self.d_ff, self.d_model, bias=True)
        self.dropout = nn.Dropout(self.drop_out_rate)

    def forward(self, x):
        x = self.relu(self.linear_1(x))  # (B, L, d_ff)
        x = self.dropout(x)
        x = self.linear_2(x)  # (B, L, d_model)

        return x


class LayerNormalization(nn.Module):
    def __init__(self, config, eps=1e-6):
        super().__init__()
        self.config = config
        self.d_model = self.config.d_model
        self.eps = eps
        self.layer = nn.LayerNorm([self.d_model], elementwise_affine=True, eps=self.eps)

    def forward(self, x):
        x = self.layer(x)

        return x


class PositionalEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.seq_len = self.config.seq_len
        self.d_model = self.config.d_model
        self.device = self.config.device
        # Make initial positional encoding matrix with 0
        pe_matrix = torch.zeros(self.seq_len, self.d_model)  # (L, d_model)

        # Calculating position encoding values
        for pos in range(self.seq_len):
            for i in range(self.d_model):
                if i % 2 == 0:
                    pe_matrix[pos, i] = math.sin(
                        pos / (10000 ** (2 * i / self.d_model))
                    )
                elif i % 2 == 1:
                    pe_matrix[pos, i] = math.cos(
                        pos / (10000 ** (2 * i / self.d_model))
                    )

        pe_matrix = pe_matrix.unsqueeze(0)  # (1, L, d_model)
        self.positional_encoding = pe_matrix.to(device=self.device).requires_grad_(
            False
        )

    def forward(self, x):
        x = x * math.sqrt(self.d_model)  # (B, L, d_model)
        x = x + self.positional_encoding  # (B, L, d_model)

        return x


class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attention = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (B, L, d_model)
        attn_scores = self.attention(x)  # (B, L, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (B, L, 1)
        pooled_output = torch.sum(x * attn_weights, dim=1)  # (B, d_model)
        return attn_weights, pooled_output
