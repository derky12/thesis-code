import torch
import torch.nn as nn
import torch.nn.functional as F
import configs
import atomInSmiles


class TranslationModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        vocab_path = config.vocab_path
        self.vocab = torch.load(vocab_path, weights_only=False)
        self.use_tanh = config.use_tanh
        self.ignore_vae = config.ignore_vae

        if self.use_tanh:
            assert self.ignore_vae, "--ignore_vae should be set along with --use_tanh"

        # Special symbols
        for ss in ("bos", "eos", "unk", "pad"):
            setattr(self, ss, getattr(self.vocab, ss))

        # Word embeddings layer
        n_vocab, d_emb = len(self.vocab), self.vocab.vectors.size(1)
        self.x_emb = nn.Embedding(n_vocab, d_emb, self.pad)
        self.x_emb.weight.data.copy_(self.vocab.vectors)
        if config.freeze_embeddings:
            self.x_emb.weight.requires_grad = False

        # Encoder
        if config.q_cell == "gru":
            self.encoder_rnn = nn.GRU(
                d_emb,
                config.q_d_h,
                num_layers=config.q_n_layers,
                batch_first=True,
                dropout=config.q_dropout if config.q_n_layers > 1 else 0,
                bidirectional=config.q_bidir,
            )
        else:
            raise ValueError("Invalid q_cell type, should be one of the ('gru',)")

        q_d_last = config.q_d_h * (2 if config.q_bidir else 1)

        # Decoder
        if config.d_cell == "gru":
            self.decoder_rnn = nn.GRU(
                d_emb,
                config.d_d_h,
                num_layers=config.d_n_layers,
                batch_first=True,
                dropout=config.d_dropout if config.d_n_layers > 1 else 0,
            )
        else:
            raise ValueError("Invalid d_cell type, should be one of the ('gru',)")

        self.decoder_fc = nn.Linear(config.d_d_h, n_vocab)

    @property
    def device(self):
        return next(self.parameters()).device

    def string2tensor(self, string, device="model"):
        ids = self.vocab.string2ids(string, add_bos=True, add_eos=True)
        tensor = torch.tensor(
            ids,
            dtype=torch.long,
            device=self.device if device == "model" else device,
        )

        return tensor

    def tensor2string(self, tensor):
        ids = tensor.tolist()
        string = self.vocab.ids2string(ids, rem_bos=True, rem_eos=True)

        return string

    def forward(self, randomized_smiles_list, canonical_smiles_list):

        canonical_smiles_tensor_list = [
            self.string2tensor(smiles) for smiles in canonical_smiles_list
        ]
        randomized_smiles_tensor_list = [
            self.string2tensor(smiles) for smiles in randomized_smiles_list
        ]

        # Encoder: x -> z, kl_loss
        h = self.forward_encoder(randomized_smiles_tensor_list)

        # Decoder: x, z -> recon_loss

        recon_loss = self.forward_decoder(canonical_smiles_tensor_list, h)

        # regression_loss = self.regression_loss(mu, logPs)

        return h, recon_loss

    def forward_encoder(self, x):
        x = [self.x_emb(i_x) for i_x in x]
        x = nn.utils.rnn.pack_sequence(x)

        _, h = self.encoder_rnn(x, None)

        h = h[-(1 + int(self.encoder_rnn.bidirectional)) :]
        h = torch.cat(h.split(1), dim=-1).squeeze(0)
        # print(f"{h.shape=}")

        return h  # h.shape = [batch_size, 1024]

    def forward_decoder(self, x, h):
        lengths = [len(i_x) for i_x in x]

        x = nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=self.pad)
        x_emb = self.x_emb(x)

        x_input = nn.utils.rnn.pack_padded_sequence(
            x_emb, lengths, batch_first=True, enforce_sorted=False
        )

        # Combine bidirectional hidden states and reshape for decoder GRU
        if self.encoder_rnn.bidirectional:
            h = h.view(h.size(0), 2, -1).sum(1)  # Combine bidirectional states
        h = h.unsqueeze(0).repeat(self.decoder_rnn.num_layers, 1, 1)
        output, _ = self.decoder_rnn(x_input, h)

        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        y = self.decoder_fc(output)

        recon_loss = F.cross_entropy(
            y[:, :-1].contiguous().view(-1, y.size(-1)),
            x[:, 1:].contiguous().view(-1),
            ignore_index=self.pad,
        )

        return recon_loss

    def generate_smiles(self, h, n_batch=1, max_len=100, temp=1.0):
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            if self.encoder_rnn.bidirectional:
                h = h.view(h.size(0), 2, -1).sum(1)  # Combine bidirectional states
                # print(f"0,{h.shape=}")

            # Reshape h for the decoder
            h = h.unsqueeze(0).repeat(self.decoder_rnn.num_layers, 1, 1)

            # Check if the shape of h matches the decoder's expected input
            expected_shape = (
                self.decoder_rnn.num_layers,
                n_batch,
                self.decoder_rnn.hidden_size,
            )
            assert (
                h.shape == expected_shape
            ), f"Expected h shape {expected_shape}, got {h.shape}"
            # print(f"1,{h.shape=}")

            generated_smiles = self.sample(n_batch, max_len, h, temp)
        self.train()  # Set the model back to training mode
        return generated_smiles

    def sample(self, n_batch, max_len=100, h=None, temp=1.0):
        """Generating n_batch samples in eval mode"""
        with torch.no_grad():
            w = torch.tensor(self.bos, device=self.device).repeat(n_batch)
            x = torch.tensor([self.pad], device=self.device).repeat(n_batch, max_len)
            x[:, 0] = self.bos
            end_pads = torch.tensor([max_len], device=self.device).repeat(n_batch)
            eos_mask = torch.zeros(n_batch, dtype=torch.bool, device=self.device)

            # Generating cycle
            for i in range(1, max_len):
                x_emb = self.x_emb(w).unsqueeze(1)
                o, h = self.decoder_rnn(x_emb, h)
                y = self.decoder_fc(o.squeeze(1))
                y = F.softmax(y / temp, dim=-1)

                w = torch.multinomial(y, 1)[:, 0]
                x[~eos_mask, i] = w[~eos_mask]
                i_eos_mask = ~eos_mask & (w == self.eos)
                end_pads[i_eos_mask] = i + 1
                eos_mask = eos_mask | i_eos_mask

            # Converting `x` to list of tensors
            new_x = []
            for i in range(x.size(0)):
                new_x.append(x[i, : end_pads[i]])

            ais_tokens_list = [self.tensor2string(i_x) for i_x in new_x]
            smi_list = []
            for ais_tokens in ais_tokens_list:
                try:
                    smi = atomInSmiles.decode(ais_tokens)
                    smi_list.append(smi)
                except:
                    smi_list.append("invalid")

            return smi_list
